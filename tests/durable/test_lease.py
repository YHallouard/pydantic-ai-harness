"""Tests for EnvironmentActivities (acquire/release lease lifecycle).

Called directly as plain async functions -- `activity.defn` only attaches
registration metadata, so exercising the lifecycle doesn't require a running
Temporal worker. Two `EnvironmentActivities` instances sharing one
`GitSnapshotStore` simulate two pods for the convergence/fencing scenarios.
"""

from __future__ import annotations

import unittest.mock
from pathlib import Path

import anyio
import pytest
from temporalio.exceptions import ApplicationError

from pydantic_ai_harness.durable import (
    AcquireEnvParams,
    ForkEnvironmentParams,
    GitSnapshotStore,
    MergeEnvironmentParams,
)
from pydantic_ai_harness.durable.temporal import EnvironmentActivities, ReadEnvFileParams, WriteEnvFileParams

pytestmark = pytest.mark.anyio


class TestAcquireFresh:
    async def test_acquire_fences_and_restores_a_fresh_environment(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path / 'store')
        acts = EnvironmentActivities(store=store, env_queue='env-q1', workspaces_base=tmp_path / 'ws')

        lease = await acts.acquire_environment(AcquireEnvParams(env_id='env-1'))

        assert lease.env_id == 'env-1'
        assert lease.env_queue == 'env-q1'
        assert lease.epoch == 0
        assert 'env-1' in acts.held_env_ids
        assert (tmp_path / 'ws' / 'env-1').is_dir()

    async def test_second_env_gets_its_own_workspace(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path / 'store')
        acts = EnvironmentActivities(store=store, env_queue='env-q1', workspaces_base=tmp_path / 'ws')

        await acts.acquire_environment(AcquireEnvParams(env_id='env-1'))
        await acts.acquire_environment(AcquireEnvParams(env_id='env-2'))

        assert acts.held_env_ids == frozenset({'env-1', 'env-2'})


class TestReacquire:
    async def test_reacquire_same_worker_returns_same_lease_without_bumping_epoch(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path / 'store')
        acts = EnvironmentActivities(store=store, env_queue='env-q1', workspaces_base=tmp_path / 'ws')

        first = await acts.acquire_environment(AcquireEnvParams(env_id='env-1'))
        second = await acts.acquire_environment(AcquireEnvParams(env_id='env-1'))

        assert second == first

    async def test_reacquire_after_being_fenced_out_purges_local_state_and_converges(self, tmp_path: Path) -> None:
        """Simulates a schedule-to-start timeout that the worker didn't notice: another
        pod fenced env-1 out from under it. The next local acquire must detect the stale
        head via `is_current` and converge on the new owner rather than serving a stale lease."""
        store = GitSnapshotStore(tmp_path / 'store')
        pod_a = EnvironmentActivities(store=store, env_queue='env-q-a', workspaces_base=tmp_path / 'ws-a')
        pod_b = EnvironmentActivities(store=store, env_queue='env-q-b', workspaces_base=tmp_path / 'ws-b')

        await pod_a.acquire_environment(AcquireEnvParams(env_id='env-1'))
        # pod_b re-provisions env-1, fencing pod_a out (failed_queue tells it env-q-a just failed).
        b_lease = await pod_b.acquire_environment(AcquireEnvParams(env_id='env-1', failed_queue='env-q-a'))
        assert b_lease.env_queue == 'env-q-b'

        # pod_a, unaware it was fenced, tries to reacquire locally.
        reacquired = await pod_a.acquire_environment(AcquireEnvParams(env_id='env-1'))
        assert reacquired.env_queue == 'env-q-b'
        assert 'env-1' not in pod_a.held_env_ids

    async def test_reacquire_after_fence_out_keeps_workspace_for_warm_restore(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The stale re-acquire path drops the in-memory hold but leaves the workspace on disk,
        so a later restore converges it warm rather than rebuilding from scratch."""
        store = GitSnapshotStore(tmp_path / 'store')
        pod_a = EnvironmentActivities(store=store, env_queue='env-q-a', workspaces_base=tmp_path / 'ws-a')
        pod_b = EnvironmentActivities(store=store, env_queue='env-q-b', workspaces_base=tmp_path / 'ws-b')

        await pod_a.acquire_environment(AcquireEnvParams(env_id='env-1'))
        await pod_b.acquire_environment(AcquireEnvParams(env_id='env-1', failed_queue='env-q-a'))

        discarded: list[Path] = []

        async def _spy(workspace: Path) -> None:
            discarded.append(workspace)

        monkeypatch.setattr(store, 'discard_workspace', _spy)
        await pod_a.acquire_environment(AcquireEnvParams(env_id='env-1'))
        assert discarded == []
        assert (tmp_path / 'ws-a' / 'env-1').exists()


class TestConvergence:
    async def test_second_acquirer_converges_on_live_lease_instead_of_fencing(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path / 'store')
        pod_a = EnvironmentActivities(store=store, env_queue='env-q-a', workspaces_base=tmp_path / 'ws-a')
        pod_b = EnvironmentActivities(store=store, env_queue='env-q-b', workspaces_base=tmp_path / 'ws-b')

        a_lease = await pod_a.acquire_environment(AcquireEnvParams(env_id='env-1'))
        # pod_b has no reason to believe env-q-a failed (failed_queue=None): it must
        # converge on pod_a's lease rather than fencing it out.
        b_lease = await pod_b.acquire_environment(AcquireEnvParams(env_id='env-1'))

        assert b_lease == a_lease
        assert 'env-1' not in pod_b.held_env_ids

    async def test_failed_queue_matching_current_owner_still_fences(self, tmp_path: Path) -> None:
        """If the only live lease *is* the one the caller just saw fail, converging
        would hand back a dead queue forever -- fence a new one instead."""
        store = GitSnapshotStore(tmp_path / 'store')
        pod_a = EnvironmentActivities(store=store, env_queue='env-q-a', workspaces_base=tmp_path / 'ws-a')
        pod_b = EnvironmentActivities(store=store, env_queue='env-q-b', workspaces_base=tmp_path / 'ws-b')

        await pod_a.acquire_environment(AcquireEnvParams(env_id='env-1'))
        b_lease = await pod_b.acquire_environment(AcquireEnvParams(env_id='env-1', failed_queue='env-q-a'))

        assert b_lease.env_queue == 'env-q-b'
        assert 'env-1' in pod_b.held_env_ids


class TestBusyBounce:
    async def test_acquire_beyond_capacity_raises_retryable_application_error(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path / 'store')
        acts = EnvironmentActivities(
            store=store, env_queue='env-q1', workspaces_base=tmp_path / 'ws', max_concurrent_environments=1
        )
        await acts.acquire_environment(AcquireEnvParams(env_id='env-1'))

        with pytest.raises(ApplicationError) as exc_info:
            await acts.acquire_environment(AcquireEnvParams(env_id='env-2'))
        assert exc_info.value.non_retryable is False

    async def test_capacity_check_does_not_block_reacquiring_an_already_held_env(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path / 'store')
        acts = EnvironmentActivities(
            store=store, env_queue='env-q1', workspaces_base=tmp_path / 'ws', max_concurrent_environments=1
        )
        await acts.acquire_environment(AcquireEnvParams(env_id='env-1'))
        # Re-acquiring the same env_id must not count as a new claim against capacity.
        again = await acts.acquire_environment(AcquireEnvParams(env_id='env-1'))
        assert again.env_id == 'env-1'


class TestRelease:
    async def test_release_pushes_final_snapshot_and_forgets_local_state(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path / 'store')
        acts = EnvironmentActivities(store=store, env_queue='env-q1', workspaces_base=tmp_path / 'ws')
        await acts.acquire_environment(AcquireEnvParams(env_id='env-1'))
        (tmp_path / 'ws' / 'env-1' / 'note.txt').write_text('hi', encoding='utf-8')

        await acts.release_environment('env-1')

        assert 'env-1' not in acts.held_env_ids
        assert not (tmp_path / 'ws' / 'env-1').exists()
        assert await store.get_lease('env-1') is None

    async def test_release_then_acquire_fences_a_genuinely_new_lease(self, tmp_path: Path) -> None:
        """`release` erases the lease record entirely, so epoch (informational only --
        the store's ref-CAS is the real fencing mechanism) restarts at 0 rather than
        continuing to climb; what matters is it's a fresh fence, not a converged one."""
        store = GitSnapshotStore(tmp_path / 'store')
        acts = EnvironmentActivities(store=store, env_queue='env-q1', workspaces_base=tmp_path / 'ws')
        await acts.acquire_environment(AcquireEnvParams(env_id='env-1'))
        await acts.release_environment('env-1')

        second = await acts.acquire_environment(AcquireEnvParams(env_id='env-1'))
        assert second.epoch == 0
        assert 'env-1' in acts.held_env_ids

    async def test_release_of_unheld_env_is_a_no_op(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path / 'store')
        acts = EnvironmentActivities(store=store, env_queue='env-q1', workspaces_base=tmp_path / 'ws')
        await acts.release_environment('never-acquired')
        assert acts.held_env_ids == frozenset()

    async def test_released_workspace_snapshot_is_preserved_for_next_restore(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path / 'store')
        acts = EnvironmentActivities(store=store, env_queue='env-q1', workspaces_base=tmp_path / 'ws')
        await acts.acquire_environment(AcquireEnvParams(env_id='env-1'))
        (tmp_path / 'ws' / 'env-1' / 'note.txt').write_text('preserved', encoding='utf-8')
        await acts.release_environment('env-1')

        restored = tmp_path / 'restored'
        await store.restore('env-1', restored)
        assert (restored / 'note.txt').read_text(encoding='utf-8') == 'preserved'

    async def test_reacquire_after_release_still_discards_the_workspace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unlike the stale re-acquire path, a release is a deliberate final give-up (a root env
        releases once at workflow completion; a branch child's one-shot env_id is essentially
        never re-acquired) -- so, unlike that path, release still discards the workspace: warm
        retention here would never pay off and would grow disk unboundedly per delegation."""
        store = GitSnapshotStore(tmp_path / 'store')
        acts = EnvironmentActivities(store=store, env_queue='env-q1', workspaces_base=tmp_path / 'ws')
        await acts.acquire_environment(AcquireEnvParams(env_id='env-1'))
        (tmp_path / 'ws' / 'env-1' / 'note.txt').write_text('kept', encoding='utf-8')

        discarded: list[Path] = []
        original = store.discard_workspace

        async def _spy(workspace: Path) -> None:
            discarded.append(workspace)
            await original(workspace)

        monkeypatch.setattr(store, 'discard_workspace', _spy)
        await acts.release_environment('env-1')
        assert discarded == [tmp_path / 'ws' / 'env-1']


class TestSnapshotHeld:
    async def test_pushes_a_snapshot_without_releasing_the_lease(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path / 'store')
        acts = EnvironmentActivities(store=store, env_queue='env-q1', workspaces_base=tmp_path / 'ws')
        await acts.acquire_environment(AcquireEnvParams(env_id='env-1'))
        (tmp_path / 'ws' / 'env-1' / 'note.txt').write_text('mid-flight', encoding='utf-8')

        await acts.snapshot_held('env-1')

        assert 'env-1' in acts.held_env_ids
        assert await store.get_lease('env-1') is not None
        restored = tmp_path / 'restored'
        await store.restore('env-1', restored)
        assert (restored / 'note.txt').read_text(encoding='utf-8') == 'mid-flight'

    async def test_is_a_no_op_for_an_env_this_worker_does_not_hold(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path / 'store')
        acts = EnvironmentActivities(store=store, env_queue='env-q1', workspaces_base=tmp_path / 'ws')

        await acts.snapshot_held('never-acquired')

        assert acts.held_env_ids == frozenset()
        assert await store.get_lease('never-acquired') is None


class TestForkEnvironment:
    async def test_fork_makes_child_restorable_with_parent_content(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path / 'store')
        acts = EnvironmentActivities(store=store, env_queue='env-q1', workspaces_base=tmp_path / 'ws')
        await acts.acquire_environment(AcquireEnvParams(env_id='parent'))
        (tmp_path / 'ws' / 'parent' / 'shared.txt').write_text('from-parent', encoding='utf-8')
        await acts.snapshot_held('parent')

        await acts.fork_environment(ForkEnvironmentParams(parent_env_id='parent', child_env_id='child-1'))

        child_lease = await acts.acquire_environment(AcquireEnvParams(env_id='child-1'))
        assert child_lease.env_id == 'child-1'
        assert (tmp_path / 'ws' / 'child-1' / 'shared.txt').read_text(encoding='utf-8') == 'from-parent'


class TestMergeEnvironment:
    async def test_merge_of_unheld_env_raises_non_retryable(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path / 'store')
        acts = EnvironmentActivities(store=store, env_queue='env-q1', workspaces_base=tmp_path / 'ws')

        with pytest.raises(ApplicationError) as exc_info:
            await acts.merge_environment(
                MergeEnvironmentParams(held_env_id='parent', other_env_id='child-1', mode='land')
            )
        assert exc_info.value.non_retryable is True

    async def test_land_merges_child_work_into_the_live_held_parent_workspace(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path / 'store')
        acts = EnvironmentActivities(store=store, env_queue='env-q1', workspaces_base=tmp_path / 'ws')
        await acts.acquire_environment(AcquireEnvParams(env_id='parent'))
        (tmp_path / 'ws' / 'parent' / 'shared.txt').write_text('v1', encoding='utf-8')
        await acts.snapshot_held('parent')
        await acts.fork_environment(ForkEnvironmentParams(parent_env_id='parent', child_env_id='child-1'))
        await acts.acquire_environment(AcquireEnvParams(env_id='child-1'))
        (tmp_path / 'ws' / 'child-1' / 'from-child.txt').write_text('child work', encoding='utf-8')
        await acts.snapshot_held('child-1')

        result = await acts.merge_environment(
            MergeEnvironmentParams(held_env_id='parent', other_env_id='child-1', mode='land')
        )

        assert result.conflicts == []
        # The live held workspace has the merged content in place -- no re-restore needed.
        assert (tmp_path / 'ws' / 'parent' / 'from-child.txt').read_text(encoding='utf-8') == 'child work'

    async def test_land_conflict_leaves_held_parent_workspace_untouched(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path / 'store')
        acts = EnvironmentActivities(store=store, env_queue='env-q1', workspaces_base=tmp_path / 'ws')
        await acts.acquire_environment(AcquireEnvParams(env_id='parent'))
        (tmp_path / 'ws' / 'parent' / 'shared.txt').write_text('base', encoding='utf-8')
        await acts.snapshot_held('parent')
        await acts.fork_environment(ForkEnvironmentParams(parent_env_id='parent', child_env_id='child-1'))
        await acts.acquire_environment(AcquireEnvParams(env_id='child-1'))
        (tmp_path / 'ws' / 'child-1' / 'shared.txt').write_text('from-child', encoding='utf-8')
        await acts.snapshot_held('child-1')

        (tmp_path / 'ws' / 'parent' / 'shared.txt').write_text('from-parent', encoding='utf-8')
        await acts.snapshot_held('parent')

        result = await acts.merge_environment(
            MergeEnvironmentParams(held_env_id='parent', other_env_id='child-1', mode='land')
        )

        assert result.conflicts == ['shared.txt']
        assert (tmp_path / 'ws' / 'parent' / 'shared.txt').read_text(encoding='utf-8') == 'from-parent'

    async def test_materialize_conflict_commits_markers_into_the_live_held_child_workspace(
        self, tmp_path: Path
    ) -> None:
        store = GitSnapshotStore(tmp_path / 'store')
        acts = EnvironmentActivities(store=store, env_queue='env-q1', workspaces_base=tmp_path / 'ws')
        await acts.acquire_environment(AcquireEnvParams(env_id='parent'))
        (tmp_path / 'ws' / 'parent' / 'shared.txt').write_text('base', encoding='utf-8')
        await acts.snapshot_held('parent')
        await acts.fork_environment(ForkEnvironmentParams(parent_env_id='parent', child_env_id='child-1'))
        await acts.acquire_environment(AcquireEnvParams(env_id='child-1'))
        (tmp_path / 'ws' / 'child-1' / 'shared.txt').write_text('from-child', encoding='utf-8')
        await acts.snapshot_held('child-1')

        (tmp_path / 'ws' / 'parent' / 'shared.txt').write_text('from-parent', encoding='utf-8')
        await acts.snapshot_held('parent')

        result = await acts.merge_environment(
            MergeEnvironmentParams(held_env_id='child-1', other_env_id='parent', mode='materialize')
        )

        assert result.conflicts == ['shared.txt']
        content = (tmp_path / 'ws' / 'child-1' / 'shared.txt').read_text(encoding='utf-8')
        assert '<<<<<<<' in content
        assert '>>>>>>>' in content


class TestGetEnvironmentQueue:
    async def test_returns_the_queue_holding_a_fenced_environment(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path / 'store')
        acts = EnvironmentActivities(store=store, env_queue='env-q1', workspaces_base=tmp_path / 'ws')
        await acts.acquire_environment(AcquireEnvParams(env_id='env-1'))

        queue = await acts.get_environment_queue('env-1')

        assert queue == 'env-q1'

    async def test_returns_none_for_an_environment_never_acquired(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path / 'store')
        acts = EnvironmentActivities(store=store, env_queue='env-q1', workspaces_base=tmp_path / 'ws')

        assert await acts.get_environment_queue('never-acquired') is None

    async def test_reads_the_store_not_local_process_state(self, tmp_path: Path) -> None:
        """Works from any worker that shares the store -- it's a fence-record read, not
        a lookup in this process's own `_held`."""
        store = GitSnapshotStore(tmp_path / 'store')
        pod_a = EnvironmentActivities(store=store, env_queue='env-q-a', workspaces_base=tmp_path / 'ws-a')
        pod_b = EnvironmentActivities(store=store, env_queue='env-q-b', workspaces_base=tmp_path / 'ws-b')
        await pod_a.acquire_environment(AcquireEnvParams(env_id='env-1'))

        assert await pod_b.get_environment_queue('env-1') == 'env-q-a'


class TestWriteEnvironmentFile:
    async def test_writes_a_file_into_the_held_workspace(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path / 'store')
        acts = EnvironmentActivities(store=store, env_queue='env-q1', workspaces_base=tmp_path / 'ws')
        await acts.acquire_environment(AcquireEnvParams(env_id='env-1'))

        result = await acts.write_environment_file(
            WriteEnvFileParams(env_id='env-1', path='note.txt', content='hello', op_id='update-1')
        )

        assert 'note.txt' in result
        assert (tmp_path / 'ws' / 'env-1' / 'note.txt').read_text(encoding='utf-8') == 'hello'

    async def test_write_is_snapshotted_like_a_normal_mutating_op(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path / 'store')
        acts = EnvironmentActivities(store=store, env_queue='env-q1', workspaces_base=tmp_path / 'ws')
        await acts.acquire_environment(AcquireEnvParams(env_id='env-1'))

        await acts.write_environment_file(
            WriteEnvFileParams(env_id='env-1', path='note.txt', content='hello', op_id='update-1')
        )

        restored = tmp_path / 'restored'
        await store.restore('env-1', restored)
        assert (restored / 'note.txt').read_text(encoding='utf-8') == 'hello'

    async def test_retry_with_same_op_id_does_not_reapply(self, tmp_path: Path) -> None:
        """A Temporal activity retry replays the same op_id -- the journal must return
        the original result rather than re-running the write with whatever (possibly
        different) params the retry happened to carry."""
        store = GitSnapshotStore(tmp_path / 'store')
        acts = EnvironmentActivities(store=store, env_queue='env-q1', workspaces_base=tmp_path / 'ws')
        await acts.acquire_environment(AcquireEnvParams(env_id='env-1'))

        first = await acts.write_environment_file(
            WriteEnvFileParams(env_id='env-1', path='note.txt', content='first', op_id='update-1')
        )
        second = await acts.write_environment_file(
            WriteEnvFileParams(env_id='env-1', path='note.txt', content='second', op_id='update-1')
        )

        assert second == first
        assert (tmp_path / 'ws' / 'env-1' / 'note.txt').read_text(encoding='utf-8') == 'first'

    async def test_raises_non_retryable_when_this_worker_does_not_hold_the_env(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path / 'store')
        acts = EnvironmentActivities(store=store, env_queue='env-q1', workspaces_base=tmp_path / 'ws')

        with pytest.raises(ApplicationError) as exc_info:
            await acts.write_environment_file(
                WriteEnvFileParams(env_id='never-acquired', path='note.txt', content='x', op_id='update-1')
            )
        assert exc_info.value.non_retryable is True

    async def test_rejects_path_traversal_above_the_workspace(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path / 'store')
        acts = EnvironmentActivities(store=store, env_queue='env-q1', workspaces_base=tmp_path / 'ws')
        await acts.acquire_environment(AcquireEnvParams(env_id='env-1'))

        with pytest.raises(ApplicationError) as exc_info:
            await acts.write_environment_file(
                WriteEnvFileParams(env_id='env-1', path='../outside.txt', content='x', op_id='update-1')
            )
        assert exc_info.value.non_retryable is True
        assert not (tmp_path / 'ws' / 'outside.txt').exists()

    async def test_concurrent_writes_to_the_same_held_workspace_are_serialized(self, tmp_path: Path) -> None:
        """Same `env_lock` a mutating tool call takes -- two external writes into one
        held workspace must not interleave."""
        store = GitSnapshotStore(tmp_path / 'store')
        acts = EnvironmentActivities(store=store, env_queue='env-q1', workspaces_base=tmp_path / 'ws')
        await acts.acquire_environment(AcquireEnvParams(env_id='env-1'))
        events: list[str] = []
        started = anyio.Event()

        real_write_text = Path.write_text

        def _slow_write_text(self: Path, *args: object, **kwargs: object) -> int:
            if self.name == 'slow.txt':
                events.append('slow-start')
                started.set()
            result = real_write_text(self, *args, **kwargs)  # type: ignore[arg-type]
            if self.name == 'slow.txt':
                events.append('slow-end')
            return result

        async def _write_slow() -> None:
            await acts.write_environment_file(
                WriteEnvFileParams(env_id='env-1', path='slow.txt', content='a', op_id='op-a')
            )

        async def _write_fast() -> None:
            await started.wait()
            events.append('fast-start')
            await acts.write_environment_file(
                WriteEnvFileParams(env_id='env-1', path='fast.txt', content='b', op_id='op-b')
            )
            events.append('fast-end')

        with unittest.mock.patch.object(Path, 'write_text', _slow_write_text):
            async with anyio.create_task_group() as tg:
                tg.start_soon(_write_slow)
                tg.start_soon(_write_fast)

        # fast's write must wait for slow's entire critical section (same env_lock).
        assert events.index('slow-end') < events.index('fast-end')


class TestReadEnvironmentFile:
    async def test_reads_a_file_from_the_held_workspace(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path / 'store')
        acts = EnvironmentActivities(store=store, env_queue='env-q1', workspaces_base=tmp_path / 'ws')
        await acts.acquire_environment(AcquireEnvParams(env_id='env-1'))
        (tmp_path / 'ws' / 'env-1' / 'note.txt').write_text('hello', encoding='utf-8')

        content = await acts.read_environment_file(ReadEnvFileParams(env_id='env-1', path='note.txt'))

        assert content == 'hello'

    async def test_sees_a_write_made_immediately_before_it(self, tmp_path: Path) -> None:
        """Read-after-write consistency within one held workspace: a write via
        write_environment_file followed by a read must see that write's content."""
        store = GitSnapshotStore(tmp_path / 'store')
        acts = EnvironmentActivities(store=store, env_queue='env-q1', workspaces_base=tmp_path / 'ws')
        await acts.acquire_environment(AcquireEnvParams(env_id='env-1'))

        await acts.write_environment_file(
            WriteEnvFileParams(env_id='env-1', path='note.txt', content='fresh', op_id='update-1')
        )
        content = await acts.read_environment_file(ReadEnvFileParams(env_id='env-1', path='note.txt'))

        assert content == 'fresh'

    async def test_raises_non_retryable_when_this_worker_does_not_hold_the_env(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path / 'store')
        acts = EnvironmentActivities(store=store, env_queue='env-q1', workspaces_base=tmp_path / 'ws')

        with pytest.raises(ApplicationError) as exc_info:
            await acts.read_environment_file(ReadEnvFileParams(env_id='never-acquired', path='note.txt'))
        assert exc_info.value.non_retryable is True

    async def test_raises_non_retryable_when_the_file_does_not_exist(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path / 'store')
        acts = EnvironmentActivities(store=store, env_queue='env-q1', workspaces_base=tmp_path / 'ws')
        await acts.acquire_environment(AcquireEnvParams(env_id='env-1'))

        with pytest.raises(ApplicationError) as exc_info:
            await acts.read_environment_file(ReadEnvFileParams(env_id='env-1', path='missing.txt'))
        assert exc_info.value.non_retryable is True

    async def test_rejects_path_traversal_above_the_workspace(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path / 'store')
        acts = EnvironmentActivities(store=store, env_queue='env-q1', workspaces_base=tmp_path / 'ws')
        await acts.acquire_environment(AcquireEnvParams(env_id='env-1'))

        with pytest.raises(ApplicationError) as exc_info:
            await acts.read_environment_file(ReadEnvFileParams(env_id='env-1', path='../outside.txt'))
        assert exc_info.value.non_retryable is True
