"""Tests for EnvironmentActivities (acquire/release lease lifecycle).

Called directly as plain async functions -- `activity.defn` only attaches
registration metadata, so exercising the lifecycle doesn't require a running
Temporal worker. Two `EnvironmentActivities` instances sharing one
`GitSnapshotStore` simulate two pods for the convergence/fencing scenarios.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from temporalio.exceptions import ApplicationError

from pydantic_ai_harness.durable import (
    AcquireEnvParams,
    ForkEnvironmentParams,
    GitSnapshotStore,
    MergeEnvironmentParams,
)
from pydantic_ai_harness.durable.temporal import EnvironmentActivities

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
