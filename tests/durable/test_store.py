"""Tests for GitSnapshotStore -- real git subprocess, no mocking."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import anyio
import pytest

from pydantic_ai_harness.durable import EnvironmentLease, FenceConflict, GitSnapshotStore, LeaseRecord, SnapshotRejected

pytestmark = pytest.mark.anyio


class TestFence:
    async def test_fence_creates_lease_record(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path)
        head = await store.fence('env-1', queue='env-queue-1')
        assert head.epoch == 0

        record = await store.get_lease('env-1')
        assert record is not None
        assert record.env_id == 'env-1'
        assert record.env_queue == 'env-queue-1'
        assert record.epoch == 0

    async def test_second_fence_bumps_epoch_and_moves_head(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path)
        first = await store.fence('env-1', queue='q1')
        second = await store.fence('env-1', queue='q2')
        assert second.epoch == first.epoch + 1
        assert second.sha != first.sha

        record = await store.get_lease('env-1')
        assert record is not None
        assert record.env_queue == 'q2'

    async def test_two_concurrent_fences_only_one_wins(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path)
        await store.fence('env-1', queue='seed')

        outcomes: list[str] = []

        async def _fence(queue: str) -> None:
            try:
                await store.fence('env-1', queue=queue)
                outcomes.append('ok')
            except FenceConflict:
                outcomes.append('conflict')

        async with anyio.create_task_group() as tg:
            tg.start_soon(_fence, 'q-a')
            tg.start_soon(_fence, 'q-b')

        assert sorted(outcomes) == ['conflict', 'ok']

    async def test_is_current_reflects_latest_fence(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path)
        head = await store.fence('env-1', queue='q1')
        assert await store.is_current('env-1', head.sha) is True

        new_head = await store.fence('env-1', queue='q2')
        assert await store.is_current('env-1', head.sha) is False
        assert await store.is_current('env-1', new_head.sha) is True

    async def test_get_lease_returns_none_for_unknown_env(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path)
        assert await store.get_lease('never-fenced') is None

    async def test_release_erases_lease_record_without_touching_snapshot(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path)
        await store.fence('env-1', queue='q1')
        workspace = tmp_path / 'workspace'
        await store.restore('env-1', workspace)
        (workspace / 'f.txt').write_text('content', encoding='utf-8')
        await store.push('env-1', workspace)

        await store.release('env-1')
        assert await store.get_lease('env-1') is None

        restored = tmp_path / 'restored'
        await store.restore('env-1', restored)
        assert (restored / 'f.txt').read_text(encoding='utf-8') == 'content'

    async def test_release_of_unknown_env_is_a_no_op(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path)
        await store.release('never-fenced')
        assert await store.get_lease('never-fenced') is None


class TestLeaseRecord:
    def test_to_lease_projects_the_public_lease_shape(self) -> None:
        record = LeaseRecord(env_id='env-1', env_queue='env-queue-1', epoch=2, fenced_at=datetime.now(timezone.utc))
        assert record.to_lease() == EnvironmentLease(env_id='env-1', env_queue='env-queue-1', epoch=2)


class TestPushRestore:
    async def test_push_then_restore_round_trips_workspace_contents(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path)
        await store.fence('env-1', queue='q1')

        workspace = tmp_path / 'workspace'
        await store.restore('env-1', workspace)
        (workspace / 'notes.txt').write_text('hello', encoding='utf-8')
        await store.push('env-1', workspace)

        restored = tmp_path / 'restored'
        await store.restore('env-1', restored)
        assert (restored / 'notes.txt').read_text(encoding='utf-8') == 'hello'

    async def test_restore_before_any_push_yields_empty_workspace(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path)
        await store.fence('env-1', queue='q1')

        workspace = tmp_path / 'workspace'
        await store.restore('env-1', workspace)
        # The workspace holds only what the agent puts there -- the git dir is a sibling.
        assert list(workspace.iterdir()) == []
        assert not (workspace / '.git').exists()

    async def test_restore_of_never_fenced_env_yields_empty_workspace(self, tmp_path: Path) -> None:
        """Restoring an env_id that was never fenced (no branch ref at all) is a no-op
        beyond creating the sibling git dir for a freshly created (empty) bare repo."""
        store = GitSnapshotStore(tmp_path)
        workspace = tmp_path / 'workspace'
        await store.restore('never-fenced', workspace)
        assert list(workspace.iterdir()) == []
        assert not (workspace / '.git').exists()

    async def test_restore_into_leftover_workspace_is_idempotent(self, tmp_path: Path) -> None:
        """A dead pod can leave `into` + sibling git dir without discard_workspace; restore
        must wipe and re-materialize instead of failing on `remote origin already exists`."""
        store = GitSnapshotStore(tmp_path)
        await store.fence('env-1', queue='q1')

        workspace = tmp_path / 'workspace'
        await store.restore('env-1', workspace)
        (workspace / 'notes.txt').write_text('hello', encoding='utf-8')
        await store.push('env-1', workspace)

        (workspace / 'stale.txt').write_text('should-be-wiped', encoding='utf-8')
        await store.restore('env-1', workspace)
        assert (workspace / 'notes.txt').read_text(encoding='utf-8') == 'hello'
        assert not (workspace / 'stale.txt').exists()

    async def test_push_rejected_after_fence_invalidates_stale_workspace(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path)
        await store.fence('env-1', queue='q1')

        workspace = tmp_path / 'workspace'
        await store.restore('env-1', workspace)
        (workspace / 'a.txt').write_text('a', encoding='utf-8')
        await store.push('env-1', workspace)

        stale = tmp_path / 'stale'
        await store.restore('env-1', stale)

        # A new owner fences (simulating re-provisioning on another pod) and pushes its own snapshot.
        await store.fence('env-1', queue='q2')
        fresh = tmp_path / 'fresh'
        await store.restore('env-1', fresh)
        (fresh / 'b.txt').write_text('b', encoding='utf-8')
        await store.push('env-1', fresh)

        # The stale workspace, based on the pre-fence head, can no longer push.
        (stale / 'c.txt').write_text('c', encoding='utf-8')
        with pytest.raises(SnapshotRejected):
            await store.push('env-1', stale)


class TestWarmRestore:
    async def test_warm_reuse_converges_to_head_pushed_elsewhere(self, tmp_path: Path) -> None:
        """A workspace still on disk, re-restored after another pod pushed a new head, converges
        to that head (tracked files updated, new files appear) and drops untracked leftovers."""
        store = GitSnapshotStore(tmp_path)
        await store.fence('env-1', queue='q1')

        local = tmp_path / 'local'
        await store.restore('env-1', local)
        (local / 'keep.txt').write_text('v1', encoding='utf-8')
        await store.push('env-1', local)

        # Another pod fences and pushes a newer head.
        await store.fence('env-1', queue='q2')
        other = tmp_path / 'other'
        await store.restore('env-1', other)
        (other / 'keep.txt').write_text('v2', encoding='utf-8')
        (other / 'added.txt').write_text('new', encoding='utf-8')
        await store.push('env-1', other)

        # Warm re-restore into the original workspace (never discarded).
        (local / 'untracked.txt').write_text('stale', encoding='utf-8')
        await store.restore('env-1', local)
        assert (local / 'keep.txt').read_text(encoding='utf-8') == 'v2'
        assert (local / 'added.txt').read_text(encoding='utf-8') == 'new'
        assert not (local / 'untracked.txt').exists()

    async def test_warm_reuse_does_not_wipe_the_workspace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The warm path fetches the delta in place: `discard_workspace` is never called."""
        store = GitSnapshotStore(tmp_path)
        await store.fence('env-1', queue='q1')
        local = tmp_path / 'local'
        await store.restore('env-1', local)
        (local / 'keep.txt').write_text('hello', encoding='utf-8')
        await store.push('env-1', local)

        calls: list[Path] = []
        original = store.discard_workspace

        async def _spy(workspace: Path) -> None:
            calls.append(workspace)
            await original(workspace)

        monkeypatch.setattr(store, 'discard_workspace', _spy)
        await store.restore('env-1', local)
        assert calls == []
        assert (local / 'keep.txt').read_text(encoding='utf-8') == 'hello'

    async def test_warm_reuse_overwrites_uncommitted_local_changes(self, tmp_path: Path) -> None:
        """The bare repo head is the authority: a warm workspace's un-pushed edits are discarded."""
        store = GitSnapshotStore(tmp_path)
        await store.fence('env-1', queue='q1')
        local = tmp_path / 'local'
        await store.restore('env-1', local)
        (local / 'f.txt').write_text('committed', encoding='utf-8')
        await store.push('env-1', local)

        (local / 'f.txt').write_text('uncommitted-edit', encoding='utf-8')
        await store.restore('env-1', local)
        assert (local / 'f.txt').read_text(encoding='utf-8') == 'committed'

    async def test_foreign_origin_falls_back_to_cold_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A workspace whose sibling git dir points at a different env's repo is not warm:
        restore wipes and re-materializes from the right repo."""
        store = GitSnapshotStore(tmp_path)
        await store.fence('env-a', queue='q1')
        await store.fence('env-b', queue='q1')

        shared = tmp_path / 'shared'
        await store.restore('env-a', shared)
        (shared / 'a.txt').write_text('a', encoding='utf-8')
        await store.push('env-a', shared)
        await store.restore('env-a', shared)  # settle origin -> env-a's repo

        calls: list[Path] = []
        original = store.discard_workspace

        async def _spy(workspace: Path) -> None:
            calls.append(workspace)
            await original(workspace)

        monkeypatch.setattr(store, 'discard_workspace', _spy)
        # Restoring env-b into the same path: origin points at env-a -> cold fallback.
        await store.restore('env-b', shared)
        assert calls == [shared]
        assert not (shared / 'a.txt').exists()

    async def test_warm_workspace_of_never_fenced_env_converges_to_empty(self, tmp_path: Path) -> None:
        """A warm workspace whose env has no head (never fenced) converges to empty, leaving no
        orphan git dir behind."""
        store = GitSnapshotStore(tmp_path)
        local = tmp_path / 'local'
        # First restore is cold: creates the sibling git dir, no head to check out.
        await store.restore('never-fenced', local)
        (local / 'scratch.txt').write_text('x', encoding='utf-8')

        # Second restore is warm (git dir + matching origin) but current head is still None.
        await store.restore('never-fenced', local)
        assert list(local.iterdir()) == []
        assert not (tmp_path / '.local.git').exists()


class TestFork:
    async def test_fork_child_starts_from_parent_snapshot(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path)
        await store.fence('parent', queue='q1')
        parent_ws = tmp_path / 'parent_ws'
        await store.restore('parent', parent_ws)
        (parent_ws / 'shared.txt').write_text('from-parent', encoding='utf-8')
        await store.push('parent', parent_ws)

        await store.fork('parent', 'child-1')
        child_ws = tmp_path / 'child_ws'
        await store.restore('child-1', child_ws)
        assert (child_ws / 'shared.txt').read_text(encoding='utf-8') == 'from-parent'

    async def test_fork_child_and_parent_snapshots_are_independent(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path)
        await store.fence('parent', queue='q1')
        parent_ws = tmp_path / 'parent_ws'
        await store.restore('parent', parent_ws)
        (parent_ws / 'shared.txt').write_text('v1', encoding='utf-8')
        await store.push('parent', parent_ws)

        await store.fork('parent', 'child-1')

        child_ws = tmp_path / 'child_ws'
        await store.restore('child-1', child_ws)
        (child_ws / 'shared.txt').write_text('from-child', encoding='utf-8')
        await store.push('child-1', child_ws)

        parent_ws_2 = tmp_path / 'parent_ws_2'
        await store.restore('parent', parent_ws_2)
        assert (parent_ws_2 / 'shared.txt').read_text(encoding='utf-8') == 'v1'

    async def test_fork_of_never_fenced_parent_raises(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path)
        with pytest.raises(RuntimeError, match='no snapshot yet'):
            await store.fork('parent', 'child-1')


class TestMerge:
    async def test_ff_merge_lands_child_content_into_parent(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path)
        await store.fence('parent', queue='q1')
        parent_ws = tmp_path / 'parent_ws'
        await store.restore('parent', parent_ws)
        (parent_ws / 'shared.txt').write_text('v1', encoding='utf-8')
        await store.push('parent', parent_ws)

        await store.fork('parent', 'child-1')
        child_ws = tmp_path / 'child_ws'
        await store.restore('child-1', child_ws)
        (child_ws / 'from-child.txt').write_text('child work', encoding='utf-8')
        await store.push('child-1', child_ws)

        result = await store.merge('parent', parent_ws, 'child-1', keep_conflicts=False)
        assert result.conflicts == []
        assert (parent_ws / 'from-child.txt').read_text(encoding='utf-8') == 'child work'
        assert (parent_ws / 'shared.txt').read_text(encoding='utf-8') == 'v1'

        parent_ws_2 = tmp_path / 'parent_ws_2'
        await store.restore('parent', parent_ws_2)
        assert (parent_ws_2 / 'from-child.txt').read_text(encoding='utf-8') == 'child work'

    async def test_three_way_merge_with_no_conflict(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path)
        await store.fence('parent', queue='q1')
        parent_ws = tmp_path / 'parent_ws'
        await store.restore('parent', parent_ws)
        (parent_ws / 'shared.txt').write_text('v1', encoding='utf-8')
        await store.push('parent', parent_ws)

        await store.fork('parent', 'child-1')
        child_ws = tmp_path / 'child_ws'
        await store.restore('child-1', child_ws)
        (child_ws / 'child-only.txt').write_text('from child', encoding='utf-8')
        await store.push('child-1', child_ws)

        (parent_ws / 'parent-only.txt').write_text('from parent', encoding='utf-8')
        await store.push('parent', parent_ws)

        result = await store.merge('parent', parent_ws, 'child-1', keep_conflicts=False)
        assert result.conflicts == []
        assert (parent_ws / 'child-only.txt').read_text(encoding='utf-8') == 'from child'
        assert (parent_ws / 'parent-only.txt').read_text(encoding='utf-8') == 'from parent'

    async def test_conflict_without_keep_conflicts_aborts_and_leaves_head_untouched(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path)
        await store.fence('parent', queue='q1')
        parent_ws = tmp_path / 'parent_ws'
        await store.restore('parent', parent_ws)
        (parent_ws / 'shared.txt').write_text('base', encoding='utf-8')
        await store.push('parent', parent_ws)

        await store.fork('parent', 'child-1')
        child_ws = tmp_path / 'child_ws'
        await store.restore('child-1', child_ws)
        (child_ws / 'shared.txt').write_text('from-child', encoding='utf-8')
        await store.push('child-1', child_ws)

        (parent_ws / 'shared.txt').write_text('from-parent', encoding='utf-8')
        await store.push('parent', parent_ws)
        parent_head_before = await store.get_lease('parent')
        assert parent_head_before is not None

        result = await store.merge('parent', parent_ws, 'child-1', keep_conflicts=False)
        assert result.conflicts == ['shared.txt']
        assert (parent_ws / 'shared.txt').read_text(encoding='utf-8') == 'from-parent'

        parent_head_after = await store.get_lease('parent')
        assert parent_head_after is not None
        assert parent_head_after.epoch == parent_head_before.epoch

    async def test_conflict_with_keep_conflicts_commits_markers(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path)
        await store.fence('parent', queue='q1')
        parent_ws = tmp_path / 'parent_ws'
        await store.restore('parent', parent_ws)
        (parent_ws / 'shared.txt').write_text('base', encoding='utf-8')
        await store.push('parent', parent_ws)

        await store.fork('parent', 'child-1')
        child_ws = tmp_path / 'child_ws'
        await store.restore('child-1', child_ws)
        (child_ws / 'shared.txt').write_text('from-child', encoding='utf-8')
        await store.push('child-1', child_ws)

        (parent_ws / 'shared.txt').write_text('from-parent', encoding='utf-8')
        await store.push('parent', parent_ws)

        result = await store.merge('child-1', child_ws, 'parent', keep_conflicts=True)
        assert result.conflicts == ['shared.txt']
        content = (child_ws / 'shared.txt').read_text(encoding='utf-8')
        assert '<<<<<<<' in content
        assert '=======' in content
        assert '>>>>>>>' in content

        child_ws_2 = tmp_path / 'child_ws_2'
        await store.restore('child-1', child_ws_2)
        assert '<<<<<<<' in (child_ws_2 / 'shared.txt').read_text(encoding='utf-8')

    async def test_retry_after_materialize_is_a_fast_forward(self, tmp_path: Path) -> None:
        """After a `materialize` merge, the parent's head is an ancestor of the child branch --
        landing the (resolved) child back into the parent should succeed cleanly."""
        store = GitSnapshotStore(tmp_path)
        await store.fence('parent', queue='q1')
        parent_ws = tmp_path / 'parent_ws'
        await store.restore('parent', parent_ws)
        (parent_ws / 'shared.txt').write_text('base', encoding='utf-8')
        await store.push('parent', parent_ws)

        await store.fork('parent', 'child-1')
        child_ws = tmp_path / 'child_ws'
        await store.restore('child-1', child_ws)
        (child_ws / 'shared.txt').write_text('from-child', encoding='utf-8')
        await store.push('child-1', child_ws)

        (parent_ws / 'shared.txt').write_text('from-parent', encoding='utf-8')
        await store.push('parent', parent_ws)

        materialize_result = await store.merge('child-1', child_ws, 'parent', keep_conflicts=True)
        assert materialize_result.conflicts == ['shared.txt']

        (child_ws / 'shared.txt').write_text('resolved', encoding='utf-8')
        await store.push('child-1', child_ws)

        land_result = await store.merge('parent', parent_ws, 'child-1', keep_conflicts=False)
        assert land_result.conflicts == []
        assert (parent_ws / 'shared.txt').read_text(encoding='utf-8') == 'resolved'
