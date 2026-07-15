"""Tests for OpJournal and guarded_mutating."""

from __future__ import annotations

import itertools
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import anyio
import pytest
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.durable import (
    MAX_RESULT,
    GitSnapshotStore,
    JournalEntry,
    JournalSkipped,
    OpJournal,
    SnapshotPolicy,
    env_id_from_ctx,
    guarded_mutating,
)
from pydantic_ai_harness.durable._journal import _ENV_LOCKS, discard_env_lock

_tool_call_ids = (f'call_{i}' for i in itertools.count())


def _ctx(
    *, run_id: str = 'test-run', tool_call_id: str | None = None, metadata: dict[str, Any] | None = None
) -> RunContext[None]:
    return RunContext[None](
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
        run_id=run_id,
        tool_call_id=tool_call_id or next(_tool_call_ids),
        metadata=metadata,
    )


def _ids(ctx: RunContext[None]) -> tuple[str, str | None]:
    """Derive the (op_id, env_id) pair a real tool call would pass to guarded_mutating,
    exactly like filesystem/_toolset.py and shell/_toolset.py do -- this is the same
    ctx-to-plain-parameter translation `guarded_mutating` no longer does internally."""
    return f'{ctx.run_id}:{ctx.tool_call_id}', env_id_from_ctx(ctx)


class TestOpJournal:
    def test_seen_returns_none_for_unknown_op_id(self, tmp_path: Path) -> None:
        journal = OpJournal(tmp_path)
        assert journal.seen('unknown') is None

    def test_record_then_seen_returns_recorded_result(self, tmp_path: Path) -> None:
        journal = OpJournal(tmp_path)
        journal.record('op-1', 'write_file', 'the result')
        recorded = journal.seen('op-1')
        assert recorded is not None
        assert recorded.result == 'the result'
        assert recorded.truncated is False

    def test_record_persists_across_new_journal_instance(self, tmp_path: Path) -> None:
        """The journal is a file, not in-memory state -- a fresh OpJournal for the
        same workspace (e.g. a new activity worker) sees prior entries."""
        OpJournal(tmp_path).record('op-1', 'write_file', 'the result')
        recorded = OpJournal(tmp_path).seen('op-1')
        assert recorded is not None
        assert recorded.result == 'the result'

    def test_journal_file_lives_under_durable_env(self, tmp_path: Path) -> None:
        OpJournal(tmp_path).record('op-1', 'write_file', 'result')
        assert (tmp_path / '.durable_env' / 'journal').is_file()

    def test_each_line_is_a_serialized_journal_entry(self, tmp_path: Path) -> None:
        """The on-disk format is one `JournalEntry` per line, round-tripped via
        `model_dump_json`/`model_validate_json` rather than a hand-built dict."""
        OpJournal(tmp_path).record('op-1', 'write_file', 'the result')
        line = (tmp_path / '.durable_env' / 'journal').read_text(encoding='utf-8').splitlines()[0]
        entry = JournalEntry.model_validate_json(line)
        assert entry.op_id == 'op-1'
        assert entry.tool == 'write_file'
        assert entry.result == 'the result'
        assert entry.truncated is False

    def test_record_truncates_oversized_result(self, tmp_path: Path) -> None:
        journal = OpJournal(tmp_path)
        oversized = 'x' * (MAX_RESULT + 100)
        journal.record('op-1', 'write_file', oversized)
        recorded = journal.seen('op-1')
        assert recorded is not None
        assert recorded.truncated is True
        assert len(recorded.result) == MAX_RESULT

    def test_record_does_not_truncate_result_at_the_limit(self, tmp_path: Path) -> None:
        journal = OpJournal(tmp_path)
        exact = 'x' * MAX_RESULT
        journal.record('op-1', 'write_file', exact)
        recorded = journal.seen('op-1')
        assert recorded is not None
        assert recorded.truncated is False

    def test_multiple_entries_independently_recorded(self, tmp_path: Path) -> None:
        journal = OpJournal(tmp_path)
        journal.record('op-1', 'write_file', 'first')
        journal.record('op-2', 'edit_file', 'second')
        assert journal.seen('op-1').result == 'first'  # type: ignore[union-attr]
        assert journal.seen('op-2').result == 'second'  # type: ignore[union-attr]


class TestGuardedMutating:
    async def test_first_call_applies_and_journals(self, tmp_path: Path) -> None:
        calls: list[int] = []

        async def apply() -> str:
            calls.append(1)
            return 'applied'

        op_id, env_id = _ids(_ctx(tool_call_id='op-1'))
        result = await guarded_mutating(op_id=op_id, env_id=env_id, root=tmp_path, tool='write_file', apply=apply)
        assert result == 'applied'
        assert len(calls) == 1
        assert OpJournal(tmp_path).seen('test-run:op-1') is not None  # type: ignore[union-attr]

    async def test_replay_same_op_id_returns_recorded_result_without_reapplying(self, tmp_path: Path) -> None:
        calls: list[int] = []

        async def apply() -> str:
            calls.append(1)
            return f'applied {len(calls)}'

        op_id, env_id = _ids(_ctx(tool_call_id='op-1'))
        first = await guarded_mutating(op_id=op_id, env_id=env_id, root=tmp_path, tool='write_file', apply=apply)
        second = await guarded_mutating(op_id=op_id, env_id=env_id, root=tmp_path, tool='write_file', apply=apply)
        assert first == second == 'applied 1'
        assert len(calls) == 1

    async def test_different_op_id_applies_again(self, tmp_path: Path) -> None:
        calls: list[int] = []

        async def apply() -> str:
            calls.append(1)
            return f'applied {len(calls)}'

        op_id_1, env_id_1 = _ids(_ctx(tool_call_id='op-1'))
        op_id_2, env_id_2 = _ids(_ctx(tool_call_id='op-2'))
        first = await guarded_mutating(op_id=op_id_1, env_id=env_id_1, root=tmp_path, tool='write_file', apply=apply)
        second = await guarded_mutating(op_id=op_id_2, env_id=env_id_2, root=tmp_path, tool='write_file', apply=apply)
        assert first == 'applied 1'
        assert second == 'applied 2'
        assert len(calls) == 2

    async def test_apply_exception_not_journaled(self, tmp_path: Path) -> None:
        """A failed op leaves no journal entry -- retrying the same op_id re-applies."""
        calls: list[int] = []

        async def apply() -> str:
            calls.append(1)
            if len(calls) == 1:
                raise ValueError('boom')
            return 'applied on retry'

        op_id, env_id = _ids(_ctx(tool_call_id='op-1'))
        with pytest.raises(ValueError, match='boom'):
            await guarded_mutating(op_id=op_id, env_id=env_id, root=tmp_path, tool='write_file', apply=apply)
        assert OpJournal(tmp_path).seen('test-run:op-1') is None

        result = await guarded_mutating(op_id=op_id, env_id=env_id, root=tmp_path, tool='write_file', apply=apply)
        assert result == 'applied on retry'
        assert len(calls) == 2

    async def test_crash_between_apply_and_record_replays_on_retry(self, tmp_path: Path) -> None:
        """Documents the crash window: if the process dies after `apply()` completes
        but before `journal.record()` runs, nothing is journaled -- a retry with the
        same op_id can't tell this apart from a fresh call and re-applies
        (at-least-once locally). The end-to-end exactly-once guarantee needs the
        joint snapshot+journal commit sub-issue 3 adds.
        """
        calls: list[int] = []

        async def apply() -> str:
            calls.append(1)
            return 'done'

        # Simulate the crashed first attempt: apply() ran, but the process died
        # before guarded_mutating reached journal.record().
        await apply()
        assert len(calls) == 1
        assert OpJournal(tmp_path).seen('test-run:op-1') is None

        op_id, env_id = _ids(_ctx(tool_call_id='op-1'))
        result = await guarded_mutating(op_id=op_id, env_id=env_id, root=tmp_path, tool='write_file', apply=apply)
        assert result == 'done'
        assert len(calls) == 2

    async def test_journal_skipped_returns_result_without_recording(self, tmp_path: Path) -> None:
        """`JournalSkipped` (e.g. an ambiguous timeout) returns its result but leaves
        no journal entry, so a retry with the same op_id tries again."""
        calls: list[int] = []

        async def apply() -> str:
            calls.append(1)
            if len(calls) == 1:
                raise JournalSkipped('[Command timed out after 30s]')
            return 'completed on retry'

        op_id, env_id = _ids(_ctx(tool_call_id='op-1'))
        first = await guarded_mutating(op_id=op_id, env_id=env_id, root=tmp_path, tool='run_command', apply=apply)
        assert first == '[Command timed out after 30s]'
        assert OpJournal(tmp_path).seen('test-run:op-1') is None

        second = await guarded_mutating(op_id=op_id, env_id=env_id, root=tmp_path, tool='run_command', apply=apply)
        assert second == 'completed on retry'
        assert len(calls) == 2

    async def test_truncated_replay_raises_model_retry(self, tmp_path: Path) -> None:
        """A truncated journal entry can't be replayed faithfully -- rather than
        return a silently-partial result, guarded_mutating raises ModelRetry."""
        journal = OpJournal(tmp_path)
        journal.record('test-run:op-1', 'write_file', 'x' * (MAX_RESULT + 1))

        async def apply() -> str:  # pragma: no cover -- must not be called
            raise AssertionError('apply() should not run for a truncated replay')

        op_id, env_id = _ids(_ctx(tool_call_id='op-1'))
        with pytest.raises(ModelRetry, match='no longer available'):
            await guarded_mutating(op_id=op_id, env_id=env_id, root=tmp_path, tool='write_file', apply=apply)

    async def test_concurrent_calls_on_same_root_are_serialized(self, tmp_path: Path) -> None:
        events: list[str] = []

        async def make_apply(name: str) -> str:
            events.append(f'{name}-start')
            await anyio.sleep(0.01)
            events.append(f'{name}-end')
            return name

        async def _call(name: str) -> None:
            op_id, env_id = _ids(_ctx(tool_call_id=f'op-{name}'))
            await guarded_mutating(
                op_id=op_id, env_id=env_id, root=tmp_path, tool='write_file', apply=lambda: make_apply(name)
            )

        async with anyio.create_task_group() as tg:
            tg.start_soon(_call, 'a')
            tg.start_soon(_call, 'b')

        # Serialized: one call's start/end must not interleave with the other's.
        assert events in (
            ['a-start', 'a-end', 'b-start', 'b-end'],
            ['b-start', 'b-end', 'a-start', 'a-end'],
        )

    async def test_concurrent_calls_on_different_roots_are_not_serialized(self, tmp_path: Path) -> None:
        """A call on root_b must not wait for a lock held by an in-flight call on root_a.

        Uses a bound on wall-clock time rather than exact task-scheduling order
        (which anyio backends don't guarantee across `start_soon` calls) --
        b must finish quickly even while a's lock is held for much longer.
        """
        root_a = tmp_path / 'a'
        root_b = tmp_path / 'b'
        root_a.mkdir()
        root_b.mkdir()
        a_holds_lock = anyio.Event()
        release_a = anyio.Event()

        async def apply_a() -> str:
            a_holds_lock.set()
            await release_a.wait()
            return 'a'

        async def apply_b() -> str:
            return 'b'

        async def _call(root: Path, tool_call_id: str, apply: Callable[[], Awaitable[str]]) -> None:
            op_id, env_id = _ids(_ctx(tool_call_id=tool_call_id))
            await guarded_mutating(op_id=op_id, env_id=env_id, root=root, tool='write_file', apply=apply)

        async with anyio.create_task_group() as tg:
            tg.start_soon(_call, root_a, 'op-a', apply_a)
            await a_holds_lock.wait()
            with anyio.fail_after(1):
                await _call(root_b, 'op-b', apply_b)
            release_a.set()


class TestGuardedMutatingSnapshot:
    """`store`/`policy` wiring: a fresh `per_op` application pushes a snapshot before returning."""

    async def test_per_op_pushes_a_snapshot_when_a_lease_is_present(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path / 'store')
        workspace = tmp_path / 'ws'
        workspace.mkdir()
        await store.restore('env-1', into=workspace)  # empty restore -- just to have a repo to push into
        op_id, env_id = _ids(
            _ctx(tool_call_id='op-1', metadata={'durable_env': {'env_id': 'env-1', 'env_queue': 'q', 'epoch': 0}})
        )

        async def apply() -> str:
            (workspace / 'note.txt').write_text('hi', encoding='utf-8')
            return 'wrote note.txt'

        await guarded_mutating(
            op_id=op_id,
            env_id=env_id,
            root=workspace,
            tool='write_file',
            apply=apply,
            store=store,
            policy=SnapshotPolicy(mode='per_op'),
        )

        restored = tmp_path / 'restored'
        await store.restore('env-1', restored)
        assert (restored / 'note.txt').read_text(encoding='utf-8') == 'hi'

    async def test_replay_does_not_push_again(self, tmp_path: Path) -> None:
        """A journaled replay returns the cached result without re-touching the store."""
        store = GitSnapshotStore(tmp_path / 'store')
        workspace = tmp_path / 'ws'
        workspace.mkdir()
        await store.restore('env-1', into=workspace)
        op_id, env_id = _ids(
            _ctx(tool_call_id='op-1', metadata={'durable_env': {'env_id': 'env-1', 'env_queue': 'q', 'epoch': 0}})
        )
        pushes: list[Path] = []

        class _CountingStore(GitSnapshotStore):
            async def push(self, env_id: str, workspace: Path) -> None:
                pushes.append(workspace)
                await super().push(env_id, workspace)

        counting_store = _CountingStore(tmp_path / 'store')

        async def apply() -> str:
            return 'applied'

        await guarded_mutating(
            op_id=op_id,
            env_id=env_id,
            root=workspace,
            tool='write_file',
            apply=apply,
            store=counting_store,
            policy=SnapshotPolicy(),
        )
        await guarded_mutating(
            op_id=op_id,
            env_id=env_id,
            root=workspace,
            tool='write_file',
            apply=apply,
            store=counting_store,
            policy=SnapshotPolicy(),
        )

        assert len(pushes) == 1

    async def test_no_push_without_a_store(self, tmp_path: Path) -> None:
        op_id, env_id = _ids(
            _ctx(tool_call_id='op-1', metadata={'durable_env': {'env_id': 'env-1', 'env_queue': 'q', 'epoch': 0}})
        )

        async def apply() -> str:
            return 'applied'

        result = await guarded_mutating(
            op_id=op_id, env_id=env_id, root=tmp_path, tool='write_file', apply=apply, store=None
        )
        assert result == 'applied'

    async def test_no_push_when_policy_is_not_per_op(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path / 'store')
        workspace = tmp_path / 'ws'
        workspace.mkdir()
        await store.restore('env-1', into=workspace)
        op_id, env_id = _ids(
            _ctx(tool_call_id='op-1', metadata={'durable_env': {'env_id': 'env-1', 'env_queue': 'q', 'epoch': 0}})
        )

        async def apply() -> str:
            (workspace / 'note.txt').write_text('hi', encoding='utf-8')
            return 'wrote note.txt'

        await guarded_mutating(
            op_id=op_id,
            env_id=env_id,
            root=workspace,
            tool='write_file',
            apply=apply,
            store=store,
            policy=SnapshotPolicy(mode='per_step'),
        )

        assert await store.get_lease('env-1') is None

    async def test_no_push_without_a_lease_in_ctx_metadata(self, tmp_path: Path) -> None:
        """A toolset can be configured with a store while running a lease-less (local) call --
        e.g. mixed usage during a migration. No lease means nothing to push to."""
        store = GitSnapshotStore(tmp_path / 'store')
        workspace = tmp_path / 'ws'
        workspace.mkdir()
        op_id, env_id = _ids(_ctx(tool_call_id='op-1', metadata=None))

        async def apply() -> str:
            return 'applied'

        result = await guarded_mutating(
            op_id=op_id,
            env_id=env_id,
            root=workspace,
            tool='write_file',
            apply=apply,
            store=store,
            policy=SnapshotPolicy(),
        )
        assert result == 'applied'

    async def test_no_push_when_durable_env_missing_from_metadata(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path / 'store')
        workspace = tmp_path / 'ws'
        workspace.mkdir()
        op_id, env_id = _ids(_ctx(tool_call_id='op-1', metadata={'other_key': 'value'}))

        async def apply() -> str:
            return 'applied'

        result = await guarded_mutating(
            op_id=op_id,
            env_id=env_id,
            root=workspace,
            tool='write_file',
            apply=apply,
            store=store,
            policy=SnapshotPolicy(),
        )
        assert result == 'applied'

    async def test_no_push_when_env_id_is_not_a_string(self, tmp_path: Path) -> None:
        store = GitSnapshotStore(tmp_path / 'store')
        workspace = tmp_path / 'ws'
        workspace.mkdir()
        op_id, env_id = _ids(_ctx(tool_call_id='op-1', metadata={'durable_env': {'env_id': 42}}))

        async def apply() -> str:
            return 'applied'

        result = await guarded_mutating(
            op_id=op_id,
            env_id=env_id,
            root=workspace,
            tool='write_file',
            apply=apply,
            store=store,
            policy=SnapshotPolicy(),
        )
        assert result == 'applied'


class TestDiscardEnvLock:
    async def test_discards_the_lock_created_for_a_root(self, tmp_path: Path) -> None:
        async def apply() -> str:
            return 'applied'

        op_id, env_id = _ids(_ctx(tool_call_id='op-1'))
        await guarded_mutating(op_id=op_id, env_id=env_id, root=tmp_path, tool='write_file', apply=apply)
        assert tmp_path in _ENV_LOCKS

        discard_env_lock(tmp_path)
        assert tmp_path not in _ENV_LOCKS

    def test_is_a_no_op_for_a_root_with_no_lock(self, tmp_path: Path) -> None:
        discard_env_lock(tmp_path / 'never-mutated')  # does not raise
