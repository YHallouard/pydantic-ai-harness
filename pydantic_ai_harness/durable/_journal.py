"""Idempotency journal and per-environment lock for mutating tool calls."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
from pydantic import BaseModel
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import RunContext

from pydantic_ai_harness.durable._store import SnapshotPolicy, SnapshotStore

MAX_RESULT = 64_000
"""Journal entries larger than this are truncated; the full result is not replayable."""

_JOURNAL_PATH = Path('.durable_env') / 'journal'


class JournalEntry(BaseModel):
    """One applied mutating op, serialized as a single JSONL line.

    The on-disk contract for the journal: `model_dump_json()` writes a line,
    `model_validate_json()` reads it back, so the shape is validated on the way
    in and out rather than hand-built and hand-parsed from a dict.
    """

    op_id: str
    tool: str
    result: str
    truncated: bool
    ts: float


@dataclass(frozen=True)
class RecordedResult:
    result: str
    truncated: bool


class JournalSkipped(Exception):
    """Raise from a `guarded_mutating` `apply` callable to return a result without journaling it.

    For an ambiguous completion -- e.g. a command that timed out and may or may
    not have applied its effects -- journaling would permanently cache that
    outcome and block a legitimate retry from ever trying again. Raising this
    instead returns `result` to the caller exactly like a normal completion,
    but leaves no journal entry, so a retry with the same op_id re-runs `apply`.
    """

    def __init__(self, result: str) -> None:
        super().__init__(result)
        self.result = result


class OpJournal:
    """JSONL append-only idempotency log for mutating environment-bound tool calls.

    Lives inside the workspace itself (`<root>/.durable_env/journal`) so it
    versions *with* the workspace under a durable execution engine's snapshot
    store -- the journal and the files it accounts for move together. JSONL is
    append-only (no rewrites), which keeps the on-disk diff clean for whatever
    versions the workspace (git shadow repo, in sub-issue 3).
    """

    def __init__(self, workspace: Path) -> None:
        self._path = workspace / _JOURNAL_PATH
        self._index: dict[str, RecordedResult] | None = None

    def _load_index(self) -> dict[str, RecordedResult]:
        index: dict[str, RecordedResult] = {}
        if self._path.exists():
            for line in self._path.read_text(encoding='utf-8').splitlines():
                if not line:  # pragma: no cover -- defensive against out-of-band file edits
                    continue
                entry = JournalEntry.model_validate_json(line)
                index[entry.op_id] = RecordedResult(result=entry.result, truncated=entry.truncated)
        return index

    def seen(self, op_id: str) -> RecordedResult | None:
        """Return the recorded result for `op_id`, or `None` if it hasn't been applied yet."""
        if self._index is None:
            self._index = self._load_index()
        return self._index.get(op_id)

    def record(self, op_id: str, tool: str, result: str) -> None:
        """Append a completed op to the journal, truncating an oversized result.

        A truncated entry still marks the op as seen -- `seen()` returns it with
        `truncated=True` -- so a replay can't silently return a partial result;
        callers must reject it instead (see `guarded_mutating`).
        """
        truncated = len(result) > MAX_RESULT
        stored = result[:MAX_RESULT]
        entry = JournalEntry(op_id=op_id, tool=tool, result=stored, truncated=truncated, ts=time.time())
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open('a', encoding='utf-8') as f:
            f.write(entry.model_dump_json() + '\n')
        if self._index is not None:
            self._index[op_id] = RecordedResult(result=stored, truncated=truncated)


_ENV_LOCKS: dict[Path, anyio.Lock] = {}
"""Per-resolved-root lock, keyed by the root path itself (not env_id) so the
local/non-durable path is protected too, not just engine-routed calls. Cleared
per environment by `discard_env_lock`, called from `EnvironmentActivities` when a
worker gives up a workspace; without that a long-lived worker would retain one
`anyio.Lock` per workspace root it ever provisioned.
"""


def discard_env_lock(root: Path) -> None:
    """Drop the cached per-workspace lock for `root` once its environment is released.

    Called after a worker has given up `root` (workflow done with it, or a
    re-provision replaced it), so no mutating op is in flight for it. A no-op if
    no lock was ever created for `root` (a durable run that never mutated).
    """
    _ENV_LOCKS.pop(root, None)


def env_lock(root: Path) -> anyio.Lock:
    """Return the per-workspace-root lock for `root`, creating it if needed.

    Shared by `guarded_mutating` and `EnvironmentActivities.merge_environment`
    (`_lease.py`) so a concurrent mutating tool call and a merge into the same
    held workspace serialize against each other instead of racing on files or
    git state.
    """
    return _ENV_LOCKS.setdefault(root, anyio.Lock())


def _env_id_from_ctx(ctx: RunContext[Any]) -> str | None:
    """Read the acquired lease's `env_id` from `ctx.metadata['durable_env']`, if any.

    `None` outside a durable-execution run (no lease has ever been acquired) --
    the caller treats that as "nothing to snapshot", not an error, since a
    toolset can be configured with a store while running locally/untested.
    """
    metadata = ctx.metadata
    if metadata is None:
        return None
    durable_env = metadata.get('durable_env')
    if durable_env is None:
        return None
    env_id = durable_env.get('env_id')
    return env_id if isinstance(env_id, str) else None


async def guarded_mutating(
    *,
    ctx: RunContext[Any],
    root: Path,
    tool: str,
    apply: Callable[[], Awaitable[str]],
    store: SnapshotStore | None = None,
    policy: SnapshotPolicy | None = None,
) -> str:
    """Serialize a mutating environment-bound tool call and dedupe retries via the journal.

    Concurrent mutations on one workspace race on files and the journal, so
    every mutating call for a given resolved `root` takes a lock before
    touching anything. `op_id = f'{run_id}:{tool_call_id}'` is stable across a
    Temporal activity retry *and* a workflow-level re-execution after
    re-provisioning -- unlike an activity id, which is not.

    Crash-window analysis (each window degrades to at-least-once locally, which
    the journal check below turns into exactly-once for anything durable enough
    to be recorded):
    - between `apply()` and `journal.record()`: a retry re-applies -- acceptable,
      since nothing has claimed the op completed yet.
    - between `journal.record()` and the snapshot push below: same -- the
      journal and workspace both restart from the last consistent snapshot on
      crash.
    - after the snapshot push: the op is captured in the snapshot, so any retry
      or re-execution is deduplicated by the `journal.seen()` lookup below.
    The end-to-end invariant: "result delivered to the caller" implies "op is
    in the latest snapshot" -- established here for `per_op` (the default),
    which pushes before returning. `per_step`/`content_hash` are recognized
    policy values (see `SnapshotPolicy`) but don't push from this function;
    wiring their debounced/triggered push is follow-up work, not sub-issue 3.

    `apply` can raise `JournalSkipped(result)` for an ambiguous completion (e.g.
    a timeout) that shouldn't be permanently cached; see `JournalSkipped`.

    `store`/`policy` are set by `configure_durability` (worker-side, by
    `run_env_worker`); both are `None` for a local, non-durable run, which
    skips the push entirely -- same for a durable toolset whose run never
    acquired a lease (`ctx.metadata['durable_env']` absent).
    """
    async with env_lock(root):
        op_id = f'{ctx.run_id}:{ctx.tool_call_id}'
        journal = OpJournal(root)
        recorded = journal.seen(op_id)
        if recorded is not None:
            if recorded.truncated:
                raise ModelRetry(
                    'The result of this already-applied operation is no longer available; re-read the file instead.'
                )
            return recorded.result
        try:
            result = await apply()
        except JournalSkipped as skipped:
            return skipped.result
        journal.record(op_id, tool, result)
        if store is not None and policy is not None and policy.mode == 'per_op':
            env_id = _env_id_from_ctx(ctx)
            if env_id is not None:
                await store.push(env_id, root)
        return result
