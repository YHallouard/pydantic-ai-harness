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
local/non-durable path is protected too, not just Temporal-routed calls. This
leak is bounded but real -- one `Lock` per workspace root ever seen by this
process -- and unbounded release/cleanup belongs to `DurableEnvironment`
(sub-issue 3), which owns the worker's environment lifecycle.
"""


async def guarded_mutating(
    *,
    ctx: RunContext[Any],
    root: Path,
    tool: str,
    apply: Callable[[], Awaitable[str]],
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
    - between `journal.record()` and a future snapshot commit (sub-issue 3):
      same -- the journal and workspace both restart from the last consistent
      snapshot on crash.
    - after a snapshot commit: the op is captured in the snapshot, so any retry
      or re-execution is deduplicated by the `journal.seen()` lookup below.
    The end-to-end invariant sub-issue 3 establishes: "result delivered to the
    caller" implies "op is in the latest snapshot."

    `apply` can raise `JournalSkipped(result)` for an ambiguous completion (e.g.
    a timeout) that shouldn't be permanently cached; see `JournalSkipped`.
    """
    async with _ENV_LOCKS.setdefault(root, anyio.Lock()):
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
        return result
