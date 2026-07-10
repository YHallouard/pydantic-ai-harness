"""Shared contract for environment-bound toolsets (FileSystem, Shell, CodeMode)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, TypeAlias

from pydantic_ai.tools import RunContext, ToolSelector

from pydantic_ai_harness.durable._store import SnapshotPolicy, SnapshotStore

RootDirSource: TypeAlias = 'str | Path | Callable[[RunContext[Any]], str | Path]'
"""A root/cwd source: a fixed path, or a callable resolved per call."""


def env_bound_metadata(*, mutating: bool) -> dict[str, Any]:
    """Build the `ToolDefinition.metadata` tags for an environment-bound tool.

    `env_bound` marks a tool as needing to execute where its workspace lives --
    the durable-execution routing story `DurableEnvironment` (sub-issue 3) builds
    on, using the same `metadata['temporal']` vocabulary as pydantic-ai#4977's
    capability path. `mutating` marks whether the tool changes the workspace:
    mutating calls go through the per-env lock and idempotency journal: see
    `pydantic_ai_harness.durable.OpJournal`. Read-only calls skip that work.
    """
    return {'env_bound': True, 'mutating': mutating}


class EnvironmentBound(Protocol):
    """Structural contract implemented by FileSystem, Shell, and CodeMode toolsets.

    Lets an orchestrator -- Temporal routing, an audit layer, approval policies --
    identify and configure environment-bound toolsets without importing their
    concrete classes.
    """

    def env_bound_tools(self) -> ToolSelector[Any]:
        """Return a selector matching every tool this toolset owns.

        Every tool an environment-bound toolset exposes is env_bound (see
        `env_bound_metadata`), so this is `'all'` for the three toolsets --
        it exists as a stable, class-agnostic way for an orchestrator to select
        them without inspecting `ToolDefinition.metadata` itself.
        """
        ...  # pragma: no cover -- Protocol method body, never executed

    def set_env_root(self, root: RootDirSource) -> None:
        """Rebind the resolved root/cwd/mount source after construction.

        Used by a durable-execution orchestrator (e.g. `DurableEnvironment`,
        sub-issue 3) to point the toolset at a workspace root that's only known
        once a lease is assigned, without rebuilding the toolset.
        """
        ...  # pragma: no cover -- Protocol method body, never executed

    def configure_durability(self, store: SnapshotStore | None, policy: SnapshotPolicy) -> None:
        """Wire a snapshot store and policy into this toolset's mutating-op path.

        `store=None` is a no-op -- the local, non-durable path. Mutating
        operations still go through the idempotency journal and per-env lock
        (see `pydantic_ai_harness.durable.OpJournal`), but nothing is
        snapshotted. Configured by `DurableEnvironment` (sub-issue 3); harness
        toolsets don't call this themselves.
        """
        ...  # pragma: no cover -- Protocol method body, never executed
