"""DurableEnvironment capability: lease acquisition and routing for env-bound tools.

Engine-neutral by construction: everything Temporal-specific (or DBOS-, or
Prefect-specific) lives behind the injected `EnvironmentPlacement` driver --
see `pydantic_ai_harness.durable._placement`. This module owns the
engine-independent run-side behavior:

- decide *when* a lease is needed (first env-bound tool call in a run) and
  memoize it for the rest of the run;
- expose the lease to the engine's activity boundary via
  `ctx.metadata['durable_env']`;
- re-acquire and retry (bounded) when the placement target dies mid-run.

The capability never touches a `SnapshotStore` directly -- durable run-side
code must stay deterministic and can't do real I/O. The store/policy it
carries are read worker-side (by `DurableEnvironmentPlugin` in
`pydantic_ai_harness.durable.temporal`), which wires them into the env-bound
toolsets via `configure_durability`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import anyio
from pydantic_ai import AbstractToolset
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import ToolsetTool, WrapperToolset

from pydantic_ai_harness.durable._placement import EnvironmentPlacement
from pydantic_ai_harness.durable._store import EnvironmentLease, SnapshotPolicy, SnapshotStore

_MAX_REPROVISIONS = 3


def _is_env_bound(tool: ToolsetTool[Any]) -> bool:
    metadata = tool.tool_def.metadata
    return bool(metadata and metadata.get('env_bound'))


@dataclass
class DurableEnvironment(AbstractCapability[AgentDepsT]):
    """Lease-backed workspace durability for environment-bound tools.

    Gives environment-bound tools (FileSystem, Shell, CodeMode) a workspace
    that survives pod failure under a durable-execution engine, without moving
    model/MCP activities off the engine's shared queue.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import FileSystem, Shell
    from pydantic_ai_harness.durable import GitSnapshotStore
    from pydantic_ai_harness.durable.temporal import TemporalPlacement

    agent = Agent(
        'openai:gpt-5.2',
        name='coder',
        capabilities=[
            FileSystem(), Shell(),
            DurableEnvironment(
                placement=TemporalPlacement(),
                store=GitSnapshotStore('/var/snapshots'),
            ),
        ],
    )
    ```

    `store`/`snapshot_policy` are read worker-side (see the module docstring);
    the run-side behavior here only handles lease acquisition and routing,
    both delegated to `placement`.

    Outside a durable run (`placement.active()` is false), this capability
    does nothing: env-bound tools run against whatever static root they were
    constructed with.
    """

    placement: EnvironmentPlacement
    """The engine driver this capability leases and routes through."""

    store: SnapshotStore | None = None
    """The `SnapshotStore` backing this agent's environments, read worker-side."""

    snapshot_policy: SnapshotPolicy | Literal['per_op', 'per_step', 'content_hash'] = 'per_op'
    """When to snapshot after a mutating operation. A bare string is shorthand for `SnapshotPolicy(mode=...)`."""

    def __post_init__(self) -> None:
        if isinstance(self.snapshot_policy, str):
            self.snapshot_policy = SnapshotPolicy(mode=self.snapshot_policy)

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        return _DurableEnvWrapper(wrapped=toolset, placement=self.placement)


@dataclass
class _DurableEnvWrapper(WrapperToolset[AgentDepsT]):
    """Routes env-bound tool calls through an engine-acquired lease.

    One instance per run (`get_wrapper_toolset` is called fresh each run), so the
    memoized lease and its lock are naturally scoped to this run -- no cross-run
    leakage, no need for `for_run`-driven state resets.
    """

    placement: EnvironmentPlacement

    _lease: EnvironmentLease | None = field(default=None, init=False, repr=False)
    _lease_lock: anyio.Lock = field(default_factory=anyio.Lock, init=False, repr=False)

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        if not _is_env_bound(tool) or not self.placement.active():
            return await super().call_tool(name, tool_args, ctx, tool)

        passthrough = super().call_tool
        lease = await self._ensure_lease()
        for attempt in range(_MAX_REPROVISIONS + 1):
            if ctx.metadata is None:
                ctx.metadata = {}
            ctx.metadata['durable_env'] = lease.model_dump()
            try:
                return await self.placement.route_call(
                    name,
                    tool_args,
                    ctx,
                    tool,
                    lease=lease,
                    wrapped=self.wrapped,
                    fallback=lambda: passthrough(name, tool_args, ctx, tool),
                )
            except Exception as e:
                if attempt == _MAX_REPROVISIONS or not self.placement.is_placement_failure(e):
                    raise
                lease = await self._reacquire(failed_queue=lease.env_queue)
        raise AssertionError('unreachable: loop above always returns or raises')  # pragma: no cover

    async def _ensure_lease(self) -> EnvironmentLease:
        async with self._lease_lock:
            if self._lease is None:
                self._lease = await self.placement.acquire(failed_queue=None)
            return self._lease

    async def _reacquire(self, *, failed_queue: str) -> EnvironmentLease:
        async with self._lease_lock:
            self._lease = await self.placement.acquire(failed_queue=failed_queue)
            return self._lease
