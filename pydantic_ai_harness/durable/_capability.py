"""DurableEnvironment capability: lease acquisition and routing for env-bound tools.

Split cleanly along the workflow/worker boundary Temporal itself imposes:

- **Workflow side** (this module): `_DurableEnvWrapper` runs inside the sandboxed
  workflow. It never touches a `SnapshotStore` directly (workflow code must be
  deterministic and can't do real I/O) -- it only calls the `acquire_environment`/
  `release_environment` *activities* and writes the resulting lease into
  `ctx.metadata['durable_env']`, which is the seam pydantic-ai core's
  `resolve_tool_activity_config` reads to route the env-bound tool's own activity
  to the leased queue.
- **Worker side** (`run_env_worker`, `EnvironmentActivities`): owns the real
  `SnapshotStore` and does the actual git/filesystem work, outside the sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Literal

import anyio
from pydantic_ai import AbstractToolset
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import ToolsetTool, WrapperToolset
from temporalio import workflow
from temporalio.exceptions import ActivityError, TimeoutType
from temporalio.exceptions import TimeoutError as TemporalTimeoutError

from pydantic_ai_harness.durable._store import AcquireEnvParams, EnvironmentLease, SnapshotPolicy

_MAX_REPROVISIONS = 3

_ACQUIRE_SCHEDULE_TO_START_TIMEOUT = timedelta(seconds=10)
"""Short by design: a schedule-to-start timeout here is how a fenced-out pod is detected quickly."""

_ACQUIRE_START_TO_CLOSE_TIMEOUT = timedelta(seconds=30)


def _in_temporal_workflow() -> bool:  # pragma: no cover -- trivial delegation, always mocked in tests
    return workflow.in_workflow()


def _is_env_bound(tool: ToolsetTool[Any]) -> bool:
    metadata = tool.tool_def.metadata
    return bool(metadata and metadata.get('env_bound'))


def _is_schedule_to_start_timeout(error: ActivityError) -> bool:
    cause = error.cause
    return isinstance(cause, TemporalTimeoutError) and cause.type == TimeoutType.SCHEDULE_TO_START


@dataclass
class DurableEnvironment(AbstractCapability[AgentDepsT]):
    """Lease-backed workspace durability for environment-bound tools.

    Gives environment-bound tools (FileSystem, Shell, CodeMode) a workspace that
    survives pod failure under `TemporalDurability`, without moving model/MCP
    activities off the shared task queue.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai.durable_exec.temporal import TemporalDurability
    from pydantic_ai_harness import FileSystem, Shell
    from pydantic_ai_harness.durable import GitSnapshotStore
    from pydantic_ai_harness.durable.temporal import DurableEnvironment

    agent = Agent(
        'openai:gpt-5.2',
        name='coder',
        capabilities=[
            FileSystem(), Shell(),
            DurableEnvironment(store=GitSnapshotStore('/var/snapshots'), snapshot_policy='per_op'),
            TemporalDurability(),
        ],
    )
    ```

    `store`/`snapshot_policy` are read worker-side by `run_env_worker`, which wires
    them into the env-bound toolsets via `configure_durability` -- this capability's
    own workflow-side behavior (lease acquisition and routing) doesn't touch them
    directly, since workflow code can't do the real I/O a store performs.

    Outside a Temporal workflow, this capability does nothing: env-bound tools run
    against whatever static root they were constructed with.
    """

    store: Any | None = None
    """The `SnapshotStore` backing this agent's environments, read by `run_env_worker`."""

    snapshot_policy: SnapshotPolicy | Literal['per_op', 'per_step', 'content_hash'] = 'per_op'
    """When to snapshot after a mutating operation. A bare string is shorthand for `SnapshotPolicy(mode=...)`."""

    def __post_init__(self) -> None:
        if isinstance(self.snapshot_policy, str):
            self.snapshot_policy = SnapshotPolicy(mode=self.snapshot_policy)

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        return _DurableEnvWrapper(wrapped=toolset)


@dataclass
class _DurableEnvWrapper(WrapperToolset[AgentDepsT]):
    """Routes env-bound tool calls through a Temporal-acquired lease.

    One instance per run (`get_wrapper_toolset` is called fresh each run), so the
    memoized lease and its lock are naturally scoped to this run -- no cross-run
    leakage, no need for `for_run`-driven state resets.
    """

    _lease: EnvironmentLease | None = field(default=None, init=False, repr=False)
    _lease_lock: anyio.Lock = field(default_factory=anyio.Lock, init=False, repr=False)

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        if not _is_env_bound(tool) or not _in_temporal_workflow():
            return await super().call_tool(name, tool_args, ctx, tool)

        lease = await self._ensure_lease()
        for attempt in range(_MAX_REPROVISIONS + 1):
            if ctx.metadata is None:
                ctx.metadata = {}
            ctx.metadata['durable_env'] = lease.model_dump()
            try:
                return await super().call_tool(name, tool_args, ctx, tool)
            except ActivityError as e:
                if attempt == _MAX_REPROVISIONS or not _is_schedule_to_start_timeout(e):
                    raise
                lease = await self._reacquire(failed_queue=lease.env_queue)
        raise AssertionError('unreachable: loop above always returns or raises')  # pragma: no cover

    async def _ensure_lease(self) -> EnvironmentLease:
        async with self._lease_lock:
            if self._lease is None:
                self._lease = await self._acquire(failed_queue=None)
            return self._lease

    async def _reacquire(self, *, failed_queue: str) -> EnvironmentLease:
        async with self._lease_lock:
            self._lease = await self._acquire(failed_queue=failed_queue)
            return self._lease

    async def _acquire(self, *, failed_queue: str | None) -> EnvironmentLease:
        env_id = workflow.info().workflow_id
        params = AcquireEnvParams(env_id=env_id, failed_queue=failed_queue)
        return await workflow.execute_activity(
            'acquire_environment',
            params,
            result_type=EnvironmentLease,
            schedule_to_start_timeout=_ACQUIRE_SCHEDULE_TO_START_TIMEOUT,
            start_to_close_timeout=_ACQUIRE_START_TO_CLOSE_TIMEOUT,
        )
