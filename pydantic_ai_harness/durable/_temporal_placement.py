"""Temporal implementation of the `EnvironmentPlacement` protocol.

Requires `temporalio` (imported here, not gated -- this module is only
reachable through `pydantic_ai_harness.durable.temporal`, which already gates
the import).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from pydantic_ai import AbstractToolset
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import ToolsetTool
from temporalio import workflow
from temporalio.exceptions import ActivityError, TimeoutType
from temporalio.exceptions import TimeoutError as TemporalTimeoutError

from pydantic_ai_harness.durable._store import AcquireEnvParams, EnvironmentLease

_ACQUIRE_SCHEDULE_TO_START_TIMEOUT = timedelta(seconds=10)
"""Short by design: a schedule-to-start timeout here is how a fenced-out pod is detected quickly."""

_ACQUIRE_START_TO_CLOSE_TIMEOUT = timedelta(seconds=30)


@dataclass
class TemporalPlacement:
    """`EnvironmentPlacement` driver for Temporal.

    Workflow-side counterpart of `EnvironmentActivities`: `acquire` calls the
    `acquire_environment` activity (registered on the shared queue by
    `DurableEnvironmentPlugin`), and a schedule-to-start timeout on any routed
    call is the signal that the leased pod is dead or fenced out.
    """

    def active(self) -> bool:
        return workflow.in_workflow()

    async def acquire(self, *, failed_queue: str | None) -> EnvironmentLease:
        params = AcquireEnvParams(env_id=workflow.info().workflow_id, failed_queue=failed_queue)
        return await workflow.execute_activity(
            'acquire_environment',
            params,
            result_type=EnvironmentLease,
            schedule_to_start_timeout=_ACQUIRE_SCHEDULE_TO_START_TIMEOUT,
            start_to_close_timeout=_ACQUIRE_START_TO_CLOSE_TIMEOUT,
        )

    def is_placement_failure(self, exc: Exception) -> bool:
        if not isinstance(exc, ActivityError):
            return False
        cause = exc.cause
        return isinstance(cause, TemporalTimeoutError) and cause.type == TimeoutType.SCHEDULE_TO_START

    async def route_call(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
        *,
        lease: EnvironmentLease,
        wrapped: AbstractToolset[Any],
        fallback: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Dispatch the tool call to `lease.env_queue` instead of the shared queue.

        A no-op pass-through here: the caller (`_DurableEnvWrapper`) has already
        written `lease.model_dump()` to `ctx.metadata['durable_env']`, and
        pydantic-ai's own `resolve_tool_activity_config` reads that back for any
        tool tagged `env_bound` and injects `task_queue=env_queue` itself -- see
        its docstring in `durable_exec/temporal/_toolset.py`. Nothing left for a
        Temporal-specific override to do; this is the routing seam pydantic-ai
        #4977 (`temporal-durability-cap`) provides, no longer a harness prototype.
        """
        return await fallback()
