"""Temporal implementation of the `EnvironmentPlacement` protocol.

Requires `temporalio` (imported here, not gated -- this module is only
reachable through `pydantic_ai_harness.durable.temporal`, which already gates
the import).
"""

from __future__ import annotations

import asyncio
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

ACQUIRE_TASK_QUEUE = 'durable-env-acquire'
"""Default task queue for a dedicated `acquire_environment` worker gated by
`CapacityGatedSlotSupplier`. A constant rather than config so workflows and the acquire worker
agree on the queue without sharing configuration; pass it as `TemporalPlacement.host_task_queue`."""

_ACQUIRE_SCHEDULE_TO_START_TIMEOUT = timedelta(seconds=10)
"""Short by design: a schedule-to-start timeout on a *routed* call (a tool activity on a sticky
`env_queue`) is how a fenced-out or dead pod is detected quickly. On the acquire call itself the
same timeout means "no acquire worker has free capacity right now", which `acquire` absorbs and
retries rather than surfacing as a placement failure."""

_ACQUIRE_START_TO_CLOSE_TIMEOUT = timedelta(seconds=30)

_ACQUIRE_RETRY_INITIAL_INTERVAL = timedelta(seconds=1)
_ACQUIRE_RETRY_MAX_INTERVAL = timedelta(minutes=1)


@dataclass
class TemporalPlacement:
    """`EnvironmentPlacement` driver for Temporal.

    Workflow-side counterpart of `EnvironmentActivities`: `acquire` calls the
    `acquire_environment` activity (registered on the host queue by
    `DurableEnvironmentPlugin`, or on a dedicated `ACQUIRE_TASK_QUEUE`), and a
    schedule-to-start timeout on a *routed tool call* (on a sticky `env_queue`)
    is the signal that the leased pod is dead or fenced out. The same timeout on
    the acquire call means the env fleet is at capacity; `acquire` treats that as
    queueing and waits, so it never reaches the reprovision path.
    """

    host_task_queue: str | None = None
    """Task queue where `acquire_environment` is polled. Unset schedules `acquire`
    on the calling workflow's own task queue (requires the plugin mounted there);
    set it to a dedicated queue (e.g. `ACQUIRE_TASK_QUEUE`) when acquire runs on a
    separate, capacity-gated worker, or to any queue where the plugin's host
    activities are registered in a multi-queue topology."""

    def active(self) -> bool:
        return workflow.in_workflow()

    async def acquire(self, *, failed_queue: str | None) -> EnvironmentLease:
        env_id = workflow.info().workflow_id
        params = AcquireEnvParams(env_id=env_id, failed_queue=failed_queue)
        backoff = _ACQUIRE_RETRY_INITIAL_INTERVAL
        while True:
            try:
                return await workflow.execute_activity(
                    'acquire_environment',
                    params,
                    result_type=EnvironmentLease,
                    task_queue=self.host_task_queue,
                    schedule_to_start_timeout=_ACQUIRE_SCHEDULE_TO_START_TIMEOUT,
                    start_to_close_timeout=_ACQUIRE_START_TO_CLOSE_TIMEOUT,
                )
            except ActivityError as exc:
                if not self.is_placement_failure(exc):
                    raise
                # A schedule-to-start timeout on the acquire call means no acquire worker had
                # free capacity in time (under `CapacityGatedSlotSupplier`, full workers stop
                # polling). That is queueing, not a placement failure: wait and retry rather than
                # reprovisioning or failing the run. `asyncio.sleep` in a workflow is durable.
                workflow.logger.info('acquire_environment not scheduled; env fleet at capacity, retrying')
                await asyncio.sleep(backoff.total_seconds())
                backoff = min(backoff * 2, _ACQUIRE_RETRY_MAX_INTERVAL)

    def is_placement_failure(self, exc: Exception) -> bool:
        """Whether `exc` is a schedule-to-start timeout (a routed call reaching a dead/fenced pod).

        Used two ways with disjoint inputs: `_DurableEnvWrapper`'s reprovision loop passes tool-call
        errors (a `True` here means "reacquire"), and `acquire` passes its own acquire errors (a
        `True` there means "at capacity, wait"). Acquire timeouts never escape `acquire`'s own loop,
        so the reprovision loop only ever sees the dead-pod meaning.
        """
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
