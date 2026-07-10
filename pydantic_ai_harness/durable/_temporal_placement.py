"""Temporal implementation of the `EnvironmentPlacement` protocol.

Requires `temporalio` (imported here, not gated -- this module is only
reachable through `pydantic_ai_harness.durable.temporal`, which already gates
the import).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from pydantic_ai import AbstractToolset

# Prototype-only imports of pydantic-ai internals: `TemporalFunctionToolset` is the
# temporalized toolset that dispatches a tool through a Temporal activity, and
# `CallToolParams` is the payload its activity deserializes. Both are private today
# because per-tool task-queue routing does not yet exist upstream (pydantic-ai
# #4977). `route_call` below reimplements the activity-config merge from
# `TemporalFunctionToolset.call_tool` purely to add `task_queue`. Delete this whole
# routing path -- and these imports -- once #4977 lands and the core reads
# `ctx.metadata['durable_env']` itself.
from pydantic_ai.durable_exec.temporal._function_toolset import (  # pyright: ignore[reportPrivateImportUsage]
    TemporalFunctionToolset,
)
from pydantic_ai.durable_exec.temporal._toolset import (  # pyright: ignore[reportPrivateImportUsage]
    CallToolParams,
    CallToolResult,
)
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import ToolsetTool
from temporalio import workflow
from temporalio.exceptions import ActivityError, TimeoutType
from temporalio.exceptions import TimeoutError as TemporalTimeoutError
from temporalio.workflow import ActivityConfig

from pydantic_ai_harness.durable._store import AcquireEnvParams, EnvironmentLease

_ACQUIRE_SCHEDULE_TO_START_TIMEOUT = timedelta(seconds=10)
"""Short by design: a schedule-to-start timeout here is how a fenced-out pod is detected quickly."""

_ACQUIRE_START_TO_CLOSE_TIMEOUT = timedelta(seconds=30)


def _walk(toolset: Any) -> Iterator[Any]:
    """Yield `toolset` and every toolset nested under it via `wrapped`/`toolsets`.

    `AbstractToolset.apply` can't be used here: `WrapperToolset.apply` descends
    to the wrapped leaf without visiting the wrapper itself, so it skips the very
    `TemporalWrapperToolset` nodes we need to find. Typed `Any` because the tree
    mixes wrapper (`.wrapped`) and combined (`.toolsets`) shapes reached by
    `getattr`.
    """
    yield toolset
    wrapped = getattr(toolset, 'wrapped', None)
    if wrapped is not None:
        yield from _walk(wrapped)
    for child in getattr(toolset, 'toolsets', ()) or ():
        yield from _walk(child)


async def _temporal_toolset_for(
    name: str, ctx: RunContext[Any], wrapped: AbstractToolset[Any]
) -> TemporalFunctionToolset[Any] | None:
    """Find the `TemporalFunctionToolset` under `wrapped` that owns tool `name`.

    Env-bound toolsets (FileSystem, Shell) are plain `FunctionToolset`s, which
    temporalize to `TemporalFunctionToolset`; tool names are unique across an
    agent's toolsets, so the first match is the one.
    """
    for candidate in _walk(wrapped):
        if isinstance(candidate, TemporalFunctionToolset):
            tfs: TemporalFunctionToolset[Any] = candidate  # pyright: ignore[reportUnknownVariableType]
            if name in await tfs.get_tools(ctx):
                return tfs
    return None


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

        Prototype standing in for pydantic-ai #4977: it reimplements the
        activity-config merge from `TemporalFunctionToolset.call_tool` and adds
        `task_queue=lease.env_queue`, so the env-bound tool runs on the pod that
        holds the workspace. Falls back to the un-routed call when the tool isn't
        dispatched through a Temporal activity (nothing to re-target), so it never
        invents a routing target.
        """
        tfs = await _temporal_toolset_for(name, ctx, wrapped)
        if tfs is None:
            return await fallback()

        tool_activity_config = tfs.tool_activity_config.get(name, {})
        if tool_activity_config is False:
            return await fallback()

        activity_config: ActivityConfig = {
            'summary': f'call tool: {tfs.id}:{name}',
            **tfs.activity_config,
            **tool_activity_config,
            'task_queue': lease.env_queue,
        }
        params = CallToolParams(
            name=name,
            tool_args=tool_args,
            serialized_run_context=tfs.run_context_type.serialize_run_context(ctx),
            tool_def=None,
        )
        result: CallToolResult = await workflow.execute_activity(
            activity=tfs.call_tool_activity, args=[params, ctx.deps], **activity_config
        )
        return tfs._unwrap_call_tool_result(result)  # pyright: ignore[reportPrivateUsage]
