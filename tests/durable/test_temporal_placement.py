"""Tests for TemporalPlacement, the Temporal `EnvironmentPlacement` driver.

Uses unittest.mock to patch Temporal workflow APIs (`workflow.in_workflow`,
`workflow.info`, `workflow.execute_activity`) since the driver runs inside a
sandboxed workflow context we can't instantiate in a unit test.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai import AbstractToolset, CombinedToolset, FunctionToolset, WrapperToolset
from pydantic_ai.durable_exec.temporal._function_toolset import TemporalFunctionToolset
from pydantic_ai.durable_exec.temporal._toolset import _ToolReturn
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import ToolsetTool
from pydantic_ai.usage import RunUsage
from temporalio.exceptions import ActivityError, ApplicationError, TimeoutType
from temporalio.exceptions import TimeoutError as TemporalTimeoutError
from temporalio.workflow import ActivityConfig

from pydantic_ai_harness.durable import AcquireEnvParams, EnvironmentLease
from pydantic_ai_harness.durable.temporal import TemporalPlacement

pytestmark = pytest.mark.anyio

_MODULE = 'pydantic_ai_harness.durable._temporal_placement'

_ACTIVITY_ERROR_KWARGS: dict[str, Any] = {
    'scheduled_event_id': 1,
    'started_event_id': 1,
    'identity': 'test-worker',
    'activity_type': 'write_file',
    'activity_id': '1',
    'retry_state': None,
}


def _make_schedule_to_start_error() -> ActivityError:
    timeout = TemporalTimeoutError('timed out', type=TimeoutType.SCHEDULE_TO_START, last_heartbeat_details=[])
    error = ActivityError('activity failed', **_ACTIVITY_ERROR_KWARGS)
    error.__cause__ = timeout
    return error


class TestActive:
    def test_true_inside_a_workflow(self) -> None:
        with patch(f'{_MODULE}.workflow.in_workflow', return_value=True):
            assert TemporalPlacement().active() is True

    def test_false_outside_a_workflow(self) -> None:
        assert TemporalPlacement().active() is False


class TestAcquire:
    async def test_calls_the_acquire_activity_with_the_workflow_id(self) -> None:
        lease = EnvironmentLease(env_id='wf-123', env_queue='env-q1', epoch=0)
        mock_info = MagicMock()
        mock_info.workflow_id = 'wf-123'
        mock_execute = AsyncMock(return_value=lease)

        with (
            patch(f'{_MODULE}.workflow.info', return_value=mock_info),
            patch(f'{_MODULE}.workflow.execute_activity', mock_execute),
        ):
            result = await TemporalPlacement().acquire(failed_queue='env-dead')

        assert result is lease
        mock_execute.assert_awaited_once()
        call = mock_execute.await_args
        assert call is not None
        assert call.args[0] == 'acquire_environment'
        assert call.args[1] == AcquireEnvParams(env_id='wf-123', failed_queue='env-dead')
        assert call.kwargs['result_type'] is EnvironmentLease
        assert call.kwargs['schedule_to_start_timeout'] is not None
        assert call.kwargs['start_to_close_timeout'] is not None


class TestIsPlacementFailure:
    def test_true_for_a_schedule_to_start_timeout(self) -> None:
        assert TemporalPlacement().is_placement_failure(_make_schedule_to_start_error()) is True

    def test_false_for_an_activity_error_with_another_cause(self) -> None:
        error = ActivityError('activity failed', **_ACTIVITY_ERROR_KWARGS)
        error.__cause__ = ApplicationError('bad input', non_retryable=True)
        assert TemporalPlacement().is_placement_failure(error) is False

    def test_false_for_a_non_activity_error(self) -> None:
        assert TemporalPlacement().is_placement_failure(ValueError('nope')) is False


def _build_ctx() -> RunContext[object]:
    return RunContext[object](
        deps=None,
        model=MagicMock(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
        pending_messages=[],
        metadata=None,
    )


async def write_file(path: str) -> str:
    """Write a file (test stub)."""
    return f'wrote {path}'  # pragma: no cover -- never actually invoked; routing is mocked


async def other_tool(x: int) -> int:
    """An unrelated tool on a different toolset (test stub)."""
    return x  # pragma: no cover -- never actually invoked; routing is mocked


@dataclass
class _Passthrough(WrapperToolset[object]):
    """A non-temporalized wrapper, so `_walk` must descend past it into `.wrapped`."""

    @property
    def id(self) -> str | None:  # pragma: no cover
        return None


def _temporalize(
    toolset: FunctionToolset[object], *, config: ActivityConfig | None = None, per_tool: dict[str, Any] | None = None
) -> TemporalFunctionToolset[object]:
    return TemporalFunctionToolset(
        toolset,
        activity_name_prefix='test',
        activity_config=config or ActivityConfig(start_to_close_timeout=timedelta(seconds=30)),
        tool_activity_config=per_tool or {},
        deps_type=object,
    )


@dataclass
class _PlainToolset(AbstractToolset[object]):
    """A leaf toolset that is not temporalized -- nothing for routing to re-target."""

    @property
    def id(self) -> str | None:  # pragma: no cover
        return None

    async def get_tools(self, ctx: RunContext[object]) -> dict[str, ToolsetTool[object]]:  # pragma: no cover
        return {}

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[object], tool: ToolsetTool[object]
    ) -> Any:  # pragma: no cover
        return None


_LEASE = EnvironmentLease(env_id='wf-123', env_queue='env-sticky', epoch=0)


class TestRouteCall:
    async def test_routes_the_call_to_the_lease_env_queue(self) -> None:
        tfs = _temporalize(FunctionToolset(tools=[write_file], id='fs'))
        ctx = _build_ctx()

        with patch(f'{_MODULE}.workflow.execute_activity', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = _ToolReturn(result='written')
            result = await TemporalPlacement().route_call(
                'write_file',
                {'path': '/a'},
                ctx,
                MagicMock(),
                lease=_LEASE,
                wrapped=tfs,
                fallback=AsyncMock(),
            )

        assert result == 'written'
        mock_execute.assert_awaited_once()
        assert mock_execute.await_args is not None
        kwargs = mock_execute.await_args.kwargs
        assert kwargs['task_queue'] == 'env-sticky'
        assert kwargs['activity'] is tfs.call_tool_activity
        params = kwargs['args'][0]
        assert params.name == 'write_file'
        assert params.tool_args == {'path': '/a'}

    async def test_merges_base_and_per_tool_activity_config(self) -> None:
        base = ActivityConfig(start_to_close_timeout=timedelta(seconds=30))
        tfs = _temporalize(
            FunctionToolset(tools=[write_file], id='fs'),
            config=base,
            per_tool={'write_file': ActivityConfig(schedule_to_close_timeout=timedelta(seconds=99))},
        )
        ctx = _build_ctx()

        with patch(f'{_MODULE}.workflow.execute_activity', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = _ToolReturn(result='ok')
            await TemporalPlacement().route_call(
                'write_file',
                {'path': '/a'},
                ctx,
                MagicMock(),
                lease=_LEASE,
                wrapped=tfs,
                fallback=AsyncMock(),
            )

        assert mock_execute.await_args is not None
        kwargs = mock_execute.await_args.kwargs
        assert kwargs['start_to_close_timeout'] == timedelta(seconds=30)  # from base
        assert kwargs['schedule_to_close_timeout'] == timedelta(seconds=99)  # from per-tool
        assert kwargs['task_queue'] == 'env-sticky'  # injected by the router

    async def test_finds_the_toolset_nested_in_a_combined_toolset(self) -> None:
        tfs = _temporalize(FunctionToolset(tools=[write_file], id='fs'))
        combined: CombinedToolset[object] = CombinedToolset([_PlainToolset(), tfs])
        ctx = _build_ctx()

        with patch(f'{_MODULE}.workflow.execute_activity', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = _ToolReturn(result='nested')
            result = await TemporalPlacement().route_call(
                'write_file',
                {'path': '/a'},
                ctx,
                MagicMock(),
                lease=_LEASE,
                wrapped=combined,
                fallback=AsyncMock(),
            )

        assert result == 'nested'
        mock_execute.assert_awaited_once()

    async def test_descends_past_wrappers_and_skips_toolsets_without_the_tool(self) -> None:
        # Exercises _walk descending into `.wrapped`, and skipping a temporalized
        # toolset that owns a different tool before matching the right one.
        tfs_other = _temporalize(FunctionToolset(tools=[other_tool], id='other'))
        tfs = _temporalize(FunctionToolset(tools=[write_file], id='fs'))
        tree = _Passthrough(wrapped=CombinedToolset([tfs_other, tfs]))
        ctx = _build_ctx()

        with patch(f'{_MODULE}.workflow.execute_activity', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = _ToolReturn(result='deep')
            result = await TemporalPlacement().route_call(
                'write_file',
                {'path': '/a'},
                ctx,
                MagicMock(),
                lease=_LEASE,
                wrapped=tree,
                fallback=AsyncMock(),
            )

        assert result == 'deep'
        assert mock_execute.await_args is not None
        assert mock_execute.await_args.kwargs['activity'] is tfs.call_tool_activity

    async def test_falls_back_when_no_temporalized_toolset_owns_the_tool(self) -> None:
        fallback = AsyncMock(return_value='fell-back')
        ctx = _build_ctx()

        with patch(f'{_MODULE}.workflow.execute_activity', new_callable=AsyncMock) as mock_execute:
            result = await TemporalPlacement().route_call(
                'write_file',
                {'path': '/a'},
                ctx,
                MagicMock(),
                lease=_LEASE,
                wrapped=_PlainToolset(),
                fallback=fallback,
            )

        assert result == 'fell-back'
        fallback.assert_awaited_once()
        mock_execute.assert_not_awaited()

    async def test_falls_back_when_the_tool_activity_is_disabled(self) -> None:
        tfs = _temporalize(FunctionToolset(tools=[write_file], id='fs'), per_tool={'write_file': False})
        fallback = AsyncMock(return_value='fell-back')
        ctx = _build_ctx()

        with patch(f'{_MODULE}.workflow.execute_activity', new_callable=AsyncMock) as mock_execute:
            result = await TemporalPlacement().route_call(
                'write_file',
                {'path': '/a'},
                ctx,
                MagicMock(),
                lease=_LEASE,
                wrapped=tfs,
                fallback=fallback,
            )

        assert result == 'fell-back'
        fallback.assert_awaited_once()
        mock_execute.assert_not_awaited()
