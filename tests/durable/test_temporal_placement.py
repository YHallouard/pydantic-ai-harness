"""Tests for TemporalPlacement, the Temporal `EnvironmentPlacement` driver.

Uses unittest.mock to patch Temporal workflow APIs (`workflow.in_workflow`,
`workflow.info`, `workflow.execute_activity`) since the driver runs inside a
sandboxed workflow context we can't instantiate in a unit test.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai import AbstractToolset
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from temporalio.exceptions import ActivityError, ApplicationError, TimeoutType
from temporalio.exceptions import TimeoutError as TemporalTimeoutError

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


_LEASE = EnvironmentLease(env_id='wf-123', env_queue='env-sticky', epoch=0)


class TestRouteCall:
    """`route_call` is a pass-through: pydantic-ai's own `resolve_tool_activity_config`
    (from #4977's `temporal-durability-cap`) reads `ctx.metadata['durable_env']['env_queue']`
    and routes the activity itself once `_DurableEnvWrapper` has written the lease there --
    see `TemporalPlacement.route_call`'s docstring. Nothing here re-targets the call, so
    the only contract to test is "always defers to fallback, untouched".
    """

    async def test_always_defers_to_fallback(self) -> None:
        fallback = AsyncMock(return_value='fallback-result')
        ctx = _build_ctx()

        result = await TemporalPlacement().route_call(
            'write_file',
            {'path': '/a'},
            ctx,
            MagicMock(),
            lease=_LEASE,
            wrapped=MagicMock(spec=AbstractToolset),
            fallback=fallback,
        )

        assert result == 'fallback-result'
        fallback.assert_awaited_once_with()
