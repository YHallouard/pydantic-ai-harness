"""Tests for TemporalPlacement, the Temporal `EnvironmentPlacement` driver.

Uses unittest.mock to patch Temporal workflow APIs (`workflow.in_workflow`,
`workflow.info`, `workflow.execute_activity`) since the driver runs inside a
sandboxed workflow context we can't instantiate in a unit test.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
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


class TestRouteCall:
    async def test_delegates_to_the_fallback(self) -> None:
        fallback = AsyncMock(return_value='routed')
        lease = EnvironmentLease(env_id='wf-123', env_queue='env-q1', epoch=0)

        result = await TemporalPlacement().route_call(
            'write_file',
            {'path': '/a'},
            MagicMock(),
            MagicMock(),
            lease=lease,
            wrapped=MagicMock(),
            fallback=fallback,
        )

        assert result == 'routed'
        fallback.assert_awaited_once()
