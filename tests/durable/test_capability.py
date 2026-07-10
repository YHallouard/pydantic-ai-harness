"""Tests for DurableEnvironment capability and its _DurableEnvWrapper.

Uses unittest.mock to patch Temporal workflow APIs (`workflow.in_workflow`,
`workflow.info`, `workflow.execute_activity`) since the wrapper runs inside a
sandboxed workflow context we can't instantiate in a unit test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai import AbstractToolset
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import ToolsetTool, WrapperToolset
from pydantic_ai.usage import RunUsage
from pydantic_core import SchemaValidator, core_schema
from temporalio.exceptions import ActivityError, ApplicationError, TimeoutType
from temporalio.exceptions import TimeoutError as TemporalTimeoutError

from pydantic_ai_harness.durable import EnvironmentLease, SnapshotPolicy
from pydantic_ai_harness.durable.temporal import DurableEnvironment

pytestmark = pytest.mark.anyio

_ANY_VALIDATOR = SchemaValidator(schema=core_schema.any_schema())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_def(name: str, *, env_bound: bool = False) -> ToolDefinition:
    metadata = {'env_bound': True} if env_bound else None
    return ToolDefinition(name=name, description=f'{name} tool', metadata=metadata)


@dataclass
class _FakeToolset(AbstractToolset[object]):
    """Minimal toolset that records calls and returns canned results."""

    tool_defs: list[ToolDefinition]
    results: dict[str, Any]
    calls: list[tuple[str, dict[str, Any]]]

    @property
    def id(self) -> str | None:  # pragma: no cover
        return None

    async def get_tools(self, ctx: RunContext[object]) -> dict[str, ToolsetTool[object]]:
        return {
            td.name: ToolsetTool(
                toolset=self,
                tool_def=td,
                max_retries=1,
                args_validator=_ANY_VALIDATOR,
            )
            for td in self.tool_defs
        }

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[object], tool: ToolsetTool[object]
    ) -> Any:
        self.calls.append((name, tool_args))
        result = self.results.get(name)
        if callable(result):
            return result()
        return result


def _build_ctx(*, metadata: dict[str, Any] | None = None) -> RunContext[object]:
    return RunContext[object](
        deps=None,
        model=MagicMock(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
        pending_messages=[],
        metadata=metadata,
    )


def _make_lease(env_queue: str = 'env-q1', epoch: int = 0) -> EnvironmentLease:
    return EnvironmentLease(env_id='wf-123', env_queue=env_queue, epoch=epoch)


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


def _make_non_retryable_error() -> ActivityError:
    cause = ApplicationError('bad input', non_retryable=True)
    error = ActivityError('activity failed', **_ACTIVITY_ERROR_KWARGS)
    error.__cause__ = cause
    return error


# ---------------------------------------------------------------------------
# DurableEnvironment dataclass
# ---------------------------------------------------------------------------


class TestDurableEnvironmentInit:
    def test_string_snapshot_policy_is_coerced_to_model(self) -> None:
        cap = DurableEnvironment[object](snapshot_policy='per_step')
        assert isinstance(cap.snapshot_policy, SnapshotPolicy)
        assert cap.snapshot_policy.mode == 'per_step'

    def test_snapshot_policy_model_is_preserved(self) -> None:
        policy = SnapshotPolicy(mode='content_hash')
        cap = DurableEnvironment[object](snapshot_policy=policy)
        assert cap.snapshot_policy is policy

    def test_get_wrapper_toolset_returns_wrapper(self) -> None:
        cap = DurableEnvironment[object]()
        fake = _FakeToolset(tool_defs=[], results={}, calls=[])
        wrapper = cap.get_wrapper_toolset(fake)
        assert isinstance(wrapper, WrapperToolset)


# ---------------------------------------------------------------------------
# _DurableEnvWrapper -- passthrough paths
# ---------------------------------------------------------------------------


class TestPassthrough:
    async def test_non_env_bound_tool_passes_through_without_lease(self) -> None:
        tool_def = _make_tool_def('search', env_bound=False)
        fake = _FakeToolset(tool_defs=[tool_def], results={'search': 'found'}, calls=[])
        cap = DurableEnvironment[object]()
        wrapper = cap.get_wrapper_toolset(fake)
        assert wrapper is not None

        ctx = _build_ctx()
        tools = await wrapper.get_tools(ctx)
        result = await wrapper.call_tool('search', {'q': 'hello'}, ctx, tools['search'])

        assert result == 'found'
        assert fake.calls == [('search', {'q': 'hello'})]
        assert ctx.metadata is None  # no lease injected

    async def test_env_bound_tool_outside_workflow_passes_through(self) -> None:
        tool_def = _make_tool_def('write_file', env_bound=True)
        fake = _FakeToolset(tool_defs=[tool_def], results={'write_file': 'ok'}, calls=[])
        cap = DurableEnvironment[object]()
        wrapper = cap.get_wrapper_toolset(fake)
        assert wrapper is not None

        ctx = _build_ctx()
        tools = await wrapper.get_tools(ctx)

        with patch('pydantic_ai_harness.durable._capability._in_temporal_workflow', return_value=False):
            result = await wrapper.call_tool('write_file', {'path': '/a'}, ctx, tools['write_file'])

        assert result == 'ok'
        assert ctx.metadata is None


# ---------------------------------------------------------------------------
# _DurableEnvWrapper -- lease acquisition path
# ---------------------------------------------------------------------------

_WORKFLOW_MODULE = 'pydantic_ai_harness.durable._capability'


class TestLeaseAcquisition:
    async def test_env_bound_tool_acquires_lease_and_injects_metadata(self) -> None:
        tool_def = _make_tool_def('write_file', env_bound=True)
        fake = _FakeToolset(tool_defs=[tool_def], results={'write_file': 'written'}, calls=[])
        cap = DurableEnvironment[object]()
        wrapper = cap.get_wrapper_toolset(fake)
        assert wrapper is not None

        ctx = _build_ctx()
        tools = await wrapper.get_tools(ctx)
        lease = _make_lease()

        mock_info = MagicMock()
        mock_info.workflow_id = 'wf-123'

        with (
            patch(f'{_WORKFLOW_MODULE}._in_temporal_workflow', return_value=True),
            patch(f'{_WORKFLOW_MODULE}.workflow.info', return_value=mock_info),
            patch(f'{_WORKFLOW_MODULE}.workflow.execute_activity', new_callable=AsyncMock, return_value=lease),
        ):
            result = await wrapper.call_tool('write_file', {'path': '/a'}, ctx, tools['write_file'])

        assert result == 'written'
        assert ctx.metadata is not None
        assert ctx.metadata['durable_env'] == lease.model_dump()

    async def test_lease_is_memoized_across_calls(self) -> None:
        tool_def = _make_tool_def('write_file', env_bound=True)
        fake = _FakeToolset(tool_defs=[tool_def], results={'write_file': 'ok'}, calls=[])
        cap = DurableEnvironment[object]()
        wrapper = cap.get_wrapper_toolset(fake)
        assert wrapper is not None

        ctx = _build_ctx()
        tools = await wrapper.get_tools(ctx)
        lease = _make_lease()

        mock_info = MagicMock()
        mock_info.workflow_id = 'wf-123'
        mock_execute = AsyncMock(return_value=lease)

        with (
            patch(f'{_WORKFLOW_MODULE}._in_temporal_workflow', return_value=True),
            patch(f'{_WORKFLOW_MODULE}.workflow.info', return_value=mock_info),
            patch(f'{_WORKFLOW_MODULE}.workflow.execute_activity', mock_execute),
        ):
            await wrapper.call_tool('write_file', {'path': '/a'}, ctx, tools['write_file'])
            await wrapper.call_tool('write_file', {'path': '/b'}, ctx, tools['write_file'])

        # acquire_environment called only once -- lease is memoized
        mock_execute.assert_awaited_once()
        assert len(fake.calls) == 2


# ---------------------------------------------------------------------------
# _DurableEnvWrapper -- re-provision on schedule-to-start timeout
# ---------------------------------------------------------------------------


class TestReprovision:
    async def test_schedule_to_start_timeout_triggers_reacquire(self) -> None:
        tool_def = _make_tool_def('write_file', env_bound=True)
        call_count = 0

        def tool_side_effect() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _make_schedule_to_start_error()
            return 'ok'

        fake = _FakeToolset(tool_defs=[tool_def], results={'write_file': tool_side_effect}, calls=[])
        cap = DurableEnvironment[object]()
        wrapper = cap.get_wrapper_toolset(fake)
        assert wrapper is not None

        ctx = _build_ctx()
        tools = await wrapper.get_tools(ctx)
        lease_a = _make_lease(env_queue='env-q-a')
        lease_b = _make_lease(env_queue='env-q-b')

        mock_info = MagicMock()
        mock_info.workflow_id = 'wf-123'
        mock_execute = AsyncMock(side_effect=[lease_a, lease_b])

        with (
            patch(f'{_WORKFLOW_MODULE}._in_temporal_workflow', return_value=True),
            patch(f'{_WORKFLOW_MODULE}.workflow.info', return_value=mock_info),
            patch(f'{_WORKFLOW_MODULE}.workflow.execute_activity', mock_execute),
        ):
            result = await wrapper.call_tool('write_file', {'path': '/a'}, ctx, tools['write_file'])

        assert result == 'ok'
        # Two acquire calls: initial + re-acquire after timeout
        assert mock_execute.await_count == 2
        # Metadata reflects the second lease
        assert ctx.metadata is not None
        assert ctx.metadata['durable_env']['env_queue'] == 'env-q-b'

    async def test_non_retryable_error_propagates_immediately(self) -> None:
        tool_def = _make_tool_def('write_file', env_bound=True)

        def tool_side_effect() -> str:
            raise _make_non_retryable_error()

        fake = _FakeToolset(tool_defs=[tool_def], results={'write_file': tool_side_effect}, calls=[])
        cap = DurableEnvironment[object]()
        wrapper = cap.get_wrapper_toolset(fake)
        assert wrapper is not None

        ctx = _build_ctx()
        tools = await wrapper.get_tools(ctx)
        lease = _make_lease()

        mock_info = MagicMock()
        mock_info.workflow_id = 'wf-123'

        with (
            patch(f'{_WORKFLOW_MODULE}._in_temporal_workflow', return_value=True),
            patch(f'{_WORKFLOW_MODULE}.workflow.info', return_value=mock_info),
            patch(f'{_WORKFLOW_MODULE}.workflow.execute_activity', new_callable=AsyncMock, return_value=lease),
        ):
            with pytest.raises(ActivityError):
                await wrapper.call_tool('write_file', {'path': '/a'}, ctx, tools['write_file'])

    async def test_max_reprovisions_exhausted_raises(self) -> None:
        tool_def = _make_tool_def('write_file', env_bound=True)

        def tool_side_effect() -> str:
            raise _make_schedule_to_start_error()

        fake = _FakeToolset(tool_defs=[tool_def], results={'write_file': tool_side_effect}, calls=[])
        cap = DurableEnvironment[object]()
        wrapper = cap.get_wrapper_toolset(fake)
        assert wrapper is not None

        ctx = _build_ctx()
        tools = await wrapper.get_tools(ctx)

        mock_info = MagicMock()
        mock_info.workflow_id = 'wf-123'
        # 1 initial + 3 re-acquire = 4 leases total before giving up
        leases = [_make_lease(env_queue=f'q-{i}') for i in range(4)]
        mock_execute = AsyncMock(side_effect=leases)

        with (
            patch(f'{_WORKFLOW_MODULE}._in_temporal_workflow', return_value=True),
            patch(f'{_WORKFLOW_MODULE}.workflow.info', return_value=mock_info),
            patch(f'{_WORKFLOW_MODULE}.workflow.execute_activity', mock_execute),
        ):
            with pytest.raises(ActivityError):
                await wrapper.call_tool('write_file', {'path': '/a'}, ctx, tools['write_file'])

        # 4 acquire calls: initial + 3 retries
        assert mock_execute.await_count == 4


# ---------------------------------------------------------------------------
# _DurableEnvWrapper -- metadata handling
# ---------------------------------------------------------------------------


class TestMetadataHandling:
    async def test_existing_metadata_is_preserved(self) -> None:
        tool_def = _make_tool_def('write_file', env_bound=True)
        fake = _FakeToolset(tool_defs=[tool_def], results={'write_file': 'ok'}, calls=[])
        cap = DurableEnvironment[object]()
        wrapper = cap.get_wrapper_toolset(fake)
        assert wrapper is not None

        ctx = _build_ctx(metadata={'user_key': 'preserved'})
        tools = await wrapper.get_tools(ctx)
        lease = _make_lease()

        mock_info = MagicMock()
        mock_info.workflow_id = 'wf-123'

        with (
            patch(f'{_WORKFLOW_MODULE}._in_temporal_workflow', return_value=True),
            patch(f'{_WORKFLOW_MODULE}.workflow.info', return_value=mock_info),
            patch(f'{_WORKFLOW_MODULE}.workflow.execute_activity', new_callable=AsyncMock, return_value=lease),
        ):
            await wrapper.call_tool('write_file', {'path': '/a'}, ctx, tools['write_file'])

        assert ctx.metadata is not None
        assert ctx.metadata['user_key'] == 'preserved'
        assert 'durable_env' in ctx.metadata

    async def test_null_metadata_is_initialized(self) -> None:
        tool_def = _make_tool_def('write_file', env_bound=True)
        fake = _FakeToolset(tool_defs=[tool_def], results={'write_file': 'ok'}, calls=[])
        cap = DurableEnvironment[object]()
        wrapper = cap.get_wrapper_toolset(fake)
        assert wrapper is not None

        ctx = _build_ctx(metadata=None)
        tools = await wrapper.get_tools(ctx)
        lease = _make_lease()

        mock_info = MagicMock()
        mock_info.workflow_id = 'wf-123'

        with (
            patch(f'{_WORKFLOW_MODULE}._in_temporal_workflow', return_value=True),
            patch(f'{_WORKFLOW_MODULE}.workflow.info', return_value=mock_info),
            patch(f'{_WORKFLOW_MODULE}.workflow.execute_activity', new_callable=AsyncMock, return_value=lease),
        ):
            await wrapper.call_tool('write_file', {'path': '/a'}, ctx, tools['write_file'])

        assert ctx.metadata is not None
        assert 'durable_env' in ctx.metadata
