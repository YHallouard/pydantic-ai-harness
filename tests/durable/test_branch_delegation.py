"""Unit tests for `_branch_delegation`'s lease-resolution and driver-gating branches.

Uses `unittest.mock` to patch the `workflow` module (`workflow.in_workflow`,
`workflow.info`, `workflow.execute_activity`), same approach as
`test_temporal_placement.py`, since these functions only behave correctly inside
a sandboxed Temporal workflow context that a unit test can't instantiate.
`run_with_self_heal`'s own fork/land/self-heal loop is covered live against a
real Temporal test server in `test_branch_delegation_integration.py`; this file
covers the branches around it that don't need a live server: `_parent_lease`'s
three outcomes and `TemporalBranchDelegation.run_delegation`'s early return.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.durable._branch_delegation import (
    TemporalBranchDelegation,
    resolve_parent_environment_lease,
)
from pydantic_ai_harness.durable._store import EnvironmentLease

pytestmark = pytest.mark.anyio

_MODULE = 'pydantic_ai_harness.durable._branch_delegation'


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


class TestResolveParentEnvironmentLease:
    async def test_none_outside_a_workflow(self) -> None:
        with patch(f'{_MODULE}.workflow.in_workflow', return_value=False):
            assert await resolve_parent_environment_lease(host_task_queue='shared') is None

    async def test_none_when_this_workflow_has_no_parent(self) -> None:
        mock_info = MagicMock()
        mock_info.parent = None
        with (
            patch(f'{_MODULE}.workflow.in_workflow', return_value=True),
            patch(f'{_MODULE}.workflow.info', return_value=mock_info),
        ):
            assert await resolve_parent_environment_lease(host_task_queue='shared') is None

    async def test_none_when_the_host_reports_no_held_environment_for_the_parent(self) -> None:
        mock_info = MagicMock()
        mock_info.parent.workflow_id = 'parent-wf'
        mock_execute = AsyncMock(return_value=None)
        with (
            patch(f'{_MODULE}.workflow.in_workflow', return_value=True),
            patch(f'{_MODULE}.workflow.info', return_value=mock_info),
            patch(f'{_MODULE}.workflow.execute_activity', mock_execute),
        ):
            assert await resolve_parent_environment_lease(host_task_queue='shared') is None

    async def test_builds_a_lease_from_the_parent_workflow_id_and_reported_queue(self) -> None:
        mock_info = MagicMock()
        mock_info.parent.workflow_id = 'parent-wf'
        mock_execute = AsyncMock(return_value='parent-env-queue')
        with (
            patch(f'{_MODULE}.workflow.in_workflow', return_value=True),
            patch(f'{_MODULE}.workflow.info', return_value=mock_info),
            patch(f'{_MODULE}.workflow.execute_activity', mock_execute),
        ):
            lease = await resolve_parent_environment_lease(host_task_queue='shared')

        assert lease == EnvironmentLease(env_id='parent-wf', env_queue='parent-env-queue', epoch=0)
        call = mock_execute.await_args
        assert call is not None
        assert call.args[0] == 'get_environment_queue'
        assert call.args[1] == 'parent-wf'
        assert call.kwargs['task_queue'] == 'shared'


class TestTemporalBranchDelegationParentLease:
    async def test_uses_the_metadata_lease_when_present(self) -> None:
        ctx = _build_ctx(metadata={'durable_env': {'env_id': 'e1', 'env_queue': 'q1', 'epoch': 0}})
        driver = TemporalBranchDelegation(host_task_queue='shared')

        lease = await driver._parent_lease(ctx)

        assert lease == EnvironmentLease(env_id='e1', env_queue='q1', epoch=0)

    async def test_falls_back_to_host_resolution_when_metadata_is_absent(self) -> None:
        """No `durable_env` metadata (the nested-child-workflow case) with
        `host_task_queue` set must ask the host worker instead of giving up."""
        ctx = _build_ctx(metadata=None)
        driver = TemporalBranchDelegation(host_task_queue='shared')
        resolved = EnvironmentLease(env_id='parent-wf', env_queue='parent-q', epoch=0)

        with patch(f'{_MODULE}.resolve_parent_environment_lease', AsyncMock(return_value=resolved)) as mock_resolve:
            lease = await driver._parent_lease(ctx)

        assert lease == resolved
        mock_resolve.assert_awaited_once_with(host_task_queue='shared')

    async def test_none_when_metadata_is_absent_and_no_host_task_queue_is_configured(self) -> None:
        ctx = _build_ctx(metadata=None)
        driver = TemporalBranchDelegation(host_task_queue=None)

        assert await driver._parent_lease(ctx) is None


class TestTemporalBranchDelegationRunDelegation:
    async def test_returns_none_when_no_parent_lease_can_be_resolved(self) -> None:
        """`run_delegation` returning `None` (not raising) is the documented signal
        `SubAgentToolset.delegate_task` uses to fall back to a plain, non-isolated run."""
        ctx = _build_ctx(metadata=None)
        driver = TemporalBranchDelegation(host_task_queue=None)

        async def run_once(task: str) -> tuple[str, bool]:
            raise AssertionError('run_once must not be called when no parent workspace is attached')

        result = await driver.run_delegation(ctx, run_once=run_once, task='do it', max_merge_retries=1)

        assert result is None
