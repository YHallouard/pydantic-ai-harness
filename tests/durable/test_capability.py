"""Tests for the DurableEnvironment capability and its _DurableEnvWrapper.

Drives the wrapper through a fake `EnvironmentPlacement` -- the engine-neutral
seam the capability delegates to -- so no Temporal mocking is needed here.
`TemporalPlacement`'s own behavior is covered in `test_temporal_placement.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic_ai import AbstractToolset
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import ToolsetTool, WrapperToolset
from pydantic_ai.usage import RunUsage
from pydantic_core import SchemaValidator, core_schema

from pydantic_ai_harness.durable import DurableEnvironment, EnvironmentLease, SnapshotPolicy

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


class _PlacementFailure(Exception):
    """The failure shape `_FakePlacement.is_placement_failure` recognizes."""


@dataclass
class _FakePlacement:
    """`EnvironmentPlacement` fake: hands out canned leases and delegates routing to `fallback`."""

    leases: list[EnvironmentLease]
    active_flag: bool = True
    acquires: list[str | None] = field(default_factory=list[str | None], init=False)

    def active(self) -> bool:
        return self.active_flag

    async def acquire(self, *, failed_queue: str | None) -> EnvironmentLease:
        self.acquires.append(failed_queue)
        return self.leases[len(self.acquires) - 1]

    def is_placement_failure(self, exc: Exception) -> bool:
        return isinstance(exc, _PlacementFailure)

    async def route_call(self, *args: Any, **kwargs: Any) -> Any:
        return await kwargs['fallback']()


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


def _make_capability(placement: _FakePlacement, **kwargs: Any) -> DurableEnvironment[object]:
    return DurableEnvironment[object](placement=placement, **kwargs)


# ---------------------------------------------------------------------------
# DurableEnvironment dataclass
# ---------------------------------------------------------------------------


class TestDurableEnvironmentInit:
    def test_string_snapshot_policy_is_coerced_to_model(self) -> None:
        cap = _make_capability(_FakePlacement(leases=[]), snapshot_policy='per_step')
        assert isinstance(cap.snapshot_policy, SnapshotPolicy)
        assert cap.snapshot_policy.mode == 'per_step'

    def test_snapshot_policy_model_is_preserved(self) -> None:
        policy = SnapshotPolicy(mode='content_hash')
        cap = _make_capability(_FakePlacement(leases=[]), snapshot_policy=policy)
        assert cap.snapshot_policy is policy

    def test_get_wrapper_toolset_returns_wrapper(self) -> None:
        cap = _make_capability(_FakePlacement(leases=[]))
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
        placement = _FakePlacement(leases=[_make_lease()])
        wrapper = _make_capability(placement).get_wrapper_toolset(fake)
        assert wrapper is not None

        ctx = _build_ctx()
        tools = await wrapper.get_tools(ctx)
        result = await wrapper.call_tool('search', {'q': 'hello'}, ctx, tools['search'])

        assert result == 'found'
        assert fake.calls == [('search', {'q': 'hello'})]
        assert placement.acquires == []
        assert ctx.metadata is None  # no lease injected

    async def test_env_bound_tool_outside_durable_context_passes_through(self) -> None:
        tool_def = _make_tool_def('write_file', env_bound=True)
        fake = _FakeToolset(tool_defs=[tool_def], results={'write_file': 'ok'}, calls=[])
        placement = _FakePlacement(leases=[_make_lease()], active_flag=False)
        wrapper = _make_capability(placement).get_wrapper_toolset(fake)
        assert wrapper is not None

        ctx = _build_ctx()
        tools = await wrapper.get_tools(ctx)
        result = await wrapper.call_tool('write_file', {'path': '/a'}, ctx, tools['write_file'])

        assert result == 'ok'
        assert placement.acquires == []
        assert ctx.metadata is None


# ---------------------------------------------------------------------------
# _DurableEnvWrapper -- lease acquisition path
# ---------------------------------------------------------------------------


class TestLeaseAcquisition:
    async def test_env_bound_tool_acquires_lease_and_injects_metadata(self) -> None:
        tool_def = _make_tool_def('write_file', env_bound=True)
        fake = _FakeToolset(tool_defs=[tool_def], results={'write_file': 'written'}, calls=[])
        lease = _make_lease()
        placement = _FakePlacement(leases=[lease])
        wrapper = _make_capability(placement).get_wrapper_toolset(fake)
        assert wrapper is not None

        ctx = _build_ctx()
        tools = await wrapper.get_tools(ctx)
        result = await wrapper.call_tool('write_file', {'path': '/a'}, ctx, tools['write_file'])

        assert result == 'written'
        assert placement.acquires == [None]
        assert ctx.metadata is not None
        assert ctx.metadata['durable_env'] == lease.model_dump()

    async def test_lease_is_memoized_across_calls(self) -> None:
        tool_def = _make_tool_def('write_file', env_bound=True)
        fake = _FakeToolset(tool_defs=[tool_def], results={'write_file': 'ok'}, calls=[])
        placement = _FakePlacement(leases=[_make_lease()])
        wrapper = _make_capability(placement).get_wrapper_toolset(fake)
        assert wrapper is not None

        ctx = _build_ctx()
        tools = await wrapper.get_tools(ctx)
        await wrapper.call_tool('write_file', {'path': '/a'}, ctx, tools['write_file'])
        await wrapper.call_tool('write_file', {'path': '/b'}, ctx, tools['write_file'])

        # acquire called only once -- lease is memoized
        assert placement.acquires == [None]
        assert len(fake.calls) == 2


# ---------------------------------------------------------------------------
# _DurableEnvWrapper -- re-provision on placement failure
# ---------------------------------------------------------------------------


class TestReprovision:
    async def test_placement_failure_triggers_reacquire(self) -> None:
        tool_def = _make_tool_def('write_file', env_bound=True)
        call_count = 0

        def tool_side_effect() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _PlacementFailure()
            return 'ok'

        fake = _FakeToolset(tool_defs=[tool_def], results={'write_file': tool_side_effect}, calls=[])
        placement = _FakePlacement(leases=[_make_lease(env_queue='env-q-a'), _make_lease(env_queue='env-q-b')])
        wrapper = _make_capability(placement).get_wrapper_toolset(fake)
        assert wrapper is not None

        ctx = _build_ctx()
        tools = await wrapper.get_tools(ctx)
        result = await wrapper.call_tool('write_file', {'path': '/a'}, ctx, tools['write_file'])

        assert result == 'ok'
        # Two acquire calls: initial + re-acquire naming the failed queue
        assert placement.acquires == [None, 'env-q-a']
        # Metadata reflects the second lease
        assert ctx.metadata is not None
        assert ctx.metadata['durable_env']['env_queue'] == 'env-q-b'

    async def test_non_placement_error_propagates_immediately(self) -> None:
        tool_def = _make_tool_def('write_file', env_bound=True)

        def tool_side_effect() -> str:
            raise ValueError('bad input')

        fake = _FakeToolset(tool_defs=[tool_def], results={'write_file': tool_side_effect}, calls=[])
        placement = _FakePlacement(leases=[_make_lease()])
        wrapper = _make_capability(placement).get_wrapper_toolset(fake)
        assert wrapper is not None

        ctx = _build_ctx()
        tools = await wrapper.get_tools(ctx)

        with pytest.raises(ValueError):
            await wrapper.call_tool('write_file', {'path': '/a'}, ctx, tools['write_file'])

        assert placement.acquires == [None]

    async def test_max_reprovisions_exhausted_raises(self) -> None:
        tool_def = _make_tool_def('write_file', env_bound=True)

        def tool_side_effect() -> str:
            raise _PlacementFailure()

        fake = _FakeToolset(tool_defs=[tool_def], results={'write_file': tool_side_effect}, calls=[])
        # 1 initial + 3 re-acquire = 4 leases total before giving up
        placement = _FakePlacement(leases=[_make_lease(env_queue=f'q-{i}') for i in range(4)])
        wrapper = _make_capability(placement).get_wrapper_toolset(fake)
        assert wrapper is not None

        ctx = _build_ctx()
        tools = await wrapper.get_tools(ctx)

        with pytest.raises(_PlacementFailure):
            await wrapper.call_tool('write_file', {'path': '/a'}, ctx, tools['write_file'])

        # 4 acquire calls: initial + 3 retries, each naming the queue that just failed
        assert placement.acquires == [None, 'q-0', 'q-1', 'q-2']


# ---------------------------------------------------------------------------
# _DurableEnvWrapper -- metadata handling
# ---------------------------------------------------------------------------


class TestMetadataHandling:
    async def test_existing_metadata_is_preserved(self) -> None:
        tool_def = _make_tool_def('write_file', env_bound=True)
        fake = _FakeToolset(tool_defs=[tool_def], results={'write_file': 'ok'}, calls=[])
        placement = _FakePlacement(leases=[_make_lease()])
        wrapper = _make_capability(placement).get_wrapper_toolset(fake)
        assert wrapper is not None

        ctx = _build_ctx(metadata={'user_key': 'preserved'})
        tools = await wrapper.get_tools(ctx)
        await wrapper.call_tool('write_file', {'path': '/a'}, ctx, tools['write_file'])

        assert ctx.metadata is not None
        assert ctx.metadata['user_key'] == 'preserved'
        assert 'durable_env' in ctx.metadata

    async def test_null_metadata_is_initialized(self) -> None:
        tool_def = _make_tool_def('write_file', env_bound=True)
        fake = _FakeToolset(tool_defs=[tool_def], results={'write_file': 'ok'}, calls=[])
        placement = _FakePlacement(leases=[_make_lease()])
        wrapper = _make_capability(placement).get_wrapper_toolset(fake)
        assert wrapper is not None

        ctx = _build_ctx(metadata=None)
        tools = await wrapper.get_tools(ctx)
        await wrapper.call_tool('write_file', {'path': '/a'}, ctx, tools['write_file'])

        assert ctx.metadata is not None
        assert 'durable_env' in ctx.metadata


# ---------------------------------------------------------------------------
# _DurableEnvWrapper -- what the placement driver receives
# ---------------------------------------------------------------------------


@dataclass
class _RecordingPlacement(_FakePlacement):
    """Fake placement that records the routing arguments instead of delegating blindly."""

    routed: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]], init=False)

    async def route_call(self, *args: Any, **kwargs: Any) -> Any:
        self.routed.append({'args': args, **{k: v for k, v in kwargs.items() if k != 'fallback'}})
        return await kwargs['fallback']()


class TestRouting:
    async def test_route_call_receives_lease_and_wrapped_toolset(self) -> None:
        tool_def = _make_tool_def('write_file', env_bound=True)
        fake = _FakeToolset(tool_defs=[tool_def], results={'write_file': 'ok'}, calls=[])
        lease = _make_lease()
        placement = _RecordingPlacement(leases=[lease])
        wrapper = _make_capability(placement).get_wrapper_toolset(fake)
        assert wrapper is not None

        ctx = _build_ctx()
        tools = await wrapper.get_tools(ctx)
        result = await wrapper.call_tool('write_file', {'path': '/a'}, ctx, tools['write_file'])

        assert result == 'ok'
        (routed,) = placement.routed
        assert routed['args'][0] == 'write_file'
        assert routed['args'][1] == {'path': '/a'}
        assert routed['lease'] is lease
        assert routed['wrapped'] is fake
        # the fallback actually delegated to the wrapped toolset
        assert fake.calls == [('write_file', {'path': '/a'})]
