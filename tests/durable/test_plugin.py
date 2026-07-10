"""Tests for `DurableEnvironmentPlugin` and its pure-logic helpers.

The plugin's full lifecycle (mounting the sticky `Worker`, draining, snapshotting)
runs against a live Temporal server and is covered by the Temporal integration
tests, not here. These tests cover construction: capability discovery, toolset
wiring, and the `_environment_bound_toolsets`/`_default_root` walkers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic_ai import AbstractToolset, Agent
from pydantic_ai.durable_exec.temporal import TemporalAgent
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolSelector
from pydantic_ai.toolsets import ToolsetTool, WrapperToolset
from pydantic_ai.toolsets.combined import CombinedToolset
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness import FileSystem
from pydantic_ai_harness.durable import DurableEnvironment, SnapshotPolicy, SnapshotStore
from pydantic_ai_harness.durable._plugin import _default_root, _environment_bound_toolsets
from pydantic_ai_harness.durable.temporal import (
    DurableEnvironmentPlugin,
    EnvironmentActivities,
    TemporalPlacement,
)

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Fakes for the pure-logic helper tests
# ---------------------------------------------------------------------------


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


@dataclass
class _PlainToolset(AbstractToolset[object]):
    """A leaf toolset that does not implement `EnvironmentBound`."""

    @property
    def id(self) -> str | None:  # pragma: no cover -- unused by the walker
        return None

    async def get_tools(self, ctx: RunContext[object]) -> dict[str, ToolsetTool[object]]:  # pragma: no cover
        return {}

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[object], tool: ToolsetTool[object]
    ) -> Any:  # pragma: no cover
        raise NotImplementedError


@dataclass
class _FakeEnvBoundToolset(AbstractToolset[object]):
    """A leaf toolset implementing `EnvironmentBound`, recording what it was configured with."""

    configured: list[tuple[SnapshotStore | None, SnapshotPolicy]] = field(
        default_factory=list[tuple[SnapshotStore | None, SnapshotPolicy]]
    )
    roots: list[Any] = field(default_factory=list[Any])

    @property
    def id(self) -> str | None:  # pragma: no cover -- unused by the walker
        return None

    async def get_tools(self, ctx: RunContext[object]) -> dict[str, ToolsetTool[object]]:  # pragma: no cover
        return {}

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[object], tool: ToolsetTool[object]
    ) -> Any:  # pragma: no cover
        raise NotImplementedError

    def env_bound_tools(self) -> ToolSelector[object]:
        return 'all'

    def set_env_root(self, root: Any) -> None:
        self.roots.append(root)

    def configure_durability(self, store: SnapshotStore | None, policy: SnapshotPolicy) -> None:
        self.configured.append((store, policy))


# ---------------------------------------------------------------------------
# _environment_bound_toolsets
# ---------------------------------------------------------------------------


class TestEnvironmentBoundToolsets:
    def test_collects_a_top_level_env_bound_toolset(self) -> None:
        env_bound = _FakeEnvBoundToolset()
        found = _environment_bound_toolsets([[env_bound]])
        assert found == [env_bound]

    def test_skips_a_plain_toolset(self) -> None:
        found = _environment_bound_toolsets([[_PlainToolset()]])
        assert found == []

    def test_finds_an_env_bound_toolset_nested_inside_a_wrapper(self) -> None:
        env_bound = _FakeEnvBoundToolset()
        wrapper: WrapperToolset[object] = WrapperToolset(env_bound)
        found = _environment_bound_toolsets([[wrapper]])
        assert found == [env_bound]

    def test_finds_env_bound_toolsets_nested_inside_a_combined_toolset(self) -> None:
        env_bound = _FakeEnvBoundToolset()
        combined: CombinedToolset[object] = CombinedToolset([_PlainToolset(), env_bound])
        found = _environment_bound_toolsets([[combined]])
        assert found == [env_bound]

    def test_collects_across_multiple_agents(self) -> None:
        env_bound_1 = _FakeEnvBoundToolset()
        env_bound_2 = _FakeEnvBoundToolset()
        found = _environment_bound_toolsets([[env_bound_1], [env_bound_2]])
        assert found == [env_bound_1, env_bound_2]

    def test_no_agents_yields_no_toolsets(self) -> None:
        assert _environment_bound_toolsets([]) == []

    def test_found_toolset_accepts_root_and_durability_configuration(self) -> None:
        env_bound = _FakeEnvBoundToolset()
        (found,) = _environment_bound_toolsets([[env_bound]])
        root = _default_root(Path('/workspaces'))
        found.set_env_root(root)
        found.configure_durability(None, SnapshotPolicy())
        assert env_bound.roots == [root]
        assert env_bound.configured == [(None, SnapshotPolicy())]


# ---------------------------------------------------------------------------
# _default_root
# ---------------------------------------------------------------------------


class TestDefaultRoot:
    def test_resolves_workspace_from_lease_env_id_in_ctx_metadata(self) -> None:
        resolve = _default_root(Path('/workspaces'))
        ctx = _build_ctx(metadata={'durable_env': {'env_id': 'wf-42'}})
        assert resolve(ctx) == Path('/workspaces/wf-42')

    def test_raises_without_metadata(self) -> None:
        resolve = _default_root(Path('/workspaces'))
        with pytest.raises(AssertionError):
            resolve(_build_ctx())


# ---------------------------------------------------------------------------
# DurableEnvironmentPlugin -- construction and capability discovery
# ---------------------------------------------------------------------------


def _agent_with(capabilities: list[Any]) -> TemporalAgent[None, str]:
    return TemporalAgent(Agent(TestModel(), name='coder', capabilities=capabilities))


def _store() -> SnapshotStore:
    return MagicMock(spec=SnapshotStore)


class TestPluginDiscovery:
    def test_reads_store_and_wires_env_bound_toolsets(self) -> None:
        store = _store()
        agent = _agent_with([FileSystem(), DurableEnvironment(placement=TemporalPlacement(), store=store)])

        DurableEnvironmentPlugin([agent])

        (fs_toolset,) = _environment_bound_toolsets([agent.wrapped.toolsets])
        assert fs_toolset._durability_store is store  # pyright: ignore[reportPrivateUsage]
        assert fs_toolset._durability_policy == SnapshotPolicy(mode='per_op')  # pyright: ignore[reportPrivateUsage]

    def test_builds_the_lease_activities_for_the_discovered_store(self) -> None:
        store = _store()
        agent = _agent_with([DurableEnvironment(placement=TemporalPlacement(), store=store)])
        plugin = DurableEnvironmentPlugin([agent])
        assert isinstance(plugin._activities, EnvironmentActivities)  # pyright: ignore[reportPrivateUsage]
        assert plugin._activities.held_env_ids == frozenset()  # pyright: ignore[reportPrivateUsage]

    def test_raises_when_no_durable_environment_capability(self) -> None:
        agent = _agent_with([FileSystem()])
        with pytest.raises(ValueError, match='no DurableEnvironment capability'):
            DurableEnvironmentPlugin([agent])

    def test_raises_when_the_capability_has_no_store(self) -> None:
        agent = _agent_with([DurableEnvironment(placement=TemporalPlacement(), store=None)])
        with pytest.raises(ValueError, match='no `store`'):
            DurableEnvironmentPlugin([agent])

    def test_raises_when_agents_disagree_on_the_store(self) -> None:
        agent_a = _agent_with([DurableEnvironment(placement=TemporalPlacement(), store=_store())])
        agent_b = _agent_with([DurableEnvironment(placement=TemporalPlacement(), store=_store())])
        with pytest.raises(ValueError, match='disagree'):
            DurableEnvironmentPlugin([agent_a, agent_b])

    def test_raises_when_agents_disagree_on_the_policy(self) -> None:
        store = _store()
        agent_a = _agent_with(
            [DurableEnvironment(placement=TemporalPlacement(), store=store, snapshot_policy='per_op')]
        )
        agent_b = _agent_with(
            [DurableEnvironment(placement=TemporalPlacement(), store=store, snapshot_policy='per_step')]
        )
        with pytest.raises(ValueError, match='disagree'):
            DurableEnvironmentPlugin([agent_a, agent_b])

    def test_two_agents_sharing_one_store_and_policy_are_accepted(self) -> None:
        store = _store()
        agent_a = _agent_with([DurableEnvironment(placement=TemporalPlacement(), store=store)])
        agent_b = _agent_with([DurableEnvironment(placement=TemporalPlacement(), store=store)])
        plugin = DurableEnvironmentPlugin([agent_a, agent_b])
        assert isinstance(plugin._activities, EnvironmentActivities)  # pyright: ignore[reportPrivateUsage]


class TestPluginActivitySlots:
    def test_default_activity_slots_are_twice_the_environment_cap(self) -> None:
        agent = _agent_with([DurableEnvironment(placement=TemporalPlacement(), store=_store())])
        plugin = DurableEnvironmentPlugin([agent], max_concurrent_environments=3)
        assert plugin._max_concurrent_activities == 6  # pyright: ignore[reportPrivateUsage]

    def test_explicit_activity_slots_are_respected(self) -> None:
        agent = _agent_with([DurableEnvironment(placement=TemporalPlacement(), store=_store())])
        plugin = DurableEnvironmentPlugin([agent], max_concurrent_environments=3, max_concurrent_activities=5)
        assert plugin._max_concurrent_activities == 5  # pyright: ignore[reportPrivateUsage]
