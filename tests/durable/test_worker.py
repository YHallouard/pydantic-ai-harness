"""Tests for `_environment_bound_toolsets` and `_default_root` (the pure-logic
parts of `run_env_worker`).

`run_env_worker` itself mounts real Temporal `Worker`s and is exercised by the
Temporal integration tests, not here (see tests/test_durable_environment.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic_ai import AbstractToolset
from pydantic_ai.tools import RunContext, ToolSelector
from pydantic_ai.toolsets import ToolsetTool, WrapperToolset
from pydantic_ai.toolsets.combined import CombinedToolset
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.durable._store import SnapshotPolicy, SnapshotStore
from pydantic_ai_harness.durable._worker import _default_root, _environment_bound_toolsets

pytestmark = pytest.mark.anyio


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
    """A leaf toolset implementing the `EnvironmentBound` protocol, recording what it was configured with."""

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
        wrapper = WrapperToolset(wrapped=env_bound)

        found = _environment_bound_toolsets([[wrapper]])

        assert found == [env_bound]

    def test_finds_env_bound_toolsets_nested_inside_a_combined_toolset(self) -> None:
        env_bound_1 = _FakeEnvBoundToolset()
        env_bound_2 = _FakeEnvBoundToolset()
        combined = CombinedToolset(toolsets=[env_bound_1, _PlainToolset(), env_bound_2])

        found = _environment_bound_toolsets([[combined]])

        assert found == [env_bound_1, env_bound_2]

    def test_collects_across_multiple_agents(self) -> None:
        env_bound_1 = _FakeEnvBoundToolset()
        env_bound_2 = _FakeEnvBoundToolset()

        found = _environment_bound_toolsets([[env_bound_1], [env_bound_2]])

        assert found == [env_bound_1, env_bound_2]

    def test_no_agents_yields_no_toolsets(self) -> None:
        assert _environment_bound_toolsets([]) == []

    def test_found_toolset_accepts_root_and_durability_configuration(self) -> None:
        """`run_env_worker` calls `set_env_root`/`configure_durability` on what this walker finds --
        verify the fake actually implements that half of the protocol too."""
        env_bound = _FakeEnvBoundToolset()
        assert env_bound.env_bound_tools() == 'all'

        root = _default_root(Path('/workspaces'))
        policy = SnapshotPolicy(mode='per_op')
        env_bound.set_env_root(root)
        env_bound.configure_durability(None, policy)

        assert env_bound.roots == [root]
        assert env_bound.configured == [(None, policy)]


class TestDefaultRoot:
    def test_resolves_workspace_from_lease_env_id_in_ctx_metadata(self) -> None:
        resolve = _default_root(Path('/workspaces'))
        ctx = _build_ctx(metadata={'durable_env': {'env_id': 'wf-123', 'env_queue': 'env-q1', 'epoch': 0}})

        assert resolve(ctx) == Path('/workspaces/wf-123')

    def test_raises_without_metadata(self) -> None:
        resolve = _default_root(Path('/workspaces'))
        ctx = _build_ctx(metadata=None)

        with pytest.raises(AssertionError, match='durable_env lease'):
            resolve(ctx)
