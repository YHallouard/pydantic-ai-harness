"""`DurableEnvironmentPlugin`: runs the sticky env-queue worker for durable agents.

Requires `temporalio` (imported here, not gated -- this module is only reachable
through `pydantic_ai_harness.durable.temporal`, which already gates the import).

The plugin owns *only* the environment: it attaches to whatever `Worker` the user
already runs (their own topology -- a dedicated model queue for rate limiting, a
separate MCP queue, and so on), registers the `acquire_environment`/
`release_environment` activities on it, and starts one extra sticky `env-{uuid}`
worker where env-bound tool calls land once this pod holds a lease. The user's
queues are untouched; nothing here mounts a "shared" queue on their behalf.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

from pydantic_ai import AbstractToolset
from pydantic_ai.durable_exec.temporal import AgentPlugin, TemporalAgent
from pydantic_ai.tools import RunContext
from temporalio.plugin import SimplePlugin
from temporalio.worker import Worker

from pydantic_ai_harness.durable._capability import DurableEnvironment
from pydantic_ai_harness.durable._lease import EnvironmentActivities
from pydantic_ai_harness.durable._store import SnapshotPolicy, SnapshotStore

__all__ = ['DurableEnvironmentPlugin']

_DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT = timedelta(seconds=30)


def _environment_bound_toolsets(toolsets: Sequence[Sequence[AbstractToolset[Any]]]) -> list[Any]:
    """Walk each toolset tree in `toolsets` for `EnvironmentBound` leaves.

    Takes plain toolset sequences rather than agents -- callers pass
    `agent.wrapped.toolsets` for each agent (the plain `Agent`'s concrete
    toolsets, e.g. `FileSystemToolset`, which execute inside the activity),
    not `agent.toolsets` (the temporalized activity-routing wrappers built at
    `TemporalAgent` construction, which offer no seam back to the originals).
    """
    found: list[Any] = []

    def _collect(toolset: Any) -> None:
        if hasattr(toolset, 'configure_durability') and hasattr(toolset, 'set_env_root'):
            found.append(toolset)

    for group in toolsets:
        for toolset in group:
            toolset.apply(_collect)
    return found


def _default_root(workspaces_base: Path) -> Callable[[RunContext[Any]], Path]:
    """Build the default `root_dir` resolver: `workspaces_base / lease.env_id`, read from `ctx.metadata`.

    A plain function factory (not a closure inside the plugin) so it's
    unit-testable without spinning up real `Worker`s.
    """

    def resolve(ctx: RunContext[Any]) -> Path:
        metadata = ctx.metadata
        assert metadata is not None, 'env-bound tool called without a durable_env lease in ctx.metadata'
        env_id = metadata['durable_env']['env_id']
        assert isinstance(env_id, str)
        return workspaces_base / env_id

    return resolve


def _discover_durability(agents: Sequence[TemporalAgent[Any, Any]]) -> tuple[SnapshotStore, SnapshotPolicy]:
    """Read the single `store`/`snapshot_policy` the agents' `DurableEnvironment` capabilities agree on.

    The capability is the one source of truth for durability config (the plugin
    doesn't take its own `store` argument), so conflicting stores or policies
    across agents is a configuration error, not something to silently pick from.
    """
    store: SnapshotStore | None = None
    policy: SnapshotPolicy | None = None
    for agent in agents:
        capabilities: list[Any] = []
        agent.wrapped.root_capability.apply(capabilities.append)
        for capability in capabilities:
            if not isinstance(capability, DurableEnvironment):
                continue
            if capability.store is None:
                raise ValueError('DurableEnvironment capability has no `store`; nothing for the plugin to snapshot.')
            cap_policy = (
                capability.snapshot_policy
                if isinstance(capability.snapshot_policy, SnapshotPolicy)
                else SnapshotPolicy(mode=capability.snapshot_policy)
            )
            if store is not None and (capability.store is not store or cap_policy != policy):
                raise ValueError('agents disagree on DurableEnvironment store/snapshot_policy; use one shared config.')
            store, policy = capability.store, cap_policy
    if store is None or policy is None:
        raise ValueError('no DurableEnvironment capability found on the agents; add one with a `store`.')
    return store, policy


class DurableEnvironmentPlugin(SimplePlugin):
    """Temporal worker plugin that runs the sticky env-queue for a set of durable agents.

    Attach it to the `Worker` whose pod should host the workspaces:

    ```python
    from temporalio.worker import Worker
    from pydantic_ai.durable_exec.temporal import AgentPlugin
    from pydantic_ai_harness.durable.temporal import DurableEnvironmentPlugin

    env_plugin = DurableEnvironmentPlugin([temporal_agent], workspaces_base=Path('/workspaces'))
    async with Worker(
        client,
        task_queue='agent-main',
        workflows=[MyWorkflow],
        plugins=[AgentPlugin(temporal_agent), env_plugin],
    ):
        await asyncio.Future()  # your own run/shutdown handling
    ```

    The plugin registers `acquire_environment`/`release_environment` on that host
    worker and, while it runs, mounts one extra `env-{uuid}` worker that serves the
    agents' tool activities. On shutdown it drains that worker and pushes a final
    snapshot for every workspace still held (a safety net for `per_step`/
    `content_hash` policies; `per_op` has already pushed by then). `store` and
    `snapshot_policy` come from each agent's `DurableEnvironment` capability, not
    from the plugin.
    """

    def __init__(
        self,
        agents: Sequence[TemporalAgent[Any, Any]],
        *,
        workspaces_base: Path = Path('/workspaces'),
        max_concurrent_environments: int = 4,
        max_concurrent_activities: int | None = None,
        graceful_shutdown_timeout: timedelta = _DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT,
    ) -> None:
        self._agents = list(agents)
        self._env_queue = f'env-{uuid.uuid4().hex}'
        self._graceful_shutdown_timeout = graceful_shutdown_timeout
        # A held environment can have several tool activities in flight at once
        # (the agent may batch tool calls), so the sticky worker's activity slots
        # are sized above the environment count rather than equal to it.
        self._max_concurrent_activities = (
            max_concurrent_activities if max_concurrent_activities is not None else max_concurrent_environments * 2
        )

        store, policy = _discover_durability(self._agents)
        self._activities = EnvironmentActivities(
            store=store,
            env_queue=self._env_queue,
            workspaces_base=workspaces_base,
            max_concurrent_environments=max_concurrent_environments,
        )

        root_resolver = _default_root(workspaces_base)
        for toolset in _environment_bound_toolsets([agent.wrapped.toolsets for agent in self._agents]):
            toolset.set_env_root(root_resolver)
            toolset.configure_durability(store, policy)

        super().__init__(  # pyright: ignore[reportUnknownMemberType]
            name='DurableEnvironmentPlugin',
            activities=[self._activities.acquire_environment, self._activities.release_environment],
        )

    async def run_worker(self, worker: Worker, next: Callable[[Worker], Awaitable[None]]) -> None:
        """Run the host worker with a sticky env-queue worker mounted alongside it.

        The sticky worker shares the host worker's `client`; its `__aexit__`
        performs Temporal's graceful drain (stop polling, wait up to
        `graceful_shutdown_timeout` for in-flight activities). After it drains,
        push a final snapshot for every workspace this pod still holds.
        """
        async with Worker(
            worker.client,
            task_queue=self._env_queue,
            plugins=[AgentPlugin(agent) for agent in self._agents],
            max_concurrent_activities=self._max_concurrent_activities,
            graceful_shutdown_timeout=self._graceful_shutdown_timeout,
        ):
            await next(worker)

        for env_id in self._activities.held_env_ids:
            await self._activities.snapshot_held(env_id)
