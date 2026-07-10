"""`run_env_worker`: mounts the shared + sticky env task queues for a set of durable agents.

Requires `temporalio` (imported here, not gated -- this module is only reachable
through `pydantic_ai_harness.durable.temporal`, which already gates the import).
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import uuid
from collections.abc import Callable, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic_ai import AbstractToolset
from pydantic_ai.durable_exec.temporal import AgentPlugin, TemporalAgent
from pydantic_ai.tools import RunContext
from temporalio.client import Client
from temporalio.worker import Worker

from pydantic_ai_harness.durable._lease import EnvironmentActivities
from pydantic_ai_harness.durable._store import SnapshotPolicy, SnapshotStore

__all__ = ['run_env_worker']

_DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT = timedelta(seconds=30)


def _environment_bound_toolsets(toolsets: Sequence[Sequence[AbstractToolset[Any]]]) -> list[Any]:
    """Walk each toolset tree in `toolsets` for `EnvironmentBound` leaves.

    Takes plain toolset sequences rather than agents -- `run_env_worker` passes
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

    A plain function factory (not a closure inside `run_env_worker`) so it's
    unit-testable without spinning up real `Worker`s.
    """

    def resolve(ctx: RunContext[Any]) -> Path:
        metadata = ctx.metadata
        assert metadata is not None, 'env-bound tool called without a durable_env lease in ctx.metadata'
        env_id = metadata['durable_env']['env_id']
        assert isinstance(env_id, str)
        return workspaces_base / env_id

    return resolve


async def run_env_worker(
    client: Client,
    *,
    agents: Sequence[TemporalAgent[Any, Any]],
    shared_task_queue: str,
    workflows: Sequence[type],
    store: SnapshotStore,
    snapshot_policy: SnapshotPolicy | Literal['per_op', 'per_step', 'content_hash'] = 'per_op',
    workspaces_base: Path = Path('/workspaces'),
    max_concurrent_environments: int = 4,
    graceful_shutdown_timeout: timedelta = _DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Run the shared-queue and sticky env-queue workers for `agents`, until SIGTERM.

    Wires `store`/`snapshot_policy` into every `EnvironmentBound` toolset found
    on `agents` (worker-side; the `DurableEnvironment` capability itself only
    handles lease acquisition and routing from inside the workflow sandbox --
    see its module docstring). Mounts two `Worker`s on `client`, matching the
    canonical `worker_specific_task_queues` pattern: `shared_task_queue` (agent
    workflows + model/MCP/tool activities + `acquire/release_environment`) and
    a fresh `env-{uuid}` queue (the same tool activities again, routed here
    once this pod holds the lease -- see `DurableEnvironment`).

    On SIGTERM: stop polling both queues, wait up to `graceful_shutdown_timeout`
    for in-flight activities to finish, then push a final snapshot for every
    workspace this pod still holds (`EnvironmentActivities.snapshot_held`) --
    a safety net for `per_step`/`content_hash` policies (see its docstring).
    Blocks until that drain completes; run it as the process's main coroutine.

    `stop_event`: inject an `asyncio.Event` to trigger the drain programmatically
    (used by tests) instead of installing a SIGTERM handler on the running loop.
    """
    policy = snapshot_policy if isinstance(snapshot_policy, SnapshotPolicy) else SnapshotPolicy(mode=snapshot_policy)
    env_queue = f'env-{uuid.uuid4().hex}'
    acts = EnvironmentActivities(
        store=store,
        env_queue=env_queue,
        workspaces_base=workspaces_base,
        max_concurrent_environments=max_concurrent_environments,
    )

    root_resolver = _default_root(workspaces_base)
    for toolset in _environment_bound_toolsets([agent.wrapped.toolsets for agent in agents]):
        toolset.set_env_root(root_resolver)
        toolset.configure_durability(store, policy)

    agent_plugins = [AgentPlugin(agent) for agent in agents]

    stop = stop_event if stop_event is not None else asyncio.Event()
    if stop_event is None:
        loop = asyncio.get_running_loop()
        with contextlib.suppress(NotImplementedError):  # pragma: no cover -- no signal handlers on Windows
            loop.add_signal_handler(signal.SIGTERM, stop.set)

    async with (
        Worker(
            client,
            task_queue=shared_task_queue,
            workflows=list(workflows),
            activities=[acts.acquire_environment, acts.release_environment],
            plugins=agent_plugins,
            graceful_shutdown_timeout=graceful_shutdown_timeout,
        ),
        Worker(
            client,
            task_queue=env_queue,
            plugins=agent_plugins,
            max_concurrent_activities=max_concurrent_environments * 2,
            graceful_shutdown_timeout=graceful_shutdown_timeout,
        ),
    ):
        await stop.wait()

    for env_id in acts.held_env_ids:
        await acts.snapshot_held(env_id)
