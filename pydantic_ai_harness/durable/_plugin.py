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

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

from pydantic_ai import AbstractToolset, Agent
from pydantic_ai.durable_exec.temporal import AgentPlugin
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
    `agent.toolsets` for each agent, the construction-time list `TemporalDurability`
    discovers and temporalizes leaves from (`FileSystemToolset`, etc). This is the
    same list `TemporalDurability.get_wrapper_toolset` matches against by `ts.id`
    to swap in the activity-routing wrapper at run time -- it never replaces
    `agent.toolsets` itself, so walking it here always finds the concrete toolsets.
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


def _discover_durability(agents: Sequence[Agent[Any, Any]]) -> tuple[SnapshotStore, SnapshotPolicy]:
    """Read the single `store`/`snapshot_policy` the agents' `DurableEnvironment` capabilities agree on.

    The capability is the one source of truth for durability config (the plugin
    doesn't take its own `store` argument), so conflicting stores or policies
    across agents is a configuration error, not something to silently pick from.
    """
    store: SnapshotStore | None = None
    policy: SnapshotPolicy | None = None
    for agent in agents:
        capabilities: list[Any] = []
        agent.root_capability.apply(capabilities.append)
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

    Attach it to the `Worker` whose pod should host the workspaces. `agent` needs a
    `TemporalDurability` capability (alongside `DurableEnvironment` and whatever
    env-bound toolsets) for `AgentPlugin`/this plugin to find:

    ```python
    from temporalio.worker import Worker
    from pydantic_ai.durable_exec.temporal import AgentPlugin, TemporalDurability
    from pydantic_ai_harness.durable.temporal import DurableEnvironmentPlugin

    env_plugin = DurableEnvironmentPlugin([agent], workspaces_base=Path('/workspaces'))
    async with Worker(
        client,
        task_queue='agent-main',
        workflows=[MyWorkflow],
        plugins=[AgentPlugin(agent), env_plugin],
    ):
        await asyncio.Future()  # your own run/shutdown handling
    ```

    The plugin registers `acquire_environment`/`release_environment`/
    `fork_environment`/`merge_environment`/`get_environment_queue`/
    `write_environment_file`/`read_environment_file` on that host worker and,
    while it runs, mounts one extra `env-{uuid}` worker that serves the
    agents' tool activities. `merge_environment`/`write_environment_file`/
    `read_environment_file` are also registered explicitly on that sticky
    worker (`AgentPlugin` there only registers each agent's own activities) --
    a sub-agent delegation with `workspace='branch'` routes its land/materialize
    merges to whichever sticky queue holds the relevant lease, not the host
    queue, and so does anything reading or writing a held environment from
    outside the agent graph (`get_environment_queue` resolves that queue
    first; see `EnvironmentActivities.write_environment_file`/
    `read_environment_file`). On shutdown it drains the sticky worker and
    pushes a final snapshot for every workspace still held (a safety net for
    `per_step`/`content_hash` policies; `per_op` has already pushed by then).
    `store` and `snapshot_policy` come from each agent's `DurableEnvironment`
    capability, not from the plugin.

    `Worker.__aexit__` cancels a plugin's `run_worker` continuation as soon as
    the *host* worker's own poll loop stops -- it does not wait for a plugin to
    finish async work scheduled after `await next(worker)`. Draining the sticky
    worker and pushing snapshots is exactly that kind of work, so it runs
    shielded from that cancellation (see `run_worker`); call `wait_drained()`
    after the host `async with Worker(...)` block exits to wait for it
    deterministically instead of assuming it already happened.

    ### Capacity-aware acquire (optional)

    By default `acquire_environment` also runs on the host worker above: a full pod (`held ==
    max_concurrent_environments`) still polls it, accepts the task, then bounces it with a
    retryable `env worker at capacity` error -- fine at low fleet sizes, but the bounce rate
    grows with fleet size under contention (see `pydantic_ai_harness.durable.temporal`'s
    `ACQUIRE_TASK_QUEUE` docs). To opt into capacity-gated acquire instead -- a full pod stops
    polling entirely, so acquire tasks queue visibly on the server rather than being bounced --
    run a second, dedicated `Worker` in the same process, serving *only*
    `acquire_environment` on `ACQUIRE_TASK_QUEUE`, tuned with `CapacityGatedSlotSupplier`:

    ```python
    from temporalio.worker import FixedSizeSlotSupplier, Worker, WorkerTuner
    from pydantic_ai_harness.durable.temporal import (
        ACQUIRE_TASK_QUEUE, CapacityGatedSlotSupplier, DurableEnvironmentPlugin,
    )

    env_plugin = DurableEnvironmentPlugin([agent], workspaces_base=Path('/workspaces'))

    # Same process as the host worker above: the gate reads this pod's own held leases.
    acquire_worker = Worker(
        client,
        task_queue=ACQUIRE_TASK_QUEUE,
        activities=[env_plugin.environment_activities.acquire_environment],
        tuner=WorkerTuner.create_composite(
            workflow_supplier=FixedSizeSlotSupplier(2),
            activity_supplier=CapacityGatedSlotSupplier(env_plugin.environment_activities, num_slots=8),
            local_activity_supplier=FixedSizeSlotSupplier(2),
            nexus_supplier=FixedSizeSlotSupplier(2),
        ),
    )
    # Workflow side: TemporalPlacement(host_task_queue=ACQUIRE_TASK_QUEUE)
    ```

    `release_environment` must stay on the host worker, never on the gated one: a full pod that
    can't run releases would deadlock (nobody frees capacity, nobody polls to free it). This
    wiring is opt-in and additive -- `acquire_environment` keeps running on the host worker too
    (`Worker(tuner=...)` is mutually exclusive with `max_concurrent_activities`, so this must be a
    *second* `Worker`, not an option on the one above), so a deployment that skips it keeps
    today's bounce-and-retry behavior unchanged.
    """

    def __init__(
        self,
        agents: Sequence[Agent[Any, Any]],
        *,
        workspaces_base: Path = Path('/workspaces'),
        max_concurrent_environments: int = 4,
        max_concurrent_activities: int | None = None,
        graceful_shutdown_timeout: timedelta = _DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT,
    ) -> None:
        self._agents = list(agents)
        self._env_queue = f'env-{uuid.uuid4().hex}'
        self._graceful_shutdown_timeout = graceful_shutdown_timeout
        self._drained = asyncio.Event()
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
        for toolset in _environment_bound_toolsets([agent.toolsets for agent in self._agents]):
            toolset.set_env_root(root_resolver)
            toolset.configure_durability(store, policy)

        super().__init__(  # pyright: ignore[reportUnknownMemberType]
            name='DurableEnvironmentPlugin',
            activities=[
                self._activities.acquire_environment,
                self._activities.release_environment,
                self._activities.fork_environment,
                self._activities.merge_environment,
                self._activities.get_environment_queue,
                self._activities.write_environment_file,
                self._activities.read_environment_file,
            ],
        )

    @property
    def environment_activities(self) -> EnvironmentActivities:
        """The shared `EnvironmentActivities` instance backing this pod's leases.

        Pass it to a `CapacityGatedSlotSupplier` and register
        `plugin.environment_activities.acquire_environment` on a dedicated `ACQUIRE_TASK_QUEUE`
        worker in the same process, so the capacity gate reads this pod's own held leases (see
        `pydantic_ai_harness.durable.temporal`). Named distinctly from `SimplePlugin.activities`
        (the registered-activity list) to avoid shadowing it.
        """
        return self._activities

    async def run_worker(self, worker: Worker, next: Callable[[Worker], Awaitable[None]]) -> None:
        """Run the host worker with a sticky env-queue worker mounted alongside it.

        `Worker.__aexit__` marks its context manager done -- and cancels this
        plugin's continuation -- as soon as the *host* worker's own internal poll
        loop stops, not once this whole method returns. Everything below the
        `await next(worker)` line (draining the sticky worker, pushing snapshots)
        would otherwise race that cancellation and be cut short mid-drain. Running
        it under `asyncio.shield` detaches it from that cancellation: it keeps
        running to completion as its own task even if this call site is
        cancelled. `wait_drained()` lets a caller await that completion instead
        of assuming the host's `async with` block exiting means it's done.
        """
        self._drained = asyncio.Event()
        await asyncio.shield(self._drain_and_snapshot(worker, next))

    async def _drain_and_snapshot(self, worker: Worker, next: Callable[[Worker], Awaitable[None]]) -> None:
        async with Worker(
            worker.client,
            task_queue=self._env_queue,
            plugins=[AgentPlugin(agent) for agent in self._agents],
            activities=[
                self._activities.merge_environment,
                self._activities.write_environment_file,
                self._activities.read_environment_file,
            ],
            max_concurrent_activities=self._max_concurrent_activities,
            graceful_shutdown_timeout=self._graceful_shutdown_timeout,
        ):
            await next(worker)

        for env_id in self._activities.held_env_ids:
            await self._activities.snapshot_held(env_id)
        self._drained.set()

    async def wait_drained(self) -> None:
        """Wait for the sticky worker to drain and the final snapshots to be pushed.

        `run_worker`'s cleanup runs shielded from the host `Worker`'s own
        cancellation (see its docstring), so it is not guaranteed to have
        finished the instant the host's `async with Worker(...)` block exits.
        Call this afterward when you need that guarantee -- e.g. before
        asserting on a snapshot in a test, or before a process actually exits.

        Calling this from the same task that entered the host's `async with
        Worker(...)` block absorbs one spurious `CancelledError`: exiting that
        block cancels `run_worker`'s continuation once the host's own poll loop
        stops (see `run_worker`), and `Worker.__aenter__`'s wrapper turns that
        into a delayed cancellation of the *caller's* task too, landing on
        whatever its next `await` happens to be. The shielded cleanup keeps
        running regardless, so retrying past that one cancellation converges on
        it actually finishing rather than surfacing an artifact of the plugin
        chain as if it were a real cancellation request.
        """
        while True:
            try:
                await self._drained.wait()
                return
            except asyncio.CancelledError:
                if self._drained.is_set():
                    return
                continue
