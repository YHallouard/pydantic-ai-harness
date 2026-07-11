"""Lease lifecycle: the acquire/release Temporal activities that hand out environment leases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from temporalio import activity
from temporalio.exceptions import ApplicationError

from pydantic_ai_harness.durable._journal import discard_env_lock
from pydantic_ai_harness.durable._store import AcquireEnvParams, EnvironmentLease, SnapshotStore


@dataclass
class HeldEnv:
    """Process-local record of an environment this worker currently holds.

    Not a Pydantic model: purely local state, never serialized or sent across
    an activity boundary (only `EnvironmentLease` crosses that boundary).
    """

    lease: EnvironmentLease
    workspace: Path
    head: str
    """The snapshot head sha this workspace was restored from -- the value `SnapshotStore.is_current` checks."""


class EnvironmentActivities:
    """Temporal activities backing the lease lifecycle: one instance per env-worker process.

    Registered on the host worker by `DurableEnvironmentPlugin`, which also runs
    this worker's own sticky `env_queue` -- the queue env-bound tool activities
    get routed to once this worker holds the lease. pydantic-ai's own
    `resolve_tool_activity_config` reads `ctx.metadata['durable_env']['env_queue']`
    and places them there directly (pydantic-ai #4977, `temporal-durability-cap`);
    `TemporalPlacement.route_call` no longer needs to reimplement this.
    """

    def __init__(
        self,
        *,
        store: SnapshotStore,
        env_queue: str,
        workspaces_base: Path,
        max_concurrent_environments: int = 4,
    ) -> None:
        self._store = store
        self._env_queue = env_queue
        self._workspaces_base = workspaces_base
        self._max_concurrent_environments = max_concurrent_environments
        self._held: dict[str, HeldEnv] = {}

    @property
    def held_env_ids(self) -> frozenset[str]:
        """Environments this worker currently holds a fenced lease for."""
        return frozenset(self._held)

    @activity.defn(name='acquire_environment')
    async def acquire_environment(self, params: AcquireEnvParams) -> EnvironmentLease:
        """Return a lease for `params.env_id`, fencing a fresh one only if nobody else holds it.

        Three paths, in order:

        1. **Re-acquire**: this worker already holds `env_id` locally. Re-validate
           against the store first -- a schedule-to-start timeout elsewhere can mean
           this worker was fenced out without noticing -- and only then return the
           held lease without bumping the epoch.
        2. **Converge**: the store has a live lease for `env_id` held by some *other*
           queue that isn't the one the caller just saw fail (`params.failed_queue`).
           Return that lease as-is, without fencing. This is what makes concurrent
           re-acquisition (a parent and a re-provisioning sub-agent racing after a pod
           death) converge on one winner instead of fencing each other out in a loop.
        3. **Provision**: fence a new claim, restore the snapshot locally, and hold it.
        """
        held = self._held.get(params.env_id)
        if held is not None:
            if await self._store.is_current(params.env_id, held.head):
                return held.lease
            del self._held[params.env_id]
            await self._store.discard_workspace(held.workspace)
            discard_env_lock(held.workspace)

        record = await self._store.get_lease(params.env_id)
        if record is not None and record.env_queue not in (params.failed_queue, self._env_queue):
            return record.to_lease()

        if len(self._held) >= self._max_concurrent_environments:
            raise ApplicationError('env worker at capacity', non_retryable=False)

        head = await self._store.fence(params.env_id, queue=self._env_queue)
        workspace = self._workspaces_base / params.env_id
        await self._store.restore(params.env_id, into=workspace)
        lease = EnvironmentLease(env_id=params.env_id, env_queue=self._env_queue, epoch=head.epoch)
        self._held[params.env_id] = HeldEnv(lease=lease, workspace=workspace, head=head.sha)
        return lease

    @activity.defn(name='release_environment')
    async def release_environment(self, env_id: str) -> None:
        """Snapshot and give up `env_id`, so a later acquirer fences fresh instead of converging on it.

        Scoped to workflow completion (or an explicit call), not per-`agent.run` --
        a conversational workflow that calls `agent.run` multiple times needs the
        workspace to survive between those calls.
        """
        held = self._held.pop(env_id, None)
        if held is None:
            return
        await self._store.push(env_id, held.workspace)
        await self._store.release(env_id)
        await self._store.discard_workspace(held.workspace)
        discard_env_lock(held.workspace)

    async def snapshot_held(self, env_id: str) -> None:
        """Push a final snapshot for `env_id` without releasing the lease or forgetting local state.

        Not a Temporal activity -- called directly by `run_env_worker`'s SIGTERM
        drain, after both `Worker`s have stopped polling. A safety net for
        `per_step`/`content_hash` snapshot policies, where a mutating op may not
        have been pushed synchronously (`per_op` already has by the time the
        call returns, see `SnapshotPolicy`). A no-op for an `env_id` this worker
        doesn't hold.
        """
        held = self._held.get(env_id)
        if held is None:
            return
        await self._store.push(env_id, held.workspace)
