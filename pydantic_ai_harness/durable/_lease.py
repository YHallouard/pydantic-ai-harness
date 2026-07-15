"""Lease lifecycle: the acquire/release Temporal activities that hand out environment leases."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel
from temporalio import activity
from temporalio.exceptions import ApplicationError

from pydantic_ai_harness.durable._journal import discard_env_lock, env_lock, guarded_mutating
from pydantic_ai_harness.durable._store import (
    AcquireEnvParams,
    EnvironmentLease,
    ForkEnvironmentParams,
    MergeEnvironmentParams,
    MergeResult,
    SnapshotPolicy,
    SnapshotStore,
)


class WriteEnvFileParams(BaseModel):
    """Params for the `write_environment_file` Temporal activity."""

    env_id: str
    path: str
    content: str
    op_id: str
    """Idempotency key for the write, supplied by the caller (no `RunContext` here to derive one
    from). Must be stable across a Temporal activity retry, exactly like a tool call's
    `f'{run_id}:{tool_call_id}'` -- e.g. an update-handler-scoped id derived deterministically
    from the workflow's own update id."""


class ReadEnvFileParams(BaseModel):
    """Params for the `read_environment_file` Temporal activity."""

    env_id: str
    path: str


def _resolve_within_workspace(workspace: Path, path: str) -> Path:
    """Resolve `path` under `workspace`, rejecting traversal above it.

    Mirrors `FileSystemToolset._resolve_path`'s containment check (symlink-resolved,
    `is_relative_to`), minus the `allowed_patterns`/`protected_patterns` gates -- those are a
    specific agent's own least-privilege config, not meaningful for a caller that isn't a tool.
    """
    candidate = (workspace / path).resolve()
    real = Path(os.path.realpath(candidate))
    if not real.is_relative_to(Path(os.path.realpath(workspace))):
        raise ApplicationError(f'Path {path!r} resolves outside the environment workspace.', non_retryable=True)
    return real


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
        default_policy: SnapshotPolicy | None = None,
    ) -> None:
        self._store = store
        self._env_queue = env_queue
        self._workspaces_base = workspaces_base
        self._max_concurrent_environments = max_concurrent_environments
        self._default_policy = default_policy if default_policy is not None else SnapshotPolicy()
        self._held: dict[str, HeldEnv] = {}

    @property
    def held_env_ids(self) -> frozenset[str]:
        """Environments this worker currently holds a fenced lease for."""
        return frozenset(self._held)

    @activity.defn(name='get_environment_queue')
    async def get_environment_queue(self, env_id: str) -> str | None:
        """Return the sticky queue holding `env_id`'s lease, or `None` if never acquired.

        A thin read over the store's persisted fence record (`SnapshotStore.get_lease`) --
        works regardless of which worker answers, since the record lives in the store, not
        this process's `_held`. This is the first of two calls a caller outside the agent
        graph makes to reach a held environment: resolve the queue here, then route
        `write_environment_file`/`read_environment_file` to it directly.
        """
        record = await self._store.get_lease(env_id)
        return record.env_queue if record is not None else None

    @activity.defn(name='write_environment_file')
    async def write_environment_file(self, params: WriteEnvFileParams) -> str:
        """Write a file into `params.env_id`'s held workspace from outside the agent graph.

        Must run on the worker that actually holds `params.env_id` -- the caller resolves
        that queue via `get_environment_queue` first and routes here with
        `task_queue=env_queue`. Reuses `guarded_mutating` (not just `env_lock`) with the
        caller-supplied `op_id`, so a Temporal activity retry is deduplicated by the same
        journal a normal `write_file` tool call uses, and the write is snapshotted the same
        way (`per_op` pushes before returning).
        """
        held = self._held.get(params.env_id)
        if held is None:
            raise ApplicationError(
                f'write_environment_file: {params.env_id!r} is not held by this worker', non_retryable=True
            )

        async def _apply() -> str:
            resolved = _resolve_within_workspace(held.workspace, params.path)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(params.content, encoding='utf-8')
            return f'Wrote {len(params.content)} chars to {params.path}'

        return await guarded_mutating(
            op_id=params.op_id,
            env_id=params.env_id,
            root=held.workspace,
            tool='write_environment_file',
            apply=_apply,
            store=self._store,
            policy=self._default_policy,
        )

    @activity.defn(name='read_environment_file')
    async def read_environment_file(self, params: ReadEnvFileParams) -> str:
        """Read a file from `params.env_id`'s held workspace from outside the agent graph.

        Same routing requirement as `write_environment_file`. Serialized by the same
        per-workspace lock a mutating tool call/write would take, so a read never sees a
        half-applied write; no journal entry (a pure read has nothing to dedupe).
        """
        held = self._held.get(params.env_id)
        if held is None:
            raise ApplicationError(
                f'read_environment_file: {params.env_id!r} is not held by this worker', non_retryable=True
            )
        async with env_lock(held.workspace):
            resolved = _resolve_within_workspace(held.workspace, params.path)
            if not resolved.is_file():
                raise ApplicationError(
                    f'read_environment_file: {params.path!r} not found in {params.env_id!r}', non_retryable=True
                )
            return resolved.read_text(encoding='utf-8')

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

    @activity.defn(name='fork_environment')
    async def fork_environment(self, params: ForkEnvironmentParams) -> None:
        """Branch `params.child_env_id` off `params.parent_env_id`'s current head.

        A thin wrapper over `SnapshotStore.fork`. Doesn't need a held
        workspace -- only the shared bare repo, which any worker can reach --
        so unlike `merge_environment` this has no pod affinity and can run on
        the host queue.
        """
        await self._store.fork(params.parent_env_id, params.child_env_id)

    @activity.defn(name='merge_environment')
    async def merge_environment(self, params: MergeEnvironmentParams) -> MergeResult:
        """Merge `params.other_env_id`'s branch into `params.held_env_id`'s held workspace.

        Must be routed to whichever sticky `env_queue` currently holds
        `params.held_env_id`'s lease -- the merge needs the actual checked-out
        work-tree, not just the bare repo. Guarded by the same per-workspace
        lock as `guarded_mutating`, so a merge and a concurrent mutating tool
        call for the same held environment serialize instead of racing.

        `mode='land'` (child -> parent, the nominal case): a conflict aborts
        the merge, leaving `held_env_id`'s head untouched; the conflicting
        paths are returned for the caller to act on (the self-heal loop in
        `delegate_task`). `mode='materialize'` (parent -> child, self-heal
        step 2): a conflict is committed and pushed as-is, markers included,
        so the sub-agent sees them as ordinary file content and the conflict
        state survives a killed pod.
        """
        held = self._held.get(params.held_env_id)
        if held is None:
            raise ApplicationError(
                f'merge_environment: {params.held_env_id!r} is not held by this worker', non_retryable=True
            )
        async with env_lock(held.workspace):
            return await self._store.merge(
                params.held_env_id, held.workspace, params.other_env_id, keep_conflicts=params.mode == 'materialize'
            )
