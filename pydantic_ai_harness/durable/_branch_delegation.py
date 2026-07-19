"""Fork/land/self-heal orchestration for a `'branch'`-workspace sub-agent delegation.

Reachable only by lazy import from `SubAgentToolset.delegate_task`
(`pydantic_ai_harness.experimental.subagents`), which must stay importable
without `temporalio` installed even though this module isn't -- mirrors
`_temporal_placement.py`'s own "imported here, not gated" stance.

Runs entirely as workflow code inside the child workflow that
`'nested_agent_run'` (`SubAgentToolset`'s tag on `delegate_task`) causes
`TemporalDurability` to start for this delegation -- fork, the sub-agent's own
run, land, and any self-heal rounds are all steps of that one child workflow,
durable and replayable together.

When `delegate_task` is offloaded to that child workflow, `ctx.metadata['durable_env']`
is usually unset (the parent's lease lives in the parent workflow, not the
child's). Callers pass `host_task_queue` so `resolve_parent_environment_lease`
can look up the parent workflow's held environment via `get_environment_queue`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Literal

from temporalio import workflow

from pydantic_ai_harness.durable._store import (
    AcquireEnvParams,
    EnvironmentLease,
    ForkEnvironmentParams,
    MergeEnvironmentParams,
    MergeResult,
)

_FORK_TIMEOUT = timedelta(seconds=30)
_ACQUIRE_TIMEOUT = timedelta(seconds=30)
_MERGE_TIMEOUT = timedelta(seconds=60)
_RELEASE_TIMEOUT = timedelta(seconds=30)


async def resolve_parent_environment_lease(*, host_task_queue: str) -> EnvironmentLease | None:
    """Look up the parent workflow's held environment lease from a nested child workflow."""
    if not workflow.in_workflow():
        return None
    parent = workflow.info().parent
    if parent is None:
        return None
    parent_env_id = parent.workflow_id
    env_queue = await workflow.execute_activity(
        'get_environment_queue',
        parent_env_id,
        task_queue=host_task_queue,
        start_to_close_timeout=_ACQUIRE_TIMEOUT,
    )
    if env_queue is None:
        return None
    return EnvironmentLease(env_id=parent_env_id, env_queue=env_queue, epoch=0)


async def run_with_self_heal(
    *,
    parent_lease: EnvironmentLease,
    run_once: Callable[[str], Awaitable[tuple[str, bool]]],
    task: str,
    max_merge_retries: int,
    host_task_queue: str | None = None,
) -> str:
    """Fork a branch for this delegation off `parent_lease`, run it, land it back.

    `run_once` runs the sub-agent for one task string and reports whether it
    actually completed (`True`, its output is real and worth merging) or
    degraded softly (`True` steering text, nothing to merge -- see
    `SubAgentToolset.delegate_task`'s `run_once`).

    A clean land returns the sub-agent's output as-is. A conflicted land
    starts the self-heal loop, up to `max_merge_retries` rounds: materialize
    the parent's current head into the child's own branch (conflict markers
    committed directly into its files, durable against a killed pod), relaunch
    the sub-agent to resolve them if any markers actually landed, and retry
    the land. Exhausting the retries still in conflict falls back to reporting
    the conflicting paths to the parent model, with the parent's workspace
    left untouched (the land that would have changed it was never completed).

    The child branch env is always released afterwards. The first env-bound tool
    call acquires it into the env-worker's `_held` map; without an explicit
    release, repeated delegations fill `max_concurrent_environments` (default 4)
    and the next acquire fails with `env worker at capacity`.
    """
    child_env_id = workflow.info().workflow_id
    fork_kwargs: dict[str, object] = {'start_to_close_timeout': _FORK_TIMEOUT}
    if host_task_queue is not None:
        fork_kwargs['task_queue'] = host_task_queue
    await workflow.execute_activity(
        'fork_environment',
        ForkEnvironmentParams(parent_env_id=parent_lease.env_id, child_env_id=child_env_id),
        **fork_kwargs,
    )

    try:
        output, completed = await run_once(task)
        if not completed:
            return output

        attempts = 0
        while True:
            land = await _merge(
                held_env_id=parent_lease.env_id,
                other_env_id=child_env_id,
                mode='land',
                task_queue=parent_lease.env_queue,
            )
            if not land.conflicts:
                return output
            if attempts >= max_merge_retries:
                return _conflict_message(land.conflicts, attempts)
            attempts += 1

            acquire_kwargs: dict[str, object] = {
                'result_type': EnvironmentLease,
                'start_to_close_timeout': _ACQUIRE_TIMEOUT,
            }
            if host_task_queue is not None:
                acquire_kwargs['task_queue'] = host_task_queue
            child_lease = await workflow.execute_activity(
                'acquire_environment',
                AcquireEnvParams(env_id=child_env_id),
                **acquire_kwargs,
            )
            materialize = await _merge(
                held_env_id=child_env_id,
                other_env_id=parent_lease.env_id,
                mode='materialize',
                task_queue=child_lease.env_queue,
            )
            if materialize.conflicts:  # pragma: no branch -- see below
                output, completed = await run_once(_resolution_task(materialize.conflicts))
                if not completed:
                    return output
            # `materialize` merges the same base/two-heads pair as the `land` above it,
            # just in the opposite direction -- git's 3-way merge conflict detection is
            # symmetric for a given base and two heads, so a `land` conflict here always
            # implies a `materialize` conflict too. The `no branch` pragma reflects that
            # invariant, not untested code: there is no path through this function where
            # `materialize.conflicts` is empty. Loop back to retry the land either way.
    finally:
        await _release_child_environment(
            child_env_id,
            host_task_queue=host_task_queue,
            fallback_queue=parent_lease.env_queue,
        )


async def _release_child_environment(
    child_env_id: str,
    *,
    host_task_queue: str | None,
    fallback_queue: str,
) -> None:
    """Release the branch env's local lease slot after land/abort.

    Prefer `host_task_queue` (where `DurableEnvironmentPlugin` registers
    `release_environment`); fall back to `fallback_queue` for single-queue
    tests. No-op when the child was never acquired.
    """
    task_queue = host_task_queue if host_task_queue is not None else fallback_queue
    await workflow.execute_activity(
        'release_environment',
        child_env_id,
        task_queue=task_queue,
        start_to_close_timeout=_RELEASE_TIMEOUT,
    )


async def _merge(
    *, held_env_id: str, other_env_id: str, mode: Literal['land', 'materialize'], task_queue: str
) -> MergeResult:
    return await workflow.execute_activity(
        'merge_environment',
        MergeEnvironmentParams(held_env_id=held_env_id, other_env_id=other_env_id, mode=mode),
        result_type=MergeResult,
        task_queue=task_queue,
        start_to_close_timeout=_MERGE_TIMEOUT,
    )


def _resolution_task(conflicts: list[str]) -> str:
    paths = ', '.join(conflicts)
    return (
        'Your previous changes conflict with concurrent edits to the shared workspace. '
        'Resolve the conflict markers (<<<<<<< / ======= / >>>>>>>) in the following files '
        f'using your usual tools, then finish: {paths}'
    )


def _conflict_message(conflicts: list[str], attempts: int) -> str:
    paths = ', '.join(conflicts)
    plural = 'attempt' if attempts == 1 else 'attempts'
    return (
        f"This delegation's changes still conflict with concurrent edits to {paths} after "
        f'{attempts} resolution {plural}. The shared workspace is unchanged; none of this '
        f"delegation's work was merged."
    )
