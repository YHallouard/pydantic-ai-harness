"""Live-Temporal coverage for `_branch_delegation.run_with_self_heal`.

`run_with_self_heal` takes the sub-agent run itself (`run_once`) as an
injected callable specifically so its fork/land/self-heal control flow can be
tested without a real model-driven `Agent` -- these tests fake `run_once` with
a couple of test-only activities that write directly into the workspace paths
`acquire_environment`/`merge_environment` operate on (`workspaces_base /
env_id`, the same convention `test_lease.py` uses). Exercising the loop with a
real sub-agent resolving conflicts via `SubAgentToolset` end-to-end is
commit 20's "tests mode branch" scope, not this one.
"""

from __future__ import annotations

import shutil
from collections.abc import AsyncIterator, Iterator
from datetime import timedelta
from pathlib import Path

import pytest

try:
    from pydantic_ai.durable_exec.temporal import PydanticAIPlugin
    from temporalio import activity, workflow
    from temporalio.client import Client
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker
except ImportError:  # pragma: lax no cover
    pytest.skip('temporalio not installed', allow_module_level=True)

from pydantic import BaseModel

from pydantic_ai_harness.durable._branch_delegation import run_with_self_heal
from pydantic_ai_harness.durable._lease import EnvironmentActivities
from pydantic_ai_harness.durable._store import AcquireEnvParams, EnvironmentLease, GitSnapshotStore

pytestmark = pytest.mark.anyio

TEMPORAL_PORT = 7246  # avoid conflict with the code_mode (7244) and durable (7245) suites
TASK_QUEUE = 'branch-delegation-main'

# Module-level with fixed (not mkdtemp) paths -- the Temporal workflow sandbox
# re-imports this module to load the workflow/activity definitions, same
# constraint documented in `test_temporal_integration.py`.
_BASE = Path('/tmp/pah_branch_delegation_it')
_STORE_DIR = _BASE / 'store'
_WORKSPACES = _BASE / 'ws'
_STORE = GitSnapshotStore(_STORE_DIR)
_ACTS = EnvironmentActivities(store=_STORE, env_queue=TASK_QUEUE, workspaces_base=_WORKSPACES)


class _WriteParams(BaseModel):
    env_id: str
    path: str
    content: str


class _SelfHealWorkflowParams(BaseModel):
    parent_env_id: str
    parent_env_queue: str
    max_merge_retries: int
    concurrent_parent_edit: bool
    """Stand-in for "someone else advances the parent while this delegation runs":
    `run_once`'s first call also directly writes into the parent's live workspace."""
    resolve_on_retry: bool
    """Whether `run_once`'s resolution-round call actually rewrites the conflicting
    file. `False` simulates a sub-agent that can't resolve it, to exercise the
    retries-exhausted fallback."""
    keep_parent_moving: bool = False
    """If set, `run_once`'s resolution-round call *also* advances the parent again
    (independent of `resolve_on_retry`), so the retried land still finds a real
    conflict instead of the fast-forward a materialize alone would produce --
    exercises the retries-exhausted fallback distinctly from a merely-unresolved
    round."""
    soft_fail_on_call: int | None = None
    """If set, `run_once`'s call number `n` (1 = the initial run, 2 = the first
    resolution round) reports a soft degradation instead of doing any work --
    exercises `run_with_self_heal` returning immediately without attempting a
    merge (call 1) or a further land retry (call 2)."""


@activity.defn(name='test_write_file')
async def _write_file(params: _WriteParams) -> None:
    """Acquire (or converge on) `params.env_id`, write `params.path`, push.

    Stands in for a sub-agent's env-bound tool call -- the real one lazily
    acquires the same way via `_DurableEnvWrapper`/`TemporalPlacement`.
    """
    await _ACTS.acquire_environment(AcquireEnvParams(env_id=params.env_id))
    (_WORKSPACES / params.env_id / params.path).write_text(params.content, encoding='utf-8')
    await _ACTS.snapshot_held(params.env_id)


@workflow.defn
class _SelfHealWorkflow:
    @workflow.run
    async def run(self, params: _SelfHealWorkflowParams) -> str:
        calls = {'n': 0}

        async def run_once(task: str) -> tuple[str, bool]:
            calls['n'] += 1
            child_env_id = workflow.info().workflow_id
            if calls['n'] == params.soft_fail_on_call:
                return f'soft failure on call {calls["n"]}', False
            if calls['n'] == 1:
                await workflow.execute_activity(
                    'test_write_file',
                    _WriteParams(env_id=child_env_id, path='shared.txt', content='from-child'),
                    start_to_close_timeout=timedelta(seconds=30),
                )
                if params.concurrent_parent_edit:
                    await workflow.execute_activity(
                        'test_write_file',
                        _WriteParams(env_id=params.parent_env_id, path='shared.txt', content='from-parent'),
                        start_to_close_timeout=timedelta(seconds=30),
                    )
                return 'child output', True
            if params.resolve_on_retry:
                await workflow.execute_activity(
                    'test_write_file',
                    _WriteParams(env_id=child_env_id, path='shared.txt', content='resolved'),
                    start_to_close_timeout=timedelta(seconds=30),
                )
            if params.keep_parent_moving:
                # The parent keeps advancing too -- the retried land must still find a
                # real conflict, not a fast-forward over stale conflict markers.
                await workflow.execute_activity(
                    'test_write_file',
                    _WriteParams(env_id=params.parent_env_id, path='shared.txt', content=f'from-parent-{calls["n"]}'),
                    start_to_close_timeout=timedelta(seconds=30),
                )
            return 'resolved output', True

        return await run_with_self_heal(
            parent_lease=EnvironmentLease(env_id=params.parent_env_id, env_queue=params.parent_env_queue, epoch=0),
            run_once=run_once,
            task='do it',
            max_merge_retries=params.max_merge_retries,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'  # Temporal's client/worker are asyncio-only


@pytest.fixture(autouse=True)
def _clean_base() -> Iterator[None]:
    shutil.rmtree(_BASE, ignore_errors=True)
    _WORKSPACES.mkdir(parents=True, exist_ok=True)
    _ACTS._held.clear()  # pyright: ignore[reportPrivateUsage]
    yield
    shutil.rmtree(_BASE, ignore_errors=True)


@pytest.fixture
async def client() -> AsyncIterator[Client]:
    async with await WorkflowEnvironment.start_local(  # pyright: ignore[reportUnknownMemberType]
        port=TEMPORAL_PORT,
        dev_server_extra_args=['--dynamic-config-value', 'frontend.enableServerVersionCheck=false'],
    ):
        yield await Client.connect(f'localhost:{TEMPORAL_PORT}', plugins=[PydanticAIPlugin()])


async def _seed_parent(env_id: str, path: str, content: str) -> None:
    await _ACTS.acquire_environment(AcquireEnvParams(env_id=env_id))
    (_WORKSPACES / env_id / path).write_text(content, encoding='utf-8')
    await _ACTS.snapshot_held(env_id)


def _worker(client: Client) -> Worker:
    return Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[_SelfHealWorkflow],
        activities=[_ACTS.acquire_environment, _ACTS.fork_environment, _ACTS.merge_environment, _write_file],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_clean_land_returns_output_without_self_heal(client: Client) -> None:
    await _seed_parent('parent-clean', 'other.txt', 'p')

    async with _worker(client):
        output = await client.execute_workflow(
            _SelfHealWorkflow.run,
            args=[
                _SelfHealWorkflowParams(
                    parent_env_id='parent-clean',
                    parent_env_queue=TASK_QUEUE,
                    max_merge_retries=1,
                    concurrent_parent_edit=False,
                    resolve_on_retry=False,
                )
            ],
            id='self-heal-clean-1',
            task_queue=TASK_QUEUE,
        )

    assert output == 'child output'
    assert (_WORKSPACES / 'parent-clean' / 'shared.txt').read_text(encoding='utf-8') == 'from-child'


async def test_conflict_self_heals_and_lands_after_one_retry(client: Client) -> None:
    await _seed_parent('parent-heal', 'shared.txt', 'base')

    async with _worker(client):
        output = await client.execute_workflow(
            _SelfHealWorkflow.run,
            args=[
                _SelfHealWorkflowParams(
                    parent_env_id='parent-heal',
                    parent_env_queue=TASK_QUEUE,
                    max_merge_retries=1,
                    concurrent_parent_edit=True,
                    resolve_on_retry=True,
                )
            ],
            id='self-heal-resolve-1',
            task_queue=TASK_QUEUE,
        )

    assert output == 'resolved output'
    assert (_WORKSPACES / 'parent-heal' / 'shared.txt').read_text(encoding='utf-8') == 'resolved'


async def test_conflict_self_heals_via_fast_forward_even_without_a_real_resolution(client: Client) -> None:
    """Even if the resolution round is a no-op, the retried land still succeeds: the
    materialize step already made the parent's head an ancestor of the child branch,
    so the retry is a fast-forward regardless of whether the sub-agent changed anything."""
    await _seed_parent('parent-ff', 'shared.txt', 'base')

    async with _worker(client):
        output = await client.execute_workflow(
            _SelfHealWorkflow.run,
            args=[
                _SelfHealWorkflowParams(
                    parent_env_id='parent-ff',
                    parent_env_queue=TASK_QUEUE,
                    max_merge_retries=1,
                    concurrent_parent_edit=True,
                    resolve_on_retry=False,
                )
            ],
            id='self-heal-ff-1',
            task_queue=TASK_QUEUE,
        )

    assert output == 'resolved output'
    # The resolution round never touched the file -- it still has materialize's
    # conflict markers, committed as ordinary content by the fast-forward.
    landed = (_WORKSPACES / 'parent-ff' / 'shared.txt').read_text(encoding='utf-8')
    assert 'from-child' in landed
    assert 'from-parent' in landed
    assert '<<<<<<<' in landed


async def test_conflict_exhausts_retries_and_leaves_parent_untouched(client: Client) -> None:
    await _seed_parent('parent-exhaust', 'shared.txt', 'base')

    async with _worker(client):
        output = await client.execute_workflow(
            _SelfHealWorkflow.run,
            args=[
                _SelfHealWorkflowParams(
                    parent_env_id='parent-exhaust',
                    parent_env_queue=TASK_QUEUE,
                    max_merge_retries=1,
                    concurrent_parent_edit=True,
                    resolve_on_retry=False,
                    keep_parent_moving=True,
                )
            ],
            id='self-heal-exhaust-1',
            task_queue=TASK_QUEUE,
        )

    assert 'shared.txt' in output
    assert 'unchanged' in output
    # The land that would have overwritten it never completed -- the parent's content
    # is whatever the (still-unresolved) concurrent edits last left it at.
    assert (_WORKSPACES / 'parent-exhaust' / 'shared.txt').read_text(encoding='utf-8') == 'from-parent-2'


async def test_soft_failure_on_initial_run_returns_immediately_without_a_merge_attempt(client: Client) -> None:
    await _seed_parent('parent-soft-1', 'other.txt', 'p')

    async with _worker(client):
        output = await client.execute_workflow(
            _SelfHealWorkflow.run,
            args=[
                _SelfHealWorkflowParams(
                    parent_env_id='parent-soft-1',
                    parent_env_queue=TASK_QUEUE,
                    max_merge_retries=1,
                    concurrent_parent_edit=False,
                    resolve_on_retry=False,
                    soft_fail_on_call=1,
                )
            ],
            id='self-heal-soft-1',
            task_queue=TASK_QUEUE,
        )

    assert output == 'soft failure on call 1'
    # Nothing landed -- the delegation degraded before it produced anything to merge.
    assert not (_WORKSPACES / 'parent-soft-1' / 'shared.txt').exists()


async def test_soft_failure_during_resolution_round_returns_immediately(client: Client) -> None:
    await _seed_parent('parent-soft-2', 'shared.txt', 'base')

    async with _worker(client):
        output = await client.execute_workflow(
            _SelfHealWorkflow.run,
            args=[
                _SelfHealWorkflowParams(
                    parent_env_id='parent-soft-2',
                    parent_env_queue=TASK_QUEUE,
                    max_merge_retries=1,
                    concurrent_parent_edit=True,
                    resolve_on_retry=False,
                    soft_fail_on_call=2,
                )
            ],
            id='self-heal-soft-2',
            task_queue=TASK_QUEUE,
        )

    assert output == 'soft failure on call 2'
    # The retried land never ran -- the parent still has the concurrent edit as-is.
    assert (_WORKSPACES / 'parent-soft-2' / 'shared.txt').read_text(encoding='utf-8') == 'from-parent'
