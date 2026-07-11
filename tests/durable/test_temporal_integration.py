"""End-to-end Temporal tests for the durable-environment stack.

Runs a real agent (FileSystem + DurableEnvironment) inside a Temporal workflow
against a local dev server (`WorkflowEnvironment.start_local()`), exercising the
pieces that unit tests can't: `TemporalPlacement`'s activity routing to the
sticky env queue, `DurableEnvironmentPlugin.run_worker` mounting/draining that
queue, and the snapshot push landing in the store.

Fencing, lease convergence, and journal dedup are covered at the unit level
(`test_lease.py`, `test_journal.py`) and not re-tested here.
"""

from __future__ import annotations

import shutil
from collections.abc import AsyncIterator, Iterator
from datetime import timedelta
from pathlib import Path

import pytest

try:
    from pydantic_ai.durable_exec.temporal import AgentPlugin, PydanticAIPlugin, TemporalAgent
    from temporalio import workflow
    from temporalio.client import Client
    from temporalio.common import RetryPolicy
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker
    from temporalio.workflow import ActivityConfig
except ImportError:  # pragma: lax no cover
    pytest.skip('temporalio not installed', allow_module_level=True)

from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_harness import FileSystem
from pydantic_ai_harness.durable import DurableEnvironment, GitSnapshotStore
from pydantic_ai_harness.durable.temporal import DurableEnvironmentPlugin, TemporalPlacement

pytestmark = pytest.mark.anyio

TEMPORAL_PORT = 7245  # avoid conflict with the code_mode suite (7244)
TASK_QUEUE = 'durable-env-main'
BASE_ACTIVITY_CONFIG = ActivityConfig(
    start_to_close_timeout=timedelta(seconds=60),
    retry_policy=RetryPolicy(maximum_attempts=1),
)

# Module-level with fixed (not mkdtemp) paths: the Temporal workflow sandbox
# re-imports this module to load the workflow definition, and forbids the
# non-deterministic `tempfile.mkdtemp` at import. Real directory setup/teardown
# happens in the `_clean_base` fixture, which runs outside the sandbox.
_BASE = Path('/tmp/pah_durable_it')
_STORE_DIR = _BASE / 'store'
_WORKSPACES = _BASE / 'ws'
_STORE = GitSnapshotStore(_STORE_DIR)


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
    yield
    shutil.rmtree(_BASE, ignore_errors=True)


@pytest.fixture
async def client() -> AsyncIterator[Client]:
    async with await WorkflowEnvironment.start_local(  # pyright: ignore[reportUnknownMemberType]
        port=TEMPORAL_PORT,
        dev_server_extra_args=['--dynamic-config-value', 'frontend.enableServerVersionCheck=false'],
    ):
        yield await Client.connect(f'localhost:{TEMPORAL_PORT}', plugins=[PydanticAIPlugin()])


# ---------------------------------------------------------------------------
# Agent and workflow (module-level -- Temporal requirement)
# ---------------------------------------------------------------------------


def _write_then_finish(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
    """Emit a `write_file` call on the first turn, then a final text on the second."""
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolReturnPart) and part.tool_name == 'write_file':
                    return ModelResponse(parts=[TextPart(content='done')])
    return ModelResponse(
        parts=[
            ToolCallPart(
                tool_name='write_file',
                args={'path': 'notes.txt', 'content': 'hello durable'},
                tool_call_id='tc_write_1',
            )
        ]
    )


durable_agent = Agent(
    FunctionModel(_write_then_finish),
    name='durable_coder',
    capabilities=[
        FileSystem(),
        DurableEnvironment(placement=TemporalPlacement(), store=_STORE, snapshot_policy='per_op'),
    ],
)

temporal_durable_agent = TemporalAgent(durable_agent, activity_config=BASE_ACTIVITY_CONFIG)


@workflow.defn
class DurableWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await temporal_durable_agent.run(prompt)
        return str(result.output)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _scheduled_activity_queues(client: Client, workflow_id: str) -> list[tuple[str, str]]:
    """Return `(activity_type, task_queue)` for every ActivityTaskScheduled event."""
    handle = client.get_workflow_handle(workflow_id)
    history = await handle.fetch_history()
    scheduled: list[tuple[str, str]] = []
    for event in history.events:
        if event.HasField('activity_task_scheduled_event_attributes'):
            attrs = event.activity_task_scheduled_event_attributes
            scheduled.append((attrs.activity_type.name, attrs.task_queue.name))
    return scheduled


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_env_bound_tool_is_routed_to_the_sticky_queue_and_snapshotted(client: Client) -> None:
    env_plugin = DurableEnvironmentPlugin([temporal_durable_agent], workspaces_base=_WORKSPACES)
    workflow_id = 'durable-nominal-1'

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurableWorkflow],
        plugins=[AgentPlugin(temporal_durable_agent), env_plugin],
    ):
        output = await client.execute_workflow(
            DurableWorkflow.run,
            args=['write a note'],
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )
    # The host Worker's `async with` block exiting does not guarantee the
    # plugin's sticky-worker drain and snapshot push have finished -- see
    # DurableEnvironmentPlugin.run_worker's docstring.
    await env_plugin.wait_drained()

    assert output == 'done'

    # The file landed in this env's workspace (env_id == workflow_id).
    workspace_file = _WORKSPACES / workflow_id / 'notes.txt'
    assert workspace_file.read_text(encoding='utf-8') == 'hello durable'

    # per_op snapshot: restoring from the store into a fresh dir reproduces it.
    restored = _WORKSPACES / 'restored-check'
    await _STORE.restore(workflow_id, restored)
    assert (restored / 'notes.txt').read_text(encoding='utf-8') == 'hello durable'

    # Routing: the acquire ran on the shared queue, the write_file activity on the
    # sticky env queue.
    scheduled = await _scheduled_activity_queues(client, workflow_id)
    acquire_queues = {q for name, q in scheduled if name == 'acquire_environment'}
    write_queues = {q for name, q in scheduled if 'call_tool' in name}
    assert acquire_queues == {TASK_QUEUE}
    assert write_queues == {env_plugin._env_queue}  # pyright: ignore[reportPrivateUsage]


async def test_two_workflows_get_independent_workspaces(client: Client) -> None:
    env_plugin = DurableEnvironmentPlugin([temporal_durable_agent], workspaces_base=_WORKSPACES)

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurableWorkflow],
        plugins=[AgentPlugin(temporal_durable_agent), env_plugin],
    ):
        outputs = [
            await client.execute_workflow(
                DurableWorkflow.run, args=['note'], id=f'durable-concurrent-{i}', task_queue=TASK_QUEUE
            )
            for i in range(2)
        ]
    await env_plugin.wait_drained()

    assert outputs == ['done', 'done']
    for i in range(2):
        assert (_WORKSPACES / f'durable-concurrent-{i}' / 'notes.txt').read_text(encoding='utf-8') == 'hello durable'
