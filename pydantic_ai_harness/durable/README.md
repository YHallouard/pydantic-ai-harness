# durable

Shared primitives for environment-bound toolsets (`FileSystem`, `Shell`, `CodeMode`) under durable execution. Not a capability itself -- there's nothing to add to `Agent(capabilities=[...])` here. The three toolsets use these internally, and the `DurableEnvironment` capability (sub-issue 3 of the durable-environment tracking issue) will build on them.

## `env_bound_metadata`

Every tool an environment-bound toolset exposes carries `ToolDefinition.metadata` tags:

```python
{'env_bound': True, 'mutating': bool}
```

`env_bound` marks a tool as needing to execute where its workspace lives -- the routing story `DurableEnvironment` builds on, using the same vocabulary as pydantic-ai#4977's `metadata['temporal']` path. `mutating` marks whether the tool changes the workspace: mutating calls go through the per-env lock and idempotency journal below; read-only calls skip that work entirely.

## `OpJournal` and `guarded_mutating`

Durable engines retry activities assuming idempotence. `guarded_mutating` wraps a mutating tool call so a retry is safe:

- **Per-env lock.** Every mutating call for a given resolved root takes an `anyio.Lock` before touching anything, so concurrent mutations on one workspace can't race on files or the journal.
- **Idempotency journal.** Each op records `op_id = f'{run_id}:{tool_call_id}'` to `<root>/.durable_env/journal` (JSONL, append-only, versioned with the workspace). Before applying, `guarded_mutating` checks whether `op_id` is already journaled and returns the recorded result instead of re-applying.
- **`JournalSkipped`.** An `apply` callable can raise `JournalSkipped(result)` for an ambiguous completion (e.g. a command that timed out and may or may not have applied its effects) -- the result is returned normally, but nothing is journaled, so a retry with the same `op_id` tries again instead of replaying a cached "maybe" forever.

```python
from pydantic_ai_harness.durable import guarded_mutating

async def write_file(self, ctx, path, content):
    async def _apply() -> str:
        ...  # the actual write
        return 'Wrote ...'

    return await guarded_mutating(ctx=ctx, root=resolved_root, tool='write_file', apply=_apply)
```

This gives at-least-once application with exactly-once-as-observed replay locally. The end-to-end exactly-once guarantee -- "result delivered to the caller" implies "op is in the latest snapshot" -- needs the joint snapshot+journal commit `DurableEnvironment` adds (sub-issue 3); see `guarded_mutating`'s docstring for the crash-window analysis.

## `EnvironmentBound`

A structural `Protocol` implemented by `FileSystemToolset`, `ShellToolset`, and `CodeModeToolset`, so an orchestrator (Temporal routing, an audit layer, approval policies) can identify and configure environment-bound toolsets without importing their concrete classes:

- `env_bound_tools()` -- a `ToolSelector` matching every tool the toolset owns (`'all'` for the three toolsets today).
- `set_env_root(root)` -- rebind the resolved root/cwd/mount source after construction, once a durable-execution orchestrator has assigned a workspace.
- `configure_durability(store, policy)` -- wire a snapshot store and policy into the mutating-op path. `store=None` is a no-op (the local path already used above); `SnapshotStore`/`SnapshotPolicy` are the shapes `DurableEnvironment` configures at worker start.

## `DurableEnvironment` and `EnvironmentPlacement`

The capability that makes a workspace survive pod failure under a durable-execution engine, without moving model/MCP activities off the engine's shared queue:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import FileSystem, Shell
from pydantic_ai_harness.durable import DurableEnvironment, GitSnapshotStore
from pydantic_ai_harness.durable.temporal import TemporalPlacement

agent = Agent(
    'openai:gpt-5.2',
    name='coder',
    capabilities=[
        FileSystem(), Shell(),
        DurableEnvironment(
            placement=TemporalPlacement(),
            store=GitSnapshotStore('/var/snapshots'),
            snapshot_policy='per_op',
        ),
    ],
)
```

The capability itself is engine-neutral: it decides *when* a lease is needed (the first env-bound tool call in a run), memoizes it, exposes it via `ctx.metadata['durable_env']`, and re-acquires it (bounded) when the placement target dies. Everything engine-specific -- durable-context detection, the durable acquire step, failure classification, and call placement -- is behind the injected `EnvironmentPlacement` driver (`pydantic_ai_harness.durable._placement`). An engine without a placement problem (e.g. DBOS, where tools run in the same process as the workflow) only needs `acquire` to restore the workspace and can leave routing to the plain fallback.

Run-side only: the capability never touches a `SnapshotStore` directly (durable run code must stay deterministic); the store/policy it carries are read worker-side, outside the sandbox.

## `pydantic_ai_harness.durable.temporal`

Requires the `temporal` optional group. Kept out of this package's own eager imports so `FileSystem`/`Shell`/`CodeMode` stay importable without `temporalio` installed.

### `TemporalPlacement`

The Temporal `EnvironmentPlacement` driver: `acquire` calls the `acquire_environment` activity on the shared queue, and a schedule-to-start timeout on a routed call is how a dead or fenced-out pod is detected (the sticky queue has no poller anymore, so the activity never starts).

### `run_env_worker`

Mounts the two `Worker`s a pod needs: the shared task queue (agent workflows, model/MCP/tool activities, `acquire_environment`/`release_environment`) and a sticky `env-{uuid}` queue that env-bound tool calls get routed to once this pod holds the lease -- the canonical Temporal `worker_specific_task_queues` pattern.

```python
from pydantic_ai.durable_exec.temporal import TemporalAgent
from pydantic_ai_harness.durable import GitSnapshotStore
from pydantic_ai_harness.durable.temporal import run_env_worker

await run_env_worker(
    client,
    agents=[TemporalAgent(agent)],
    shared_task_queue='agent-io',
    workflows=[MyWorkflow],
    store=GitSnapshotStore('/var/snapshots'),
)
```

It also wires `store`/`snapshot_policy` into every `EnvironmentBound` toolset it finds on `agents` (worker-side `configure_durability`/`set_env_root` -- the concrete toolsets, not the temporalized activity-routing wrappers) and blocks until SIGTERM, then drains: stop polling both queues, wait for in-flight activities, push a final snapshot for every workspace this pod still holds.
