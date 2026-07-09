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
- `configure_durability(store, policy)` -- wire a snapshot store and policy into the mutating-op path. `store=None` is a no-op (the local path already used above); `SnapshotStore`/`SnapshotPolicy` are placeholder shapes fleshed out by `DurableEnvironment` (sub-issue 3).
