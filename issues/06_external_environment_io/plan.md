# Plan -- external read/write access to a held durable environment

Target: `pydantic_ai_harness/durable/{_journal.py,_lease.py,_plugin.py}`.

## Step 0 -- `guarded_mutating` can't be reused as-is (blocker, found while planning)

`guarded_mutating(*, ctx: RunContext[Any], root, tool, apply, store, policy)` derives both
`op_id = f'{ctx.run_id}:{ctx.tool_call_id}'` and (internally, via `_env_id_from_ctx(ctx)`) the
`env_id` for the snapshot push from a `RunContext`. `write_environment_file` has no `RunContext` --
it's called from outside the agent graph (a Temporal `@workflow.update`, an HTTP handler). Reusing
the function without duplicating its lock+journal+push logic means widening its contract:

1. `guarded_mutating`'s signature becomes `(*, op_id: str, env_id: str | None, root, tool, apply,
   store, policy)` -- `ctx` dropped, `op_id`/`env_id` become plain parameters the caller derives.
2. The three existing call sites in `filesystem/_toolset.py` (`write_file`, `edit_file`,
   `create_directory`) compute `op_id=f'{ctx.run_id}:{ctx.tool_call_id}'` and
   `env_id=_env_id_from_ctx(ctx)` themselves before calling -- two lines added per call site, no
   behavior change (same values, computed one frame up).
3. `_env_id_from_ctx` stays in `_journal.py` (still the natural home -- it reads
   `ctx.metadata['durable_env']`, which `filesystem/_toolset.py` already imports the module for),
   just called one level higher.
4. `write_environment_file` (the new activity, no `ctx` at all) passes `env_id` directly -- it's
   already a plain parameter, no extraction needed -- and derives `op_id` from its own caller-
   supplied `op_id` parameter (not synthesized locally: the *caller* -- e.g. CSE's `update_pv`
   workflow handler -- must supply one that's stable across a Temporal activity retry, the same
   requirement `op_id = f'{run_id}:{tool_call_id}'` satisfies for a tool call).

This is a real widening of `guarded_mutating`'s contract (two new required kwargs replacing one),
not a hack bolted beside it -- it turns a tool-call-shaped primitive into an engine-neutral one,
consistent with the "prefer the most generic input types possible" guideline. Existing tests for
`write_file`/`edit_file`/`create_directory` must keep passing unchanged (behavior, not signature,
is under test there per this repo's testing conventions -- drive through `Agent(...)`, not the
private `guarded_mutating` call).

## Step 1 -- `get_environment_queue`

`_lease.py`, on `EnvironmentActivities`:

```python
@activity.defn(name='get_environment_queue')
async def get_environment_queue(self, env_id: str) -> str | None:
    """Return the sticky queue holding `env_id`'s lease, or None if never acquired.

    A thin read over the store's fence record -- works regardless of which worker
    answers, since the record is persisted by the store, not this process's `_held`.
    """
    record = await self._store.get_lease(env_id)
    return record.env_queue if record is not None else None
```

Registered on the **host** queue only (like `acquire_environment`) -- a caller doesn't yet know
which sticky queue to ask, that's the whole point of this activity.

## Step 2 -- `write_environment_file` / `read_environment_file`

`_lease.py`, on `EnvironmentActivities`. Both resolve the workspace via `self._held[env_id]` --
raising a clear, non-retryable error if this worker doesn't hold the lease (the caller routed here
by `task_queue=env_queue` from step 1, so this should only happen if the environment was released
between the two calls -- a real, reportable condition, not a transient one):

```python
@dataclass
class WriteEnvFileParams:
    env_id: str
    path: str
    content: str
    op_id: str

@activity.defn(name='write_environment_file')
async def write_environment_file(self, params: WriteEnvFileParams) -> str:
    held = self._held.get(params.env_id)
    if held is None:
        raise ApplicationError(f'{params.env_id!r} is not held by this worker', non_retryable=True)

    async def _apply() -> str:
        resolved = held.workspace / params.path  # same traversal guard as FileSystemToolset -- see open question below
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(params.content, encoding='utf-8')
        return f'Wrote {len(params.content)} chars to {params.path}'

    return await guarded_mutating(
        op_id=params.op_id, env_id=params.env_id, root=held.workspace, tool='write_environment_file',
        apply=_apply, store=self._store, policy=self._default_policy,
    )

@activity.defn(name='read_environment_file')
async def read_environment_file(self, params: ReadEnvFileParams) -> str:
    held = self._held.get(params.env_id)
    if held is None:
        raise ApplicationError(f'{params.env_id!r} is not held by this worker', non_retryable=True)
    async with env_lock(held.workspace):  # don't read mid-write
        resolved = held.workspace / params.path
        if not resolved.is_file():
            raise ApplicationError(f'{params.path!r} not found in {params.env_id!r}', non_retryable=True)
        return resolved.read_text(encoding='utf-8')
```

Both registered on **host + sticky** (same treatment as `merge_environment` in `_plugin.py`).

**Path safety, found while planning**: `FileSystemToolset._safe_resolve` does traversal
prevention (`allowed_patterns`/`denied_patterns`/symlink resolution) that `write_environment_file`
has no equivalent for -- it isn't a `FileSystemToolset` instance, it's a bare activity. Minimal
guard needed: reject any `path` that resolves (via `Path.resolve()`) outside `held.workspace`,
mirroring the "no traversal above root" invariant `FileSystemToolset` already documents, without
pulling in `allowed_patterns`/`protected_patterns` (those are the toolset's own least-privilege
config, not meaningful for a caller that isn't a specific agent's tool).

**`EnvironmentActivities` needs a default `SnapshotPolicy`** to pass to `guarded_mutating` --
today `store`/`policy` are only known toolset-side (`configure_durability`), not on
`EnvironmentActivities` itself (constructed with `store`, but no `policy` field, see `__init__`
sub-issue 3). Add `default_policy: SnapshotPolicy = SnapshotPolicy(mode='per_op')` to
`EnvironmentActivities.__init__`, matching `DurableEnvironment`'s own default.

## Step 3 -- registration (`_plugin.py`)

Mirror the existing `merge_environment`/`fork_environment` wiring exactly: `get_environment_queue`
goes wherever `acquire_environment` is registered (host); `write_environment_file`/
`read_environment_file` go wherever `merge_environment` is registered (host + sticky).

## Step 4 -- tests

`tests/durable/test_lease.py` (mirroring existing `EnvironmentActivities` tests):

- `get_environment_queue` returns the right queue for a held/fenced env, `None` for one never
  acquired.
- `write_environment_file` on the holding worker: file appears, content correct, journal entry
  recorded under the given `op_id`.
- Same `op_id` retried: second call returns the journaled result without re-applying (assert via a
  side-effect counter in a stubbed `apply`, same pattern the existing `guarded_mutating` tests use).
- `write_environment_file`/`read_environment_file` on a worker that doesn't hold `env_id`: clear
  non-retryable error, not a silent no-op or a hang.
- `write_environment_file` racing a concurrent tool call's `write_file` on the same workspace:
  serialized by `env_lock`, no interleaved writes (same lock object, assert via ordering).
- Traversal rejection: `path='../outside.txt'` raises before touching the filesystem.
- `read_environment_file` sees a write made immediately before it (same `env_id`, sequential
  calls) -- read-after-write consistency within one held workspace.

`tests/durable/test_journal.py` (or wherever `guarded_mutating` is directly tested): update for the
new `op_id`/`env_id` kwargs; confirm `filesystem/_toolset.py`'s three call sites still pass their
existing test suite unchanged (behavior parity, not a signature test).

## Gates

`make lint && make typecheck && make test` (per this repo's own `CLAUDE.md`) before considering
this issue done.
