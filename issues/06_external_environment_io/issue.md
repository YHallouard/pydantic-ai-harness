# External read/write access to a held durable environment

> Repo: **pydantic-ai-harness** -- suggested labels: `enhancement`, `capability`
> Part of: [Tracking] Durable execution for environment-bound capabilities (sub-issue 9)
> Depends on: sub-issue 4 (`DurableEnvironment`)

## Goal

Let something *outside* the agent's own tool-call graph -- an HTTP handler, a Temporal
`@workflow.update`, a webhook -- read or write a file in an environment a `DurableEnvironment`
capability already holds a lease for, without going through a synthetic tool call.

## Problem

Today the only path into a held environment is `_DurableEnvWrapper.call_tool`
(`durable/_capability.py`), reachable only from inside the agent graph with a real
`RunContext`/`ToolsetTool`. Everything the routing needs -- the held lease
(`EnvironmentActivities._held`), the snapshot store, the sticky env-queue name -- is private, with
no injection point. A consumer that needs to push or pull a file from outside a tool call (a
live-editing UI, an attachment upload mid-session, a debugging endpoint) has no supported way to do
it: the workspace is reachable exclusively through the agent's own tool loop.

This is not a narrow one-consumer need. Anywhere a durable agent's workspace is also the system of
record for something a human or another service edits concurrently -- a live document, a shared
scratch space, an uploaded attachment -- the same gap shows up: the environment is held by a
specific worker (identified by `env_queue`), and nothing outside the tool-call path can address
that worker or its held state.

## Proposal

Three additions, deliberately narrow (reads and writes only, no new snapshot semantics):

1. `EnvironmentActivities.get_environment_queue(env_id) -> str | None`: a thin read over
   `SnapshotStore.get_lease(env_id)` (already public on the `SnapshotStore` protocol) -- returns
   the sticky `env_queue` a caller must route subsequent calls to, or `None` if the environment was
   never acquired. Registered on the host queue, like `acquire_environment`.
2. `EnvironmentActivities.write_environment_file(env_id, path, content, op_id) -> str`: registered
   on **both** the host queue and the sticky `env_queue` (same treatment `merge_environment`
   already gets). Resolves the held workspace via `_held[env_id]`, and reuses `guarded_mutating`
   (not just `env_lock`) with the caller-supplied `op_id`, so a Temporal-activity retry is
   deduplicated by the same idempotency journal a normal `write_file` tool call uses, and the write
   shows up in the same `.durable_env/journal` audit trail. A snapshot push is included, same as
   any other mutating op under `per_op` policy.
3. `EnvironmentActivities.read_environment_file(env_id, path) -> str`: same routing (host + sticky),
   no journal entry needed (a pure read).

None of this touches `_DurableEnvWrapper`, tool routing, or the lease/fencing protocol -- it's a
second, narrower entry point into the same held workspace, for callers that aren't inside the agent
graph.

## Non-goal

A generic "arbitrary RPC into a held environment" surface. This is scoped to read/write a single
file by path, matching exactly what `FileSystemToolset.read_file`/`write_file` already do
internally -- not a new capability.

## Open questions

- Exact behavior of `get_environment_queue`/`write_environment_file`/`read_environment_file` when
  `env_id` was genuinely never acquired (fresh session, typo) vs. was acquired and later released
  (`release_environment` already ran) -- both should return/raise something a caller can
  distinguish from "still being acquired, retry."
- Whether `write_environment_file` needs an `expected_hash`-style optimistic-concurrency guard for
  callers racing a live agent tool call on the same path, or whether the existing `env_lock`
  serialization is sufficient (no new invariant beyond "last write wins, in lock order").

## Related

Sub-issue 4 (`issues/03_durable_environment_capability`), `_journal.py`'s `guarded_mutating`/
`OpJournal` (the exact-once contract this reuses).
