# `GitSnapshotStore` -- fenced snapshot store (03a)

> Repo: **pydantic-ai-harness** -- suggested labels: `enhancement`, `capability`
> Part of: [Tracking] Durable execution for environment-bound capabilities (sub-issue 4)
> Split from: `issues/03_durable_environment_capability/` (2026-07-20), PR 1 of 3
> Depends on: sub-issues 1-2 (harness). **Autonomous**: testable without Temporal, and the
> prerequisite of 03b and 03c.

## Goal

The snapshot store that gives a workspace a durable head per `env_id`, with native epoch
fencing: a pod that lost its lease cannot push anymore, so a false-positive failure
detection degrades to a clean handover instead of a split-brain. This is the foundation
03b's lease activities and 03c's Temporal integration both build on.

Commit: `33b6f73` (`GitSnapshotStore` fence CAS).

## Mechanics

- **Contracts** (Pydantic `BaseModel`, serialized natively by Temporal's pydantic data
  converter across activity boundaries): `EnvironmentLease` (`env_id`, `env_queue`,
  `epoch` -- `model_dump()` must satisfy the `DurableEnvironmentLease` `TypedDict`
  pydantic-ai core reads in `ctx.metadata['durable_env']`), `Head` (`sha`, `epoch`),
  `LeaseRecord` (current lease + `fenced_at`, `to_lease()`), `SnapshotPolicy`
  (`per_op`/`per_step`/`content_hash`, default `per_op`), `AcquireEnvParams`.
- **Repo model**: one bare repo per root env (the workflow), one branch per env (`main`
  for the parent, `{tool_call_id}` for children) -- `fork` = `git branch`, which requires
  the shared repo. The `LeaseRecord` lives in the same store.
- **`GitSnapshotStore(base_path)`**: `receive.denyNonFastforwards=true`. **The CAS is
  native git**: `fence(env_id)` pushes an empty "fence" commit on the branch; a fenced
  pod's local head is now behind, its next push is a non-fast-forward rejected by git
  itself. The lease `epoch` becomes an informational counter (logs/tests), not the
  critical mechanism.
- **`SnapshotStore` protocol**: `fence`, `get_lease` (get-or-provision support),
  `is_current` (fencing validation for idempotent re-acquire), `push`, `restore`,
  `fork` (for sub-issue 6 / branch workspaces).
- **`S3SnapshotStore`**: tarball + `head` object with `If-Match` ETag preconditions for
  the CAS.
- `per_op` for mutating ops is the default *for correctness*, not performance: it
  preserves the invariant *completed in history => present in the snapshot*.
  `per_step`/`content_hash` are opt-in for reconstructible workspaces (documented
  divergence window on pod death).

## Scope

- Source: `pydantic_ai_harness/durable/_store.py`
- Tests: `tests/durable/test_store.py`

**Perimeter note**: `925b6b0` (keep the git dir out of the workspace) also touches
`_store.py` but stays in 03c -- it is an additive fix 03c applies on top of the 03a
base, not part of 03a itself.

## Acceptance

- Two concurrent `fence` calls: exactly one wins.
- A fenced pod's next push is rejected (non-fast-forward).
- Push/restore round-trip reproduces the workspace.
- `fork` creates the child branch off the parent head.
- The lease record written by `fence` is readable via `get_lease`.

## Related

`issues/03_durable_environment_capability/issue.md` (design d'ensemble), #115 (Hermes
git checkpoints), #277 (S3 StepStore).

**Warm restore (replié depuis l'ancienne issue 08, 2026-07-22)** : `restore` n'est plus
systématiquement destructeur -- un chemin warm réutilise le git dir sibling quand il correspond déjà
au bon bare repo (`fetch` du delta + `checkout --force` + `clean -fdx`), sans changer le contrat
public de `SnapshotStore.restore` ni le rôle du fence CAS. Voir `plan.md` Étape 4. La contrepartie
côté `_lease.py` (chemin de re-acquisition stale qui ne jette plus le workspace) vit en 03b. Ce repli
a remplacé l'ancienne issue de suivi 08 : une amélioration stricte, sans changement de contrat, du
code déjà présent dans 03 n'avait pas à être livrée en suivi séparé après un `restore` sciemment
destructeur.
