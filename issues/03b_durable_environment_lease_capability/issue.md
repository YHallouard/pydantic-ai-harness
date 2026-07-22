# Lease + engine-neutral `DurableEnvironment` capability (03b)

> Repo: **pydantic-ai-harness** -- suggested labels: `enhancement`, `capability`
> Part of: [Tracking] Durable execution for environment-bound capabilities (sub-issue 4)
> Split from: `issues/03_durable_environment_capability/` (2026-07-20), PR 2 of 3
> Depends on: 03a (snapshot store). **Testable without pydantic-ai#4977** and without
> the real Temporal routing seam -- this is the pivot PR of the split.

## Goal

The lease lifecycle and the engine-neutral capability that makes a workspace survive
pod failure: `DurableEnvironment` acquires an `EnvironmentLease`, transports it through
`ctx.metadata['durable_env']`, and delegates everything engine-specific to an
`EnvironmentPlacement` driver. Temporal specifics are out of scope (03c).

Commits: `73cf809` (lease activities), `6b3ecfd` (`temporalio` import gate),
`8ef5d78` (`DurableEnvironment` capability), `eef2db8` (historical intermediate, see
perimeter note), `a447692` (wire `configure_durability`), `e7c8b94`
(`EnvironmentPlacement` driver).

## Mechanics

- **Lease contract**: `EnvironmentLease` from 03a. Transport is
  `RunContext.metadata['durable_env']` -- the one thing that propagates across
  capability hooks and is already serialized by `TemporalRunContext` into every
  activity. Written as `lease.model_dump()`, re-validated with
  `EnvironmentLease.model_validate(...)` worker-side: the transport is plain data
  (JSON round-trip), the contract is the model.
- **Lease lifecycle** (`durable/_lease.py`): acquisition memoized behind a shared
  future (parallel env-bound calls in one batch must not double-acquire); *idempotent
  re-acquire* -- a pod that already holds the `env_id` first validates its local head
  via `store.is_current(...)` (a fenced pod still believes it holds the lease) and only
  then returns the existing lease **without an epoch bump**; a "busy bounce"
  (retryable `ApplicationError` at capacity) spreads placement; `release_environment`
  scoped to **workflow completion**, not run end, plus a TTL janitor for terminated
  workflows. Get-or-provision: converge on the lease recorded by the last `fence`
  unless it is the failed one -- without it, parent and sub-agent re-provisioning at
  the same time fence each other in ping-pong.
- **`temporalio` gate**: `6b3ecfd` keeps the `try: import temporalio` guard directly
  inside `_lease.py` -- the lease module must stay importable without `temporalio`
  installed. No public `temporal.py` module in this PR (see perimeter note).
- **Capability** (`durable/_capability.py`): `DurableEnvironment(AbstractCapability)`
  takes a driver `placement: EnvironmentPlacement` (`durable/_placement.py`) that
  isolates the engine. `for_agent()` **does not mutate toolsets** (non-mutating
  contract of #4977; a toolset shared between a durable and a non-durable agent would
  be clobbered). Workflow-side wrapper: memoized lease written into
  `ctx.metadata['durable_env']` at the first env-bound call of the run; **no release
  at run end** (a conversational workflow runs several `agent.run` on the same
  workspace). `env_id` = `workflow.info().workflow_id` in a workflow, else
  `ctx.run_id`.
- **`configure_durability` wiring** (`a447692`): integration points in
  `filesystem/_toolset.py`, `shell/_toolset.py`, `code_mode/_toolset.py` -- worker-side
  configuration (store/policy), root resolution in the toolset itself via
  `ctx.metadata['durable_env']['env_id']` (the `EnvironmentBound` protocol default,
  static-root fallback without a lease).
- **Graceful no-op** outside Temporal: without a workflow context the capability does
  nothing and `root_dir` falls back to its static value.

## Scope

- Source: `durable/_lease.py`, `durable/_capability.py`, `durable/_placement.py`,
  `durable/__init__.py` (engine-neutral exports), plus the `configure_durability`
  integration points in `filesystem/_toolset.py`, `shell/_toolset.py`,
  `code_mode/_toolset.py`.
- Tests: `tests/durable/test_lease.py`, `tests/durable/test_capability.py`,
  `tests/durable/test_journal.py`.

**Perimeter notes** (corrections vs. the initial commit-by-commit split):

- `eef2db8` (`run_env_worker`) no longer exists in the tree -- the final design went
  straight to `DurableEnvironmentPlugin` (03c). It is an historical intermediate step
  of 03b with no final deliverable of its own: it stays out of the PR scope. 03b ships
  the usable lease activities directly, without a dedicated worker helper.
- `durable/temporal.py` (the module with the `try: import temporalio` gate) is **not**
  in 03b despite `6b3ecfd`: it re-exports 03c symbols (`TemporalPlacement`,
  `DurableEnvironmentPlugin`, `TemporalBranchDelegation`) on top of
  `EnvironmentActivities`. It moves entirely to 03c.

## Acceptance

- Lease acquire/release round-trip through the store (no Temporal required).
- Memoized acquisition: parallel env-bound calls in one batch acquire once.
- Idempotent re-acquire validates `is_current` and does not bump the epoch; a fenced
  holder purges and re-provisions.
- Concurrent re-acquire after a pod failure converges on a single lease (no fence
  ping-pong -- count fence commits).
- Capability is a graceful no-op outside a workflow context.
- Model-request activities never leave the shared queue (asserted in tests).

## Related

`issues/03_durable_environment_capability/issue.md` (design d'ensemble), 03a, 03c,
pydantic-ai#4977.

**Warm re-acquisition (replié depuis l'ancienne issue 08, 2026-07-22)** : le chemin de
re-acquisition stale d'`acquire_environment` (fencé ailleurs, `is_current` faux) cesse d'appeler
`discard_workspace` avant le `restore` qui suit -- le workspace chaud converge en warm (voir 03a
Étape 4). Scope volontairement limité à ce seul chemin, **pas** `release_environment` : un enfant de
délégation `branch` a un `env_id` à usage unique (`workflow.info().workflow_id`, jamais réacquis) et
release après chaque délégation ; y retenir le workspace ferait croître le disque sans borne pour un
bénéfice qui ne se matérialise essentiellement jamais. Une régression de
`test_branch_delegation_integration.py` a révélé ce point pendant l'implémentation. Le chemin stale,
lui, est borné par le nombre d'environnements tenus au moment d'un fencing et réellement susceptible
d'être réutilisé (reprovision du même env, moments après).

**Follow-up (2026-07-21)** : le driver `EnvironmentPlacement` gagne une notion de capacité
observable (`at_capacity`/`wait_for_capacity` sur `EnvironmentActivities`) dans
`issues/09_capacity_aware_acquire_placement/`, consommée par un `CustomSlotSupplier` Temporal
côté 03c -- le protocole engine-neutre lui-même n'est pas modifié, seule l'implémentation
`EnvironmentActivities` s'enrichit.
