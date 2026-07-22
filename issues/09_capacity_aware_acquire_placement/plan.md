<!-- checkout_from: issues/03c_durable_environment_temporal_integration/ -->
<!-- note: l'ancienne issue 08 (warm restore) a été repliée dans 03a/03b le 2026-07-22 ; ce plan
     ne checkout plus depuis 08. Le warm restore fait partie de la base sur laquelle 09 s'appuie. -->
> Trois changements couplés, un seul comportement correct (voir Step 0) : un `CustomSlotSupplier`
> gaté par la capacité réelle du pod (`held < max`), une task queue dédiée à
> `acquire_environment` pour que ce gate n'affame pas les autres activités du worker hôte, et une
> boucle d'attente dans `TemporalPlacement.acquire` qui reclassifie le `schedule_to_start` de
> l'acquire (saturation = file d'attente, attendre) de celui d'une `env_queue` sticky (pod
> mort/fencé = reprovisionner). Le fence git-CAS reste la seule autorité de correction ; tout ceci
> n'est que préférence de placement. Séparation stricte : la harness fournit les primitives
> (supplier, constante de queue, boucle acquire), l'utilisateur câble le worker et son tuner.

# Plan -- capacity-aware acquire placement (gated slot supplier + dedicated acquire queue)

Target: `pydantic_ai_harness/durable/{_lease.py,_temporal_slots.py,_temporal_placement.py,_plugin.py,temporal.py}`.

## Statut (2026-07-21) : DONE (non commité -- branche `feat/env-capacity-and-incremental-restore`)

## Step 0 -- the gate cannot be per-activity (blocker, found while planning)

Vérifié dans le `temporalio` installé (1.27.1, `temporalio/worker/_tuning.py`) : `SlotReserveContext`
(lignes 78-96) porte `slot_type`/`task_queue`/`worker_identity`/`is_sticky` mais **pas
`activity_type`** -- ce nom n'apparaît que plus tard, dans `SlotMarkUsedContext`. Un gate à
`reserve_slot` est donc **par type de slot**, pas par activité : sur le worker hôte de
l'utilisateur, gater les slots d'activité affamerait *toutes* les activités qui y tournent, pas
seulement `acquire_environment`. Deux conséquences, toutes deux structurantes :

1. Le worker gaté doit servir **uniquement** `acquire_environment` -- d'où une task queue dédiée
   avec son propre `Worker`, dont le supplier d'activité est le gate. L'utilisateur câble ce
   worker ; la harness fournit la classe du supplier et la constante de nom de queue.
2. `release_environment` ne doit **jamais** être enregistrée sur la queue gatée : un pod plein qui
   ne pourrait plus exécuter de release ferait un deadlock (personne ne libère de capacité,
   personne ne polle pour la libérer). `release_environment` reste sur le worker hôte
   (enregistrement `SimplePlugin` existant, inchangé).

Confirmé aussi dans le SDK installé : `Worker(tuner=...)` est **mutuellement exclusif** avec
`max_concurrent_activities=`/`max_concurrent_workflow_tasks=`/etc (`ValueError` si les deux sont
passés, `_worker.py`) -- le worker d'acquire dédié doit passer `tuner` seul. Et
`WorkerTuner.create_composite` exige les **quatre** suppliers (workflow/activity/local-activity/nexus).

## Step 1 -- observable capacity on `EnvironmentActivities` (`_lease.py`)

Le supplier gate sur `_held` de ce process -- la même instance `EnvironmentActivities` que le
plugin partage déjà entre le worker hôte et le worker sticky, donc le gate reflète la vraie
capacité du pod :

```python
def __init__(self, ...) -> None:
    ...
    self._held: dict[str, HeldEnv] = {}
    self._capacity_available = asyncio.Event()
    self._capacity_available.set()  # empty `_held` == not at capacity

@property
def at_capacity(self) -> bool:
    """Whether this worker already holds `max_concurrent_environments` leases."""
    return len(self._held) >= self._max_concurrent_environments

async def wait_for_capacity(self) -> None:
    """Block until this worker could hold one more environment. Cancellation-safe."""
    await self._capacity_available.wait()

def _sync_capacity_event(self) -> None:
    if self.at_capacity:
        self._capacity_available.clear()
    else:
        self._capacity_available.set()
```

Transitions : `_sync_capacity_event()` appelé après chaque mutation de `_held` -- insertion en fin
d'`acquire_environment`, suppression dans `release_environment`, et dans le chemin de
re-acquisition stale. Le gate `ApplicationError('env worker at capacity')` reste en place comme
backstop de course (deux pollers, un slot libre) : le supplier le rend quasi inatteignable, pas
inutile.

## Step 2 -- `CapacityGatedSlotSupplier` (nouveau `_temporal_slots.py`)

API vérifiée contre le `temporalio` installé (1.27.1) : `reserve_slot` est async et peut bloquer
(seule `asyncio.CancelledError` est une exception acceptable), `try_reserve_slot` ne doit jamais
bloquer, `mark_slot_used`/`release_slot` sont synchrones.

```python
"""Capacity-gated activity slot supplier for a dedicated `acquire_environment` worker."""

from __future__ import annotations

import asyncio

from temporalio.worker import (
    CustomSlotSupplier,
    SlotMarkUsedContext,
    SlotPermit,
    SlotReleaseContext,
    SlotReserveContext,
)

from pydantic_ai_harness.durable._lease import EnvironmentActivities


class CapacityGatedSlotSupplier(CustomSlotSupplier):
    """Activity slot supplier that stops polling while the env worker is at capacity.

    Wire it as the `activity_supplier` of a `Worker` that serves *only* `acquire_environment`
    on `ACQUIRE_TASK_QUEUE`. While the worker holds `max_concurrent_environments` leases,
    `reserve_slot` blocks, so the worker stops polling: an acquire task then stays visibly
    pending on the server (a usable autoscaling signal) instead of being accepted then bounced.

    The gate can only apply per slot *type*, not per activity (`SlotReserveContext` carries no
    activity name) -- a worker using this supplier must serve `acquire_environment` alone;
    `release_environment` must stay on a different, ungated worker.
    """

    def __init__(self, activities: EnvironmentActivities, *, num_slots: int) -> None:
        self._activities = activities
        self._slots = asyncio.Semaphore(num_slots)

    async def reserve_slot(self, ctx: SlotReserveContext) -> SlotPermit:
        await self._activities.wait_for_capacity()
        await self._slots.acquire()
        return SlotPermit()

    def try_reserve_slot(self, ctx: SlotReserveContext) -> SlotPermit | None:
        return None  # no eager dispatch: this worker runs no workflows

    def mark_slot_used(self, ctx: SlotMarkUsedContext) -> None:
        pass

    def release_slot(self, ctx: SlotReleaseContext) -> None:
        self._slots.release()
```

## Step 3 -- acquire retry loop + queue constant (`_temporal_placement.py`)

Aujourd'hui `acquire` laisse remonter tout schedule-to-start brut, et `is_placement_failure`
retourne `True` pour n'importe lequel. Sous le gate, une flotte saturée signifie que personne ne
polle la queue d'acquire -- ce timeout est de la mise en file, pas un échec. L'absorber à
l'intérieur d'`acquire` fait que les seuls schedule-to-start atteignant `is_placement_failure` sont
ceux des tool-calls routés sur une `env_queue` sticky -- qui signifient toujours "pod mort/fencé",
toujours `True`, toujours déclencheurs de `_reacquire` :

```python
ACQUIRE_TASK_QUEUE = 'durable-env-acquire'
"""Default task queue for the dedicated `acquire_environment` worker (see
`CapacityGatedSlotSupplier`). A constant, not config, so workflows and workers agree without
sharing configuration."""

_ACQUIRE_RETRY_INITIAL_INTERVAL = timedelta(seconds=1)
_ACQUIRE_RETRY_MAX_INTERVAL = timedelta(minutes=1)

async def acquire(self, *, failed_queue: str | None) -> EnvironmentLease:
    env_id = workflow.info().workflow_id
    params = AcquireEnvParams(env_id=env_id, failed_queue=failed_queue)
    backoff = _ACQUIRE_RETRY_INITIAL_INTERVAL
    while True:
        try:
            return await workflow.execute_activity(
                'acquire_environment', params, result_type=EnvironmentLease,
                task_queue=self._acquire_task_queue(env_id),
                schedule_to_start_timeout=_ACQUIRE_SCHEDULE_TO_START_TIMEOUT,
                start_to_close_timeout=_ACQUIRE_START_TO_CLOSE_TIMEOUT,
            )
        except ActivityError as exc:
            if not self.is_placement_failure(exc):
                raise
            workflow.logger.info('acquire_environment not scheduled; env fleet at capacity, retrying')
            await asyncio.sleep(backoff.total_seconds())
            backoff = min(backoff * 2, _ACQUIRE_RETRY_MAX_INTERVAL)
```

(`asyncio.sleep` dans un workflow est patché par temporalio et durable.) `is_placement_failure`
elle-même est inchangée -- son *contrat* se resserre (documenté dans son docstring) : la boucle de
reprovision de `_capability.py` et la boucle d'`acquire` lui passent des entrées disjointes, donc
les timeouts d'acquire n'échappent jamais à leur propre boucle.

## Step 4 -- expose the shared activities (`_plugin.py`, `temporal.py`)

L'utilisateur a besoin de l'instance `EnvironmentActivities` du plugin pour câbler le worker
d'acquire dans le **même process** (le gate lit le `_held` de ce pod précis) :

```python
@property
def environment_activities(self) -> EnvironmentActivities:
    """The shared activities instance -- pass to `CapacityGatedSlotSupplier` and register
    `environment_activities.acquire_environment` on your dedicated acquire-queue `Worker`."""
    return self._activities
```

(Nommée `environment_activities`, pas `activities` : `SimplePlugin.__init__` assigne déjà
`self.activities` à la liste d'activités enregistrées -- une property `activities` en lecture
seule entrerait en collision avec cette assignation, `AttributeError: property 'activities' ...
has no setter`, découvert en lançant la suite de tests complète.)

Câblage utilisateur complet, documenté dans le docstring de `DurableEnvironmentPlugin` :

```python
from temporalio.worker import FixedSizeSlotSupplier, Worker, WorkerTuner
from pydantic_ai_harness.durable.temporal import (
    ACQUIRE_TASK_QUEUE, CapacityGatedSlotSupplier, DurableEnvironmentPlugin,
)

env_plugin = DurableEnvironmentPlugin([agent], workspaces_base=Path('/workspaces'))

# Same process as the host worker -- the gate reads this pod's own held leases.
acquire_worker = Worker(
    client,
    task_queue=ACQUIRE_TASK_QUEUE,
    activities=[env_plugin.environment_activities.acquire_environment],
    tuner=WorkerTuner.create_composite(
        workflow_supplier=FixedSizeSlotSupplier(2),
        activity_supplier=CapacityGatedSlotSupplier(env_plugin.environment_activities, num_slots=8),
        local_activity_supplier=FixedSizeSlotSupplier(2),
        nexus_supplier=FixedSizeSlotSupplier(2),
    ),
)
# ... et workflow-side: TemporalPlacement(host_task_queue=ACQUIRE_TASK_QUEUE)
```

`acquire_environment` reste enregistrée sur le worker hôte aussi (`SimplePlugin`, inchangé) : un
utilisateur qui ne câble pas la queue dédiée garde le comportement actuel (plus la boucle de
retry), donc c'est opt-in, non-breaking. `CapacityGatedSlotSupplier` et `ACQUIRE_TASK_QUEUE`
exportés depuis `durable/temporal.py` (`__all__`).

## Step 5 -- tests

`tests/durable/test_temporal_slots.py` (nouveau, épinglé `anyio_backend='asyncio'`) :

- `reserve_slot` résout immédiatement sous la capacité ; bloque une fois `at_capacity` vrai.
- Le `reserve_slot` bloqué résout après qu'un `release_environment` libère un slot.
- Cap `num_slots` : N réservations concurrentes résolvent, la N+1ᵉ attend, un `release_slot` la
  débloque.
- `try_reserve_slot` retourne `None` ; `mark_slot_used` est un no-op.
- Annulation d'un `reserve_slot` bloqué : propage `CancelledError` sans fuir de slot (un
  `reserve_slot` suivant résout toujours).

`tests/durable/test_lease.py::TestCapacitySignal` : `at_capacity`/`wait_for_capacity` set/clear à
chaque transition (provision jusqu'au max → bloqué ; release → débloqué ; re-acquisition stale →
débloqué).

`tests/durable/test_temporal_placement.py::TestAcquire` : retry sur schedule-to-start puis succès
(compte les tentatives, vérifie le sleep) ; pas de retry sur une autre erreur ; `is_placement_failure`
inchangé pour les autres appelants.

## Gates

`make lint && make typecheck && make test` (per this repo's own `CLAUDE.md`) before considering
this issue done. Vérifié : 148 tests passent, 100% couverture de branches sur `_lease.py`,
`_temporal_slots.py`, `_temporal_placement.py`.
