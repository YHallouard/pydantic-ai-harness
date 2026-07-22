# Callable `host_task_queue` for sharded acquire queues

> Repo: **pydantic-ai-harness** -- suggested labels: `enhancement`, `performance`, `durable`,
> `temporal`, `speculative`
> Part of: [Tracking] Durable execution for environment-bound capabilities (placement/capacity
> follow-up)
> Depends on: `issues/09_capacity_aware_acquire_placement`

## Statut (2026-07-21) : DONE (non commité -- branche `feat/env-capacity-and-incremental-restore`),
mais **ne pas merger avant mesure** -- voir "Non-goal".

Implémenté : `TemporalPlacement.host_task_queue: str | Callable[[str], str] | None`,
`_acquire_task_queue(env_id)`. Tests : `tests/durable/test_temporal_placement.py::TestAcquire`
(callable, déterminisme entre retries).

## Goal

Permettre de sharder la queue d'acquire (issue 09) sur plusieurs workers indépendamment gatés,
pour le cas où une seule `ACQUIRE_TASK_QUEUE` devient un goulot de head-of-line blocking à très
grande échelle (beaucoup d'environnements concurrents, un seul worker d'acquire gaté ne suffit
plus à absorber le débit).

## Problem

Avec l'issue 09, tous les `acquire_environment` d'une flotte convergent sur une seule
`ACQUIRE_TASK_QUEUE`. Un seul `CapacityGatedSlotSupplier`/`Worker` gaté peut devenir un point de
contention si le nombre d'acquisitions concurrentes dépasse ce qu'un seul worker peut traiter en
série (chaque acceptation reste un `fence`+`restore`, quelques secondes).

## Proposal

Élargir `host_task_queue` pour accepter un callable déterministe `env_id -> queue` en plus d'un
`str` :

```python
host_task_queue: str | Callable[[str], str] | None = None
```

`acquire` résout la queue via un helper interne appelé à chaque tentative de la boucle de retry
(le résultat doit être stable pour un `env_id` donné entre les tentatives -- déterminisme requis
par le modèle de workflow Temporal). Exemple d'usage :
`lambda env_id: f'{ACQUIRE_TASK_QUEUE}-{hash(env_id) % 4}'`. Le câblage des workers shardés (un
`Worker` par shard, chacun avec son propre `CapacityGatedSlotSupplier` puisque `_held` est
process-local) reste entièrement côté utilisateur.

## Non-goal

**Merger avant d'avoir mesuré un vrai goulot.** Cette issue est volontairement petite (une dizaine
de lignes) et écrite maintenant pour ne pas bloquer plus tard, mais elle ajoute de la généralité
spéculative : rien à ce jour ne démontre qu'une seule `ACQUIRE_TASK_QUEUE` sature en pratique. À ne
merger qu'après une mesure de charge réelle montrant qu'un seul worker d'acquire gaté est le
goulot (et pas, par exemple, `fence`/`restore` eux-mêmes, qui resteraient le vrai plafond même avec
plusieurs queues).

## Related

`issues/09_capacity_aware_acquire_placement` (prérequis direct, même fichier
`_temporal_placement.py`).
