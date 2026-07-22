<!-- checkout_from: issues/09_capacity_aware_acquire_placement/ -->
> Suite optionnelle de 09, à ne merger qu'après mesure : une seule queue d'acquire met tous les
> acquirers en file derrière le head-of-line. Permettre de sharder par env_id
> (`durable-env-acquire-{i}`) coûte quelques lignes côté `TemporalPlacement`.

# Plan -- callable `host_task_queue` for sharded acquire queues

Target: `pydantic_ai_harness/durable/_temporal_placement.py`.

## Statut (2026-07-21) : DONE (non commité -- branche `feat/env-capacity-and-incremental-restore`),
ne pas merger avant mesure (voir issue.md).

## Step 1 -- widen `host_task_queue`

```python
host_task_queue: str | Callable[[str], str] | None = None
"""... A callable receives the env_id (the workflow id) and returns the queue -- e.g.
`lambda env_id: f'{ACQUIRE_TASK_QUEUE}-{hash(env_id) % 4}'` to spread acquire load over 4
sharded, independently gated acquire workers once a single `ACQUIRE_TASK_QUEUE` becomes a
head-of-line bottleneck. The callable must be deterministic (same env_id -> same queue on every
call): `acquire` calls it again on each retry, and Temporal requires workflow code to be
deterministic. Each shard needs its own `CapacityGatedSlotSupplier`, since `_held` -- what the
gate reads -- is process-local; wire one `Worker` per shard."""

def _acquire_task_queue(self, env_id: str) -> str | None:
    return self.host_task_queue(env_id) if callable(self.host_task_queue) else self.host_task_queue
```

`acquire` appelle `self._acquire_task_queue(env_id)` à l'intérieur de la boucle de retry (le
résultat est déterministe pour un `env_id` donné, donc stable entre tentatives -- workflow-safe).
Le câblage des workers shardés reste côté utilisateur (un `Worker` gaté par queue, avec son propre
`CapacityGatedSlotSupplier` puisque `_held` est process-local -- pas de partage de compteur entre
shards).

## Step 2 -- tests

`tests/durable/test_temporal_placement.py::TestAcquire` :

- Callable invoqué avec l'`env_id` du workflow ; résultat utilisé comme `task_queue`.
- Le callable est appelé de façon identique entre deux tentatives de la boucle de retry
  (déterminisme requis par Temporal) -- vérifié via `call_args_list` sur un mock.
- `str` plain se comporte exactement comme avant (non-régression).

## Gates

`make lint && make typecheck && make test`. Vérifié : tests ajoutés passent, 100% couverture de
branches sur `_temporal_placement.py` maintenue.
