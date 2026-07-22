# Capacity-aware acquire placement (gated slot supplier + dedicated acquire queue)

> Repo: **pydantic-ai-harness** -- suggested labels: `enhancement`, `performance`, `durable`,
> `temporal`
> Part of: [Tracking] Durable execution for environment-bound capabilities (placement/capacity
> follow-up)
> Depends on: le warm restore replié dans 03a/03b (2026-07-22, ancienne issue 08) -- il réduit `t_a`,
> le coût que ce gating rend visible ; et 03c (le placement Temporal d'origine que cette issue étend).

## Statut (2026-07-21) : DONE (non commité -- branche `feat/env-capacity-and-incremental-restore`)

Implémenté et testé (148 tests unitaires, 100% couverture de branches sur les fichiers touchés) :

- `EnvironmentActivities.at_capacity`/`wait_for_capacity` (`durable/_lease.py`) : capacité
  observable, synchronisée à chaque mutation de `_held`.
- `CapacityGatedSlotSupplier` (nouveau `durable/_temporal_slots.py`) : `CustomSlotSupplier` qui
  bloque `reserve_slot` tant que le pod est plein.
- `TemporalPlacement.acquire` (`durable/_temporal_placement.py`) : boucle de retry qui absorbe le
  schedule-to-start de l'acquire (backoff exponentiel plafonné 1 min) au lieu de le laisser
  atteindre `is_placement_failure` ; constante `ACQUIRE_TASK_QUEUE`.
- `DurableEnvironmentPlugin.environment_activities` (`durable/_plugin.py`) : expose l'instance
  partagée pour le câblage user-side, docstring avec l'exemple de wiring complet.
- Exports publics : `ACQUIRE_TASK_QUEUE`, `CapacityGatedSlotSupplier` dans `durable/temporal.py`.
- Tests : `tests/durable/test_temporal_slots.py` (nouveau), `tests/durable/test_lease.py::TestCapacitySignal`,
  `tests/durable/test_temporal_placement.py::TestAcquire` (retry cases).

## Goal

Faire cesser le "bounce" d'acquisition à saturation : aujourd'hui, un pod plein continue de happer
les tâches `acquire_environment`, les accepte, puis les rejette avec une erreur retryable. Sous
charge (analyse quantitative préalable, voir `docs/durable-environment-isolation.md` §3 dans le
repo `cse` hôte), ce mécanisme dégrade en proba `~t_b/(N·t_a)` de tomber sur un worker libre
(régime backlog), donnant un time-to-place de l'ordre de la dizaine de minutes à plusieurs heures
selon la taille de la flotte, et un signal d'autoscaling structurellement aveugle sous backoff (la
tentative en attente est sur un timer serveur, pas dans le backlog dispatchable).

## Problem

Deux défauts couplés :

1. **Pas de gating par capacité.** `acquire_environment` (`durable/_lease.py`) ne fait qu'un check
   local *après* avoir été dispatché et démarré : `if self.at_capacity: raise
   ApplicationError('env worker at capacity', non_retryable=False)`. Un pod plein continue de
   poller, d'accepter, puis de rejeter -- un bounce qui coûte un aller-retour au store git
   (`get_lease`) pour rien, et qui, à l'échelle, "hogge" les grabs Temporal parce qu'un pod plein
   revient poller bien plus vite (~0,2s) qu'un pod qui accepte réellement (~2-5s de fence+restore).
2. **`schedule_to_start` confond saturation et mort.** `TemporalPlacement.is_placement_failure`
   traite tout timeout de schedule-to-start comme "pod mort/fencé" -> reprovision ×3
   (`_MAX_REPROVISIONS`, `_capability.py`) -> échec du workflow. Appliqué à l'acquire, ce même
   timeout signifie en réalité "aucun worker n'a de capacité libre pour l'instant" -- une
   information de file d'attente normale, pas un placement failure.

## Proposal

Trois changements couplés, un seul comportement correct -- voir plan.md Step 0 pour le blocker qui
les rend inséparables (le gate côté SDK Temporal ne peut être appliqué que par *type* de slot, pas
par activité, donc le gate exige une queue dédiée qui ne sert que l'acquire) :

1. **Capacité observable** (`EnvironmentActivities.at_capacity`/`wait_for_capacity`) : un
   `asyncio.Event` synchronisé à chaque mutation de `_held`, lisible sans activité Temporal.
2. **`CapacityGatedSlotSupplier`** (`CustomSlotSupplier`) : gate `reserve_slot` sur
   `wait_for_capacity()`. Wiré sur un `Worker` dédié qui ne sert *que* `acquire_environment` sur
   `ACQUIRE_TASK_QUEUE` -- jamais `release_environment`, qui doit rester joignable pour qu'un pod
   plein puisse libérer de la capacité.
3. **Reclassification structurelle** : `TemporalPlacement.acquire` absorbe désormais le
   schedule-to-start de l'acquire dans sa propre boucle de retry (backoff exponentiel plafonné à
   1 min) -- ce timeout n'atteint plus jamais `is_placement_failure`/la boucle de reprovision.
   Seuls les timeouts de tool-calls routés sur une `env_queue` sticky (pod mort/fencé) gardent leur
   sens original.

   **Pourquoi cette pièce reste dans 09 et n'est pas repliée dans 03c** (tranché le 2026-07-22, après
   avoir envisagé le contraire) : en base 03c (sans gating) un pod plein **continue de poller** la
   queue d'acquire -- il happe, puis bounce avec une `ApplicationError` retryable (start-to-close,
   pas schedule-to-start). Un schedule-to-start sur l'acquire n'y survient donc jamais par
   saturation, seulement par flotte réellement morte/mal-configurée -- et l'échec propagé (base
   actuelle : `_ACQUIRE_SCHEDULE_TO_START_TIMEOUT = 10 s` remonté hors du `try` de reprovision de
   `_capability.py`) est le bon comportement. La boucle de retry ne devient correcte -- et
   nécessaire -- qu'une fois le gating en place, quand les pods pleins cessent de poller et que la
   saturation *devient* un schedule-to-start normal. La mettre dans la base 03c serait au contraire
   nuisible : elle masquerait une vraie panne de flotte en attendant indéfiniment au lieu d'échouer.
   La reclassification est donc couplée au gating, pas un fix de correctness indépendant de 03c.

Le câblage (worker dédié, tuner) reste **entièrement côté utilisateur** -- la harness fournit
`CapacityGatedSlotSupplier`, `ACQUIRE_TASK_QUEUE`, et
`DurableEnvironmentPlugin.environment_activities` pour le brancher ; rien n'est activé par défaut
(`acquire_environment` continue de tourner sur le worker hôte comme avant, comportement inchangé
pour qui ne câble pas le worker dédié).

## Non-goal

Un scheduler de placement centralisé, un registre de capacité partagé, ou tout mécanisme qui
ferait porter à la harness une décision d'infra (combien de pods, où ils tournent, comment ils
scalent). Le fence git-CAS reste la seule autorité de correction -- tout ce qui précède n'est
qu'une préférence d'admission, jamais une garantie de correction.

## Open questions

- **Attente d'acquire bornée ou infinie** : sous le gate, la saturation devient l'état normal --
  choix actuel : backoff infini plafonné à 1 min, avec l'attente que l'opérateur surveille le
  backlog de `ACQUIRE_TASK_QUEUE` (rendu visible justement par le gate). Alternative :
  `max_acquire_wait: timedelta | None` sur `TemporalPlacement`, échouant après une borne -- change
  le contrat (un worker d'acquire réellement mort ferait attendre indéfiniment au lieu d'échouer en
  10s). Pas tranché ; à valider avec les mainteneurs/l'utilisateur avant une éventuelle PR upstream.
- **Nom `ACQUIRE_TASK_QUEUE = 'durable-env-acquire'`** : constante arbitraire, à valider.
- **API du supplier** : `CapacityGatedSlotSupplier(activities: EnvironmentActivities, *,
  num_slots)` prend la classe concrète plutôt qu'un protocole `Callable[[], bool]` découplé --
  choix KISS (une seule implémentation existe aujourd'hui), à reconsidérer si un second besoin de
  gating apparaît.

## Related

03a/03b (warm restore replié, ancienne issue 08, qui réduit `t_a`), `issues/10_sharded_acquire_queue`
(extension optionnelle), `issues/03c_durable_environment_temporal_integration` (le placement
Temporal d'origine que cette issue étend -- la reclassification du schedule-to-start de l'acquire y a
été envisagée puis gardée ici, voir Proposal point 3).
