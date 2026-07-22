<!-- checkout_from: issues/03a_durable_environment_snapshot_store/ -->
> `_lease.py`/`_capability.py` importent `_store.py` (03a) — dépendance d'import propre.

# Plan -- lease + capability engine-neutre (03b)

Module cible : `pydantic_ai_harness/durable/`. Dépend de 03a (imports de `_store.py`
par `_lease.py`/`_capability.py` -- dépendance unidirectionnelle propre, aucune
co-modification de fichier). Testable sans Temporal réel et sans le seam de routage
#4977. C'est la PR pivot du découpage : la garder digeste en s'appuyant sur les
protocoles de 02 et le store de 03a.

## Étape 1 -- lease et activités (`73cf809`, `6b3ecfd`)

`durable/_lease.py` :

1. `EnvironmentLease` (modèle Pydantic, importé de 03a `_store.py`).
2. `acquire_environment` : step 0 re-acquire idempotent -- registre local
   `env_id -> workspace` du worker, **validation `store.is_current(env_id, head_local)`**
   (un pod fencé se croit encore titulaire ; head périmé => purge locale et chemin
   complet), puis retour de la lease existante **sans re-fence** ; get-or-provision ;
   busy bounce (`ApplicationError(retryable)` si `len(active) >=
   max_concurrent_environments`) ; sinon `fence` + restore + retour lease.
3. `release_environment(env_id)` : snapshot final + suppression locale + **effacement
   du lease record du store** (sinon le get-or-provision d'un workflow ultérieur
   convergerait vers une lease libérée) + cleanup de l'entrée locks du root.
4. Janitor : tâche asyncio du worker, TTL sur `last_used` des workspaces locaux.
5. Gate `temporalio` (`6b3ecfd`) : le `try: import temporalio` vit directement dans
   `_lease.py` -- le module reste importable sans `temporalio` installé. **Pas de
   module `temporal.py` public dans cette PR** (déplacé en 03c, voir périmètre).

État process du worker (dataclass, pas `BaseModel` : purement local, ne traverse
aucune frontière) :

```python
@dataclass
class HeldEnv:
    lease: EnvironmentLease
    workspace: Path
    head: str            # sha du head connu -- la donnée de fencing

_HELD: dict[str, HeldEnv] = {}
```

Note get-or-provision (sémantique exacte) :

1. `get_lease(env_id)` retourne la lease enregistrée par le dernier `fence`.
2. Si elle existe, est vivante et **n'est pas** `params.failed_queue` : la retourner
   telle quelle (pas de fence) -- l'appelant converge sur le holder existant. C'est le
   chemin du second re-acquire concurrent (parent ou sub-agent) après une panne.
3. Sinon : fence + provision (chemin nominal).

Test dédié : panne du pod avec parent + child workflow actifs => les deux
re-acquièrent en concurrence => **une seule** nouvelle lease, zéro fence ping-pong
(compter les fence commits).

## Étape 2 -- la capability (`8ef5d78`)

`durable/_capability.py` -- `DurableEnvironment(AbstractCapability)` :

1. `for_agent()` **ne mute pas les toolsets** (contrat non-mutant de #4977). Split :
   côté workflow, la capability ne gère que lease + transport ; côté worker,
   store/policy sont configurés au démarrage via `configure_durability` sur les
   toolsets, et la résolution du root se fait dans le toolset lui-même via
   `ctx.metadata['durable_env']['env_id']` (défaut du protocole `EnvironmentBound`,
   fallback root statique sans lease).
2. Wrapper toolset (côté workflow) : au premier call env-bound de la run, lease
   mémoïsée derrière un `asyncio.Future` partagé, écrite dans
   `ctx.metadata['durable_env']`. **Pas de release en fin de run** : le workspace doit
   survivre aux runs successives d'un même workflow ; release au scope workflow,
   janitor TTL en filet.
3. `env_id` : `workflow.info().workflow_id` si dans un workflow, sinon `ctx.run_id`
   (chemin local = no-op de toute façon).
4. Hors workflow : la capability ne fait rien.

```python
class _DurableEnvWrapper(WrapperToolset[AgentDepsT]):
    async def call_tool(self, name, tool_args, ctx, tool):
        md = tool.tool_def.metadata
        if not md or not md.get('env_bound'):
            return await super().call_tool(name, tool_args, ctx, tool)
        lease = await self._lease(ctx)                       # asyncio.Future mémoïsé par run
        ctx.metadata['durable_env'] = lease.model_dump()
        return await super().call_tool(name, tool_args, ctx, tool)
```

(Le catch `ActivityError` / boucle de re-provision vit dans le driver Temporal, 03c.)

## Étape 3 -- driver `EnvironmentPlacement` (`e7c8b94`)

`durable/_placement.py` : interface engine-neutre que la capability consomme. Tout le
spécifique Temporal (`TemporalPlacement`) vit en 03c ; ici le contrat seul, pour que la
capability reste testable avec un driver fake.

## Étape 4 -- wiring `configure_durability` (`a447692`)

Points d'intégration dans `filesystem/_toolset.py`, `shell/_toolset.py`,
`code_mode/_toolset.py` : configuration worker-side (store/policy) exposée par les
toolsets env-bound, jamais par mutation dans `for_agent()`.

## Étape 5 -- exports et tests

- `durable/__init__.py` : exports engine-neutres (`DurableEnvironment`,
  `EnvironmentPlacement`, `EnvironmentLease`, store de 03a). Aucun symbole Temporal.
- `tests/durable/test_lease.py` : acquire/release, re-acquire idempotent, busy bounce,
  get-or-provision (pas de ping-pong), janitor.
- `tests/durable/test_capability.py` : lease mémoïsée écrite dans `ctx.metadata`,
  no-op hors workflow, `env_id` fallback `run_id`, non-mutation des toolsets.
- `tests/durable/test_journal.py` : ré-exécution d'op mutante avec même
  `run_id:tool_call_id` => pas de double application.

## Périmètre -- corrections par rapport au découpage initial

- **`run_env_worker` (`eef2db8`) n'est pas un livrable de 03b.** Le design final est
  passé directement à `DurableEnvironmentPlugin` (03c) ; `eef2db8` est une étape
  intermédiaire historique, à fondre/réécrire dans le récit de 03b sans livrable
  propre, ou à omettre du périmètre PR. Le livrable de 03b est directement les
  activités lease utilisables, sans worker helper dédié.
- **`durable/temporal.py` est exclu de 03b** malgré `6b3ecfd` : il réexporte des
  symboles de 03c (`TemporalPlacement`, `DurableEnvironmentPlugin`,
  `TemporalBranchDelegation`) en plus de `EnvironmentActivities`. Rangé entièrement en
  03c. Le gate `import temporalio` de 03b reste dans `_lease.py`.

Commits livrés : `73cf809`, `6b3ecfd`, `8ef5d78`, `a447692`, `e7c8b94` (+ `eef2db8`
comme étape historique sans livrable final).
