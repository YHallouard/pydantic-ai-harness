<!-- checkout_from: issues/02_environment_bound_protocol/ -->
> `33b6f73` modifie `durable/__init__.py` et `durable/_protocol.py`, tous deux créés par 02
> (pas par 03a) — voir la section « Vérification de soumissibilité » plus bas pour le détail
> vérifié par test réel.

# Plan -- `GitSnapshotStore` (03a)

Module cible : `pydantic_ai_harness/durable/` (créé par la sous-issue 2). Autonome
vis-à-vis de Temporal : aucune dépendance, aucun import `temporalio`. Livrable = PR 1
du découpage de l'issue 03, prérequis de 03b (qui l'importe depuis
`_lease.py`/`_capability.py`) et 03c.

> **Vérification de soumissibilité (2026-07-20, testée par cherry-pick réel dans un
> worktree jetable, pas une supposition) :** "autonome vis-à-vis de Temporal" ne veut
> **pas dire soumissible seule**. `upstream/main` (`pydantic/pydantic-ai-harness`)
> n'a **aucun** paquet `durable/` -- ni l'issue 01 (`353ade5`, ids stables + root
> dynamique) ni l'issue 02 (`aa8f512`/`013bc7e`, protocole `EnvironmentBound` +
> `_protocol.py`) n'y sont mergées. `git cherry-pick 33b6f73` sur `upstream/main` seul
> échoue immédiatement (`durable/__init__.py` et `durable/_protocol.py` "deleted in
> HEAD"). **Chaîne de prérequis réelle pour soumettre 03a : 01 → 02 → 03a**, aucun des
> trois ne dépendant de #4977.
>
> Pire : **l'issue 01 elle-même ne s'applique plus proprement sur `upstream/main`
> actuel** (`upstream/main` a 15 commits d'avance sur notre `main` local). Cherry-pick
> testé : conflits de contenu réels (pas juste fichiers manquants) sur
> `pydantic_ai_harness/code_mode/_toolset.py`, `code_mode/README.md`,
> `tests/filesystem/test_filesystem.py`. Un rebase manuel de 01 contre `upstream/main`
> est nécessaire avant que quoi que ce soit dans cette chaîne soit prêt à pousser.
> Voir `docs/durable-execution-chantier.md` §6 pour la procédure complète.

## Étape 1 -- contrats typés du store

`durable/_store.py` -- tous les objets qui traversent une frontière sont des modèles
Pydantic (`BaseModel`), pas des dicts ni des dataclasses nues :

```python
class EnvironmentLease(BaseModel):
    """Contrat public -- model_dump() doit satisfaire le TypedDict
    `DurableEnvironmentLease` que pydantic-ai core lit dans ctx.metadata['durable_env']."""
    env_id: str
    env_queue: str
    epoch: int

class Head(BaseModel):
    sha: str
    epoch: int

class LeaseRecord(BaseModel):
    env_id: str
    env_queue: str
    epoch: int
    fenced_at: datetime

    def to_lease(self) -> EnvironmentLease: ...

class SnapshotPolicy(BaseModel):
    mode: Literal['per_op', 'per_step', 'content_hash'] = 'per_op'
    # extensible (debounce per_step, etc.) sans casser la signature

class AcquireEnvParams(BaseModel):
    env_id: str
    failed_queue: str | None = None
```

Ce `SnapshotPolicy` remplace le placeholder de la sous-issue 2 (même import path
`pydantic_ai_harness.durable`).

## Étape 2 -- protocole et implémentation git

```python
class SnapshotStore(Protocol):
    async def fence(self, env_id: str, *, queue: str) -> Head   # fence commit + enregistre la lease courante
    async def get_lease(self, env_id: str) -> LeaseRecord | None  # lease courante (get-or-provision)
    async def is_current(self, env_id: str, head: str) -> bool
    async def push(self, env_id: str, workspace: Path) -> None  # non-ff rejeté par le bare repo
    async def restore(self, env_id: str, into: Path) -> None    # depuis le head
    async def fork(self, parent_env_id: str, child_env_id: str) -> None        # pour sub-issue 6
```

Modèle de repo : **un bare repo par env racine** (le workflow), une **branche par env**
(`main` pour le parent, `{tool_call_id}` pour les enfants) -- `fork` = `git branch`
exige le même repo. Le `LeaseRecord` (env_queue + fenced_at) vit dans le même store.

- `GitSnapshotStore(base_path)` : `receive.denyNonFastforwards=true`. **Le CAS est
  natif git** : `fence(env_id)` = commit vide « fence » poussé sur la branche ; un pod
  fencé a un head local en retard, son prochain push est un non-fast-forward rejeté
  par git lui-même. Le champ `epoch` devient un compteur informatif (logs/tests), plus
  le mécanisme critique.
- `S3SnapshotStore` : tarball + objet `head` avec préconditions (`If-Match` ETag) pour
  le CAS.

## Étape 3 -- tests

`tests/durable/test_store.py` :

- deux `fence` concurrents : un seul gagne ;
- push d'un pod fencé rejeté (non-fast-forward) ;
- round-trip push/restore ;
- fork ;
- lease record écrit par `fence` et lu par `get_lease`.

## Étape 4 -- restore incrémental (warm path, replié depuis l'ancienne issue 08)

Replié le 2026-07-22 après ré-examen du découpage : `restore` était totalement destructeur
(`discard_workspace` systématique, puis `git init` + fetch complet + checkout), même quand le git
dir sibling `.{name}.git` contenait déjà tous les objets. `restore` gagne un chemin warm :

- `_is_warm(workspace, repo_dir)` : vrai si le workspace et son git dir sibling existent et que
  l'`origin` du sibling pointe exactement sur le bare repo de cet env (sinon reste d'un autre env
  ou état corrompu -- cold path inchangé) ;
- chemin warm : `fetch` du delta seulement, puis `checkout --force -B branch FETCH_HEAD` +
  `clean -fdx`. Le bare repo reste l'autorité : tout état local non poussé est écrasé, exactement
  comme le `rmtree` du cold path le jetait déjà. Le fence git-CAS reste la seule autorité de
  correction, quel que soit le chemin de restauration.

Tests : `tests/durable/test_store.py::TestWarmRestore` (warm converge vers un head poussé ailleurs,
untracked nettoyé, modif non commitée écrasée, cold fallback si git dir absent ou `origin`
étranger, `current is None` -> workspace vide).

Change **additif** sur `_store.py`, replié dans la PR de 03a à sa préparation (même patron que
`925b6b0` et les fixups `b77a61f`/`ccd6dd7`). La contrepartie côté lease (chemin de re-acquisition
stale qui cesse de jeter le workspace, `durable/_lease.py`) est en 03b : elle n'a de sens qu'avec ce
warm path.

## Hors périmètre

- `925b6b0` (git dir hors du workspace) touche `_store.py` mais est rangé en 03c : fix
  additif appliqué par-dessus cette base, pas un ajout à 03a.
- Aucune activité Temporal, aucune capability : c'est 03b. Aucun plugin worker : c'est
  03c.

Commit livré : `33b6f73`.
