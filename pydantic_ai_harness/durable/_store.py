"""Fenced snapshot store for environment-bound workspaces.

`GitSnapshotStore` is the reference implementation: one bare git repository per
root environment (a workflow), one branch per environment (`main` for the
root, `{tool_call_id}` for a forked sub-agent child -- sub-issue 5). Fencing
and push-rejection are both native git primitives, not something layered on
top:

- `fence` creates a new (empty) commit and lands it with `git update-ref
  <ref> <new> <old>`, which git performs as an atomic compare-and-swap against
  the ref's current value -- a losing racer's `update-ref` fails outright.
- `push` lands a workspace snapshot as a commit and pushes it without
  `--force`; a pod whose local history no longer contains the branch's current
  tip (because it was fenced out) gets git's own non-fast-forward rejection.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

import anyio
from anyio import Lock
from pydantic import BaseModel

_ZERO_SHA = '0' * 40


class EnvironmentLease(BaseModel):
    """Public contract for a held environment.

    `_DurableEnvWrapper` writes `model_dump()` to `ctx.metadata['durable_env']`;
    the engine driver reads it back to place env-bound tool calls on `env_queue`.
    On Temporal, core itself does that placement (pydantic-ai's own
    `resolve_tool_activity_config`, from #4977's `temporal-durability-cap`),
    so `TemporalPlacement.route_call` is a pass-through; the shape here is the
    public contract #4977 reads.
    """

    env_id: str
    env_queue: str
    epoch: int


class Head(BaseModel):
    """The result of a successful `fence`: the new commit and its epoch."""

    sha: str
    epoch: int


class LeaseRecord(BaseModel):
    """The lease as persisted by the store -- `fence` writes it, `get_lease` reads it."""

    env_id: str
    env_queue: str
    epoch: int
    fenced_at: datetime

    def to_lease(self) -> EnvironmentLease:
        return EnvironmentLease(env_id=self.env_id, env_queue=self.env_queue, epoch=self.epoch)


class SnapshotPolicy(BaseModel):
    """When to snapshot a workspace after a mutating operation.

    `per_op` (the default) snapshots after every mutating op -- the only mode
    that preserves "completed in the workflow history implies present in the
    snapshot" unconditionally. `per_step`/`content_hash` are opt-in, cheaper
    modes with a documented divergence window on pod death.
    """

    mode: Literal['per_op', 'per_step', 'content_hash'] = 'per_op'


class AcquireEnvParams(BaseModel):
    """Params for the `acquire_environment` Temporal activity."""

    env_id: str
    failed_queue: str | None = None


class ForkEnvironmentParams(BaseModel):
    """Params for the `fork_environment` Temporal activity."""

    parent_env_id: str
    child_env_id: str


class MergeEnvironmentParams(BaseModel):
    """Params for the `merge_environment` Temporal activity.

    `held_env_id` names the environment whose *held* workspace the merge runs
    in -- the parent for `mode='land'`, the child for `mode='materialize'`.
    `merge_environment` must be routed to whichever sticky queue holds that
    lease, since the merge needs the actual checked-out work-tree, not just
    the bare repo (see `EnvironmentActivities.merge_environment`).
    `other_env_id` is the environment whose branch is fetched and merged in --
    the child for `land`, the parent for `materialize`.
    """

    held_env_id: str
    other_env_id: str
    mode: Literal['land', 'materialize']


class MergeResult(BaseModel):
    """Result of a `SnapshotStore.merge`: the paths left in conflict, if any.

    Empty `conflicts` means the merge was committed and pushed as
    `held_env_id`'s new head. A non-empty `conflicts` with `mode='land'` means
    the merge was aborted (head untouched, nothing pushed); with
    `mode='materialize'` it means the conflict markers were committed and
    pushed as-is.
    """

    conflicts: list[str] = []


class SnapshotStore(Protocol):
    """Persists and restores environment workspace snapshots, with fencing."""

    async def fence(self, env_id: str, *, queue: str) -> Head:
        """Claim `env_id` for `queue`, invalidating any earlier claim's next `push`."""
        ...  # pragma: no cover -- Protocol method body, never executed

    async def get_lease(self, env_id: str) -> LeaseRecord | None:
        """Return the lease recorded by the most recent successful `fence`, if any."""
        ...  # pragma: no cover -- Protocol method body, never executed

    async def release(self, env_id: str) -> None:
        """Erase `env_id`'s lease record (the snapshot itself is untouched).

        Called when a worker gives up an environment it holds, so a later
        `get_lease` returns `None` and the next acquirer fences fresh instead
        of converging on a lease nobody holds anymore.
        """
        ...  # pragma: no cover -- Protocol method body, never executed

    async def is_current(self, env_id: str, head: str) -> bool:
        """Whether `head` is still `env_id`'s current snapshot head (no fence has moved it)."""
        ...  # pragma: no cover -- Protocol method body, never executed

    async def push(self, env_id: str, workspace: Path) -> None:
        """Snapshot `workspace`'s contents as the new head for `env_id`.

        Rejected (implementation-defined error) if `env_id` was fenced since
        `workspace` was last restored/pushed.
        """
        ...  # pragma: no cover -- Protocol method body, never executed

    async def restore(self, env_id: str, into: Path) -> None:
        """Materialize `env_id`'s current snapshot into the (not yet existing) `into` directory."""
        ...  # pragma: no cover -- Protocol method body, never executed

    async def discard_workspace(self, workspace: Path) -> None:
        """Delete a restored `workspace` and any store-private state kept beside it.

        Called when a worker gives up an environment it holds, so a later
        `restore` into the same path starts clean instead of adopting stale
        local state.
        """
        ...  # pragma: no cover -- Protocol method body, never executed

    async def fork(self, parent_env_id: str, child_env_id: str) -> None:
        """Branch `child_env_id` off `parent_env_id`'s current head (sub-issue 5)."""
        ...  # pragma: no cover -- Protocol method body, never executed

    async def merge(self, held_env_id: str, workspace: Path, other_env_id: str, *, keep_conflicts: bool) -> MergeResult:
        """Merge `other_env_id`'s current head into `workspace` (already `restore`d from `held_env_id`).

        A clean merge -- or, with `keep_conflicts=True`, a conflicted one too --
        is committed and pushed as `held_env_id`'s new head via `push`, with
        `other_env_id`'s head as the merge's second parent. With
        `keep_conflicts=False`, a conflict aborts instead: `held_env_id`'s head
        is left untouched and nothing is pushed.
        """
        ...  # pragma: no cover -- Protocol method body, never executed


class SnapshotRejected(Exception):
    """`push` was rejected: `env_id` was fenced (claimed by another queue) since the workspace was last synced."""

    def __init__(self, env_id: str) -> None:
        super().__init__(f'Snapshot push rejected for {env_id!r}: environment was fenced by another owner.')
        self.env_id = env_id


class FenceConflict(Exception):
    """`fence` lost a race against a concurrent `fence` call for the same `env_id`."""

    def __init__(self, env_id: str) -> None:
        super().__init__(f'Fence conflict for {env_id!r}: another caller fenced it first.')
        self.env_id = env_id


async def _run_git(*args: str, cwd: Path | None = None, check: bool = True) -> tuple[int, str, str]:
    result = await anyio.run_process(['git', *args], cwd=cwd, check=False)
    returncode = result.returncode
    stdout = result.stdout.decode().strip()
    stderr = result.stderr.decode().strip()
    if check and returncode != 0:  # pragma: no cover -- defensive against an unexpected git/filesystem failure
        raise RuntimeError(f'git {" ".join(args)} failed ({returncode}): {stderr}')
    return returncode, stdout, stderr


class GitSnapshotStore:
    """`SnapshotStore` backed by a bare git repository per root environment.

    `base_path` should be a directory reachable (as a plain filesystem path)
    from every pod that might hold an environment's lease -- e.g. a shared
    network volume. Git's own ref-locking and non-fast-forward rejection are
    what make fencing safe across concurrent pods sharing that path; nothing
    here adds distributed locking on top.
    """

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path
        self._index_path = base_path / '_branches.json'
        self._index_lock = Lock()

    def _repo_dir(self, root_env_id: str) -> Path:
        digest = hashlib.sha1(root_env_id.encode()).hexdigest()[:16]
        return self._base_path / f'{digest}.git'

    async def _load_index(self) -> dict[str, dict[str, str]]:
        if not self._index_path.exists():
            return {}
        return json.loads(self._index_path.read_text(encoding='utf-8'))

    async def _resolve(self, env_id: str) -> tuple[Path, str]:
        """Resolve `env_id` to (bare repo dir, branch), creating the repo dir for a root env."""
        index = await self._load_index()
        if env_id in index:
            entry = index[env_id]
            return Path(entry['repo']), entry['branch']
        repo_dir = self._repo_dir(env_id)
        if not repo_dir.exists():
            self._base_path.mkdir(parents=True, exist_ok=True)
            await _run_git('init', '--quiet', '--bare', str(repo_dir))
            await _run_git('config', 'receive.denyNonFastforwards', 'true', cwd=repo_dir)
        return repo_dir, 'main'

    async def _record_child(self, child_env_id: str, repo_dir: Path, branch: str) -> None:
        async with self._index_lock:
            index = await self._load_index()
            index[child_env_id] = {'repo': str(repo_dir), 'branch': branch}
            self._index_path.write_text(json.dumps(index), encoding='utf-8')

    def _lease_path(self, repo_dir: Path, branch: str) -> Path:
        return repo_dir / f'lease.{branch}.json'

    async def _current_sha(self, repo_dir: Path, branch: str) -> str | None:
        returncode, stdout, _ = await _run_git(
            '--git-dir', str(repo_dir), 'rev-parse', '--verify', f'refs/heads/{branch}', check=False
        )
        return stdout if returncode == 0 else None

    async def fence(self, env_id: str, *, queue: str) -> Head:
        repo_dir, branch = await self._resolve(env_id)
        old_sha = await self._current_sha(repo_dir, branch)
        if old_sha is None:
            _, tree, _ = await _run_git('hash-object', '-t', 'tree', '/dev/null')
            commit_args = ['commit-tree', tree, '-m', f'fence: {queue}']
        else:
            _, tree, _ = await _run_git('--git-dir', str(repo_dir), 'rev-parse', f'{old_sha}^{{tree}}')
            commit_args = ['commit-tree', tree, '-p', old_sha, '-m', f'fence: {queue}']
        _, new_sha, _ = await _run_git(
            '-c',
            'user.name=durable-environment',
            '-c',
            'user.email=durable-environment@local',
            '--git-dir',
            str(repo_dir),
            *commit_args,
        )
        returncode, _, stderr = await _run_git(
            '--git-dir',
            str(repo_dir),
            'update-ref',
            f'refs/heads/{branch}',
            new_sha,
            old_sha or _ZERO_SHA,
            check=False,
        )
        if returncode != 0:
            raise FenceConflict(env_id) from RuntimeError(stderr)

        previous = await self.get_lease(env_id)
        epoch = (previous.epoch + 1) if previous is not None else 0
        record = LeaseRecord(env_id=env_id, env_queue=queue, epoch=epoch, fenced_at=datetime.now(timezone.utc))
        self._lease_path(repo_dir, branch).write_text(record.model_dump_json(), encoding='utf-8')
        return Head(sha=new_sha, epoch=epoch)

    async def get_lease(self, env_id: str) -> LeaseRecord | None:
        repo_dir, branch = await self._resolve(env_id)
        path = self._lease_path(repo_dir, branch)
        if not path.exists():
            return None
        return LeaseRecord.model_validate_json(path.read_text(encoding='utf-8'))

    async def release(self, env_id: str) -> None:
        repo_dir, branch = await self._resolve(env_id)
        self._lease_path(repo_dir, branch).unlink(missing_ok=True)

    async def is_current(self, env_id: str, head: str) -> bool:
        repo_dir, branch = await self._resolve(env_id)
        return await self._current_sha(repo_dir, branch) == head

    def _work_git_dir(self, workspace: Path) -> Path:
        """The git directory for `workspace`, kept as a sibling so it never lands inside it.

        The workspace is what an agent's shell/filesystem tools operate on; a
        `.git` inside it would be visible to those tools (a stray `rm -rf .git`
        or the agent running its own `git init` could corrupt the snapshot repo).
        Keeping the git dir at `<parent>/.<name>.git` leaves the workspace with
        only the files the agent put there.
        """
        return workspace.parent / f'.{workspace.name}.git'

    async def restore(self, env_id: str, into: Path) -> None:
        repo_dir, branch = await self._resolve(env_id)
        git_dir = self._work_git_dir(into)
        into.mkdir(parents=True, exist_ok=True)
        await _run_git('--git-dir', str(git_dir), '--work-tree', str(into), 'init', '--quiet')
        await _run_git('--git-dir', str(git_dir), '--work-tree', str(into), 'remote', 'add', 'origin', str(repo_dir))
        current = await self._current_sha(repo_dir, branch)
        if current is None:
            return
        await _run_git('--git-dir', str(git_dir), '--work-tree', str(into), 'fetch', '--quiet', 'origin', branch)
        await _run_git(
            '--git-dir', str(git_dir), '--work-tree', str(into), 'checkout', '--quiet', '-B', branch, 'FETCH_HEAD'
        )

    async def push(self, env_id: str, workspace: Path) -> None:
        """Snapshot `workspace` as `env_id`'s new head.

        `workspace` must have been produced by `restore`, which created the
        sibling git dir (`_work_git_dir`) this reuses; pushing an arbitrary
        directory isn't supported.
        """
        _, branch = await self._resolve(env_id)
        git_dir = self._work_git_dir(workspace)
        common = ('--git-dir', str(git_dir), '--work-tree', str(workspace))
        await _run_git(*common, 'add', '-A')
        await _run_git(
            *common,
            '-c',
            'user.name=durable-environment',
            '-c',
            'user.email=durable-environment@local',
            'commit',
            '--quiet',
            '--allow-empty',
            '-m',
            'snapshot',
        )
        returncode, _, stderr = await _run_git(
            *common, 'push', '--quiet', 'origin', f'HEAD:refs/heads/{branch}', check=False
        )
        if returncode != 0:
            raise SnapshotRejected(env_id) from RuntimeError(stderr)

    async def discard_workspace(self, workspace: Path) -> None:
        """Delete `workspace` and its sibling git dir, so a re-provision starts clean.

        Callers that give up a held environment (`EnvironmentActivities`) use this
        instead of a bare `rmtree`, which would leave the `_work_git_dir` sibling
        behind to be re-adopted -- and its lease/objects re-used -- on the next
        `restore` into the same path.
        """
        shutil.rmtree(workspace, ignore_errors=True)
        shutil.rmtree(self._work_git_dir(workspace), ignore_errors=True)

    async def fork(self, parent_env_id: str, child_env_id: str) -> None:
        repo_dir, parent_branch = await self._resolve(parent_env_id)
        parent_sha = await self._current_sha(repo_dir, parent_branch)
        if parent_sha is None:
            raise RuntimeError(f'Cannot fork {parent_env_id!r}: it has no snapshot yet.')
        await _run_git('--git-dir', str(repo_dir), 'update-ref', f'refs/heads/{child_env_id}', parent_sha)
        await self._record_child(child_env_id, repo_dir, child_env_id)

    async def merge(self, held_env_id: str, workspace: Path, other_env_id: str, *, keep_conflicts: bool) -> MergeResult:
        """Merge `other_env_id`'s branch into `workspace`'s work-tree, without committing.

        `workspace` must have been produced by `restore` (like `push`), so
        `origin` already points at the shared bare repo both `held_env_id` and
        `other_env_id` live in. A fast-forward-able merge lands directly, no
        different from any other `push`. A real merge with no conflicts, or a
        conflicted one when `keep_conflicts=True`, is committed by `push` --
        its bare `commit` picks up `MERGE_HEAD` automatically, producing a
        two-parent merge commit (conflict markers included in the conflicted
        case). A conflicted merge with `keep_conflicts=False` is `abort`ed
        instead, leaving `workspace` exactly as it was.
        """
        _, other_branch = await self._resolve(other_env_id)
        git_dir = self._work_git_dir(workspace)
        common = ('--git-dir', str(git_dir), '--work-tree', str(workspace))
        await _run_git(*common, 'fetch', '--quiet', 'origin', other_branch)
        returncode, _, _ = await _run_git(*common, 'merge', '--no-commit', 'FETCH_HEAD', check=False)
        if returncode == 0:
            await self.push(held_env_id, workspace)
            return MergeResult()
        _, conflicts, _ = await _run_git(*common, 'diff', '--name-only', '--diff-filter=U')
        conflict_paths = [path for path in conflicts.splitlines() if path]
        if not keep_conflicts:
            await _run_git(*common, 'merge', '--abort')
            return MergeResult(conflicts=conflict_paths)
        await self.push(held_env_id, workspace)
        return MergeResult(conflicts=conflict_paths)
