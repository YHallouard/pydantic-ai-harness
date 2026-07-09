"""Shared primitives for environment-bound toolsets under durable execution.

Not a capability itself -- `FileSystem`, `Shell`, and `CodeMode` implement
`EnvironmentBound` and use `guarded_mutating`/`OpJournal` internally. The
`DurableEnvironment` capability (sub-issue 3) will build on these.
"""

from pydantic_ai_harness.durable._journal import MAX_RESULT, JournalSkipped, OpJournal, RecordedResult, guarded_mutating
from pydantic_ai_harness.durable._protocol import (
    EnvironmentBound,
    RootDirSource,
    SnapshotPolicy,
    SnapshotStore,
    env_bound_metadata,
)

__all__ = [
    'MAX_RESULT',
    'EnvironmentBound',
    'JournalSkipped',
    'OpJournal',
    'RecordedResult',
    'RootDirSource',
    'SnapshotPolicy',
    'SnapshotStore',
    'env_bound_metadata',
    'guarded_mutating',
]
