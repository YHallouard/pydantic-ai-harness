"""Shared primitives for environment-bound toolsets under durable execution.

Not a capability itself -- `FileSystem`, `Shell`, and `CodeMode` implement
`EnvironmentBound` and use `guarded_mutating`/`OpJournal` internally.
`DurableEnvironment` (sub-issue 3) builds on these: it owns the lease
lifecycle and wires a `SnapshotStore` into the toolsets via
`configure_durability`.
"""

from pydantic_ai_harness.durable._journal import (
    MAX_RESULT,
    JournalEntry,
    JournalSkipped,
    OpJournal,
    RecordedResult,
    guarded_mutating,
)
from pydantic_ai_harness.durable._protocol import (
    EnvironmentBound,
    RootDirSource,
    env_bound_metadata,
)
from pydantic_ai_harness.durable._store import (
    AcquireEnvParams,
    EnvironmentLease,
    FenceConflict,
    GitSnapshotStore,
    Head,
    LeaseRecord,
    SnapshotPolicy,
    SnapshotRejected,
    SnapshotStore,
)

__all__ = [
    'MAX_RESULT',
    'AcquireEnvParams',
    'EnvironmentBound',
    'EnvironmentLease',
    'FenceConflict',
    'GitSnapshotStore',
    'Head',
    'JournalEntry',
    'JournalSkipped',
    'LeaseRecord',
    'OpJournal',
    'RecordedResult',
    'RootDirSource',
    'SnapshotPolicy',
    'SnapshotRejected',
    'SnapshotStore',
    'env_bound_metadata',
    'guarded_mutating',
]
