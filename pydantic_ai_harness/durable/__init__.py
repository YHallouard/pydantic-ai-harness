"""Engine-agnostic primitives for environment-bound toolsets under durable execution.

`FileSystem`, `Shell`, and `CodeMode` implement `EnvironmentBound` and use
`guarded_mutating`/`OpJournal` internally. The `DurableEnvironment` capability
builds on these: it owns the run-side lease lifecycle, delegating everything
engine-specific to an injected `EnvironmentPlacement` driver. The Temporal
driver and worker-side wiring live in `pydantic_ai_harness.durable.temporal`
(requires the `temporal` extra); nothing in this package imports `temporalio`.
"""

from pydantic_ai_harness.durable._capability import DurableEnvironment
from pydantic_ai_harness.durable._journal import (
    MAX_RESULT,
    JournalEntry,
    JournalSkipped,
    OpJournal,
    RecordedResult,
    env_id_from_ctx,
    guarded_mutating,
)
from pydantic_ai_harness.durable._placement import EnvironmentPlacement
from pydantic_ai_harness.durable._protocol import (
    EnvironmentBound,
    RootDirSource,
    env_bound_metadata,
)
from pydantic_ai_harness.durable._store import (
    AcquireEnvParams,
    EnvironmentLease,
    FenceConflict,
    ForkEnvironmentParams,
    GitSnapshotStore,
    Head,
    LeaseRecord,
    MergeEnvironmentParams,
    MergeResult,
    SnapshotPolicy,
    SnapshotRejected,
    SnapshotStore,
)

__all__ = [
    'MAX_RESULT',
    'AcquireEnvParams',
    'DurableEnvironment',
    'EnvironmentBound',
    'EnvironmentLease',
    'EnvironmentPlacement',
    'FenceConflict',
    'ForkEnvironmentParams',
    'GitSnapshotStore',
    'Head',
    'JournalEntry',
    'JournalSkipped',
    'LeaseRecord',
    'MergeEnvironmentParams',
    'MergeResult',
    'OpJournal',
    'RecordedResult',
    'RootDirSource',
    'SnapshotPolicy',
    'SnapshotRejected',
    'SnapshotStore',
    'env_bound_metadata',
    'env_id_from_ctx',
    'guarded_mutating',
]
