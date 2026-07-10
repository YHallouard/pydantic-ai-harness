"""Temporal-specific durable-environment wiring.

Requires the `temporal` optional group. Kept out of `pydantic_ai_harness.durable`'s
own eager imports so that `FileSystem`/`Shell`/`CodeMode` (which depend on that
package for `EnvironmentBound`/`OpJournal`/`GitSnapshotStore`, none of which touch
Temporal) stay importable without `temporalio` installed.
"""

from __future__ import annotations

try:
    import temporalio  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ImportError as _import_error:  # pragma: no cover -- exercised by not having the optional extra installed
    raise ImportError(
        'Please install the `temporalio` package to use DurableEnvironment, '
        'you can use the `temporal` optional group -- `pip install "pydantic-ai-harness[temporal]"`'
    ) from _import_error

from pydantic_ai_harness.durable._capability import DurableEnvironment
from pydantic_ai_harness.durable._lease import EnvironmentActivities, HeldEnv

__all__ = [
    'DurableEnvironment',
    'EnvironmentActivities',
    'HeldEnv',
]
