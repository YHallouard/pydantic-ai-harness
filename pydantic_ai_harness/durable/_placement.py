"""Engine placement contract for the `DurableEnvironment` capability.

`DurableEnvironment` itself is engine-neutral: it decides *when* a lease is
needed (first env-bound tool call in a run), memoizes it, and re-acquires it
when the placement target dies. Everything an execution engine actually does
-- detecting the durable context, running the acquire step durably, deciding
what "the target is gone" looks like, and executing a tool call on the target
-- is behind this protocol, injected at construction:

```python
DurableEnvironment(placement=TemporalPlacement(), store=GitSnapshotStore(...))
```

`TemporalPlacement` (`pydantic_ai_harness.durable.temporal`) is the reference
implementation. An engine without a placement problem (e.g. DBOS, where tools
run in the same process as the workflow) only needs `acquire` to restore the
workspace and can make `route_call` a plain `fallback()` delegation.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from pydantic_ai import AbstractToolset
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import ToolsetTool

from pydantic_ai_harness.durable._store import EnvironmentLease


class EnvironmentPlacement(Protocol):
    """How one durable-execution engine leases workspaces and places env-bound tool calls on them."""

    def active(self) -> bool:
        """Whether the calling code is currently running under this engine's durable context.

        `False` makes `DurableEnvironment` a no-op for the call: env-bound
        tools run against whatever static root they were constructed with,
        so the same agent works unchanged in a local, non-durable run.
        """
        ...  # pragma: no cover -- Protocol method body, never executed

    async def acquire(self, *, failed_queue: str | None) -> EnvironmentLease:
        """Acquire (or converge on) this run's environment lease, durably.

        The implementation derives the environment id from its own execution
        context (for Temporal, the workflow id). `failed_queue` names the
        placement target the caller just saw fail, so the acquirer fences a
        fresh claim instead of converging back on the dead holder; `None` on
        first acquisition.
        """
        ...  # pragma: no cover -- Protocol method body, never executed

    def is_placement_failure(self, exc: Exception) -> bool:
        """Whether `exc` means the lease's placement target is gone (pod dead or fenced out).

        `True` triggers a re-acquire and retry in `DurableEnvironment` (up to
        its re-provision budget); any other exception propagates untouched.
        """
        ...  # pragma: no cover -- Protocol method body, never executed

    async def route_call(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
        *,
        lease: EnvironmentLease,
        wrapped: AbstractToolset[Any],
        fallback: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Execute the tool call on `lease`'s placement target (`lease.env_queue`).

        `wrapped` is the toolset tree under the `DurableEnvironment` wrapper,
        for implementations that need to locate the engine-specific execution
        seam for `name` inside it. `fallback` performs the un-routed call
        (plain delegation to the wrapped toolset); implementations use it when
        they have nothing to route (no placement problem, or the engine's own
        routing already reads `ctx.metadata['durable_env']`).
        """
        ...  # pragma: no cover -- Protocol method body, never executed
