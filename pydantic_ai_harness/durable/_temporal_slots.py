"""Capacity-gated activity slot supplier for a dedicated `acquire_environment` worker.

Requires `temporalio` (imported here, not gated -- this module is only reachable through
`pydantic_ai_harness.durable.temporal`, which already gates the import).
"""

from __future__ import annotations

import asyncio

from temporalio.worker import (
    CustomSlotSupplier,
    SlotMarkUsedContext,
    SlotPermit,
    SlotReleaseContext,
    SlotReserveContext,
)

from pydantic_ai_harness.durable._lease import EnvironmentActivities


class CapacityGatedSlotSupplier(CustomSlotSupplier):
    """Activity slot supplier that stops polling while the env worker is at capacity.

    Wire it as the `activity_supplier` of a `Worker` that serves *only* `acquire_environment`
    on `ACQUIRE_TASK_QUEUE` (see `TemporalPlacement`), in the same process as the
    `DurableEnvironmentPlugin` whose `EnvironmentActivities` it reads. While that worker holds
    `max_concurrent_environments` leases, `reserve_slot` blocks, so the worker stops polling: an
    acquire task then stays visibly pending on the server (a usable autoscaling signal) instead of
    being accepted and bounced with `env worker at capacity` by a pod that would reject it anyway.

    The gate can only be applied per slot *type*, not per activity: `SlotReserveContext` carries no
    activity name, so a worker using this supplier must serve `acquire_environment` alone --
    `release_environment` in particular must stay on a different (ungated) worker, or a full pod
    could never free a slot.
    """

    def __init__(self, activities: EnvironmentActivities, *, num_slots: int) -> None:
        self._activities = activities
        self._slots = asyncio.Semaphore(num_slots)

    async def reserve_slot(self, ctx: SlotReserveContext) -> SlotPermit:
        await self._activities.wait_for_capacity()
        await self._slots.acquire()
        return SlotPermit()

    def try_reserve_slot(self, ctx: SlotReserveContext) -> SlotPermit | None:
        # Never reserve eagerly: eager activity slots are only used for activities returned by a
        # workflow task on this same worker, and this worker runs no workflows -- so a conservative
        # `None` costs nothing and keeps the capacity gate the single admission point.
        return None

    def mark_slot_used(self, ctx: SlotMarkUsedContext) -> None:
        pass

    def release_slot(self, ctx: SlotReleaseContext) -> None:
        self._slots.release()
