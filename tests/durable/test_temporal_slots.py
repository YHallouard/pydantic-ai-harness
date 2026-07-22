"""Tests for CapacityGatedSlotSupplier -- gates acquire polling on real held capacity."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pydantic_ai_harness.durable import AcquireEnvParams, GitSnapshotStore
from pydantic_ai_harness.durable._lease import EnvironmentActivities
from pydantic_ai_harness.durable.temporal import CapacityGatedSlotSupplier

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'  # the slot supplier and capacity event are asyncio-only (Temporal worker side)


def _activities(tmp_path: Path, *, max_concurrent_environments: int = 1) -> EnvironmentActivities:
    store = GitSnapshotStore(tmp_path / 'store')
    return EnvironmentActivities(
        store=store,
        env_queue='env-q1',
        workspaces_base=tmp_path / 'ws',
        max_concurrent_environments=max_concurrent_environments,
    )


class TestCapacityGate:
    async def test_reserves_immediately_below_capacity(self, tmp_path: Path) -> None:
        supplier = CapacityGatedSlotSupplier(_activities(tmp_path, max_concurrent_environments=4), num_slots=4)
        permit = await asyncio.wait_for(supplier.reserve_slot(MagicMock()), timeout=1)
        assert permit is not None

    async def test_blocks_at_capacity_and_unblocks_on_release(self, tmp_path: Path) -> None:
        acts = _activities(tmp_path, max_concurrent_environments=1)
        supplier = CapacityGatedSlotSupplier(acts, num_slots=4)
        await acts.acquire_environment(AcquireEnvParams(env_id='env-1'))
        assert acts.at_capacity

        reserve = asyncio.ensure_future(supplier.reserve_slot(MagicMock()))
        await asyncio.sleep(0.05)
        assert not reserve.done()  # gated while the worker is full

        await acts.release_environment('env-1')
        permit = await asyncio.wait_for(reserve, timeout=1)
        assert permit is not None

    async def test_num_slots_caps_concurrent_reservations(self, tmp_path: Path) -> None:
        supplier = CapacityGatedSlotSupplier(_activities(tmp_path, max_concurrent_environments=10), num_slots=1)
        first = await asyncio.wait_for(supplier.reserve_slot(MagicMock()), timeout=1)
        second = asyncio.ensure_future(supplier.reserve_slot(MagicMock()))
        await asyncio.sleep(0.05)
        assert not second.done()  # only one slot, first still held

        supplier.release_slot(MagicMock())
        assert await asyncio.wait_for(second, timeout=1) is not None
        assert first is not None

    async def test_try_reserve_returns_none_and_mark_used_is_a_noop(self, tmp_path: Path) -> None:
        supplier = CapacityGatedSlotSupplier(_activities(tmp_path, max_concurrent_environments=4), num_slots=4)
        assert supplier.try_reserve_slot(MagicMock()) is None
        assert supplier.mark_slot_used(MagicMock()) is None

    async def test_cancelled_reserve_does_not_leak_a_slot(self, tmp_path: Path) -> None:
        acts = _activities(tmp_path, max_concurrent_environments=1)
        supplier = CapacityGatedSlotSupplier(acts, num_slots=1)
        await acts.acquire_environment(AcquireEnvParams(env_id='env-1'))

        blocked = asyncio.ensure_future(supplier.reserve_slot(MagicMock()))
        await asyncio.sleep(0.05)
        blocked.cancel()
        with pytest.raises(asyncio.CancelledError):
            await blocked

        # After freeing capacity a fresh reserve still resolves: the cancelled one took no slot.
        await acts.release_environment('env-1')
        permit = await asyncio.wait_for(supplier.reserve_slot(MagicMock()), timeout=1)
        assert permit is not None
