import asyncio
import time
from collections.abc import Iterable
from contextlib import asynccontextmanager
from typing import TypeVar

import numpy as np
import pytest

import xarray as xr
import xarray.testing as xrt
from xarray.tests import has_zarr_v3, requires_zarr_v3

if has_zarr_v3:
    import zarr
    from zarr.abc.store import ByteRequest, Store
    from zarr.core.buffer import Buffer, BufferPrototype
    from zarr.storage import MemoryStore
    from zarr.storage._wrapper import WrapperStore

    T_Store = TypeVar("T_Store", bound=Store)

    class LatencyStore(WrapperStore[T_Store]):
        """Works the same way as the zarr LoggingStore"""

        latency: float

        def __init__(
            self,
            store: T_Store,
            latency: float = 0.0,
        ) -> None:
            """
            Store wrapper that adds artificial latency to each get call.

            Parameters
            ----------
            store : Store
                Store to wrap
            latency : float
                Amount of artificial latency to add to each get call, in seconds.
            """
            super().__init__(store)
            self.latency = latency

        def __str__(self) -> str:
            return f"latency-{self._store}"

        def __repr__(self) -> str:
            return f"LatencyStore({self._store.__class__.__name__}, '{self._store}', latency={self.latency})"

        async def get(
            self,
            key: str,
            prototype: BufferPrototype,
            byte_range: ByteRequest | None = None,
        ) -> Buffer | None:
            await asyncio.sleep(self.latency)
            return await self._store.get(
                key=key, prototype=prototype, byte_range=byte_range
            )

        async def get_partial_values(
            self,
            prototype: BufferPrototype,
            key_ranges: Iterable[tuple[str, ByteRequest | None]],
        ) -> list[Buffer | None]:
            await asyncio.sleep(self.latency)
            return await self._store.get_partial_values(
                prototype=prototype, key_ranges=key_ranges
            )
else:
    LatencyStore = {}


@pytest.fixture
def memorystore() -> "MemoryStore":
    memorystore = zarr.storage.MemoryStore({})
    z = zarr.create_array(
        store=memorystore,
        name="foo",
        shape=(10, 10),
        chunks=(5, 5),
        dtype="f4",
        dimension_names=["x", "y"],
    )
    z[:, :] = np.random.random((10, 10))

    z = zarr.create_array(
        store=memorystore,
        name="bar",
        shape=(10,),
        chunks=(5),
        dtype="f4",
        dimension_names=["x"],
    )
    z[:] = np.random.random((10,))

    return memorystore


class AsyncTimer:
    """Context manager for timing async operations and making assertions about their execution time."""

    start_time: float
    end_time: float
    total_time: float

    @asynccontextmanager
    async def measure(self):
        """Measure the execution time of the async code within this context."""
        self.start_time = time.time()
        try:
            yield self
        finally:
            self.end_time = time.time()
            self.total_time = self.end_time - self.start_time


@requires_zarr_v3
@pytest.mark.asyncio
class TestAsyncLoad:
    N_LOADS: int = 5
    LATENCY: float = 1.0

    @pytest.fixture(params=["ds", "da", "var"])
    def xr_obj(self, request, memorystore) -> xr.Dataset | xr.DataArray | xr.Variable:
        latencystore = LatencyStore(memorystore, latency=self.LATENCY)
        ds = xr.open_zarr(latencystore, zarr_format=3, consolidated=False, chunks=None)

        match request.param:
            case "var":
                return ds["foo"].variable
            case "da":
                return ds["foo"]
            case "ds":
                return ds

    def assert_time_as_expected(self, total_time: float) -> None:
        assert total_time > self.LATENCY  # Cannot possibly be quicker than this
        assert (
            total_time < self.LATENCY * self.N_LOADS
        )  # If this isn't true we're gaining nothing from async
        assert (
            abs(total_time - self.LATENCY) < 2.0
        )  # Should take approximately LATENCY seconds, but allow some buffer

    async def test_async_load(self, xr_obj):
        async with AsyncTimer().measure() as timer:
            tasks = [xr_obj.load_async() for _ in range(self.N_LOADS)]
            results = await asyncio.gather(*tasks)

        for result in results:
            xrt.assert_identical(result, xr_obj.load())

        self.assert_time_as_expected(timer.total_time)
