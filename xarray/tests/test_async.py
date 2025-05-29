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

        # TODO only have to add this because of dumb behaviour in zarr where it raises with "ValueError: Store is not read-only but mode is 'r'"
        read_only = True

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
    z1 = zarr.create_array(
        store=memorystore,
        name="foo",
        shape=(10, 10),
        chunks=(5, 5),
        dtype="f4",
        dimension_names=["x", "y"],
    )
    z1[:, :] = np.random.random((10, 10))

    z2 = zarr.create_array(
        store=memorystore,
        name="x",
        shape=(10,),
        chunks=(5),
        dtype="f4",
        dimension_names=["x"],
    )
    z2[:] = np.arange(10)

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
    LATENCY: float = 1.0

    @pytest.fixture(params=["var", "ds", "da"])
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

    def assert_time_as_expected(
        self, total_time: float, latency: float, n_loads: int
    ) -> None:
        assert total_time > latency  # Cannot possibly be quicker than this
        assert (
            total_time < latency * n_loads
        )  # If this isn't true we're gaining nothing from async
        assert (
            abs(total_time - latency) < 2.0
        )  # Should take approximately `latency` seconds, but allow some buffer

    async def test_concurrent_load_multiple_variables(self, memorystore) -> None:
        latencystore = LatencyStore(memorystore, latency=self.LATENCY)
        ds = xr.open_zarr(latencystore, zarr_format=3, consolidated=False, chunks=None)

        # TODO up the number of variables in the dataset?
        async with AsyncTimer().measure() as timer:
            result_ds = await ds.load_async()

        xrt.assert_identical(result_ds, ds.load())

        # 2 because there are 2 lazy variables in the dataset
        self.assert_time_as_expected(
            total_time=timer.total_time, latency=self.LATENCY, n_loads=2
        )

    async def test_concurrent_load_multiple_objects(self, xr_obj) -> None:
        N_OBJECTS = 5

        async with AsyncTimer().measure() as timer:
            coros = [xr_obj.load_async() for _ in range(N_OBJECTS)]
            results = await asyncio.gather(*coros)

        for result in results:
            xrt.assert_identical(result, xr_obj.load())

        self.assert_time_as_expected(
            total_time=timer.total_time, latency=self.LATENCY, n_loads=N_OBJECTS
        )

    @pytest.mark.parametrize(
        "method,indexer",
        [
            ("sel", {"x": 2}),
            ("sel", {"x": slice(2, 4)}),
            ("sel", {"x": [2, 3]}),
            (
                "sel",
                {
                    "x": xr.DataArray([2, 3], dims="points"),
                    "y": xr.DataArray([2, 3], dims="points"),
                },
            ),
        ],
        ids=["basic-int", "basic-slice", "outer", "vectorized"],
    )
    async def test_indexing(self, memorystore, method, indexer) -> None:
        # TODO we don't need a LatencyStore for this test
        latencystore = LatencyStore(memorystore, latency=0.0)
        ds = xr.open_zarr(latencystore, zarr_format=3, consolidated=False, chunks=None)

        # TODO we're not actually testing that these indexing methods are not blocking...
        result = await getattr(ds, method)(**indexer).load_async()
        expected = getattr(ds, method)(**indexer).load()
        xrt.assert_identical(result, expected)
