from typing import TypeVar, Iterable
import asyncio
import time

import pytest
import numpy as np

from xarray.tests import has_zarr_v3, requires_zarr_v3
import xarray as xr


if has_zarr_v3:
    import zarr
    from zarr.abc.store import Store
    from zarr.storage import MemoryStore
    from zarr.storage._wrapper import WrapperStore

    from zarr.abc.store import ByteRequest
    from zarr.core.buffer import Buffer, BufferPrototype

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
            return await self._store.get(key=key, prototype=prototype, byte_range=byte_range)
        
        async def get_partial_values(
            self,
            prototype: BufferPrototype,
            key_ranges: Iterable[tuple[str, ByteRequest | None]],
        ) -> list[Buffer | None]:
            await asyncio.sleep(self.latency)
            return await self._store.get_partial_values(prototype=prototype, key_ranges=key_ranges)
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
        dimension_names=["x", "y"]
    )
    z[:, :] = np.random.random((10, 10))

    return memorystore


@requires_zarr_v3
@pytest.mark.asyncio
async def test_async_load(memorystore):
    N_DATASETS = 10
    LATENCY = 1.0

    latencystore = LatencyStore(memorystore, latency=LATENCY)
    datasets = [xr.open_zarr(latencystore, zarr_format=3, consolidated=False, chunks=None) for _ in range(N_DATASETS)]

    # TODO add async load to Dataset and DataArray as well as to Variable
    start_time = time.time()
    tasks = [ds['foo'].variable.async_load() for ds in datasets]
    results = await asyncio.gather(*tasks)
    #results = [ds['foo'].variable.load() for ds in datasets]
    total_time = time.time() - start_time
    
    assert total_time > LATENCY  # Cannot possibly be quicker than this
    assert total_time < LATENCY * N_DATASETS # If this isn't true we're gaining nothing from async
    assert abs(total_time - LATENCY) < 0.5  # Should take approximately LATENCY seconds, but allow some buffer

    print(total_time)
    assert False