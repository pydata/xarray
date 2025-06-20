import asyncio
import time
from collections.abc import Iterable
from contextlib import asynccontextmanager
from typing import Literal, TypeVar
from unittest.mock import patch

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
        attributes={"add_offset": 1, "scale_factor": 2},
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


@pytest.fixture
def store(memorystore) -> "zarr.abc.store.Store":
    # TODO we shouldn't need a LatencyStore at all for the patched tests, but we currently use it just as a way around https://github.com/zarr-developers/zarr-python/issues/3105#issuecomment-2990367167
    return LatencyStore(memorystore, latency=0.0)


def get_xr_obj(
    store: "zarr.abc.store.Store", cls_name: Literal["Variable", "DataArray", "Dataset"]
):
    ds = xr.open_zarr(store, zarr_format=3, consolidated=False, chunks=None)

    match cls_name:
        case "Variable":
            return ds["foo"].variable
        case "DataArray":
            return ds["foo"]
        case "Dataset":
            return ds


@requires_zarr_v3
@pytest.mark.asyncio
class TestAsyncLoad:
    async def test_concurrent_load_multiple_variables(self, store) -> None:
        target_class = zarr.AsyncArray
        method_name = "getitem"
        original_method = getattr(target_class, method_name)

        # TODO up the number of variables in the dataset?
        # the coordinate variable is not lazy
        N_LAZY_VARS = 1

        with patch.object(
            target_class, method_name, wraps=original_method, autospec=True
        ) as mocked_meth:
            # blocks upon loading the coordinate variables here
            ds = xr.open_zarr(store, zarr_format=3, consolidated=False, chunks=None)

            # TODO we're not actually testing that these indexing methods are not blocking...
            result_ds = await ds.load_async()

            mocked_meth.assert_called()
            assert mocked_meth.call_count >= N_LAZY_VARS
            mocked_meth.assert_awaited()

        xrt.assert_identical(result_ds, ds.load())

    # TODO apply this parametrization to the other test too?
    @pytest.mark.parametrize("cls_name", ["Variable", "DataArray", "Dataset"])
    async def test_concurrent_load_multiple_objects(self, store, cls_name) -> None:
        N_OBJECTS = 5

        # factor this mocking out of all tests as a fixture?
        target_class = zarr.AsyncArray
        method_name = "getitem"
        original_method = getattr(target_class, method_name)

        with patch.object(
            target_class, method_name, wraps=original_method, autospec=True
        ) as mocked_meth:
            xr_obj = get_xr_obj(store, cls_name)

            # TODO we're not actually testing that these indexing methods are not blocking...
            coros = [xr_obj.load_async() for _ in range(N_OBJECTS)]
            results = await asyncio.gather(*coros)

            mocked_meth.assert_called()
            assert mocked_meth.call_count >= N_OBJECTS
            mocked_meth.assert_awaited()

        for result in results:
            xrt.assert_identical(result, xr_obj.load())

    @pytest.mark.parametrize("method", ["sel", "isel"])
    @pytest.mark.parametrize(
        "indexer, zarr_class_and_method",
        [
            ({}, (zarr.AsyncArray, "getitem")),
            ({"x": 2}, (zarr.AsyncArray, "getitem")),
            ({"x": slice(2, 4)}, (zarr.AsyncArray, "getitem")),
            ({"x": [2, 3]}, (zarr.core.indexing.AsyncOIndex, "getitem")),
            (
                {
                    "x": xr.DataArray([2, 3], dims="points"),
                    "y": xr.DataArray([2, 3], dims="points"),
                },
                (zarr.core.indexing.AsyncVIndex, "getitem"),
            ),
        ],
        ids=["no-indexing", "basic-int", "basic-slice", "outer", "vectorized"],
    )
    async def test_indexing(
        self, store, method, indexer, zarr_class_and_method
    ) -> None:
        # each type of indexing ends up calling a different zarr indexing method
        target_class, method_name = zarr_class_and_method
        original_method = getattr(target_class, method_name)

        with patch.object(
            target_class, method_name, wraps=original_method, autospec=True
        ) as mocked_meth:
            ds = xr.open_zarr(
                store,
                zarr_format=3,
                consolidated=False,
                chunks=None,
            )

            # TODO we're not actually testing that these indexing methods are not blocking...
            result = await getattr(ds, method)(**indexer).load_async()

            mocked_meth.assert_called()
            mocked_meth.assert_awaited()
            assert mocked_meth.call_count > 0

        expected = getattr(ds, method)(**indexer).load()
        xrt.assert_identical(result, expected)
