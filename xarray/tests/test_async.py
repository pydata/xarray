import asyncio
from typing import Literal, TypeVar
from unittest.mock import patch

import pytest

import xarray as xr
import xarray.testing as xrt
from xarray.tests import has_zarr_v3, requires_zarr_v3
from xarray.tests.test_dataset import create_test_data

if has_zarr_v3:
    import zarr
    from zarr.abc.store import ByteRequest, Store
    from zarr.core.buffer import Buffer, BufferPrototype
    from zarr.storage import MemoryStore
    from zarr.storage._wrapper import WrapperStore

    T_Store = TypeVar("T_Store", bound=Store)

    class ReadOnlyStore(WrapperStore[T_Store]):
        """
        We shouldn't need this - but we currently do just as a way around https://github.com/zarr-developers/zarr-python/issues/3105#issuecomment-2990367167

        Works the same way as the zarr LoggingStore.
        """

        read_only = True

        def __init__(
            self,
            store: T_Store,
        ) -> None:
            super().__init__(store)

        async def get(
            self,
            key: str,
            prototype: BufferPrototype,
            byte_range: ByteRequest | None = None,
        ) -> Buffer | None:
            return await self._store.get(
                key=key, prototype=prototype, byte_range=byte_range
            )

else:
    ReadOnlyStore = {}


@pytest.fixture
def memorystore() -> "MemoryStore":
    memorystore = zarr.storage.MemoryStore({})

    ds = create_test_data()
    ds.to_zarr(memorystore, zarr_format=3, consolidated=False)

    return memorystore


@pytest.fixture
def store(memorystore) -> "zarr.abc.store.Store":
    # TODO we shouldn't this Store at all for the patched tests, but we currently use it just as a way around https://github.com/zarr-developers/zarr-python/issues/3105#issuecomment-2990367167
    return ReadOnlyStore(memorystore)


def get_xr_obj(
    store: "zarr.abc.store.Store", cls_name: Literal["Variable", "DataArray", "Dataset"]
):
    ds = xr.open_zarr(store, zarr_format=3, consolidated=False, chunks=None)

    match cls_name:
        case "Variable":
            return ds["var1"].variable
        case "DataArray":
            return ds["var1"]
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

    @pytest.mark.parametrize("cls_name", ["Variable", "DataArray", "Dataset"])
    async def test_concurrent_load_multiple_objects(self, store, cls_name) -> None:
        N_OBJECTS = 5

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

    @pytest.mark.parametrize("cls_name", ["Variable", "DataArray", "Dataset"])
    @pytest.mark.parametrize(
        "indexer, method, zarr_class_and_method",
        [
            ({}, "sel", (zarr.AsyncArray, "getitem")),
            ({}, "isel", (zarr.AsyncArray, "getitem")),
            ({"dim2": 1.0}, "sel", (zarr.AsyncArray, "getitem")),
            ({"dim2": 2}, "isel", (zarr.AsyncArray, "getitem")),
            ({"dim2": slice(1.0, 3.0)}, "sel", (zarr.AsyncArray, "getitem")),
            ({"dim2": slice(1, 3)}, "isel", (zarr.AsyncArray, "getitem")),
            ({"dim2": [1.0, 3.0]}, "sel", (zarr.core.indexing.AsyncOIndex, "getitem")),
            ({"dim2": [1, 3]}, "isel", (zarr.core.indexing.AsyncOIndex, "getitem")),
            (
                {
                    "dim1": xr.Variable(data=[2, 3], dims="points"),
                    "dim2": xr.Variable(data=[1.0, 2.0], dims="points"),
                },
                "sel",
                (zarr.core.indexing.AsyncVIndex, "getitem"),
            ),
            (
                {
                    "dim1": xr.Variable(data=[2, 3], dims="points"),
                    "dim2": xr.Variable(data=[1, 3], dims="points"),
                },
                "isel",
                (zarr.core.indexing.AsyncVIndex, "getitem"),
            ),
        ],
        ids=[
            "no-indexing-sel",
            "no-indexing-isel",
            "basic-int-sel",
            "basic-int-isel",
            "basic-slice-sel",
            "basic-slice-isel",
            "outer-sel",
            "outer-isel",
            "vectorized-sel",
            "vectorized-isel",
        ],
    )
    async def test_indexing(
        self,
        store,
        cls_name,
        method,
        indexer,
        zarr_class_and_method,
    ) -> None:
        if cls_name == "Variable" and method == "sel":
            pytest.skip("Variable doesn't have a .sel method")

        # each type of indexing ends up calling a different zarr indexing method
        target_class, method_name = zarr_class_and_method
        original_method = getattr(target_class, method_name)

        with patch.object(
            target_class, method_name, wraps=original_method, autospec=True
        ) as mocked_meth:
            xr_obj = get_xr_obj(store, cls_name)

            # TODO we're not actually testing that these indexing methods are not blocking...
            result = await getattr(xr_obj, method)(**indexer).load_async()

            mocked_meth.assert_called()
            mocked_meth.assert_awaited()
            assert mocked_meth.call_count > 0

        expected = getattr(xr_obj, method)(**indexer).load()
        xrt.assert_identical(result, expected)
