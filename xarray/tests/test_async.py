import asyncio
from importlib import import_module
from typing import Any, Literal
from unittest.mock import patch

import pytest

import xarray as xr
import xarray.testing as xrt
from xarray.tests import has_zarr, requires_zarr, has_zarr_v3_async_index, requires_zarr_v3_async_index
from xarray.tests.test_dataset import create_test_data
from xarray.tests.test_backends import ZARR_FORMATS


if has_zarr:
    import zarr
else:
    zarr = None


@pytest.fixture(scope="module", params=ZARR_FORMATS)
def store(request) -> "zarr.storage.MemoryStore":
    memorystore = zarr.storage.MemoryStore({})

    ds = create_test_data()
    ds.to_zarr(memorystore, zarr_format=request.param, consolidated=False)

    return memorystore


def get_xr_obj(
    store: "zarr.abc.store.Store", cls_name: Literal["Variable", "DataArray", "Dataset"]
):
    ds = xr.open_zarr(store, consolidated=False, chunks=None)

    match cls_name:
        case "Variable":
            return ds["var1"].variable
        case "DataArray":
            return ds["var1"]
        case "Dataset":
            return ds


def _resolve_class_from_string(class_path: str) -> type[Any]:
    """Resolve a string class path like 'zarr.AsyncArray' to the actual class."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, class_name)


@pytest.mark.asyncio
class TestAsyncLoad:
    @requires_zarr_v3_async_index
    async def test_concurrent_load_multiple_variables(self, store) -> None:
        target_class = _resolve_class_from_string("zarr.AsyncArray")
        method_name = "getitem"
        original_method = getattr(target_class, method_name)

        # TODO up the number of variables in the dataset?
        # the coordinate variable is not lazy
        N_LAZY_VARS = 1

        with patch.object(
            target_class, method_name, wraps=original_method, autospec=True
        ) as mocked_meth:
            # blocks upon loading the coordinate variables here
            ds = xr.open_zarr(store, consolidated=False, chunks=None)

            # TODO we're not actually testing that these indexing methods are not blocking...
            result_ds = await ds.load_async()

            mocked_meth.assert_called()
            assert mocked_meth.call_count >= N_LAZY_VARS
            mocked_meth.assert_awaited()

        xrt.assert_identical(result_ds, ds.load())

    @requires_zarr_v3_async_index
    @pytest.mark.parametrize("cls_name", ["Variable", "DataArray", "Dataset"])
    async def test_concurrent_load_multiple_objects(self, store, cls_name) -> None:
        N_OBJECTS = 5

        target_class = _resolve_class_from_string("zarr.AsyncArray")
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

    @requires_zarr_v3_async_index
    @pytest.mark.parametrize("cls_name", ["Variable", "DataArray", "Dataset"])
    @pytest.mark.parametrize(
        "indexer, method, zarr_class_and_method",
        [
            ({}, "sel", ("zarr.AsyncArray", "getitem")),
            ({}, "isel", ("zarr.AsyncArray", "getitem")),
            ({"dim2": 1.0}, "sel", ("zarr.AsyncArray", "getitem")),
            ({"dim2": 2}, "isel", ("zarr.AsyncArray", "getitem")),
            ({"dim2": slice(1.0, 3.0)}, "sel", ("zarr.AsyncArray", "getitem")),
            ({"dim2": slice(1, 3)}, "isel", ("zarr.AsyncArray", "getitem")),
            (
                {"dim2": [1.0, 3.0]},
                "sel",
                ("zarr.core.indexing.AsyncOIndex", "getitem"),
            ),
            ({"dim2": [1, 3]}, "isel", ("zarr.core.indexing.AsyncOIndex", "getitem")),
            (
                {
                    "dim1": xr.Variable(data=[2, 3], dims="points"),
                    "dim2": xr.Variable(data=[1.0, 2.0], dims="points"),
                },
                "sel",
                ("zarr.core.indexing.AsyncVIndex", "getitem"),
            ),
            (
                {
                    "dim1": xr.Variable(data=[2, 3], dims="points"),
                    "dim2": xr.Variable(data=[1, 3], dims="points"),
                },
                "isel",
                ("zarr.core.indexing.AsyncVIndex", "getitem"),
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
        target_class_path, method_name = zarr_class_and_method
        target_class = _resolve_class_from_string(target_class_path)
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

    @requires_zarr
    @pytest.mark.skipif(has_zarr_v3_async_index, reason="newer version of zarr has async indexing")
    @pytest.mark.parametrize(
        "indexer",
        [
            {"dim2": [1, 3]},  # tests oindexing
                {  # test vindexing
                    "dim1": xr.Variable(data=[2, 3], dims="points"),
                    "dim2": xr.Variable(data=[1, 3], dims="points"),
                },
        ],
    )
    async def test_raise_on_older_zarr_version(self, store, indexer):
        """Test that trying to use async load with insufficiently new version of zarr raises a clear error"""

        ds = xr.open_zarr(store, consolidated=False, chunks=None)

        with pytest.raises(NotImplementedError, match="async indexing"):
            await ds.isel(**indexer).load_async()

    # TODO also test raising informative error if attempting to do basic async indexing with 3.0.0 <= zarr <= 3.1.1?
