import asyncio
from importlib import import_module
from typing import Any, Literal
from unittest.mock import patch

import pytest

import xarray as xr
import xarray.testing as xrt
from xarray.tests import (
    has_zarr,
    has_zarr_v3,
    has_zarr_v3_async_oindex,
    requires_zarr,
    requires_zarr_v3,
)
from xarray.tests.test_backends import ZARR_FORMATS
from xarray.tests.test_dataset import create_test_data

if has_zarr:
    import zarr
else:
    zarr = None  # type: ignore[assignment]


@pytest.fixture(scope="module", params=ZARR_FORMATS)
def store(request) -> "zarr.storage.MemoryStore":
    memorystore = zarr.storage.MemoryStore({})

    ds = create_test_data()
    print(ds)
    ds.to_zarr(memorystore, zarr_format=request.param, consolidated=False)  # type: ignore[call-overload]

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
    @requires_zarr_v3
    async def test_concurrent_load_multiple_variables(self, store) -> None:
        target_class = zarr.AsyncArray
        method_name = "getitem"
        original_method = getattr(target_class, method_name)

        # the indexed coordinate variables is not lazy, so the create_test_dataset has 4 lazy variables in total
        N_LAZY_VARS = 4

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

    @requires_zarr_v3
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

    @requires_zarr_v3
    @pytest.mark.parametrize("cls_name", ["Variable", "DataArray", "Dataset"])
    @pytest.mark.parametrize(
        "indexer, method, target_zarr_class",
        [
            ({}, "sel", "zarr.AsyncArray"),
            ({}, "isel", "zarr.AsyncArray"),
            ({"dim2": 1.0}, "sel", "zarr.AsyncArray"),
            ({"dim2": 2}, "isel", "zarr.AsyncArray"),
            ({"dim2": slice(1.0, 3.0)}, "sel", "zarr.AsyncArray"),
            ({"dim2": slice(1, 3)}, "isel", "zarr.AsyncArray"),
            (
                {"dim2": [1.0, 3.0]},
                "sel",
                "zarr.core.indexing.AsyncOIndex",
            ),
            ({"dim2": [1, 3]}, "isel", "zarr.core.indexing.AsyncOIndex"),
            (
                {
                    "dim1": xr.Variable(data=[2, 3], dims="points"),
                    "dim2": xr.Variable(data=[1.0, 2.0], dims="points"),
                },
                "sel",
                "zarr.core.indexing.AsyncVIndex",
            ),
            (
                {
                    "dim1": xr.Variable(data=[2, 3], dims="points"),
                    "dim2": xr.Variable(data=[1, 3], dims="points"),
                },
                "isel",
                "zarr.core.indexing.AsyncVIndex",
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
        target_zarr_class,
    ) -> None:
        if not has_zarr_v3_async_oindex and target_zarr_class in (
            "zarr.core.indexing.AsyncOIndex",
            "zarr.core.indexing.AsyncVIndex",
        ):
            pytest.skip(
                "current version of zarr does not support orthogonal or vectorized async indexing"
            )

        if cls_name == "Variable" and method == "sel":
            pytest.skip("Variable doesn't have a .sel method")

        # Each type of indexing ends up calling a different zarr indexing method
        # They all use a method named .getitem, but on a different internal zarr class
        target_class = _resolve_class_from_string(target_zarr_class)
        method_name = "getitem"
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
    @pytest.mark.parametrize(
        ("indexer", "expected_err_msg"),
        [
            pytest.param(
                {"dim2": 2},
                "basic async indexing",
                marks=pytest.mark.skipif(
                    has_zarr_v3,
                    reason="current version of zarr has basic async indexing",
                ),
            ),  # tests basic indexing
            pytest.param(
                {"dim2": [1, 3]},
                "orthogonal async indexing",
                marks=pytest.mark.skipif(
                    has_zarr_v3_async_oindex,
                    reason="current version of zarr has async orthogonal indexing",
                ),
            ),  # tests oindexing
            pytest.param(
                {
                    "dim1": xr.Variable(data=[2, 3], dims="points"),
                    "dim2": xr.Variable(data=[1, 3], dims="points"),
                },
                "vectorized async indexing",
                marks=pytest.mark.skipif(
                    has_zarr_v3_async_oindex,
                    reason="current version of zarr has async vectorized indexing",
                ),
            ),  # tests vindexing
        ],
    )
    async def test_raise_on_older_zarr_version(self, store, indexer, expected_err_msg):
        """Test that trying to use async load with insufficiently new version of zarr raises a clear error"""

        ds = xr.open_zarr(store, consolidated=False, chunks=None)

        with pytest.raises(NotImplementedError, match=expected_err_msg):
            await ds.isel(**indexer).load_async()
