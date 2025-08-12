import pytest

import xarray as xr
from xarray.tests import (
    has_zarr,
    has_zarr_v3,
    has_zarr_v3_async_oindex,
    requires_zarr,
)
from xarray.tests.test_backends import ZARR_FORMATS
from xarray.tests.test_dataset import create_test_data

if has_zarr:
    import zarr
else:
    zarr = None  # type: ignore[assignment]


@pytest.fixture(params=ZARR_FORMATS)
def store(request) -> "zarr.storage.MemoryStore":
    memorystore = zarr.storage.MemoryStore({})

    ds = create_test_data()
    ds.to_zarr(memorystore, zarr_format=request.param, consolidated=False)  # type: ignore[call-overload]

    return memorystore


@pytest.mark.asyncio
class TestAsyncLoad:
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
        var = ds["var1"].variable

        with pytest.raises(NotImplementedError, match=expected_err_msg):
            await var.isel(**indexer).load_async()
