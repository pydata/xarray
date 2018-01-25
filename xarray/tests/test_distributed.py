import sys

import pytest
import xarray as xr

distributed = pytest.importorskip('distributed')
da = pytest.importorskip('dask.array')
import dask
from dask.distributed import Client
from distributed.utils_test import cluster, gen_cluster
from distributed.utils_test import loop  # flake8: noqa
from distributed.client import futures_of

from xarray.tests.test_backends import create_tmp_file, ON_WINDOWS
from xarray.tests.test_dataset import create_test_data

from . import (assert_allclose, has_scipy, has_netCDF4, has_h5netcdf,
               requires_zarr)


@gen_cluster(client=True, timeout=None)
def test_dask_distributed_netcdf_integration_test_scipy(c, s, a, b):
    chunks = {'dim1': 4, 'dim2': 3, 'dim3': 6}
    original = create_test_data().chunk(chunks)
    with create_tmp_file(allow_cleanup_failure=ON_WINDOWS) as filename:
        original.to_netcdf(filename, engine='scipy')
        with xr.open_dataset(filename, chunks=chunks,
                             engine='scipy') as restored:
            assert isinstance(restored.var1.data, da.Array)
            computed = restored.compute()
            assert_allclose(original, computed)

@gen_cluster(client=True, timeout=None)
def test_dask_distributed_netcdf_integration_test_netcdf4(c, s, a, b):
    chunks = {'dim1': 4, 'dim2': 3, 'dim3': 6}
    original = create_test_data().chunk(chunks)
    with create_tmp_file(allow_cleanup_failure=ON_WINDOWS) as filename:
        original.to_netcdf(filename, engine='netcdf4')
        with xr.open_dataset(filename, chunks=chunks,
                             engine='netcdf4') as restored:
            assert isinstance(restored.var1.data, da.Array)
            computed = restored.compute()
            assert_allclose(original, computed)


@gen_cluster(client=True, timeout=None)
def test_dask_distributed_netcdf_integration_test_h5netcdf(c, s, a, b):
    chunks = {'dim1': 4, 'dim2': 3, 'dim3': 6}
    original = create_test_data().chunk(chunks)
    with create_tmp_file(allow_cleanup_failure=ON_WINDOWS) as filename:
        original.to_netcdf(filename, engine='h5netcdf')
        with xr.open_dataset(filename, chunks=chunks,
                             engine='h5netcdf') as restored:
            assert isinstance(restored.var1.data, da.Array)
            computed = restored.compute()
            assert_allclose(original, computed)


@requires_zarr
def test_dask_distributed_zarr_integration_test(loop):
    chunks = {'dim1': 4, 'dim2': 3, 'dim3': 5}
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop) as c:
            original = create_test_data().chunk(chunks)
            with create_tmp_file(allow_cleanup_failure=ON_WINDOWS,
                                 suffix='.zarr') as filename:
                original.to_zarr(filename)
                with xr.open_zarr(filename) as restored:
                    assert isinstance(restored.var1.data, da.Array)
                    computed = restored.compute()
                    assert_allclose(original, computed)


@pytest.mark.skipif(distributed.__version__ <= '1.19.3',
                    reason='Need recent distributed version to clean up get')
@gen_cluster(client=True, timeout=None)
def test_async(c, s, a, b):
    x = create_test_data()
    assert not dask.is_dask_collection(x)
    y = x.chunk({'dim2': 4}) + 10
    assert dask.is_dask_collection(y)
    assert dask.is_dask_collection(y.var1)
    assert dask.is_dask_collection(y.var2)

    z = y.persist()
    assert str(z)

    assert dask.is_dask_collection(z)
    assert dask.is_dask_collection(z.var1)
    assert dask.is_dask_collection(z.var2)
    assert len(y.__dask_graph__()) > len(z.__dask_graph__())

    assert not futures_of(y)
    assert futures_of(z)

    future = c.compute(z)
    w = yield future
    assert not dask.is_dask_collection(w)
    assert_allclose(x + 10, w)

    assert s.task_state
