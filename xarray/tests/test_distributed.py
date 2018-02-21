""" isort:skip_file """

import sys

import pytest

dask = pytest.importorskip('dask')  # isort:skip
distributed = pytest.importorskip('distributed')  # isort:skip

from dask import array
from distributed.utils_test import cluster, gen_cluster
from distributed.utils_test import loop  # flake8: noqa
from distributed.client import futures_of

import xarray as xr
from xarray.tests.test_backends import ON_WINDOWS, create_tmp_file
from xarray.tests.test_dataset import create_test_data

from . import (
    assert_allclose, has_h5netcdf, has_netCDF4, has_scipy, requires_zarr)

# this is to stop isort throwing errors. May have been easier to just use
# `isort:skip` in retrospect


da = pytest.importorskip('dask.array')


ENGINES = []
if has_scipy:
    ENGINES.append('scipy')
if has_netCDF4:
    ENGINES.append('netcdf4')
if has_h5netcdf:
    ENGINES.append('h5netcdf')


@pytest.mark.xfail(sys.platform == 'win32',
                   reason='https://github.com/pydata/xarray/issues/1738')
@pytest.mark.parametrize('engine', ENGINES)
def test_dask_distributed_netcdf_integration_test(loop, engine):
    with cluster() as (s, _):
        with distributed.Client(s['address'], loop=loop):
            original = create_test_data()
            with create_tmp_file(allow_cleanup_failure=ON_WINDOWS) as filename:
                original.to_netcdf(filename, engine=engine)
                with xr.open_dataset(
                        filename, chunks=3, engine=engine) as restored:
                    assert isinstance(restored.var1.data, da.Array)
                    computed = restored.compute()
                    assert_allclose(original, computed)


@requires_zarr
def test_dask_distributed_zarr_integration_test(loop):
    with cluster() as (s, _):
        with distributed.Client(s['address'], loop=loop):
            original = create_test_data()
            with create_tmp_file(allow_cleanup_failure=ON_WINDOWS) as filename:
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

    assert s.tasks
