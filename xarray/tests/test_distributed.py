import pytest
import xarray as xr
from xarray.core.pycompat import suppress

distributed = pytest.importorskip('distributed')
da = pytest.importorskip('dask.array')
import dask
from distributed.utils_test import cluster, loop, gen_cluster
from distributed.client import futures_of, wait

from xarray.tests.test_backends import create_tmp_file, ON_WINDOWS
from xarray.tests.test_dataset import create_test_data

from . import assert_allclose, has_scipy, has_netCDF4, has_h5netcdf


ENGINES = []
if has_scipy:
    ENGINES.append('scipy')
if has_netCDF4:
    ENGINES.append('netcdf4')
if has_h5netcdf:
    ENGINES.append('h5netcdf')


@pytest.mark.parametrize('engine', ENGINES)
def test_dask_distributed_integration_test(loop, engine):
    with cluster() as (s, _):
        with distributed.Client(s['address'], loop=loop):
            original = create_test_data()
            with create_tmp_file(allow_cleanup_failure=ON_WINDOWS) as filename:
                original.to_netcdf(filename, engine=engine)
                with xr.open_dataset(filename, chunks=3, engine=engine) as restored:
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
