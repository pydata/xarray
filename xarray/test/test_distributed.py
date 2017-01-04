import pytest
import xarray as xr
from xarray.core.pycompat import suppress

distributed = pytest.importorskip('distributed')
da = pytest.importorskip('dask.array')
from distributed.utils_test import cluster, loop

from xarray.test.test_backends import create_tmp_file
from xarray.test.test_dataset import create_test_data

from . import assert_dataset_allclose, has_scipy, has_netCDF4, has_h5netcdf


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
        with distributed.Client(('127.0.0.1', s['port']), loop=loop):
            original = create_test_data()
            with create_tmp_file() as filename:
                original.to_netcdf(filename, engine=engine)
                restored = xr.open_dataset(filename, chunks=3, engine=engine)
                assert isinstance(restored.var1.data, da.Array)
                computed = restored.compute()
                assert_dataset_allclose(original, computed)
