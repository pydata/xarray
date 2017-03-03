import pytest
import xarray as xr
from xarray.core.pycompat import suppress

distributed = pytest.importorskip('distributed')
da = pytest.importorskip('dask.array')
from distributed.utils_test import cluster, loop

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
