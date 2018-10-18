
from xarray.backends.api import _get_default_engine
from . import requires_netCDF4, requires_pynio, requires_scipy


@requires_netCDF4
@requires_scipy
@requires_pynio
def test__get_default_engine():
    engine_remote = _get_default_engine('http://example.org/test.nc',
                                        allow_remote=True)
    assert engine_remote == 'netcdf4'

    engine = _get_default_engine('/example.gz')
    assert engine == 'scipy'

    engine = _get_default_engine('/example.grib')
    assert engine == 'pynio'

    engine = _get_default_engine('/example')
    assert engine == 'netcdf4'
