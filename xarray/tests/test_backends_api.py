import numpy as np

import xarray as xr
from xarray.backends.api import _get_default_engine

from . import assert_identical, requires_netCDF4, requires_scipy


@requires_netCDF4
@requires_scipy
def test__get_default_engine():
    engine_remote = _get_default_engine("http://example.org/test.nc", allow_remote=True)
    assert engine_remote == "netcdf4"

    engine_gz = _get_default_engine("/example.gz")
    assert engine_gz == "scipy"

    engine_default = _get_default_engine("/example")
    assert engine_default == "netcdf4"


def test_custom_engine():
    expected = xr.Dataset(
        dict(a=2 * np.arange(5)), coords=dict(x=("x", np.arange(5), dict(units="s")))
    )

    class CustomBackend(xr.backends.BackendEntrypoint):
        def open_dataset(
            filename_or_obj,
            drop_variables=None,
            **kwargs,
        ):
            return expected.copy(deep=True)

    actual = xr.open_dataset("fake_filename", engine=CustomBackend)
    assert_identical(expected, actual)
