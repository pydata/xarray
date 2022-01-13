import importlib
from distutils import version

import pytest


def _importorskip(modname, minversion=None):
    try:
        mod = importlib.import_module(modname)
        has = True
        if minversion is not None:
            if LooseVersion(mod.__version__) < LooseVersion(minversion):
                raise ImportError("Minimum version not satisfied")
    except ImportError:
        has = False
    func = pytest.mark.skipif(not has, reason=f"requires {modname}")
    return has, func


def LooseVersion(vstring):
    # Our development version is something like '0.10.9+aac7bfc'
    # This function just ignores the git commit id.
    vstring = vstring.split("+")[0]
    return version.LooseVersion(vstring)


has_zarr, requires_zarr = _importorskip("zarr")
has_h5netcdf, requires_h5netcdf = _importorskip("h5netcdf")
has_netCDF4, requires_netCDF4 = _importorskip("netCDF4")
