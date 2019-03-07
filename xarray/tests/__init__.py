import warnings
from contextlib import contextmanager
from distutils import version
import re
import importlib
from unittest import mock

import numpy as np
from numpy.testing import assert_array_equal  # noqa: F401
from xarray.core.duck_array_ops import allclose_or_equiv  # noqa
import pytest

from xarray.core import utils
from xarray.core.options import set_options
from xarray.core.indexing import ExplicitlyIndexed
from xarray.testing import (assert_equal, assert_identical,  # noqa: F401
                            assert_allclose, assert_combined_tile_ids_equal)
from xarray.plot.utils import import_seaborn

try:
    from pandas.testing import assert_frame_equal
except ImportError:
    # old location, for pandas < 0.20
    from pandas.util.testing import assert_frame_equal  # noqa: F401

# import mpl and change the backend before other mpl imports
try:
    import matplotlib as mpl
    # Order of imports is important here.
    # Using a different backend makes Travis CI work
    mpl.use('Agg')
except ImportError:
    pass


def _importorskip(modname, minversion=None):
    try:
        mod = importlib.import_module(modname)
        has = True
        if minversion is not None:
            if LooseVersion(mod.__version__) < LooseVersion(minversion):
                raise ImportError('Minimum version not satisfied')
    except ImportError:
        has = False
    func = pytest.mark.skipif(not has, reason='requires {}'.format(modname))
    return has, func


def LooseVersion(vstring):
    # Our development version is something like '0.10.9+aac7bfc'
    # This function just ignored the git commit id.
    vstring = vstring.split('+')[0]
    return version.LooseVersion(vstring)


has_matplotlib, requires_matplotlib = _importorskip('matplotlib')
has_matplotlib2, requires_matplotlib2 = _importorskip('matplotlib',
                                                      minversion='2')
has_scipy, requires_scipy = _importorskip('scipy')
has_pydap, requires_pydap = _importorskip('pydap.client')
has_netCDF4, requires_netCDF4 = _importorskip('netCDF4')
has_h5netcdf, requires_h5netcdf = _importorskip('h5netcdf')
has_pynio, requires_pynio = _importorskip('Nio')
has_pseudonetcdf, requires_pseudonetcdf = _importorskip('PseudoNetCDF')
has_cftime, requires_cftime = _importorskip('cftime')
has_nc_time_axis, requires_nc_time_axis = _importorskip('nc_time_axis',
                                                        minversion='1.2.0')
has_cftime_1_0_2_1, requires_cftime_1_0_2_1 = _importorskip(
    'cftime', minversion='1.0.2.1')
has_dask, requires_dask = _importorskip('dask')
has_bottleneck, requires_bottleneck = _importorskip('bottleneck')
has_rasterio, requires_rasterio = _importorskip('rasterio')
has_pathlib, requires_pathlib = _importorskip('pathlib')
has_zarr, requires_zarr = _importorskip('zarr', minversion='2.2')
has_np113, requires_np113 = _importorskip('numpy', minversion='1.13.0')
has_iris, requires_iris = _importorskip('iris')
has_cfgrib, requires_cfgrib = _importorskip('cfgrib')

# some special cases
has_scipy_or_netCDF4 = has_scipy or has_netCDF4
requires_scipy_or_netCDF4 = pytest.mark.skipif(
    not has_scipy_or_netCDF4, reason='requires scipy or netCDF4')
has_cftime_or_netCDF4 = has_cftime or has_netCDF4
requires_cftime_or_netCDF4 = pytest.mark.skipif(
    not has_cftime_or_netCDF4, reason='requires cftime or netCDF4')
if not has_pathlib:
    has_pathlib, requires_pathlib = _importorskip('pathlib2')
try:
    import_seaborn()
    has_seaborn = True
except ImportError:
    has_seaborn = False
requires_seaborn = pytest.mark.skipif(not has_seaborn,
                                      reason='requires seaborn')

# change some global options for tests
set_options(warn_for_unclosed_files=True)

if has_dask:
    import dask
    if LooseVersion(dask.__version__) < '0.18':
        dask.set_options(get=dask.get)
    else:
        dask.config.set(scheduler='single-threaded')

# pytest config
try:
    _SKIP_FLAKY = not pytest.config.getoption("--run-flaky")
    _SKIP_NETWORK_TESTS = not pytest.config.getoption("--run-network-tests")
except (ValueError, AttributeError):
    # Can't get config from pytest, e.g., because xarray is installed instead
    # of being run from a development version (and hence conftests.py is not
    # available). Don't run flaky tests.
    _SKIP_FLAKY = True
    _SKIP_NETWORK_TESTS = True

flaky = pytest.mark.skipif(
    _SKIP_FLAKY, reason="set --run-flaky option to run flaky tests")
network = pytest.mark.skipif(
    _SKIP_NETWORK_TESTS,
    reason="set --run-network-tests option to run tests requiring an "
    "internet connection")


@contextmanager
def raises_regex(error, pattern):
    __tracebackhide__ = True  # noqa: F841
    with pytest.raises(error) as excinfo:
        yield
    message = str(excinfo.value)
    if not re.search(pattern, message):
        raise AssertionError('exception %r did not match pattern %r'
                             % (excinfo.value, pattern))


class UnexpectedDataAccess(Exception):
    pass


class InaccessibleArray(utils.NDArrayMixin, ExplicitlyIndexed):

    def __init__(self, array):
        self.array = array

    def __getitem__(self, key):
        raise UnexpectedDataAccess("Tried accessing data")


class ReturnItem(object):

    def __getitem__(self, key):
        return key


class IndexerMaker(object):

    def __init__(self, indexer_cls):
        self._indexer_cls = indexer_cls

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        return self._indexer_cls(key)


def source_ndarray(array):
    """Given an ndarray, return the base object which holds its memory, or the
    object itself.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'DatetimeIndex.base')
        warnings.filterwarnings('ignore', 'TimedeltaIndex.base')
        base = getattr(array, 'base', np.asarray(array).base)
    if base is None:
        base = array
    return base
