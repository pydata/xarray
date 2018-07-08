from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
from contextlib import contextmanager
from distutils.version import LooseVersion
import re
import importlib

import numpy as np
from numpy.testing import assert_array_equal  # noqa: F401
from xarray.core.duck_array_ops import allclose_or_equiv
import pytest

from xarray.core import utils
from xarray.core.pycompat import PY3
from xarray.core.indexing import ExplicitlyIndexed
from xarray.testing import (assert_equal, assert_identical,  # noqa: F401
                            assert_allclose)
from xarray.plot.utils import import_seaborn

try:
    from pandas.testing import assert_frame_equal
except ImportError:
    # old location, for pandas < 0.20
    from pandas.util.testing import assert_frame_equal  # noqa: F401

try:
    import unittest2 as unittest
except ImportError:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock  # noqa: F401

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
has_dask, requires_dask = _importorskip('dask')
has_bottleneck, requires_bottleneck = _importorskip('bottleneck')
has_rasterio, requires_rasterio = _importorskip('rasterio')
has_pathlib, requires_pathlib = _importorskip('pathlib')
has_zarr, requires_zarr = _importorskip('zarr', minversion='2.2')
has_np113, requires_np113 = _importorskip('numpy', minversion='1.13.0')

# some special cases
has_scipy_or_netCDF4 = has_scipy or has_netCDF4
requires_scipy_or_netCDF4 = pytest.mark.skipif(
    not has_scipy_or_netCDF4, reason='requires scipy or netCDF4')
has_cftime_or_netCDF4 = has_cftime or has_netCDF4
requires_cftime_or_netCDF4 = pytest.mark.skipif(
    not has_cftime_or_netCDF4, reason='requires cftime or netCDF4')
if not has_pathlib:
    has_pathlib, requires_pathlib = _importorskip('pathlib2')
if has_dask:
    import dask
    if LooseVersion(dask.__version__) < '0.18':
        dask.set_options(get=dask.get)
    else:
        dask.config.set(scheduler='sync')
try:
    import_seaborn()
    has_seaborn = True
except ImportError:
    has_seaborn = False
requires_seaborn = pytest.mark.skipif(not has_seaborn,
                                      reason='requires seaborn')

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


class TestCase(unittest.TestCase):
    """
    These functions are all deprecated. Instead, use functions in xr.testing
    """
    if PY3:
        # Python 3 assertCountEqual is roughly equivalent to Python 2
        # assertItemsEqual
        def assertItemsEqual(self, first, second, msg=None):
            __tracebackhide__ = True  # noqa: F841
            return self.assertCountEqual(first, second, msg)

    @contextmanager
    def assertWarns(self, message):
        __tracebackhide__ = True  # noqa: F841
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', message)
            yield
        assert len(w) > 0
        assert any(message in str(wi.message) for wi in w)

    def assertVariableNotEqual(self, v1, v2):
        __tracebackhide__ = True  # noqa: F841
        assert not v1.equals(v2)

    def assertEqual(self, a1, a2):
        __tracebackhide__ = True  # noqa: F841
        assert a1 == a2 or (a1 != a1 and a2 != a2)

    def assertAllClose(self, a1, a2, rtol=1e-05, atol=1e-8):
        __tracebackhide__ = True  # noqa: F841
        assert allclose_or_equiv(a1, a2, rtol=rtol, atol=atol)


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
