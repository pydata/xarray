from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
from contextlib import contextmanager
from distutils.version import LooseVersion
import re
import importlib
import types

import numpy as np
from numpy.testing import assert_array_equal
from xarray.core.duck_array_ops import allclose_or_equiv
import pytest

from xarray.core import utils
from xarray.core.pycompat import PY3
from xarray.testing import assert_equal, assert_identical, assert_allclose


def _importorskip(modname, minversion=None):
    try:
        mod = importlib.import_module(modname)
        has = True
        if minversion is not None:
            if LooseVersion(mod.__version__) < LooseVersion(minversion):
                raise ImportError('Minimum version not satisfied')
    except ImportError:
        has = False
    func = pytest.mark.skipif((not has), reason='requires {}'.format(modname))
    return has, func


try:
    import unittest2 as unittest
except ImportError:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock

has_matplotlib, requires_matplotlib = _importorskip('matplotlib')
has_scipy, requires_scipy = _importorskip('scipy')
has_pydap, requires_pydap = _importorskip('pydap.client')
has_netCDF4, requires_netCDF4 = _importorskip('netCDF4')
has_h5netcdf, requires_h5netcdf = _importorskip('h5netcdf')
has_pynio, requires_pynio = _importorskip('pynio')
has_dask, requires_dask = _importorskip('dask')
has_bottleneck, requires_bottleneck = _importorskip('bottleneck')
has_rasterio, requires_rasterio = _importorskip('rasterio')
has_pathlib, requires_pathlib = _importorskip('pathlib')

# some special cases
has_scipy_or_netCDF4 = has_scipy or has_netCDF4
requires_scipy_or_netCDF4 = pytest.mark.skipif(
    not has_scipy_or_netCDF4, reason='requires scipy or netCDF4')
if not has_pathlib:
    has_pathlib, requires_pathlib = _importorskip('pathlib2')

if has_dask:
    import dask
    dask.set_options(get=dask.get)

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
    if PY3:
        # Python 3 assertCountEqual is roughly equivalent to Python 2
        # assertItemsEqual
        def assertItemsEqual(self, first, second, msg=None):
            return self.assertCountEqual(first, second, msg)

    @contextmanager
    def assertWarns(self, message):
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', message)
            yield
        assert len(w) > 0
        assert any(message in str(wi.message) for wi in w)

    def assertVariableEqual(self, v1, v2):
        assert_equal(v1, v2)

    def assertVariableIdentical(self, v1, v2):
        assert_identical(v1, v2)

    def assertVariableAllClose(self, v1, v2, rtol=1e-05, atol=1e-08):
        assert_allclose(v1, v2, rtol=rtol, atol=atol)

    def assertVariableNotEqual(self, v1, v2):
        assert not v1.equals(v2)

    def assertArrayEqual(self, a1, a2):
        assert_array_equal(a1, a2)

    def assertEqual(self, a1, a2):
        assert a1 == a2 or (a1 != a1 and a2 != a2)

    def assertAllClose(self, a1, a2, rtol=1e-05, atol=1e-8):
        assert allclose_or_equiv(a1, a2, rtol=rtol, atol=atol)

    def assertDatasetEqual(self, d1, d2):
        assert_equal(d1, d2)

    def assertDatasetIdentical(self, d1, d2):
        assert_identical(d1, d2)

    def assertDatasetAllClose(self, d1, d2, rtol=1e-05, atol=1e-08):
        assert_allclose(d1, d2, rtol=rtol, atol=atol)

    def assertCoordinatesEqual(self, d1, d2):
        assert_equal(d1, d2)

    def assertDataArrayEqual(self, ar1, ar2):
        assert_equal(ar1, ar2)

    def assertDataArrayIdentical(self, ar1, ar2):
        assert_identical(ar1, ar2)

    def assertDataArrayAllClose(self, ar1, ar2, rtol=1e-05, atol=1e-08):
        assert_allclose(ar1, ar2, rtol=rtol, atol=atol)


@contextmanager
def raises_regex(error, pattern):
    with pytest.raises(error) as excinfo:
        yield
    message = str(excinfo.value)
    if not re.match(pattern, message):
        raise AssertionError('exception %r did not match pattern %s'
                             % (excinfo.value, pattern))


class UnexpectedDataAccess(Exception):
    pass


class InaccessibleArray(utils.NDArrayMixin):

    def __init__(self, array):
        self.array = array

    def __getitem__(self, key):
        raise UnexpectedDataAccess("Tried accessing data")


class ReturnItem(object):

    def __getitem__(self, key):
        return key


def source_ndarray(array):
    """Given an ndarray, return the base object which holds its memory, or the
    object itself.
    """
    base = getattr(array, 'base', np.asarray(array).base)
    if base is None:
        base = array
    return base
