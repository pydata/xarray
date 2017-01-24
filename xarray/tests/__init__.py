from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
from contextlib import contextmanager
from distutils.version import StrictVersion

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from xarray.core import utils, nputils, ops
from xarray.core.variable import as_variable
from xarray.core.pycompat import PY3
from xarray.testing import assert_equal, assert_identical, assert_allclose

try:
    import unittest2 as unittest
except ImportError:
    import unittest

try:
    import scipy
    has_scipy = True
except ImportError:
    has_scipy = False

try:
    import pydap.client
    has_pydap = True
except ImportError:
    has_pydap = False

try:
    import netCDF4
    has_netCDF4 = True
except ImportError:
    has_netCDF4 = False


try:
    import h5netcdf
    has_h5netcdf = True
except ImportError:
    has_h5netcdf = False


try:
    import Nio
    has_pynio = True
except ImportError:
    has_pynio = False


try:
    import dask.array
    import dask
    dask.set_options(get=dask.get)
    has_dask = True
except ImportError:
    has_dask = False


try:
    import matplotlib
    has_matplotlib = True
except ImportError:
    has_matplotlib = False


try:
    import bottleneck
    if StrictVersion(bottleneck.__version__) < StrictVersion('1.0'):
        raise ImportError('Fall back to numpy')
    has_bottleneck = True
except ImportError:
    has_bottleneck = False

# slighly simpler construction that the full functions.
# Generally `pytest.importorskip('package')` inline is even easier
requires_matplotlib = pytest.mark.skipif(not has_matplotlib, reason='requires matplotlib')


def requires_scipy(test):
    return test if has_scipy else pytest.mark.skip('requires scipy')(test)


def requires_pydap(test):
    return test if has_pydap else pytest.mark.skip('requires pydap.client')(test)


def requires_netCDF4(test):
    return test if has_netCDF4 else pytest.mark.skip('requires netCDF4')(test)


def requires_h5netcdf(test):
    return test if has_h5netcdf else pytest.mark.skip('requires h5netcdf')(test)


def requires_pynio(test):
    return test if has_pynio else pytest.mark.skip('requires pynio')(test)


def requires_scipy_or_netCDF4(test):
    return (test if has_scipy or has_netCDF4
            else pytest.mark.skip('requires scipy or netCDF4')(test))


def requires_dask(test):
    return test if has_dask else pytest.mark.skip('requires dask')(test)


def requires_bottleneck(test):
    return test if has_bottleneck else pytest.mark.skip('requires bottleneck')(test)


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
