import unittest

import numpy as np
from numpy.testing import assert_array_equal

from xray import utils, DataArray
from xray.variable import as_variable

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


def requires_scipy(test):
    return test if has_scipy else unittest.skip('requires scipy')(test)


def requires_pydap(test):
    return test if has_pydap else unittest.skip('requires pydap.client')(test)


def requires_netCDF4(test):
    return test if has_netCDF4 else unittest.skip('requires netCDF4')(test)


def data_allclose_or_equiv(arr1, arr2, rtol=1e-05, atol=1e-08):
    exact_dtypes = [np.datetime64, np.timedelta64, np.string_]
    if any(any(np.issubdtype(arr.dtype, t) for t in exact_dtypes)
           or arr.dtype == object for arr in [arr1, arr2]):
        return np.array_equal(arr1, arr2)
    else:
        return utils.allclose_or_equiv(arr1, arr2, rtol=rtol, atol=atol)


class TestCase(unittest.TestCase):
    def assertVariableEqual(self, v1, v2):
        self.assertTrue(as_variable(v1).equals(v2))

    def assertVariableIdentical(self, v1, v2):
        self.assertTrue(as_variable(v1).identical(v2))

    def assertVariableAllClose(self, v1, v2, rtol=1e-05, atol=1e-08):
        self.assertEqual(v1.dimensions, v2.dimensions)
        self.assertTrue(data_allclose_or_equiv(v1.values, v2.values,
                                               rtol=rtol, atol=atol))

    def assertVariableNotEqual(self, v1, v2):
        self.assertFalse(as_variable(v1).equals(v2))

    def assertArrayEqual(self, a1, a2):
        assert_array_equal(a1, a2)

    def assertDatasetEqual(self, d1, d2):
        # this method is functionally equivalent to `assert d1 == d2`, but it
        # checks each aspect of equality separately for easier debugging
        self.assertEqual(sorted(d1.variables), sorted(d2.variables))
        for k in d1:
            v1 = d1.variables[k]
            v2 = d2.variables[k]
            self.assertVariableEqual(v1, v2)

    def assertDatasetIdentical(self, d1, d2):
        # this method is functionally equivalent to `assert d1.identical(d2)`,
        # but it checks each aspect of equality separately for easier debugging
        self.assertTrue(utils.dict_equal(d1.attrs, d2.attrs))
        self.assertEqual(sorted(d1.variables), sorted(d2.variables))
        for k in d1:
            v1 = d1.variables[k]
            v2 = d2.variables[k]
            self.assertTrue(v1.identical(v2))

    def assertDatasetAllClose(self, d1, d2, rtol=1e-05, atol=1e-08):
        self.assertEqual(sorted(d1.variables), sorted(d2.variables))
        for k in d1:
            v1 = d1.variables[k]
            v2 = d2.variables[k]
            self.assertVariableAllClose(v1, v2, rtol=rtol, atol=atol)

    def assertCoordsEqual(self, d1, d2):
        self.assertEqual(sorted(d1.coordinates), sorted(d2.coordinates))
        for k in d1.coordinates:
            v1 = d1.coordinates[k]
            v2 = d2.coordinates[k]
            self.assertVariableEqual(v1, v2)

    def assertDataArrayEqual(self, ar1, ar2):
        self.assertVariableEqual(ar1, ar2)
        self.assertCoordsEqual(ar1, ar2)

    def assertDataArrayIdentical(self, ar1, ar2):
        self.assertEqual(ar1.name, ar2.name)
        self.assertDatasetIdentical(ar1.dataset, ar2.dataset)

    def assertDataArrayAllClose(self, ar1, ar2, rtol=1e-05, atol=1e-08):
        self.assertVariableAllClose(ar1, ar2, rtol=rtol, atol=atol)
        self.assertCoordsEqual(ar1, ar2)


class ReturnItem(object):
    def __getitem__(self, key):
        return key


def source_ndarray(array):
    """Given an ndarray, return the base object which holds its memory, or the
    object itself.
    """
    base = array.base
    if base is None:
        base = array
    return base
