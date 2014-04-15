import unittest

from numpy.testing import assert_array_equal

from xray import utils, DataArray

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


class TestCase(unittest.TestCase):
    def assertVariableEqual(self, v1, v2):
        self.assertTrue(utils.variable_equal(v1, v2))

    def assertVariableAllClose(self, v1, v2, rtol=1e-05, atol=1e-08):
        self.assertTrue(utils.variable_allclose(v1, v2, rtol=rtol, atol=atol))

    def assertVariableNotEqual(self, v1, v2):
        self.assertFalse(utils.variable_equal(v1, v2))

    def assertArrayEqual(self, a1, a2):
        assert_array_equal(a1, a2)

    def assertDatasetEqual(self, d1, d2):
        # this method is functionally equivalent to `assert d1 == d2`, but it
        # checks each aspect of equality separately for easier debugging
        self.assertTrue(utils.dict_equal(d1.attributes, d2.attributes))
        self.assertEqual(sorted(d1.variables), sorted(d2.variables))
        for k in d1:
            v1 = d1.variables[k]
            v2 = d2.variables[k]
            self.assertVariableEqual(v1, v2)

    def assertDatasetAllClose(self, d1, d2, rtol=1e-05, atol=1e-08):
        self.assertTrue(utils.dict_equal(d1.attributes, d2.attributes))
        self.assertEqual(sorted(d1.variables), sorted(d2.variables))
        for k in d1:
            v1 = d1.variables[k]
            v2 = d2.variables[k]
            self.assertVariableAllClose(v1, v2, rtol=rtol, atol=atol)

    def assertDataArrayEqual(self, ar1, ar2):
        self.assertEqual(ar1.name, ar2.name)
        self.assertDatasetEqual(ar1.dataset, ar2.dataset)

    def assertDataArrayAllClose(self, ar1, ar2, rtol=1e-05, atol=1e-08):
        self.assertEqual(ar1.name, ar2.name)
        self.assertDatasetAllClose(ar1.dataset, ar2.dataset,
                                   rtol=rtol, atol=atol)

    def assertDataArrayEquiv(self, ar1, ar2):
        self.assertIsInstance(ar1, DataArray)
        self.assertIsInstance(ar2, DataArray)
        random_name = 'randomly-renamed-variable'
        self.assertDataArrayEqual(ar1.rename(random_name),
                                  ar2.rename(random_name))


class ReturnItem(object):
    def __getitem__(self, key):
        return key
