import unittest

from numpy.testing import assert_array_equal

from xray import utils


class TestCase(unittest.TestCase):
    def assertXArrayEqual(self, v1, v2):
        self.assertTrue(utils.xarray_equal(v1, v2))

    def assertXArrayNotEqual(self, v1, v2):
        self.assertFalse(utils.xarray_equal(v1, v2))

    def assertArrayEqual(self, a1, a2):
        assert_array_equal(a1, a2)

    def assertDatasetEqual(self, d1, d2):
        # this method is functionally equivalent to `assert d1 == d2`, but it
        # checks each aspect of equality separately for easier debugging
        self.assertEqual(sorted(d1.attributes.items()),
                         sorted(d2.attributes.items()))
        self.assertEqual(sorted(d1.variables), sorted(d2.variables))
        for k in d1:
            v1 = d1.variables[k]
            v2 = d2.variables[k]
            self.assertXArrayEqual(v1, v2)


class ReturnItem(object):
    def __getitem__(self, key):
        return key
