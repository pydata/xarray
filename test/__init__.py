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


class ReturnItem(object):
    def __getitem__(self, key):
        return key
