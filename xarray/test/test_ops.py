import numpy as np
from numpy import array, nan
from xarray.core import ops
from xarray.core.ops import (
    first, last, count, mean
)

from . import TestCase


class TestOps(TestCase):
    def setUp(self):
        self.x = array([[[nan,  nan,   2.,  nan],
                         [nan,   5.,   6.,  nan],
                         [8.,   9.,  10.,  nan]],

                        [[nan,  13.,  14.,  15.],
                         [nan,  17.,  18.,  nan],
                         [nan,  21.,  nan,  nan]]])

    def test_first(self):
        expected_results = [array([[nan, 13, 2, 15],
                                   [nan, 5, 6, nan],
                                   [8, 9, 10, nan]]),
                            array([[8, 5, 2, nan],
                                   [nan, 13, 14, 15]]),
                            array([[2, 5, 8],
                                   [13, 17, 21]])]
        for axis, expected in zip([0, 1, 2, -3, -2, -1],
                                  2 * expected_results):
            actual = first(self.x, axis)
            self.assertArrayEqual(expected, actual)

        expected = self.x[0]
        actual = first(self.x, axis=0, skipna=False)
        self.assertArrayEqual(expected, actual)

        expected = self.x[..., 0]
        actual = first(self.x, axis=-1, skipna=False)
        self.assertArrayEqual(expected, actual)

        with self.assertRaisesRegexp(IndexError, 'out of bounds'):
            first(self.x, 3)

    def test_last(self):
        expected_results = [array([[nan, 13, 14, 15],
                                   [nan, 17, 18, nan],
                                   [8, 21, 10, nan]]),
                            array([[8, 9, 10, nan],
                                   [nan, 21, 18, 15]]),
                            array([[2, 6, 10],
                                   [15, 18, 21]])]
        for axis, expected in zip([0, 1, 2, -3, -2, -1],
                                  2 * expected_results):
            actual = last(self.x, axis)
            self.assertArrayEqual(expected, actual)

        expected = self.x[-1]
        actual = last(self.x, axis=0, skipna=False)
        self.assertArrayEqual(expected, actual)

        expected = self.x[..., -1]
        actual = last(self.x, axis=-1, skipna=False)
        self.assertArrayEqual(expected, actual)

        with self.assertRaisesRegexp(IndexError, 'out of bounds'):
            last(self.x, 3)

    def test_count(self):
        self.assertEqual(12, count(self.x))

        expected = array([[1, 2, 3], [3, 2, 1]])
        self.assertArrayEqual(expected, count(self.x, axis=-1))

    def test_all_nan_arrays(self):
        assert np.isnan(mean([np.nan, np.nan]))
