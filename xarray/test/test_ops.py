import numpy as np
from numpy import array, nan
from xarray.core import ops
from xarray.core.ops import (
    first, last, count, mean
)
from xarray.core.nputils import interleaved_concat as interleaved_concat_numpy

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

    def test_interleaved_concat(self):
        for interleaved_concat in [interleaved_concat_numpy,
                                   ops._interleaved_concat_slow,
                                   ops.interleaved_concat]:
            x = np.arange(5)
            self.assertArrayEqual(x, interleaved_concat([x], [x]))

            arrays = np.arange(10).reshape(2, -1)
            indices = np.arange(10).reshape(2, -1, order='F')
            actual = interleaved_concat(arrays, indices)
            expected = np.array([0, 5, 1, 6, 2, 7, 3, 8, 4, 9])
            self.assertArrayEqual(expected, actual)

            arrays2 = arrays.reshape(2, 5, 1)

            actual = interleaved_concat(arrays2, indices, axis=0)
            self.assertArrayEqual(expected.reshape(10, 1), actual)

            actual = interleaved_concat(arrays2, [[0], [1]], axis=1)
            self.assertArrayEqual(arrays.T, actual)

            actual = interleaved_concat(arrays2, [slice(1), slice(1, 2)], axis=-1)
            self.assertArrayEqual(arrays.T, actual)

            with self.assertRaises(IndexError):
                interleaved_concat(arrays, indices, axis=1)
            with self.assertRaises(IndexError):
                interleaved_concat(arrays, indices, axis=-2)
            with self.assertRaises(IndexError):
                interleaved_concat(arrays2, [0, 1], axis=2)

    def test_interleaved_concat_dtypes(self):
        for interleaved_concat in [interleaved_concat_numpy,
                                   ops._interleaved_concat_slow,
                                   ops.interleaved_concat]:
            a = np.array(['a'])
            b = np.array(['bc'])
            actual = interleaved_concat([a, b], [[0], [1]])
            expected = np.array(['a', 'bc'])
            self.assertArrayEqual(expected, actual)

            c = np.array([np.nan], dtype=object)
            actual = interleaved_concat([a, b, c], [[0], [1], [2]])
            expected = np.array(['a', 'bc', np.nan], dtype=object)
            self.assertArrayEqual(expected, actual)

    def test_interleaved_indices_required(self):
        self.assertFalse(ops._interleaved_indices_required([[0]]))
        self.assertFalse(ops._interleaved_indices_required([[0, 1], [2, 3, 4]]))
        self.assertFalse(ops._interleaved_indices_required([slice(3), slice(3, 4)]))
        self.assertFalse(ops._interleaved_indices_required([slice(0, 2, 1)]))
        self.assertTrue(ops._interleaved_indices_required([[0], [2]]))
        self.assertTrue(ops._interleaved_indices_required([[1], [2, 3]]))
        self.assertTrue(ops._interleaved_indices_required([[0, 1], [2, 4]]))
        self.assertTrue(ops._interleaved_indices_required([[0, 1], [3.5, 4]]))
        self.assertTrue(ops._interleaved_indices_required([slice(None, None, 2)]))
