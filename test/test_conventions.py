import numpy as np

from xray.conventions import MaskedAndScaledArray, CharToStringArray
from . import TestCase


class TestMaskedAndScaledArray(TestCase):
    def test(self):
        x = MaskedAndScaledArray(np.arange(3), fill_value=0)
        self.assertEqual(x.dtype, np.dtype('float'))
        self.assertEqual(x.shape, (3,))
        self.assertEqual(x.size, 3)
        self.assertEqual(x.ndim, 1)
        self.assertEqual(len(x), 3)
        self.assertArrayEqual([np.nan, 1, 2], x)

        x = MaskedAndScaledArray(np.arange(3), add_offset=1)
        self.assertArrayEqual(np.arange(3) + 1, x)

        x = MaskedAndScaledArray(np.arange(3), scale_factor=2)
        self.assertArrayEqual(2 * np.arange(3), x)

        x = MaskedAndScaledArray(np.array([-99, -1, 0, 1, 2]), -99, 0.01, 1)
        expected = np.array([np.nan, 0.99, 1, 1.01, 1.02])
        self.assertArrayEqual(expected, x)

    def test_0d(self):
        x = MaskedAndScaledArray(np.array(0), fill_value=0)
        self.assertTrue(np.isnan(x))
        self.assertTrue(np.isnan(x[...]))

        x = MaskedAndScaledArray(np.array(0), fill_value=10)
        self.assertEqual(0, x[...])


class TestCharToStringArray(TestCase):
    def test(self):
        array = np.array(list('abc'))
        actual = CharToStringArray(array)
        expected = np.array('abc')
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.size, expected.size)
        self.assertEqual(actual.ndim, expected.ndim)
        with self.assertRaises(TypeError):
            len(actual)
        self.assertArrayEqual(expected, actual)
        with self.assertRaises(IndexError):
            actual[:2]
        self.assertEqual(str(actual), 'abc')

        array = np.array([list('abc'), list('cdf')])
        actual = CharToStringArray(array)
        expected = np.array(['abc', 'cdf'])
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.size, expected.size)
        self.assertEqual(actual.ndim, expected.ndim)
        self.assertEqual(len(actual), len(expected))
        self.assertArrayEqual(expected, actual)
        self.assertArrayEqual(expected[:1], actual[:1])
        with self.assertRaises(IndexError):
            actual[:, :2]
