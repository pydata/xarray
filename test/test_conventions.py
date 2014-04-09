import numpy as np
import pandas as pd
from datetime import datetime
import warnings

from xray import conventions
from . import TestCase, requires_netCDF4


class TestMaskedAndScaledArray(TestCase):
    def test(self):
        x = conventions.MaskedAndScaledArray(np.arange(3), fill_value=0)
        self.assertEqual(x.dtype, np.dtype('float'))
        self.assertEqual(x.shape, (3,))
        self.assertEqual(x.size, 3)
        self.assertEqual(x.ndim, 1)
        self.assertEqual(len(x), 3)
        self.assertArrayEqual([np.nan, 1, 2], x)

        x = conventions.MaskedAndScaledArray(np.arange(3), add_offset=1)
        self.assertArrayEqual(np.arange(3) + 1, x)

        x = conventions.MaskedAndScaledArray(np.arange(3), scale_factor=2)
        self.assertArrayEqual(2 * np.arange(3), x)

        x = conventions.MaskedAndScaledArray(np.array([-99, -1, 0, 1, 2]), -99, 0.01, 1)
        expected = np.array([np.nan, 0.99, 1, 1.01, 1.02])
        self.assertArrayEqual(expected, x)

    def test_0d(self):
        x = conventions.MaskedAndScaledArray(np.array(0), fill_value=0)
        self.assertTrue(np.isnan(x))
        self.assertTrue(np.isnan(x[...]))

        x = conventions.MaskedAndScaledArray(np.array(0), fill_value=10)
        self.assertEqual(0, x[...])


class TestCharToStringArray(TestCase):
    def test(self):
        array = np.array(list('abc'))
        actual = conventions.CharToStringArray(array)
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
        actual = conventions.CharToStringArray(array)
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


class TestDatetime(TestCase):
    @requires_netCDF4
    def test_cf_datetime(self):
        import netCDF4 as nc4
        for num_dates, units in [
                (np.arange(100), 'days since 2000-01-01'),
                (np.arange(100).reshape(10, 10), 'days since 2000-01-01'),
                (12300 + np.arange(50), 'hours since 1680-01-01 00:00:00'),
                (10, 'days since 2000-01-01'),
                ([10], 'days since 2000-01-01'),
                ([[10]], 'days since 2000-01-01'),
                ([10, 10], 'days since 2000-01-01'),
                (0, 'days since 1000-01-01'),
                ([0], 'days since 1000-01-01'),
                ([[0]], 'days since 1000-01-01'),
                (np.arange(20), 'days since 1000-01-01'),
                (np.arange(0, 100000, 10000), 'days since 1900-01-01')
                ]:
            for calendar in ['standard', 'gregorian', 'proleptic_gregorian']:
                expected = nc4.num2date(num_dates, units, calendar)
                actual = conventions.decode_cf_datetime(num_dates, units, calendar)
                if (isinstance(actual, np.ndarray)
                        and np.issubdtype(actual.dtype, np.datetime64)):
                    self.assertEqual(actual.dtype, np.dtype('M8[ns]'))
                    # For some reason, numpy 1.8 does not compare ns precision
                    # datetime64 arrays as equal to arrays of datetime objects,
                    # but it works for us precision. Thus, convert to us
                    # precision for the actual array equal comparison...
                    actual_cmp = actual.astype('M8[us]')
                else:
                    actual_cmp = actual
                self.assertArrayEqual(expected, actual_cmp)
                encoded, _, _ = conventions.encode_cf_datetime(actual, units, calendar)
                self.assertArrayEqual(num_dates, np.around(encoded))
                if (hasattr(num_dates, 'ndim') and num_dates.ndim == 1
                        and '1000' not in units):
                    # verify that wrapping with a pandas.Index works
                    # note that it *does not* currently work to even put
                    # non-datetime64 compatible dates into a pandas.Index :(
                    encoded, _, _ = conventions.encode_cf_datetime(
                        pd.Index(actual), units, calendar)
                    self.assertArrayEqual(num_dates, np.around(encoded))

    @requires_netCDF4
    def test_cf_datetime_nan(self):
        for num_dates, units, expected_list in [
                ([np.nan], 'days since 2000-01-01', ['NaT']),
                ([np.nan, 0], 'days since 2000-01-01',
                 ['NaT', '2000-01-01T00:00:00Z']),
                ([np.nan, 0, 1], 'days since 2000-01-01',
                 ['NaT', '2000-01-01T00:00:00Z', '2000-01-02T00:00:00Z']),
                ]:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'All-NaN')
                actual = conventions.decode_cf_datetime(num_dates, units)
            expected = np.array(expected_list, dtype='datetime64[ns]')
            self.assertArrayEqual(expected, actual)

    def test_guess_time_units(self):
        for dates, expected in [(pd.date_range('1900-01-01', periods=5),
                                 'days since 1900-01-01 00:00:00'),
                                (pd.date_range('1900-01-01 12:00:00', freq='H',
                                               periods=2),
                                 'hours since 1900-01-01 12:00:00'),
                                (['1900-01-01', '1900-01-02',
                                  '1900-01-02 00:00:01'],
                                 'seconds since 1900-01-01 00:00:00')]:
            self.assertEquals(expected, conventions.guess_time_units(dates))
