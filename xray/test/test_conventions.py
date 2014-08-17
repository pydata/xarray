import numpy as np
import pandas as pd
import warnings

from xray import conventions, Variable
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
        array = np.array(list('abc'), dtype='S')
        actual = conventions.CharToStringArray(array)
        expected = np.array('abc', dtype='S')
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

        array = np.array([list('abc'), list('cdf')], dtype='S')
        actual = conventions.CharToStringArray(array)
        expected = np.array(['abc', 'cdf'], dtype='S')
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
                print(num_dates, units, calendar)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', 'Unable to decode time axis')
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
    def test_decoded_cf_datetime_array(self):
        actual = conventions.DecodedCFDatetimeArray(
            np.array([0, 1, 2]), 'days since 1900-01-01', 'standard')
        expected = pd.date_range('1900-01-01', periods=3).values
        self.assertEqual(actual.dtype, np.dtype('datetime64[ns]'))
        self.assertArrayEqual(actual, expected)

        # default calendar
        actual = conventions.DecodedCFDatetimeArray(
            np.array([0, 1, 2]), 'days since 1900-01-01')
        self.assertEqual(actual.dtype, np.dtype('datetime64[ns]'))
        self.assertArrayEqual(actual, expected)

    @requires_netCDF4
    def test_slice_decoded_cf_datetime_array(self):
        actual = conventions.DecodedCFDatetimeArray(
            np.array([0, 1, 2]), 'days since 1900-01-01', 'standard')
        expected = pd.date_range('1900-01-01', periods=3).values
        self.assertEqual(actual.dtype, np.dtype('datetime64[ns]'))
        self.assertArrayEqual(actual[slice(0, 2)], expected[slice(0, 2)])

        actual = conventions.DecodedCFDatetimeArray(
            np.array([0, 1, 2]), 'days since 1900-01-01', 'standard')
        expected = pd.date_range('1900-01-01', periods=3).values
        self.assertEqual(actual.dtype, np.dtype('datetime64[ns]'))
        self.assertArrayEqual(actual[[0, 2]], expected[[0, 2]])

    @requires_netCDF4
    def test_decode_non_standard_calendar(self):
        import netCDF4 as nc4

        for calendar in ['noleap', '365_day', '360_day', 'julian', 'all_leap',
                         '366_day']:
            units = 'days since 0001-01-01'
            times = pd.date_range('2001-04-01-00', end='2001-04-30-23',
                                  freq='H')
            noleap_time = nc4.date2num(times.to_pydatetime(), units,
                                       calendar=calendar)
            expected = times.values
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'Unable to decode time axis')
                actual = conventions.decode_cf_datetime(noleap_time, units,
                                                        calendar=calendar)
            self.assertEqual(actual.dtype, np.dtype('M8[ns]'))
            self.assertArrayEqual(actual, expected)

    @requires_netCDF4
    def test_decode_non_standard_calendar_single_element(self):
        units = 'days since 0001-01-01'
        for calendar in ['noleap', '365_day', '360_day', 'julian', 'all_leap',
                         '366_day']:
            for num_time in [735368, [735368], [[735368]]]:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', 'Unable to decode time axis')
                    actual = conventions.decode_cf_datetime(num_time, units,
                                                            calendar=calendar)
                self.assertEqual(actual.dtype, np.dtype('M8[ns]'))

    @requires_netCDF4
    def test_decode_non_standard_calendar_single_element_fallback(self):
        import netCDF4 as nc4

        units = 'days since 0001-01-01'
        dt = nc4.netcdftime.datetime(2001, 2, 29)
        for calendar in ['360_day', 'all_leap', '366_day']:
            num_time = nc4.date2num(dt, units, calendar)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                actual = conventions.decode_cf_datetime(num_time, units,
                                                        calendar=calendar)
                self.assertEqual(len(w), 1)
                self.assertIn('Unable to decode time axis',
                              str(w[0].message))
            expected = np.asarray(nc4.num2date(num_time, units, calendar))
            print(num_time, calendar, actual, expected)
            self.assertEqual(actual.dtype, np.dtype('O'))
            self.assertEqual(expected, actual)

    @requires_netCDF4
    def test_decode_non_standard_calendar_multidim_time(self):
        import netCDF4 as nc4

        calendar = 'noleap'
        units = 'days since 0001-01-01'
        times1 = pd.date_range('2001-04-01', end='2001-04-05', freq='D')
        times2 = pd.date_range('2001-05-01', end='2001-05-05', freq='D')
        noleap_time1 = nc4.date2num(times1.to_pydatetime(), units,
                                    calendar=calendar)
        noleap_time2 = nc4.date2num(times2.to_pydatetime(), units,
                                    calendar=calendar)
        mdim_time = np.empty((len(noleap_time1), 2), )
        mdim_time[:, 0] = noleap_time1
        mdim_time[:, 1] = noleap_time2

        expected1 = times1.values
        expected2 = times2.values
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Unable to decode time axis')
            actual = conventions.decode_cf_datetime(mdim_time, units,
                                                    calendar=calendar)
        self.assertEqual(actual.dtype, np.dtype('M8[ns]'))
        self.assertArrayEqual(actual[:, 0], expected1)
        self.assertArrayEqual(actual[:, 1], expected2)

    @requires_netCDF4
    def test_decode_non_standard_calendar_fallback(self):
        import netCDF4 as nc4
        for year in [2010, 2011, 2012, 2013, 2014]: # insure leap year doesn't matter
            for calendar in ['360_day', '366_day', 'all_leap']:
                calendar = '360_day'
                units = 'days since {0}-01-01'.format(year)
                num_times = np.arange(100)
                expected = nc4.num2date(num_times, units, calendar)

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')
                    actual = conventions.decode_cf_datetime(num_times, units,
                                                            calendar=calendar)
                    self.assertEqual(len(w), 1)
                    self.assertIn('Unable to decode time axis',
                                  str(w[0].message))

                self.assertEqual(actual.dtype, np.dtype('O'))
                self.assertArrayEqual(actual, expected)

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
            self.assertEqual(expected, conventions.guess_time_units(dates))


class TestEncodeCFVariable(TestCase):
    def test_incompatible_attributes(self):
        invalid_vars = [
            Variable(['t'], pd.date_range('2000-01-01', periods=3),
                     {'units': 'foobar'}),
            Variable(['t'], [0, 1, 2], {'add_offset': 0}, {'add_offset': 2}),
            Variable(['t'], [0, 1, 2], {'_FillValue': 0}, {'_FillValue': 2}),
            ]
        for var in invalid_vars:
            with self.assertRaises(ValueError):
                conventions.encode_cf_variable(var)
