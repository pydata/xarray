from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import contextlib
import numpy as np
import pandas as pd
import warnings

from xarray import Variable, Dataset, open_dataset
from xarray.conventions import coding
from xarray.core import utils, indexing
from . import TestCase, requires_netCDF4, unittest
from .test_backends import CFEncodedDataTest
from xarray.core.pycompat import iteritems
from xarray.backends.memory import InMemoryDataStore
from xarray.backends.common import WritableCFDataStore
from xarray.conventions.coding import decode_cf


class TestMaskedAndScaledArray(TestCase):
    def test(self):
        x = coding.MaskedAndScaledArray(np.arange(3), fill_value=0)
        self.assertEqual(x.dtype, np.dtype('float'))
        self.assertEqual(x.shape, (3,))
        self.assertEqual(x.size, 3)
        self.assertEqual(x.ndim, 1)
        self.assertEqual(len(x), 3)
        self.assertArrayEqual([np.nan, 1, 2], x)

        x = coding.MaskedAndScaledArray(np.arange(3), add_offset=1)
        self.assertArrayEqual(np.arange(3) + 1, x)

        x = coding.MaskedAndScaledArray(np.arange(3), scale_factor=2)
        self.assertArrayEqual(2 * np.arange(3), x)

        x = coding.MaskedAndScaledArray(np.array([-99, -1, 0, 1, 2]),
                                             -99, 0.01, 1)
        expected = np.array([np.nan, 0.99, 1, 1.01, 1.02])
        self.assertArrayEqual(expected, x)

    def test_0d(self):
        x = coding.MaskedAndScaledArray(np.array(0), fill_value=0)
        self.assertTrue(np.isnan(x))
        self.assertTrue(np.isnan(x[...]))

        x = coding.MaskedAndScaledArray(np.array(0), fill_value=10)
        self.assertEqual(0, x[...])

    def test_multiple_fill_value(self):
        x = coding.MaskedAndScaledArray(
            np.arange(4), fill_value=np.array([0, 1]))
        self.assertArrayEqual([np.nan, np.nan, 2, 3], x)

        x = coding.MaskedAndScaledArray(
            np.array(0), fill_value=np.array([0, 1]))
        self.assertTrue(np.isnan(x))
        self.assertTrue(np.isnan(x[...]))


class TestCharToStringArray(TestCase):
    def test_wrapper_class(self):
        array = np.array(list('abc'), dtype='S')
        actual = coding.CharToStringArray(array)
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
        actual = coding.CharToStringArray(array)
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

    def test_char_to_string(self):
        array = np.array([['a', 'b', 'c'], ['d', 'e', 'f']])
        expected = np.array(['abc', 'def'])
        actual = coding.char_to_string(array)
        self.assertArrayEqual(actual, expected)

        expected = np.array(['ad', 'be', 'cf'])
        actual = coding.char_to_string(array.T)  # non-contiguous
        self.assertArrayEqual(actual, expected)

    def test_string_to_char(self):
        array = np.array([['ab', 'cd'], ['ef', 'gh']])
        expected = np.array([[['a', 'b'], ['c', 'd']],
                             [['e', 'f'], ['g', 'h']]])
        actual = coding.string_to_char(array)
        self.assertArrayEqual(actual, expected)

        expected = np.array([[['a', 'b'], ['e', 'f']],
                             [['c', 'd'], ['g', 'h']]])
        actual = coding.string_to_char(array.T)
        self.assertArrayEqual(actual, expected)


class TestBoolTypeArray(TestCase):
    def test_booltype_array(self):
        x = np.array([1, 0, 1, 1, 0], dtype='i1')
        bx = coding.BoolTypeArray(x)
        self.assertEqual(bx.dtype, np.bool)
        self.assertArrayEqual(bx, np.array([True, False, True, True, False],
                                           dtype=np.bool))


@np.vectorize
def _ensure_naive_tz(dt):
    if hasattr(dt, 'tzinfo'):
        return dt.replace(tzinfo=None)
    else:
        return dt


def _non_standard_netcdftime_date_types():
    from netcdftime import (
        DatetimeNoLeap, DatetimeJulian, DatetimeAllLeap, Datetime360Day)
    return {'noleap': DatetimeNoLeap,
            '365_day': DatetimeNoLeap,
            '360_day': Datetime360Day,
            'julian': DatetimeJulian,
            'all_leap': DatetimeAllLeap,
            '366_day': DatetimeAllLeap}


def _all_netcdftime_date_types():
    from netcdftime import (
        DatetimeNoLeap, DatetimeJulian, DatetimeAllLeap, Datetime360Day,
        DatetimeGregorian, DatetimeProlepticGregorian)
    return {'noleap': DatetimeNoLeap,
            '365_day': DatetimeNoLeap,
            '360_day': Datetime360Day,
            'julian': DatetimeJulian,
            'all_leap': DatetimeAllLeap,
            '366_day': DatetimeAllLeap,
            'gregorian': DatetimeGregorian,
            'proleptic_gregorian': DatetimeProlepticGregorian}


class TestDatetime(TestCase):
    @requires_netCDF4
    def test_cf_datetime(self):
        import netCDF4 as nc4
        for num_dates, units in [
                (np.arange(10), 'days since 2000-01-01'),
                (np.arange(10).reshape(2, 5), 'days since 2000-01-01'),
                (12300 + np.arange(5), 'hours since 1680-01-01 00:00:00'),
                # here we add a couple minor formatting errors to test
                # the robustness of the parsing algorithm.
                (12300 + np.arange(5), 'hour since 1680-01-01  00:00:00'),
                (12300 + np.arange(5), u'Hour  since 1680-01-01 00:00:00'),
                (12300 + np.arange(5), ' Hour  since  1680-01-01 00:00:00 '),
                (10, 'days since 2000-01-01'),
                ([10], 'daYs  since 2000-01-01'),
                ([[10]], 'days since 2000-01-01'),
                ([10, 10], 'days since 2000-01-01'),
                (np.array(10), 'days since 2000-01-01'),
                (0, 'days since 1000-01-01'),
                ([0], 'days since 1000-01-01'),
                ([[0]], 'days since 1000-01-01'),
                (np.arange(2), 'days since 1000-01-01'),
                (np.arange(0, 100000, 20000), 'days since 1900-01-01'),
                (17093352.0, 'hours since 1-1-1 00:00:0.0'),
                ([0.5, 1.5], 'hours since 1900-01-01T00:00:00'),
                (0, 'milliseconds since 2000-01-01T00:00:00'),
                (0, 'microseconds since 2000-01-01T00:00:00'),
                ]:
            for calendar in ['standard', 'gregorian', 'proleptic_gregorian']:
                expected = _ensure_naive_tz(nc4.num2date(num_dates, units, calendar))
                print(num_dates, units, calendar)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore',
                                            'Unable to decode time axis')
                    actual = coding.decode_cf_datetime(num_dates, units,
                                                            calendar)
                if (isinstance(actual, np.ndarray) and
                        np.issubdtype(actual.dtype, np.datetime64)):
                    # self.assertEqual(actual.dtype.kind, 'M')
                    # For some reason, numpy 1.8 does not compare ns precision
                    # datetime64 arrays as equal to arrays of datetime objects,
                    # but it works for us precision. Thus, convert to us
                    # precision for the actual array equal comparison...
                    actual_cmp = actual.astype('M8[us]')
                else:
                    actual_cmp = actual
                self.assertArrayEqual(expected, actual_cmp)
                encoded, _, _ = coding.encode_cf_datetime(actual, units,
                                                               calendar)
                if '1-1-1' not in units:
                    # pandas parses this date very strangely, so the original
                    # units/encoding cannot be preserved in this case:
                    # (Pdb) pd.to_datetime('1-1-1 00:00:0.0')
                    # Timestamp('2001-01-01 00:00:00')
                    self.assertArrayEqual(num_dates, np.around(encoded, 1))
                    if (hasattr(num_dates, 'ndim') and num_dates.ndim == 1 and
                            '1000' not in units):
                        # verify that wrapping with a pandas.Index works
                        # note that it *does not* currently work to even put
                        # non-datetime64 compatible dates into a pandas.Index :(
                        encoded, _, _ = coding.encode_cf_datetime(
                            pd.Index(actual), units, calendar)
                        self.assertArrayEqual(num_dates, np.around(encoded, 1))

    @requires_netCDF4
    def test_decode_cf_datetime_overflow(self):
        # checks for
        # https://github.com/pydata/pandas/issues/14068
        # https://github.com/pydata/xarray/issues/975

        from datetime import datetime
        units = 'days since 2000-01-01 00:00:00'

        # date after 2262 and before 1678
        days = (-117608, 95795)
        expected = (datetime(1677, 12, 31), datetime(2262, 4, 12))

        for i, day in enumerate(days):
            result = coding.decode_cf_datetime(day, units)
            self.assertEqual(result, expected[i])

    @requires_netCDF4
    def test_decode_cf_datetime_transition_to_invalid(self):
        # manually create dataset with not-decoded date
        from datetime import datetime
        ds = Dataset(coords={'time': [0, 266 * 365]})
        units = 'days since 2000-01-01 00:00:00'
        ds.time.attrs = dict(units=units)
        ds_decoded = coding.decode_cf(ds)

        expected = [datetime(2000, 1, 1, 0, 0),
                    datetime(2265, 10, 28, 0, 0)]

        self.assertArrayEqual(ds_decoded.time.values, expected)

    def test_decoded_cf_datetime_array(self):
        actual = coding.DecodedCFDatetimeArray(
            np.array([0, 1, 2]), 'days since 1900-01-01', 'standard')
        expected = pd.date_range('1900-01-01', periods=3).values
        self.assertEqual(actual.dtype, np.dtype('datetime64[ns]'))
        self.assertArrayEqual(actual, expected)

        # default calendar
        actual = coding.DecodedCFDatetimeArray(
            np.array([0, 1, 2]), 'days since 1900-01-01')
        self.assertEqual(actual.dtype, np.dtype('datetime64[ns]'))
        self.assertArrayEqual(actual, expected)

    def test_slice_decoded_cf_datetime_array(self):
        actual = coding.DecodedCFDatetimeArray(
            np.array([0, 1, 2]), 'days since 1900-01-01', 'standard')
        expected = pd.date_range('1900-01-01', periods=3).values
        self.assertEqual(actual.dtype, np.dtype('datetime64[ns]'))
        self.assertArrayEqual(actual[slice(0, 2)], expected[slice(0, 2)])

        actual = coding.DecodedCFDatetimeArray(
            np.array([0, 1, 2]), 'days since 1900-01-01', 'standard')
        expected = pd.date_range('1900-01-01', periods=3).values
        self.assertEqual(actual.dtype, np.dtype('datetime64[ns]'))
        self.assertArrayEqual(actual[[0, 2]], expected[[0, 2]])

    def test_decode_cf_datetime_non_standard_units(self):
        expected = pd.date_range(periods=100, start='1970-01-01', freq='h')
        # netCDFs from madis.noaa.gov use this format for their time units
        # they cannot be parsed by netcdftime, but pd.Timestamp works
        units = 'hours since 1-1-1970'
        actual = coding.decode_cf_datetime(np.arange(100), units)
        self.assertArrayEqual(actual, expected)

    def test_decode_cf_with_conflicting_fill_missing_value(self):
        var = Variable(['t'], np.arange(10),
                       {'units': 'foobar',
                        'missing_value': 0,
                        '_FillValue': 1})
        self.assertRaisesRegexp(ValueError, "_FillValue and missing_value",
                                lambda: coding.decode_cf_variable(var))

        var = Variable(['t'], np.arange(10),
                       {'units': 'foobar',
                        'missing_value': np.nan,
                        '_FillValue': np.nan})
        var = coding.decode_cf_variable(var)
        self.assertIsNotNone(var)

        var = Variable(['t'], np.arange(10),
                               {'units': 'foobar',
                                'missing_value': np.float32(np.nan),
                                '_FillValue': np.float32(np.nan)})
        var = coding.decode_cf_variable(var)
        self.assertIsNotNone(var)

    @requires_netCDF4
    def test_decode_cf_datetime_non_iso_strings(self):
        # datetime strings that are _almost_ ISO compliant but not quite,
        # but which netCDF4.num2date can still parse correctly
        expected = pd.date_range(periods=100, start='2000-01-01', freq='h')
        cases = [(np.arange(100), 'hours since 2000-01-01 0'),
                 (np.arange(100), 'hours since 2000-1-1 0'),
                 (np.arange(100), 'hours since 2000-01-01 0:00')]
        for num_dates, units in cases:
            actual = coding.decode_cf_datetime(num_dates, units)
            self.assertArrayEqual(actual, expected)

    @requires_netCDF4
    def test_decode_non_standard_calendar(self):
        from datetime import datetime
        import netCDF4 as nc4

        date_types = _non_standard_netcdftime_date_types()
        for calendar, date_type in date_types.items():
            units = 'days since 0001-01-01'
            times = [datetime(1, 4, 1, h) for h in range(1, 5)]
            noleap_time = nc4.date2num(times, units, calendar=calendar)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'Unable to decode time axis')
                actual = coding.decode_cf_datetime(noleap_time, units,
                                                   calendar=calendar)
            assert all(isinstance(value, date_type) for value in actual)
            expected = np.array(
                [date_type(1, 4, 1, h) for h in range(1, 5)])
            abs_diff = abs(actual - expected)
            # once we no longer support versions of netCDF4 older than 1.1.5,
            # we could do this check with near microsecond accuracy:
            # https://github.com/Unidata/netcdf4-python/issues/355
            self.assertTrue((abs_diff <= np.timedelta64(1, 's')).all())

    @requires_netCDF4
    def test_decode_standard_calendar_single_element_ns_range(self):
        units = 'days since 0001-01-01'
        for calendar in coding._STANDARD_CALENDARS:
            for num_time in [735368, [735368], [[735368]]]:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore',
                                            'Unable to decode time axis')
                    actual = coding.decode_cf_datetime(num_time, units,
                                                       calendar=calendar)
                assert actual.dtype == 'datetime64[ns]'

    @requires_netCDF4
    def test_decode_standard_calendar_single_element_non_ns_range(self):
        from datetime import datetime
        from netcdftime import DatetimeGregorian

        units = 'days since 0001-01-01'
        for days in [1, 1470376]:
            for calendar in coding._STANDARD_CALENDARS:
                for num_time in [days, [days], [[days]]]:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore',
                                                'Unable to decode time axis')
                        actual = coding.decode_cf_datetime(num_time, units,
                                                           calendar=calendar)
                    # Confusing, but this is how netCDF4.num2date behaves --
                    # According to the documentation this is supposed to have
                    # something to do with whether the date falls before or
                    # after the breakpoint between the Julian
                    # and Gregorian calendars (1582-10-15).
                    if calendar == 'standard' and days == 1:
                        assert isinstance(actual.item(), DatetimeGregorian)
                    else:
                        assert isinstance(actual.item(), datetime)

    @requires_netCDF4
    def test_decode_non_standard_calendar_single_element(self):
        date_types = _non_standard_netcdftime_date_types()
        units = 'days since 0001-01-01'
        for days in [1, 735368]:
            for calendar, date_type in date_types.items():
                for num_time in [days, [days], [[days]]]:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore',
                                                'Unable to decode time axis')
                        actual = coding.decode_cf_datetime(num_time, units,
                                                           calendar=calendar)
                    assert isinstance(actual.item(), date_type)

    @requires_netCDF4
    def test_decode_standard_calendar_multidim_time(self):
        import netCDF4 as nc4

        for calendar in coding._STANDARD_CALENDARS:
            units = 'days since 0001-01-01'
            times1 = pd.date_range('2001-04-01', end='2001-04-05', freq='D')
            times2 = pd.date_range('2001-05-01', end='2001-05-05', freq='D')
            noleap_time1 = nc4.date2num(times1.to_pydatetime(),
                                        units, calendar=calendar)
            noleap_time2 = nc4.date2num(times2.to_pydatetime(),
                                        units, calendar=calendar)
            mdim_time = np.empty((len(noleap_time1), 2), )
            mdim_time[:, 0] = noleap_time1
            mdim_time[:, 1] = noleap_time2

            expected1 = times1.values
            expected2 = times2.values
            actual = coding.decode_cf_datetime(mdim_time, units,
                                               calendar=calendar)
            self.assertEqual(actual.dtype, np.dtype('M8[ns]'))

            abs_diff1 = abs(actual[:, 0] - expected1)
            abs_diff2 = abs(actual[:, 1] - expected2)
            # once we no longer support versions of netCDF4 older than 1.1.5,
            # we could do this check with near microsecond accuracy:
            # https://github.com/Unidata/netcdf4-python/issues/355
            self.assertTrue((abs_diff1 <= np.timedelta64(1, 's')).all())
            self.assertTrue((abs_diff2 <= np.timedelta64(1, 's')).all())

    @requires_netCDF4
    def test_decode_non_standard_calendar_multidim_time(self):
        from datetime import datetime
        import netCDF4 as nc4

        date_types = _non_standard_netcdftime_date_types()
        for calendar, date_type in date_types.items():
            units = 'days since 0001-01-01'
            times1 = [datetime(1, 4, day) for day in range(1, 6)]
            times2 = [datetime(1, 5, day) for day in range(1, 6)]
            noleap_time1 = nc4.date2num(times1, units, calendar=calendar)
            noleap_time2 = nc4.date2num(times2, units, calendar=calendar)
            mdim_time = np.empty((len(noleap_time1), 2), )
            mdim_time[:, 0] = noleap_time1
            mdim_time[:, 1] = noleap_time2

            expected1 = np.array(
                [date_type(1, 4, day) for day in range(1, 6)])
            expected2 = np.array(
                [date_type(1, 5, day) for day in range(1, 6)])
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'Unable to decode time axis')
                actual = coding.decode_cf_datetime(mdim_time, units,
                                                   calendar=calendar)
            self.assertEqual(actual.dtype, np.dtype('O'))

            abs_diff1 = abs(actual[:, 0] - expected1)
            abs_diff2 = abs(actual[:, 1] - expected2)
            # once we no longer support versions of netCDF4 older than 1.1.5,
            # we could do this check with near microsecond accuracy:
            # https://github.com/Unidata/netcdf4-python/issues/355
            self.assertTrue((abs_diff1 <= np.timedelta64(1, 's')).all())
            self.assertTrue((abs_diff2 <= np.timedelta64(1, 's')).all())

    # I think this test may now be redundant
    @requires_netCDF4
    def test_decode_non_standard_calendar_fallback(self):
        import netCDF4 as nc4
        # ensure leap year doesn't matter
        for year in [2010, 2011, 2012, 2013, 2014]:
            for calendar in ['360_day', '366_day', 'all_leap']:
                calendar = '360_day'
                units = 'days since {0}-01-01'.format(year)
                num_times = np.arange(100)
                expected = nc4.num2date(num_times, units, calendar)

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')
                    actual = coding.decode_cf_datetime(num_times, units,
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
                actual = coding.decode_cf_datetime(num_dates, units)
            expected = np.array(expected_list, dtype='datetime64[ns]')
            self.assertArrayEqual(expected, actual)

    @requires_netCDF4
    def test_decoded_cf_datetime_array_2d(self):
        # regression test for GH1229
        array = coding.DecodedCFDatetimeArray(np.array([[0, 1], [2, 3]]),
                                                   'days since 2000-01-01')
        assert array.dtype == 'datetime64[ns]'
        expected = pd.date_range('2000-01-01', periods=4).values.reshape(2, 2)
        self.assertArrayEqual(np.asarray(array), expected)

    def test_infer_datetime_units(self):
        for dates, expected in [(pd.date_range('1900-01-01', periods=5),
                                 'days since 1900-01-01 00:00:00'),
                                (pd.date_range('1900-01-01 12:00:00', freq='H',
                                               periods=2),
                                 'hours since 1900-01-01 12:00:00'),
                                (pd.to_datetime(['1900-01-01', '1900-01-02', 'NaT']),
                                 'days since 1900-01-01 00:00:00'),
                                (pd.to_datetime(['1900-01-01',
                                                 '1900-01-02T00:00:00.005']),
                                 'seconds since 1900-01-01 00:00:00'),
                                (pd.to_datetime(['NaT', '1900-01-01']),
                                 'days since 1900-01-01 00:00:00'),
                                (pd.to_datetime(['NaT']),
                                 'days since 1970-01-01 00:00:00'),
                                ]:
            self.assertEqual(expected, coding.infer_datetime_units(dates))

    @requires_netCDF4
    def test_infer_netcdftime_datetime_units(self):
        date_types = _all_netcdftime_date_types()
        for date_type in date_types.values():
            for dates, expected in [([date_type(1900, 1, 1),
                                      date_type(1900, 1, 2)],
                                     'days since 1900-01-01 00:00:00'),
                                    ([date_type(1900, 1, 1, 12),
                                      date_type(1900, 1, 1, 13)],
                                     'seconds since 1900-01-01 12:00:00'),
                                    ([date_type(1900, 1, 1),
                                      date_type(1900, 1, 2),
                                      date_type(1900, 1, 2, 0, 0, 1)],
                                     'seconds since 1900-01-01 00:00:00'),
                                    ([date_type(1900, 1, 1),
                                      date_type(1900, 1, 2, 0, 0, 0, 5)],
                                     'days since 1900-01-01 00:00:00')]:
                self.assertEqual(expected, coding.infer_datetime_units(dates))

    def test_cf_timedelta(self):
        examples = [
            ('1D', 'days', np.int64(1)),
            (['1D', '2D', '3D'], 'days', np.array([1, 2, 3], 'int64')),
            ('1h', 'hours', np.int64(1)),
            ('1ms', 'milliseconds', np.int64(1)),
            ('1us', 'microseconds', np.int64(1)),
            (['NaT', '0s', '1s'], None, [np.nan, 0, 1]),
            (['30m', '60m'], 'hours', [0.5, 1.0]),
        ]
        if pd.__version__ >= '0.16':
            # not quite sure why, but these examples don't work on older pandas
            examples.extend([(np.timedelta64('NaT', 'ns'), 'days', np.nan),
                             (['NaT', 'NaT'], 'days', [np.nan, np.nan])])

        for timedeltas, units, numbers in examples:
            timedeltas = pd.to_timedelta(timedeltas, box=False)
            numbers = np.array(numbers)

            expected = numbers
            actual, _ = coding.encode_cf_timedelta(timedeltas, units)
            self.assertArrayEqual(expected, actual)
            self.assertEqual(expected.dtype, actual.dtype)

            if units is not None:
                expected = timedeltas
                actual = coding.decode_cf_timedelta(numbers, units)
                self.assertArrayEqual(expected, actual)
                self.assertEqual(expected.dtype, actual.dtype)

        expected = np.timedelta64('NaT', 'ns')
        actual = coding.decode_cf_timedelta(np.array(np.nan), 'days')
        self.assertArrayEqual(expected, actual)

    def test_cf_timedelta_2d(self):
        timedeltas, units, numbers = ['1D', '2D', '3D'], 'days', np.atleast_2d([1, 2, 3])

        timedeltas = np.atleast_2d(pd.to_timedelta(timedeltas, box=False))
        expected = timedeltas

        actual = coding.decode_cf_timedelta(numbers, units)
        self.assertArrayEqual(expected, actual)
        self.assertEqual(expected.dtype, actual.dtype)

    def test_infer_timedelta_units(self):
        for deltas, expected in [
                (pd.to_timedelta(['1 day', '2 days']), 'days'),
                (pd.to_timedelta(['1h', '1 day 1 hour']), 'hours'),
                (pd.to_timedelta(['1m', '2m', np.nan]), 'minutes'),
                (pd.to_timedelta(['1m3s', '1m4s']), 'seconds')]:
            self.assertEqual(expected, coding.infer_timedelta_units(deltas))

    def test_invalid_units_raises_eagerly(self):
        ds = Dataset({'time': ('time', [0, 1], {'units': 'foobar since 123'})})
        with self.assertRaisesRegexp(ValueError, 'unable to decode time'):
            decode_cf(ds)

    @requires_netCDF4
    def test_dataset_repr_with_netcdf4_datetimes(self):
        # regression test for #347
        attrs = {'units': 'days since 0001-01-01', 'calendar': 'noleap'}
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'unable to decode time')
            ds = decode_cf(Dataset({'time': ('time', [0, 1], attrs)}))
            self.assertIn('(time) object', repr(ds))

        attrs = {'units': 'days since 1900-01-01'}
        ds = decode_cf(Dataset({'time': ('time', [0, 1], attrs)}))
        self.assertIn('(time) datetime64[ns]', repr(ds))

        # this should not throw a warning (GH1111)
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            coding.DecodedCFDatetimeArray(np.asarray([722624]),
                                               "days since 0001-01-01")


class TestNativeEndiannessArray(TestCase):
    def test(self):
        x = np.arange(5, dtype='>i8')
        expected = np.arange(5, dtype='int64')
        a = coding.NativeEndiannessArray(x)
        assert a.dtype == expected.dtype
        assert a.dtype == expected[:].dtype
        self.assertArrayEqual(a, expected)


@requires_netCDF4
class TestEncodeCFVariable(TestCase):
    def test_incompatible_attributes(self):
        invalid_vars = [
            Variable(['t'], pd.date_range('2000-01-01', periods=3),
                     {'units': 'foobar'}),
            Variable(['t'], pd.to_timedelta(['1 day']), {'units': 'foobar'}),
            Variable(['t'], [0, 1, 2], {'add_offset': 0}, {'add_offset': 2}),
            Variable(['t'], [0, 1, 2], {'_FillValue': 0}, {'_FillValue': 2}),
            ]
        for var in invalid_vars:
            with self.assertRaises(ValueError):
                coding.encode_cf_variable(var)

    def test_missing_fillvalue(self):
        v = Variable(['x'], np.array([np.nan, 1, 2, 3]))
        v.encoding = {'dtype': 'int16'}
        with self.assertWarns('floating point data as an integer'):
            coding.encode_cf_variable(v)


@requires_netCDF4
class TestDecodeCF(TestCase):
    def test_dataset(self):
        original = Dataset({
            't': ('t', [0, 1, 2], {'units': 'days since 2000-01-01'}),
            'foo': ('t', [0, 0, 0], {'coordinates': 'y', 'units': 'bar'}),
            'y': ('t', [5, 10, -999], {'_FillValue': -999})
        })
        expected = Dataset({'foo': ('t', [0, 0, 0], {'units': 'bar'})},
                           {'t': pd.date_range('2000-01-01', periods=3),
                            'y': ('t', [5.0, 10.0, np.nan])})
        actual = coding.decode_cf(original)
        self.assertDatasetIdentical(expected, actual)

    def test_invalid_coordinates(self):
        # regression test for GH308
        original = Dataset({'foo': ('t', [1, 2], {'coordinates': 'invalid'})})
        actual = coding.decode_cf(original)
        self.assertDatasetIdentical(original, actual)

    def test_decode_coordinates(self):
        # regression test for GH610
        original = Dataset({'foo': ('t', [1, 2], {'coordinates': 'x'}),
                            'x': ('t', [4, 5])})
        actual = coding.decode_cf(original)
        self.assertEqual(actual.foo.encoding['coordinates'], 'x')

    def test_0d_int32_encoding(self):
        original = Variable((), np.int32(0), encoding={'dtype': 'int64'})
        expected = Variable((), np.int64(0))
        actual = coding.maybe_encode_dtype(original)
        self.assertDatasetIdentical(expected, actual)

    def test_decode_cf_with_multiple_missing_values(self):
        original = Variable(['t'], [0, 1, 2],
                            {'missing_value': np.array([0, 1])})
        expected = Variable(['t'], [np.nan, np.nan, 2], {})
        with warnings.catch_warnings(record=True) as w:
            actual = coding.decode_cf_variable(original)
            self.assertDatasetIdentical(expected, actual)
            self.assertIn('variable has multiple fill', str(w[0].message))

    def test_decode_cf_with_drop_variables(self):
        original = Dataset({
            't': ('t', [0, 1, 2], {'units': 'days since 2000-01-01'}),
            'x': ("x", [9, 8, 7], {'units': 'km'}),
            'foo': (('t', 'x'), [[0, 0, 0], [1, 1, 1], [2, 2, 2]], {'units': 'bar'}),
            'y': ('t', [5, 10, -999], {'_FillValue': -999})
        })
        expected = Dataset({
            't': pd.date_range('2000-01-01', periods=3),
            'foo': (('t', 'x'), [[0, 0, 0], [1, 1, 1], [2, 2, 2]], {'units': 'bar'}),
            'y': ('t', [5, 10, np.nan])
        })
        actual = coding.decode_cf(original, drop_variables=("x",))
        actual2 = coding.decode_cf(original, drop_variables="x")
        self.assertDatasetIdentical(expected, actual)
        self.assertDatasetIdentical(expected, actual2)


class CFEncodedInMemoryStore(WritableCFDataStore, InMemoryDataStore):
    pass


class NullWrapper(utils.NDArrayMixin):
    """
    Just for testing, this lets us create a numpy array directly
    but make it look like its not in memory yet.
    """
    def __init__(self, array):
        self.array = array

    def __getitem__(self, key):
        return self.array[indexing.orthogonal_indexer(key, self.shape)]


def null_wrap(ds):
    """
    Given a data store this wraps each variable in a NullWrapper so that
    it appears to be out of memory.
    """
    variables = dict((k, Variable(v.dims, NullWrapper(v.values), v.attrs))
                     for k, v in iteritems(ds))
    return InMemoryDataStore(variables=variables, attributes=ds.attrs)


@requires_netCDF4
class TestCFEncodedDataStore(CFEncodedDataTest, TestCase):
    @contextlib.contextmanager
    def create_store(self):
        yield CFEncodedInMemoryStore()

    @contextlib.contextmanager
    def roundtrip(self, data, save_kwargs={}, open_kwargs={},
                  allow_cleanup_failure=False):
        store = CFEncodedInMemoryStore()
        data.dump_to_store(store, **save_kwargs)
        yield open_dataset(store, **open_kwargs)

    def test_roundtrip_coordinates(self):
        raise unittest.SkipTest('cannot roundtrip coordinates yet for '
                                'CFEncodedInMemoryStore')

    def test_invalid_dataarray_names_raise(self):
        pass

    def test_encoding_kwarg(self):
        pass
