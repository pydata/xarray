from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import numpy as np
import pandas as pd

from xarray import Variable, coding
from xarray.coding.times import _import_netcdftime
from . import (
    TestCase, requires_netcdftime, assert_array_equal)
import pytest


@np.vectorize
def _ensure_naive_tz(dt):
    if hasattr(dt, 'tzinfo'):
        return dt.replace(tzinfo=None)
    else:
        return dt


class TestDatetime(TestCase):
    @requires_netcdftime
    def test_cf_datetime(self):
        nctime = _import_netcdftime()
        for num_dates, units in [
            (np.arange(10), 'days since 2000-01-01'),
            (np.arange(10).astype('float64'), 'days since 2000-01-01'),
            (np.arange(10).astype('float32'), 'days since 2000-01-01'),
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
                expected = _ensure_naive_tz(
                    nctime.num2date(num_dates, units, calendar))
                print(num_dates, units, calendar)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore',
                                            'Unable to decode time axis')
                    actual = coding.times.decode_cf_datetime(num_dates, units,
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
                assert_array_equal(expected, actual_cmp)
                encoded, _, _ = coding.times.encode_cf_datetime(actual, units,
                                                                calendar)
                if '1-1-1' not in units:
                    # pandas parses this date very strangely, so the original
                    # units/encoding cannot be preserved in this case:
                    # (Pdb) pd.to_datetime('1-1-1 00:00:0.0')
                    # Timestamp('2001-01-01 00:00:00')
                    assert_array_equal(num_dates, np.around(encoded, 1))
                    if (hasattr(num_dates, 'ndim') and num_dates.ndim == 1 and
                            '1000' not in units):
                        # verify that wrapping with a pandas.Index works
                        # note that it *does not* currently work to even put
                        # non-datetime64 compatible dates into a pandas.Index
                        encoded, _, _ = coding.times.encode_cf_datetime(
                            pd.Index(actual), units, calendar)
                        assert_array_equal(num_dates, np.around(encoded, 1))

    @requires_netcdftime
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
            result = coding.times.decode_cf_datetime(day, units)
            assert result == expected[i]

    def test_decode_cf_datetime_non_standard_units(self):
        expected = pd.date_range(periods=100, start='1970-01-01', freq='h')
        # netCDFs from madis.noaa.gov use this format for their time units
        # they cannot be parsed by netcdftime, but pd.Timestamp works
        units = 'hours since 1-1-1970'
        actual = coding.times.decode_cf_datetime(np.arange(100), units)
        assert_array_equal(actual, expected)

    @requires_netcdftime
    def test_decode_cf_datetime_non_iso_strings(self):
        # datetime strings that are _almost_ ISO compliant but not quite,
        # but which netCDF4.num2date can still parse correctly
        expected = pd.date_range(periods=100, start='2000-01-01', freq='h')
        cases = [(np.arange(100), 'hours since 2000-01-01 0'),
                 (np.arange(100), 'hours since 2000-1-1 0'),
                 (np.arange(100), 'hours since 2000-01-01 0:00')]
        for num_dates, units in cases:
            actual = coding.times.decode_cf_datetime(num_dates, units)
            assert_array_equal(actual, expected)

    @requires_netcdftime
    def test_decode_non_standard_calendar(self):
        nctime = _import_netcdftime()

        for calendar in ['noleap', '365_day', '360_day', 'julian', 'all_leap',
                         '366_day']:
            units = 'days since 0001-01-01'
            times = pd.date_range('2001-04-01-00', end='2001-04-30-23',
                                  freq='H')
            noleap_time = nctime.date2num(times.to_pydatetime(), units,
                                          calendar=calendar)
            expected = times.values
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'Unable to decode time axis')
                actual = coding.times.decode_cf_datetime(noleap_time, units,
                                                         calendar=calendar)
            assert actual.dtype == np.dtype('M8[ns]')
            abs_diff = abs(actual - expected)
            # once we no longer support versions of netCDF4 older than 1.1.5,
            # we could do this check with near microsecond accuracy:
            # https://github.com/Unidata/netcdf4-python/issues/355
            assert (abs_diff <= np.timedelta64(1, 's')).all()

    @requires_netcdftime
    def test_decode_non_standard_calendar_single_element(self):
        units = 'days since 0001-01-01'
        for calendar in ['noleap', '365_day', '360_day', 'julian', 'all_leap',
                         '366_day']:
            for num_time in [735368, [735368], [[735368]]]:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore',
                                            'Unable to decode time axis')
                    actual = coding.times.decode_cf_datetime(num_time, units,
                                                             calendar=calendar)
                assert actual.dtype == np.dtype('M8[ns]')

    @requires_netcdftime
    def test_decode_non_standard_calendar_single_element_fallback(self):
        nctime = _import_netcdftime()

        units = 'days since 0001-01-01'
        try:
            dt = nctime.netcdftime.datetime(2001, 2, 29)
        except AttributeError:
            # Must be using standalone netcdftime library
            dt = nctime.datetime(2001, 2, 29)
        for calendar in ['360_day', 'all_leap', '366_day']:
            num_time = nctime.date2num(dt, units, calendar)
            with pytest.warns(Warning, match='Unable to decode time axis'):
                actual = coding.times.decode_cf_datetime(num_time, units,
                                                         calendar=calendar)
            expected = np.asarray(nctime.num2date(num_time, units, calendar))
            assert actual.dtype == np.dtype('O')
            assert expected == actual

    @requires_netcdftime
    def test_decode_non_standard_calendar_multidim_time(self):
        nctime = _import_netcdftime()

        calendar = 'noleap'
        units = 'days since 0001-01-01'
        times1 = pd.date_range('2001-04-01', end='2001-04-05', freq='D')
        times2 = pd.date_range('2001-05-01', end='2001-05-05', freq='D')
        noleap_time1 = nctime.date2num(times1.to_pydatetime(), units,
                                       calendar=calendar)
        noleap_time2 = nctime.date2num(times2.to_pydatetime(), units,
                                       calendar=calendar)
        mdim_time = np.empty((len(noleap_time1), 2), )
        mdim_time[:, 0] = noleap_time1
        mdim_time[:, 1] = noleap_time2

        expected1 = times1.values
        expected2 = times2.values
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Unable to decode time axis')
            actual = coding.times.decode_cf_datetime(mdim_time, units,
                                                     calendar=calendar)
        assert actual.dtype == np.dtype('M8[ns]')
        assert_array_equal(actual[:, 0], expected1)
        assert_array_equal(actual[:, 1], expected2)

    @requires_netcdftime
    def test_decode_non_standard_calendar_fallback(self):
        nctime = _import_netcdftime()
        # ensure leap year doesn't matter
        for year in [2010, 2011, 2012, 2013, 2014]:
            for calendar in ['360_day', '366_day', 'all_leap']:
                calendar = '360_day'
                units = 'days since {0}-01-01'.format(year)
                num_times = np.arange(100)
                expected = nctime.num2date(num_times, units, calendar)

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')
                    actual = coding.times.decode_cf_datetime(num_times, units,
                                                             calendar=calendar)
                    assert len(w) == 1
                    assert 'Unable to decode time axis' in \
                        str(w[0].message)

                assert actual.dtype == np.dtype('O')
                assert_array_equal(actual, expected)

    @requires_netcdftime
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
                actual = coding.times.decode_cf_datetime(num_dates, units)
            expected = np.array(expected_list, dtype='datetime64[ns]')
            assert_array_equal(expected, actual)

    @requires_netcdftime
    def test_decoded_cf_datetime_array_2d(self):
        # regression test for GH1229
        variable = Variable(('x', 'y'), np.array([[0, 1], [2, 3]]),
                            {'units': 'days since 2000-01-01'})
        result = coding.times.CFDatetimeCoder().decode(variable)
        assert result.dtype == 'datetime64[ns]'
        expected = pd.date_range('2000-01-01', periods=4).values.reshape(2, 2)
        assert_array_equal(np.asarray(result), expected)

    def test_infer_datetime_units(self):
        for dates, expected in [(pd.date_range('1900-01-01', periods=5),
                                 'days since 1900-01-01 00:00:00'),
                                (pd.date_range('1900-01-01 12:00:00', freq='H',
                                               periods=2),
                                 'hours since 1900-01-01 12:00:00'),
                                (['1900-01-01', '1900-01-02',
                                  '1900-01-02 00:00:01'],
                                 'seconds since 1900-01-01 00:00:00'),
                                (pd.to_datetime(
                                    ['1900-01-01', '1900-01-02', 'NaT']),
                                 'days since 1900-01-01 00:00:00'),
                                (pd.to_datetime(['1900-01-01',
                                                 '1900-01-02T00:00:00.005']),
                                 'seconds since 1900-01-01 00:00:00'),
                                (pd.to_datetime(['NaT', '1900-01-01']),
                                 'days since 1900-01-01 00:00:00'),
                                (pd.to_datetime(['NaT']),
                                 'days since 1970-01-01 00:00:00'),
                                ]:
            assert expected == coding.times.infer_datetime_units(dates)

    def test_cf_timedelta(self):
        examples = [
            ('1D', 'days', np.int64(1)),
            (['1D', '2D', '3D'], 'days', np.array([1, 2, 3], 'int64')),
            ('1h', 'hours', np.int64(1)),
            ('1ms', 'milliseconds', np.int64(1)),
            ('1us', 'microseconds', np.int64(1)),
            (['NaT', '0s', '1s'], None, [np.nan, 0, 1]),
            (['30m', '60m'], 'hours', [0.5, 1.0]),
            (np.timedelta64('NaT', 'ns'), 'days', np.nan),
            (['NaT', 'NaT'], 'days', [np.nan, np.nan]),
        ]

        for timedeltas, units, numbers in examples:
            timedeltas = pd.to_timedelta(timedeltas, box=False)
            numbers = np.array(numbers)

            expected = numbers
            actual, _ = coding.times.encode_cf_timedelta(timedeltas, units)
            assert_array_equal(expected, actual)
            assert expected.dtype == actual.dtype

            if units is not None:
                expected = timedeltas
                actual = coding.times.decode_cf_timedelta(numbers, units)
                assert_array_equal(expected, actual)
                assert expected.dtype == actual.dtype

        expected = np.timedelta64('NaT', 'ns')
        actual = coding.times.decode_cf_timedelta(np.array(np.nan), 'days')
        assert_array_equal(expected, actual)

    def test_cf_timedelta_2d(self):
        timedeltas = ['1D', '2D', '3D']
        units = 'days'
        numbers = np.atleast_2d([1, 2, 3])

        timedeltas = np.atleast_2d(pd.to_timedelta(timedeltas, box=False))
        expected = timedeltas

        actual = coding.times.decode_cf_timedelta(numbers, units)
        assert_array_equal(expected, actual)
        assert expected.dtype == actual.dtype

    def test_infer_timedelta_units(self):
        for deltas, expected in [
                (pd.to_timedelta(['1 day', '2 days']), 'days'),
                (pd.to_timedelta(['1h', '1 day 1 hour']), 'hours'),
                (pd.to_timedelta(['1m', '2m', np.nan]), 'minutes'),
                (pd.to_timedelta(['1m3s', '1m4s']), 'seconds')]:
            assert expected == coding.times.infer_timedelta_units(deltas)
