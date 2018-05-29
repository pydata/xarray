from __future__ import absolute_import, division, print_function

from itertools import product
import warnings

import numpy as np
import pandas as pd
import pytest

from xarray import Variable, coding, set_options, DataArray, decode_cf
from xarray.coding.times import _import_cftime
from xarray.coding.variables import SerializationWarning
from xarray.core.common import contains_cftime_datetimes

from . import (assert_array_equal, has_cftime_or_netCDF4,
               requires_cftime_or_netCDF4, has_cftime, has_dask)


_NON_STANDARD_CALENDARS = {'noleap', '365_day', '360_day',
                           'julian', 'all_leap', '366_day'}
_ALL_CALENDARS = _NON_STANDARD_CALENDARS.union(
    coding.times._STANDARD_CALENDARS)
_CF_DATETIME_NUM_DATES_UNITS = [
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
    (np.int32(788961600), 'seconds since 1981-01-01')  # GH2002
]
_CF_DATETIME_TESTS = [num_dates_units + (calendar,) for num_dates_units,
                      calendar in product(_CF_DATETIME_NUM_DATES_UNITS,
                                          coding.times._STANDARD_CALENDARS)]


@np.vectorize
def _ensure_naive_tz(dt):
    if hasattr(dt, 'tzinfo'):
        return dt.replace(tzinfo=None)
    else:
        return dt


def _all_cftime_date_types():
    try:
        import cftime
    except ImportError:
        import netcdftime as cftime
    return {'noleap': cftime.DatetimeNoLeap,
            '365_day': cftime.DatetimeNoLeap,
            '360_day': cftime.Datetime360Day,
            'julian': cftime.DatetimeJulian,
            'all_leap': cftime.DatetimeAllLeap,
            '366_day': cftime.DatetimeAllLeap,
            'gregorian': cftime.DatetimeGregorian,
            'proleptic_gregorian': cftime.DatetimeProlepticGregorian}


@pytest.mark.skipif(not has_cftime_or_netCDF4, reason='cftime not installed')
@pytest.mark.parametrize(['num_dates', 'units', 'calendar'],
                         _CF_DATETIME_TESTS)
def test_cf_datetime(num_dates, units, calendar):
    cftime = _import_cftime()
    expected = _ensure_naive_tz(
        cftime.num2date(num_dates, units, calendar))
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


@requires_cftime_or_netCDF4
def test_decode_cf_datetime_overflow():
    # checks for
    # https://github.com/pydata/pandas/issues/14068
    # https://github.com/pydata/xarray/issues/975

    from datetime import datetime
    units = 'days since 2000-01-01 00:00:00'

    # date after 2262 and before 1678
    days = (-117608, 95795)
    expected = (datetime(1677, 12, 31), datetime(2262, 4, 12))

    for i, day in enumerate(days):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Unable to decode time axis')
            result = coding.times.decode_cf_datetime(day, units)
        assert result == expected[i]


def test_decode_cf_datetime_non_standard_units():
    expected = pd.date_range(periods=100, start='1970-01-01', freq='h')
    # netCDFs from madis.noaa.gov use this format for their time units
    # they cannot be parsed by cftime, but pd.Timestamp works
    units = 'hours since 1-1-1970'
    actual = coding.times.decode_cf_datetime(np.arange(100), units)
    assert_array_equal(actual, expected)


@requires_cftime_or_netCDF4
def test_decode_cf_datetime_non_iso_strings():
    # datetime strings that are _almost_ ISO compliant but not quite,
    # but which netCDF4.num2date can still parse correctly
    expected = pd.date_range(periods=100, start='2000-01-01', freq='h')
    cases = [(np.arange(100), 'hours since 2000-01-01 0'),
             (np.arange(100), 'hours since 2000-1-1 0'),
             (np.arange(100), 'hours since 2000-01-01 0:00')]
    for num_dates, units in cases:
        actual = coding.times.decode_cf_datetime(num_dates, units)
        assert_array_equal(actual, expected)


@pytest.mark.skipif(not has_cftime_or_netCDF4, reason='cftime not installed')
@pytest.mark.parametrize(
    ['calendar', 'enable_cftimeindex'],
    product(coding.times._STANDARD_CALENDARS, [False, True]))
def test_decode_standard_calendar_inside_timestamp_range(
        calendar, enable_cftimeindex):
    if enable_cftimeindex:
        pytest.importorskip('cftime')

    cftime = _import_cftime()
    units = 'days since 0001-01-01'
    times = pd.date_range('2001-04-01-00', end='2001-04-30-23',
                          freq='H')
    noleap_time = cftime.date2num(times.to_pydatetime(), units,
                                  calendar=calendar)
    expected = times.values
    expected_dtype = np.dtype('M8[ns]')

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Unable to decode time axis')
        actual = coding.times.decode_cf_datetime(
            noleap_time, units, calendar=calendar,
            enable_cftimeindex=enable_cftimeindex)
    assert actual.dtype == expected_dtype
    abs_diff = abs(actual - expected)
    # once we no longer support versions of netCDF4 older than 1.1.5,
    # we could do this check with near microsecond accuracy:
    # https://github.com/Unidata/netcdf4-python/issues/355
    assert (abs_diff <= np.timedelta64(1, 's')).all()


@pytest.mark.skipif(not has_cftime_or_netCDF4, reason='cftime not installed')
@pytest.mark.parametrize(
    ['calendar', 'enable_cftimeindex'],
    product(_NON_STANDARD_CALENDARS, [False, True]))
def test_decode_non_standard_calendar_inside_timestamp_range(
        calendar, enable_cftimeindex):
    if enable_cftimeindex:
        pytest.importorskip('cftime')

    cftime = _import_cftime()
    units = 'days since 0001-01-01'
    times = pd.date_range('2001-04-01-00', end='2001-04-30-23',
                          freq='H')
    noleap_time = cftime.date2num(times.to_pydatetime(), units,
                                  calendar=calendar)
    if enable_cftimeindex:
        expected = cftime.num2date(noleap_time, units, calendar=calendar)
        expected_dtype = np.dtype('O')
    else:
        expected = times.values
        expected_dtype = np.dtype('M8[ns]')

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Unable to decode time axis')
        actual = coding.times.decode_cf_datetime(
            noleap_time, units, calendar=calendar,
            enable_cftimeindex=enable_cftimeindex)
    assert actual.dtype == expected_dtype
    abs_diff = abs(actual - expected)
    # once we no longer support versions of netCDF4 older than 1.1.5,
    # we could do this check with near microsecond accuracy:
    # https://github.com/Unidata/netcdf4-python/issues/355
    assert (abs_diff <= np.timedelta64(1, 's')).all()


@pytest.mark.skipif(not has_cftime_or_netCDF4, reason='cftime not installed')
@pytest.mark.parametrize(
    ['calendar', 'enable_cftimeindex'],
    product(_ALL_CALENDARS, [False, True]))
def test_decode_dates_outside_timestamp_range(
        calendar, enable_cftimeindex):
    from datetime import datetime

    if enable_cftimeindex:
        pytest.importorskip('cftime')

    cftime = _import_cftime()

    units = 'days since 0001-01-01'
    times = [datetime(1, 4, 1, h) for h in range(1, 5)]
    noleap_time = cftime.date2num(times, units, calendar=calendar)
    if enable_cftimeindex:
        expected = cftime.num2date(noleap_time, units, calendar=calendar,
                                   only_use_cftime_datetimes=True)
    else:
        expected = cftime.num2date(noleap_time, units, calendar=calendar)
    expected_date_type = type(expected[0])

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Unable to decode time axis')
        actual = coding.times.decode_cf_datetime(
            noleap_time, units, calendar=calendar,
            enable_cftimeindex=enable_cftimeindex)
    assert all(isinstance(value, expected_date_type) for value in actual)
    abs_diff = abs(actual - expected)
    # once we no longer support versions of netCDF4 older than 1.1.5,
    # we could do this check with near microsecond accuracy:
    # https://github.com/Unidata/netcdf4-python/issues/355
    assert (abs_diff <= np.timedelta64(1, 's')).all()


@pytest.mark.skipif(not has_cftime_or_netCDF4, reason='cftime not installed')
@pytest.mark.parametrize(
    ['calendar', 'enable_cftimeindex'],
    product(coding.times._STANDARD_CALENDARS, [False, True]))
def test_decode_standard_calendar_single_element_inside_timestamp_range(
        calendar, enable_cftimeindex):
    if enable_cftimeindex:
        pytest.importorskip('cftime')

    units = 'days since 0001-01-01'
    for num_time in [735368, [735368], [[735368]]]:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                                    'Unable to decode time axis')
            actual = coding.times.decode_cf_datetime(
                num_time, units, calendar=calendar,
                enable_cftimeindex=enable_cftimeindex)
        assert actual.dtype == np.dtype('M8[ns]')


@pytest.mark.skipif(not has_cftime_or_netCDF4, reason='cftime not installed')
@pytest.mark.parametrize(
    ['calendar', 'enable_cftimeindex'],
    product(_NON_STANDARD_CALENDARS, [False, True]))
def test_decode_non_standard_calendar_single_element_inside_timestamp_range(
        calendar, enable_cftimeindex):
    if enable_cftimeindex:
        pytest.importorskip('cftime')

    units = 'days since 0001-01-01'
    for num_time in [735368, [735368], [[735368]]]:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                                    'Unable to decode time axis')
            actual = coding.times.decode_cf_datetime(
                num_time, units, calendar=calendar,
                enable_cftimeindex=enable_cftimeindex)
        if enable_cftimeindex:
            assert actual.dtype == np.dtype('O')
        else:
            assert actual.dtype == np.dtype('M8[ns]')


@pytest.mark.skipif(not has_cftime_or_netCDF4, reason='cftime not installed')
@pytest.mark.parametrize(
    ['calendar', 'enable_cftimeindex'],
    product(_NON_STANDARD_CALENDARS, [False, True]))
def test_decode_single_element_outside_timestamp_range(
        calendar, enable_cftimeindex):
    if enable_cftimeindex:
        pytest.importorskip('cftime')

    cftime = _import_cftime()
    units = 'days since 0001-01-01'
    for days in [1, 1470376]:
        for num_time in [days, [days], [[days]]]:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',
                                        'Unable to decode time axis')
                actual = coding.times.decode_cf_datetime(
                    num_time, units, calendar=calendar,
                    enable_cftimeindex=enable_cftimeindex)
            expected = cftime.num2date(days, units, calendar)
            assert isinstance(actual.item(), type(expected))


@pytest.mark.skipif(not has_cftime_or_netCDF4, reason='cftime not installed')
@pytest.mark.parametrize(
    ['calendar', 'enable_cftimeindex'],
    product(coding.times._STANDARD_CALENDARS, [False, True]))
def test_decode_standard_calendar_multidim_time_inside_timestamp_range(
        calendar, enable_cftimeindex):
    if enable_cftimeindex:
        pytest.importorskip('cftime')

    cftime = _import_cftime()

    units = 'days since 0001-01-01'
    times1 = pd.date_range('2001-04-01', end='2001-04-05', freq='D')
    times2 = pd.date_range('2001-05-01', end='2001-05-05', freq='D')
    noleap_time1 = cftime.date2num(times1.to_pydatetime(),
                                   units, calendar=calendar)
    noleap_time2 = cftime.date2num(times2.to_pydatetime(),
                                   units, calendar=calendar)
    mdim_time = np.empty((len(noleap_time1), 2), )
    mdim_time[:, 0] = noleap_time1
    mdim_time[:, 1] = noleap_time2

    expected1 = times1.values
    expected2 = times2.values

    actual = coding.times.decode_cf_datetime(
        mdim_time, units, calendar=calendar,
        enable_cftimeindex=enable_cftimeindex)
    assert actual.dtype == np.dtype('M8[ns]')

    abs_diff1 = abs(actual[:, 0] - expected1)
    abs_diff2 = abs(actual[:, 1] - expected2)
    # once we no longer support versions of netCDF4 older than 1.1.5,
    # we could do this check with near microsecond accuracy:
    # https://github.com/Unidata/netcdf4-python/issues/355
    assert (abs_diff1 <= np.timedelta64(1, 's')).all()
    assert (abs_diff2 <= np.timedelta64(1, 's')).all()


@pytest.mark.skipif(not has_cftime_or_netCDF4, reason='cftime not installed')
@pytest.mark.parametrize(
    ['calendar', 'enable_cftimeindex'],
    product(_NON_STANDARD_CALENDARS, [False, True]))
def test_decode_nonstandard_calendar_multidim_time_inside_timestamp_range(
        calendar, enable_cftimeindex):
    if enable_cftimeindex:
        pytest.importorskip('cftime')

    cftime = _import_cftime()

    units = 'days since 0001-01-01'
    times1 = pd.date_range('2001-04-01', end='2001-04-05', freq='D')
    times2 = pd.date_range('2001-05-01', end='2001-05-05', freq='D')
    noleap_time1 = cftime.date2num(times1.to_pydatetime(),
                                   units, calendar=calendar)
    noleap_time2 = cftime.date2num(times2.to_pydatetime(),
                                   units, calendar=calendar)
    mdim_time = np.empty((len(noleap_time1), 2), )
    mdim_time[:, 0] = noleap_time1
    mdim_time[:, 1] = noleap_time2

    if enable_cftimeindex:
        expected1 = cftime.num2date(noleap_time1, units, calendar)
        expected2 = cftime.num2date(noleap_time2, units, calendar)
        expected_dtype = np.dtype('O')
    else:
        expected1 = times1.values
        expected2 = times2.values
        expected_dtype = np.dtype('M8[ns]')

    actual = coding.times.decode_cf_datetime(
        mdim_time, units, calendar=calendar,
        enable_cftimeindex=enable_cftimeindex)

    assert actual.dtype == expected_dtype
    abs_diff1 = abs(actual[:, 0] - expected1)
    abs_diff2 = abs(actual[:, 1] - expected2)
    # once we no longer support versions of netCDF4 older than 1.1.5,
    # we could do this check with near microsecond accuracy:
    # https://github.com/Unidata/netcdf4-python/issues/355
    assert (abs_diff1 <= np.timedelta64(1, 's')).all()
    assert (abs_diff2 <= np.timedelta64(1, 's')).all()


@pytest.mark.skipif(not has_cftime_or_netCDF4, reason='cftime not installed')
@pytest.mark.parametrize(
    ['calendar', 'enable_cftimeindex'],
    product(_ALL_CALENDARS, [False, True]))
def test_decode_multidim_time_outside_timestamp_range(
        calendar, enable_cftimeindex):
    from datetime import datetime

    if enable_cftimeindex:
        pytest.importorskip('cftime')

    cftime = _import_cftime()

    units = 'days since 0001-01-01'
    times1 = [datetime(1, 4, day) for day in range(1, 6)]
    times2 = [datetime(1, 5, day) for day in range(1, 6)]
    noleap_time1 = cftime.date2num(times1, units, calendar=calendar)
    noleap_time2 = cftime.date2num(times2, units, calendar=calendar)
    mdim_time = np.empty((len(noleap_time1), 2), )
    mdim_time[:, 0] = noleap_time1
    mdim_time[:, 1] = noleap_time2

    if enable_cftimeindex:
        expected1 = cftime.num2date(noleap_time1, units, calendar,
                                    only_use_cftime_datetimes=True)
        expected2 = cftime.num2date(noleap_time2, units, calendar,
                                    only_use_cftime_datetimes=True)
    else:
        expected1 = cftime.num2date(noleap_time1, units, calendar)
        expected2 = cftime.num2date(noleap_time2, units, calendar)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Unable to decode time axis')
        actual = coding.times.decode_cf_datetime(
            mdim_time, units, calendar=calendar,
            enable_cftimeindex=enable_cftimeindex)

    assert actual.dtype == np.dtype('O')

    abs_diff1 = abs(actual[:, 0] - expected1)
    abs_diff2 = abs(actual[:, 1] - expected2)
    # once we no longer support versions of netCDF4 older than 1.1.5,
    # we could do this check with near microsecond accuracy:
    # https://github.com/Unidata/netcdf4-python/issues/355
    assert (abs_diff1 <= np.timedelta64(1, 's')).all()
    assert (abs_diff2 <= np.timedelta64(1, 's')).all()


@pytest.mark.skipif(not has_cftime_or_netCDF4, reason='cftime not installed')
@pytest.mark.parametrize(
    ['calendar', 'enable_cftimeindex'],
    product(['360_day', 'all_leap', '366_day'], [False, True]))
def test_decode_non_standard_calendar_single_element_fallback(
        calendar, enable_cftimeindex):
    if enable_cftimeindex:
        pytest.importorskip('cftime')

    cftime = _import_cftime()

    units = 'days since 0001-01-01'
    try:
        dt = cftime.netcdftime.datetime(2001, 2, 29)
    except AttributeError:
        # Must be using standalone netcdftime library
        dt = cftime.datetime(2001, 2, 29)

    num_time = cftime.date2num(dt, units, calendar)
    if enable_cftimeindex:
        actual = coding.times.decode_cf_datetime(
            num_time, units, calendar=calendar,
            enable_cftimeindex=enable_cftimeindex)
    else:
        with pytest.warns(SerializationWarning,
                          match='Unable to decode time axis'):
            actual = coding.times.decode_cf_datetime(
                num_time, units, calendar=calendar,
                enable_cftimeindex=enable_cftimeindex)

    expected = np.asarray(cftime.num2date(num_time, units, calendar))
    assert actual.dtype == np.dtype('O')
    assert expected == actual


@pytest.mark.skipif(not has_cftime_or_netCDF4, reason='cftime not installed')
@pytest.mark.parametrize(
    ['calendar', 'enable_cftimeindex'],
    product(['360_day'], [False, True]))
def test_decode_non_standard_calendar_fallback(
        calendar, enable_cftimeindex):
    if enable_cftimeindex:
        pytest.importorskip('cftime')

    cftime = _import_cftime()
    # ensure leap year doesn't matter
    for year in [2010, 2011, 2012, 2013, 2014]:
        units = 'days since {0}-01-01'.format(year)
        num_times = np.arange(100)
        expected = cftime.num2date(num_times, units, calendar)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            actual = coding.times.decode_cf_datetime(
                num_times, units, calendar=calendar,
                enable_cftimeindex=enable_cftimeindex)
            if enable_cftimeindex:
                assert len(w) == 0
            else:
                assert len(w) == 1
                assert 'Unable to decode time axis' in str(w[0].message)

        assert actual.dtype == np.dtype('O')
        assert_array_equal(actual, expected)


@pytest.mark.skipif(not has_cftime_or_netCDF4, reason='cftime not installed')
@pytest.mark.parametrize(
    ['num_dates', 'units', 'expected_list'],
    [([np.nan], 'days since 2000-01-01', ['NaT']),
     ([np.nan, 0], 'days since 2000-01-01',
      ['NaT', '2000-01-01T00:00:00Z']),
     ([np.nan, 0, 1], 'days since 2000-01-01',
      ['NaT', '2000-01-01T00:00:00Z', '2000-01-02T00:00:00Z'])])
def test_cf_datetime_nan(num_dates, units, expected_list):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'All-NaN')
        actual = coding.times.decode_cf_datetime(num_dates, units)
    expected = np.array(expected_list, dtype='datetime64[ns]')
    assert_array_equal(expected, actual)


@requires_cftime_or_netCDF4
def test_decoded_cf_datetime_array_2d():
    # regression test for GH1229
    variable = Variable(('x', 'y'), np.array([[0, 1], [2, 3]]),
                        {'units': 'days since 2000-01-01'})
    result = coding.times.CFDatetimeCoder().decode(variable)
    assert result.dtype == 'datetime64[ns]'
    expected = pd.date_range('2000-01-01', periods=4).values.reshape(2, 2)
    assert_array_equal(np.asarray(result), expected)


@pytest.mark.parametrize(
    ['dates', 'expected'],
    [(pd.date_range('1900-01-01', periods=5),
      'days since 1900-01-01 00:00:00'),
     (pd.date_range('1900-01-01 12:00:00', freq='H',
                    periods=2),
      'hours since 1900-01-01 12:00:00'),
     (pd.to_datetime(
         ['1900-01-01', '1900-01-02', 'NaT']),
      'days since 1900-01-01 00:00:00'),
     (pd.to_datetime(['1900-01-01',
                      '1900-01-02T00:00:00.005']),
      'seconds since 1900-01-01 00:00:00'),
     (pd.to_datetime(['NaT', '1900-01-01']),
      'days since 1900-01-01 00:00:00'),
     (pd.to_datetime(['NaT']),
      'days since 1970-01-01 00:00:00')])
def test_infer_datetime_units(dates, expected):
    assert expected == coding.times.infer_datetime_units(dates)


@pytest.mark.skipif(not has_cftime_or_netCDF4, reason='cftime not installed')
def test_infer_cftime_datetime_units():
    date_types = _all_cftime_date_types()
    for date_type in date_types.values():
        for dates, expected in [
                ([date_type(1900, 1, 1),
                  date_type(1900, 1, 2)],
                 'days since 1900-01-01 00:00:00.000000'),
                ([date_type(1900, 1, 1, 12),
                  date_type(1900, 1, 1, 13)],
                 'seconds since 1900-01-01 12:00:00.000000'),
                ([date_type(1900, 1, 1),
                  date_type(1900, 1, 2),
                  date_type(1900, 1, 2, 0, 0, 1)],
                 'seconds since 1900-01-01 00:00:00.000000'),
                ([date_type(1900, 1, 1),
                  date_type(1900, 1, 2, 0, 0, 0, 5)],
                 'days since 1900-01-01 00:00:00.000000'),
                ([date_type(1900, 1, 1), date_type(1900, 1, 8),
                  date_type(1900, 1, 16)],
                 'days since 1900-01-01 00:00:00.000000')]:
            assert expected == coding.times.infer_datetime_units(dates)


@pytest.mark.parametrize(
    ['timedeltas', 'units', 'numbers'],
    [('1D', 'days', np.int64(1)),
     (['1D', '2D', '3D'], 'days', np.array([1, 2, 3], 'int64')),
     ('1h', 'hours', np.int64(1)),
     ('1ms', 'milliseconds', np.int64(1)),
     ('1us', 'microseconds', np.int64(1)),
     (['NaT', '0s', '1s'], None, [np.nan, 0, 1]),
     (['30m', '60m'], 'hours', [0.5, 1.0]),
     (np.timedelta64('NaT', 'ns'), 'days', np.nan),
     (['NaT', 'NaT'], 'days', [np.nan, np.nan])])
def test_cf_timedelta(timedeltas, units, numbers):
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


def test_cf_timedelta_2d():
    timedeltas = ['1D', '2D', '3D']
    units = 'days'
    numbers = np.atleast_2d([1, 2, 3])

    timedeltas = np.atleast_2d(pd.to_timedelta(timedeltas, box=False))
    expected = timedeltas

    actual = coding.times.decode_cf_timedelta(numbers, units)
    assert_array_equal(expected, actual)
    assert expected.dtype == actual.dtype


@pytest.mark.parametrize(
    ['deltas', 'expected'],
    [(pd.to_timedelta(['1 day', '2 days']), 'days'),
     (pd.to_timedelta(['1h', '1 day 1 hour']), 'hours'),
     (pd.to_timedelta(['1m', '2m', np.nan]), 'minutes'),
     (pd.to_timedelta(['1m3s', '1m4s']), 'seconds')])
def test_infer_timedelta_units(deltas, expected):
    assert expected == coding.times.infer_timedelta_units(deltas)


@pytest.mark.skipif(not has_cftime_or_netCDF4, reason='cftime not installed')
@pytest.mark.parametrize(['date_args', 'expected'],
                         [((1, 2, 3, 4, 5, 6),
                          '0001-02-03 04:05:06.000000'),
                          ((10, 2, 3, 4, 5, 6),
                           '0010-02-03 04:05:06.000000'),
                          ((100, 2, 3, 4, 5, 6),
                           '0100-02-03 04:05:06.000000'),
                          ((1000, 2, 3, 4, 5, 6),
                           '1000-02-03 04:05:06.000000')])
def test_format_cftime_datetime(date_args, expected):
    date_types = _all_cftime_date_types()
    for date_type in date_types.values():
        result = coding.times.format_cftime_datetime(date_type(*date_args))
        assert result == expected


@pytest.mark.skipif(not has_cftime_or_netCDF4, reason='cftime not installed')
@pytest.mark.parametrize(
    ['calendar', 'enable_cftimeindex'],
    product(_ALL_CALENDARS, [False, True]))
def test_decode_cf_enable_cftimeindex(calendar, enable_cftimeindex):
    days = [1., 2., 3.]
    da = DataArray(days, coords=[days], dims=['time'], name='test')
    ds = da.to_dataset()

    for v in ['test', 'time']:
        ds[v].attrs['units'] = 'days since 2001-01-01'
        ds[v].attrs['calendar'] = calendar

    if (not has_cftime and enable_cftimeindex and
       calendar not in coding.times._STANDARD_CALENDARS):
        with pytest.raises(ValueError):
            with set_options(enable_cftimeindex=enable_cftimeindex):
                ds = decode_cf(ds)
    else:
        with set_options(enable_cftimeindex=enable_cftimeindex):
            ds = decode_cf(ds)

        if (enable_cftimeindex and
           calendar not in coding.times._STANDARD_CALENDARS):
            assert ds.test.dtype == np.dtype('O')
        else:
            assert ds.test.dtype == np.dtype('M8[ns]')


@pytest.fixture(params=_ALL_CALENDARS)
def calendar(request):
    return request.param


@pytest.fixture()
def times(calendar):
    cftime = _import_cftime()

    return cftime.num2date(
        np.arange(4), units='hours since 2000-01-01', calendar=calendar,
        only_use_cftime_datetimes=True)


@pytest.fixture()
def data(times):
    data = np.random.rand(2, 2, 4)
    lons = np.linspace(0, 11, 2)
    lats = np.linspace(0, 20, 2)
    return DataArray(data, coords=[lons, lats, times],
                     dims=['lon', 'lat', 'time'], name='data')


@pytest.fixture()
def times_3d(times):
    lons = np.linspace(0, 11, 2)
    lats = np.linspace(0, 20, 2)
    times_arr = np.random.choice(times, size=(2, 2, 4))
    return DataArray(times_arr, coords=[lons, lats, times],
                     dims=['lon', 'lat', 'time'],
                     name='data')


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
def test_contains_cftime_datetimes_1d(data):
    assert contains_cftime_datetimes(data.time)


@pytest.mark.skipif(not has_dask, reason='dask not installed')
@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
def test_contains_cftime_datetimes_dask_1d(data):
    assert contains_cftime_datetimes(data.time.chunk())


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
def test_contains_cftime_datetimes_3d(times_3d):
    assert contains_cftime_datetimes(times_3d)


@pytest.mark.skipif(not has_dask, reason='dask not installed')
@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
def test_contains_cftime_datetimes_dask_3d(times_3d):
    assert contains_cftime_datetimes(times_3d.chunk())


@pytest.mark.parametrize('non_cftime_data', [DataArray([]), DataArray([1, 2])])
def test_contains_cftime_datetimes_non_cftimes(non_cftime_data):
    assert not contains_cftime_datetimes(non_cftime_data)


@pytest.mark.skipif(not has_dask, reason='dask not installed')
@pytest.mark.parametrize('non_cftime_data', [DataArray([]), DataArray([1, 2])])
def test_contains_cftime_datetimes_non_cftimes_dask(non_cftime_data):
    assert not contains_cftime_datetimes(non_cftime_data.chunk())
