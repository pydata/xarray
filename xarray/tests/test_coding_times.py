import warnings
from itertools import product

import numpy as np
import pandas as pd
import pytest
from pandas.errors import OutOfBoundsDatetime

from xarray import DataArray, Dataset, Variable, coding, decode_cf
from xarray.coding.times import (
    cftime_to_nptime,
    decode_cf_datetime,
    encode_cf_datetime,
    to_timedelta_unboxed,
)
from xarray.coding.variables import SerializationWarning
from xarray.conventions import _update_bounds_attributes, cf_encoder
from xarray.core.common import contains_cftime_datetimes
from xarray.testing import assert_equal

from . import arm_xfail, assert_array_equal, has_cftime, requires_cftime, requires_dask

_NON_STANDARD_CALENDARS_SET = {
    "noleap",
    "365_day",
    "360_day",
    "julian",
    "all_leap",
    "366_day",
}
_ALL_CALENDARS = sorted(
    _NON_STANDARD_CALENDARS_SET.union(coding.times._STANDARD_CALENDARS)
)
_NON_STANDARD_CALENDARS = sorted(_NON_STANDARD_CALENDARS_SET)
_STANDARD_CALENDARS = sorted(coding.times._STANDARD_CALENDARS)
_CF_DATETIME_NUM_DATES_UNITS = [
    (np.arange(10), "days since 2000-01-01"),
    (np.arange(10).astype("float64"), "days since 2000-01-01"),
    (np.arange(10).astype("float32"), "days since 2000-01-01"),
    (np.arange(10).reshape(2, 5), "days since 2000-01-01"),
    (12300 + np.arange(5), "hours since 1680-01-01 00:00:00"),
    # here we add a couple minor formatting errors to test
    # the robustness of the parsing algorithm.
    (12300 + np.arange(5), "hour since 1680-01-01  00:00:00"),
    (12300 + np.arange(5), "Hour  since 1680-01-01 00:00:00"),
    (12300 + np.arange(5), " Hour  since  1680-01-01 00:00:00 "),
    (10, "days since 2000-01-01"),
    ([10], "daYs  since 2000-01-01"),
    ([[10]], "days since 2000-01-01"),
    ([10, 10], "days since 2000-01-01"),
    (np.array(10), "days since 2000-01-01"),
    (0, "days since 1000-01-01"),
    ([0], "days since 1000-01-01"),
    ([[0]], "days since 1000-01-01"),
    (np.arange(2), "days since 1000-01-01"),
    (np.arange(0, 100000, 20000), "days since 1900-01-01"),
    (17093352.0, "hours since 1-1-1 00:00:0.0"),
    ([0.5, 1.5], "hours since 1900-01-01T00:00:00"),
    (0, "milliseconds since 2000-01-01T00:00:00"),
    (0, "microseconds since 2000-01-01T00:00:00"),
    (np.int32(788961600), "seconds since 1981-01-01"),  # GH2002
    (12300 + np.arange(5), "hour since 1680-01-01 00:00:00.500000"),
]
_CF_DATETIME_TESTS = [
    num_dates_units + (calendar,)
    for num_dates_units, calendar in product(
        _CF_DATETIME_NUM_DATES_UNITS, _STANDARD_CALENDARS
    )
]


def _all_cftime_date_types():
    import cftime

    return {
        "noleap": cftime.DatetimeNoLeap,
        "365_day": cftime.DatetimeNoLeap,
        "360_day": cftime.Datetime360Day,
        "julian": cftime.DatetimeJulian,
        "all_leap": cftime.DatetimeAllLeap,
        "366_day": cftime.DatetimeAllLeap,
        "gregorian": cftime.DatetimeGregorian,
        "proleptic_gregorian": cftime.DatetimeProlepticGregorian,
    }


@requires_cftime
@pytest.mark.parametrize(["num_dates", "units", "calendar"], _CF_DATETIME_TESTS)
def test_cf_datetime(num_dates, units, calendar):
    import cftime

    expected = cftime.num2date(
        num_dates, units, calendar, only_use_cftime_datetimes=True
    )
    min_y = np.ravel(np.atleast_1d(expected))[np.nanargmin(num_dates)].year
    max_y = np.ravel(np.atleast_1d(expected))[np.nanargmax(num_dates)].year
    if min_y >= 1678 and max_y < 2262:
        expected = cftime_to_nptime(expected)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Unable to decode time axis")
        actual = coding.times.decode_cf_datetime(num_dates, units, calendar)

    abs_diff = np.asarray(abs(actual - expected)).ravel()
    abs_diff = pd.to_timedelta(abs_diff.tolist()).to_numpy()

    # once we no longer support versions of netCDF4 older than 1.1.5,
    # we could do this check with near microsecond accuracy:
    # https://github.com/Unidata/netcdf4-python/issues/355
    assert (abs_diff <= np.timedelta64(1, "s")).all()
    encoded, _, _ = coding.times.encode_cf_datetime(actual, units, calendar)
    if "1-1-1" not in units:
        # pandas parses this date very strangely, so the original
        # units/encoding cannot be preserved in this case:
        # (Pdb) pd.to_datetime('1-1-1 00:00:0.0')
        # Timestamp('2001-01-01 00:00:00')
        assert_array_equal(num_dates, np.around(encoded, 1))
        if hasattr(num_dates, "ndim") and num_dates.ndim == 1 and "1000" not in units:
            # verify that wrapping with a pandas.Index works
            # note that it *does not* currently work to even put
            # non-datetime64 compatible dates into a pandas.Index
            encoded, _, _ = coding.times.encode_cf_datetime(
                pd.Index(actual), units, calendar
            )
            assert_array_equal(num_dates, np.around(encoded, 1))


@requires_cftime
def test_decode_cf_datetime_overflow():
    # checks for
    # https://github.com/pydata/pandas/issues/14068
    # https://github.com/pydata/xarray/issues/975
    from cftime import DatetimeGregorian

    datetime = DatetimeGregorian
    units = "days since 2000-01-01 00:00:00"

    # date after 2262 and before 1678
    days = (-117608, 95795)
    expected = (datetime(1677, 12, 31), datetime(2262, 4, 12))

    for i, day in enumerate(days):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Unable to decode time axis")
            result = coding.times.decode_cf_datetime(day, units)
        assert result == expected[i]


def test_decode_cf_datetime_non_standard_units():
    expected = pd.date_range(periods=100, start="1970-01-01", freq="h")
    # netCDFs from madis.noaa.gov use this format for their time units
    # they cannot be parsed by cftime, but pd.Timestamp works
    units = "hours since 1-1-1970"
    actual = coding.times.decode_cf_datetime(np.arange(100), units)
    assert_array_equal(actual, expected)


@requires_cftime
def test_decode_cf_datetime_non_iso_strings():
    # datetime strings that are _almost_ ISO compliant but not quite,
    # but which cftime.num2date can still parse correctly
    expected = pd.date_range(periods=100, start="2000-01-01", freq="h")
    cases = [
        (np.arange(100), "hours since 2000-01-01 0"),
        (np.arange(100), "hours since 2000-1-1 0"),
        (np.arange(100), "hours since 2000-01-01 0:00"),
    ]
    for num_dates, units in cases:
        actual = coding.times.decode_cf_datetime(num_dates, units)
        abs_diff = abs(actual - expected.values)
        # once we no longer support versions of netCDF4 older than 1.1.5,
        # we could do this check with near microsecond accuracy:
        # https://github.com/Unidata/netcdf4-python/issues/355
        assert (abs_diff <= np.timedelta64(1, "s")).all()


@requires_cftime
@pytest.mark.parametrize("calendar", _STANDARD_CALENDARS)
def test_decode_standard_calendar_inside_timestamp_range(calendar):
    import cftime

    units = "days since 0001-01-01"
    times = pd.date_range("2001-04-01-00", end="2001-04-30-23", freq="H")
    time = cftime.date2num(times.to_pydatetime(), units, calendar=calendar)
    expected = times.values
    expected_dtype = np.dtype("M8[ns]")

    actual = coding.times.decode_cf_datetime(time, units, calendar=calendar)
    assert actual.dtype == expected_dtype
    abs_diff = abs(actual - expected)
    # once we no longer support versions of netCDF4 older than 1.1.5,
    # we could do this check with near microsecond accuracy:
    # https://github.com/Unidata/netcdf4-python/issues/355
    assert (abs_diff <= np.timedelta64(1, "s")).all()


@requires_cftime
@pytest.mark.parametrize("calendar", _NON_STANDARD_CALENDARS)
def test_decode_non_standard_calendar_inside_timestamp_range(calendar):
    import cftime

    units = "days since 0001-01-01"
    times = pd.date_range("2001-04-01-00", end="2001-04-30-23", freq="H")
    non_standard_time = cftime.date2num(times.to_pydatetime(), units, calendar=calendar)

    expected = cftime.num2date(
        non_standard_time, units, calendar=calendar, only_use_cftime_datetimes=True
    )
    expected_dtype = np.dtype("O")

    actual = coding.times.decode_cf_datetime(
        non_standard_time, units, calendar=calendar
    )
    assert actual.dtype == expected_dtype
    abs_diff = abs(actual - expected)
    # once we no longer support versions of netCDF4 older than 1.1.5,
    # we could do this check with near microsecond accuracy:
    # https://github.com/Unidata/netcdf4-python/issues/355
    assert (abs_diff <= np.timedelta64(1, "s")).all()


@requires_cftime
@pytest.mark.parametrize("calendar", _ALL_CALENDARS)
def test_decode_dates_outside_timestamp_range(calendar):
    import cftime
    from datetime import datetime

    units = "days since 0001-01-01"
    times = [datetime(1, 4, 1, h) for h in range(1, 5)]
    time = cftime.date2num(times, units, calendar=calendar)

    expected = cftime.num2date(
        time, units, calendar=calendar, only_use_cftime_datetimes=True
    )
    expected_date_type = type(expected[0])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Unable to decode time axis")
        actual = coding.times.decode_cf_datetime(time, units, calendar=calendar)
    assert all(isinstance(value, expected_date_type) for value in actual)
    abs_diff = abs(actual - expected)
    # once we no longer support versions of netCDF4 older than 1.1.5,
    # we could do this check with near microsecond accuracy:
    # https://github.com/Unidata/netcdf4-python/issues/355
    assert (abs_diff <= np.timedelta64(1, "s")).all()


@requires_cftime
@pytest.mark.parametrize("calendar", _STANDARD_CALENDARS)
def test_decode_standard_calendar_single_element_inside_timestamp_range(calendar):
    units = "days since 0001-01-01"
    for num_time in [735368, [735368], [[735368]]]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Unable to decode time axis")
            actual = coding.times.decode_cf_datetime(num_time, units, calendar=calendar)
        assert actual.dtype == np.dtype("M8[ns]")


@requires_cftime
@pytest.mark.parametrize("calendar", _NON_STANDARD_CALENDARS)
def test_decode_non_standard_calendar_single_element_inside_timestamp_range(calendar):
    units = "days since 0001-01-01"
    for num_time in [735368, [735368], [[735368]]]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Unable to decode time axis")
            actual = coding.times.decode_cf_datetime(num_time, units, calendar=calendar)
        assert actual.dtype == np.dtype("O")


@requires_cftime
@pytest.mark.parametrize("calendar", _NON_STANDARD_CALENDARS)
def test_decode_single_element_outside_timestamp_range(calendar):
    import cftime

    units = "days since 0001-01-01"
    for days in [1, 1470376]:
        for num_time in [days, [days], [[days]]]:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Unable to decode time axis")
                actual = coding.times.decode_cf_datetime(
                    num_time, units, calendar=calendar
                )

            expected = cftime.num2date(
                days, units, calendar, only_use_cftime_datetimes=True
            )
            assert isinstance(actual.item(), type(expected))


@requires_cftime
@pytest.mark.parametrize("calendar", _STANDARD_CALENDARS)
def test_decode_standard_calendar_multidim_time_inside_timestamp_range(calendar):
    import cftime

    units = "days since 0001-01-01"
    times1 = pd.date_range("2001-04-01", end="2001-04-05", freq="D")
    times2 = pd.date_range("2001-05-01", end="2001-05-05", freq="D")
    time1 = cftime.date2num(times1.to_pydatetime(), units, calendar=calendar)
    time2 = cftime.date2num(times2.to_pydatetime(), units, calendar=calendar)
    mdim_time = np.empty((len(time1), 2))
    mdim_time[:, 0] = time1
    mdim_time[:, 1] = time2

    expected1 = times1.values
    expected2 = times2.values

    actual = coding.times.decode_cf_datetime(mdim_time, units, calendar=calendar)
    assert actual.dtype == np.dtype("M8[ns]")

    abs_diff1 = abs(actual[:, 0] - expected1)
    abs_diff2 = abs(actual[:, 1] - expected2)
    # once we no longer support versions of netCDF4 older than 1.1.5,
    # we could do this check with near microsecond accuracy:
    # https://github.com/Unidata/netcdf4-python/issues/355
    assert (abs_diff1 <= np.timedelta64(1, "s")).all()
    assert (abs_diff2 <= np.timedelta64(1, "s")).all()


@requires_cftime
@pytest.mark.parametrize("calendar", _NON_STANDARD_CALENDARS)
def test_decode_nonstandard_calendar_multidim_time_inside_timestamp_range(calendar):
    import cftime

    units = "days since 0001-01-01"
    times1 = pd.date_range("2001-04-01", end="2001-04-05", freq="D")
    times2 = pd.date_range("2001-05-01", end="2001-05-05", freq="D")
    time1 = cftime.date2num(times1.to_pydatetime(), units, calendar=calendar)
    time2 = cftime.date2num(times2.to_pydatetime(), units, calendar=calendar)
    mdim_time = np.empty((len(time1), 2))
    mdim_time[:, 0] = time1
    mdim_time[:, 1] = time2

    if cftime.__name__ == "cftime":
        expected1 = cftime.num2date(
            time1, units, calendar, only_use_cftime_datetimes=True
        )
        expected2 = cftime.num2date(
            time2, units, calendar, only_use_cftime_datetimes=True
        )
    else:
        expected1 = cftime.num2date(time1, units, calendar)
        expected2 = cftime.num2date(time2, units, calendar)

    expected_dtype = np.dtype("O")

    actual = coding.times.decode_cf_datetime(mdim_time, units, calendar=calendar)

    assert actual.dtype == expected_dtype
    abs_diff1 = abs(actual[:, 0] - expected1)
    abs_diff2 = abs(actual[:, 1] - expected2)
    # once we no longer support versions of netCDF4 older than 1.1.5,
    # we could do this check with near microsecond accuracy:
    # https://github.com/Unidata/netcdf4-python/issues/355
    assert (abs_diff1 <= np.timedelta64(1, "s")).all()
    assert (abs_diff2 <= np.timedelta64(1, "s")).all()


@requires_cftime
@pytest.mark.parametrize("calendar", _ALL_CALENDARS)
def test_decode_multidim_time_outside_timestamp_range(calendar):
    import cftime
    from datetime import datetime

    units = "days since 0001-01-01"
    times1 = [datetime(1, 4, day) for day in range(1, 6)]
    times2 = [datetime(1, 5, day) for day in range(1, 6)]
    time1 = cftime.date2num(times1, units, calendar=calendar)
    time2 = cftime.date2num(times2, units, calendar=calendar)
    mdim_time = np.empty((len(time1), 2))
    mdim_time[:, 0] = time1
    mdim_time[:, 1] = time2

    expected1 = cftime.num2date(time1, units, calendar, only_use_cftime_datetimes=True)
    expected2 = cftime.num2date(time2, units, calendar, only_use_cftime_datetimes=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Unable to decode time axis")
        actual = coding.times.decode_cf_datetime(mdim_time, units, calendar=calendar)

    assert actual.dtype == np.dtype("O")

    abs_diff1 = abs(actual[:, 0] - expected1)
    abs_diff2 = abs(actual[:, 1] - expected2)
    # once we no longer support versions of netCDF4 older than 1.1.5,
    # we could do this check with near microsecond accuracy:
    # https://github.com/Unidata/netcdf4-python/issues/355
    assert (abs_diff1 <= np.timedelta64(1, "s")).all()
    assert (abs_diff2 <= np.timedelta64(1, "s")).all()


@requires_cftime
@pytest.mark.parametrize("calendar", ["360_day", "all_leap", "366_day"])
def test_decode_non_standard_calendar_single_element(calendar):
    import cftime

    units = "days since 0001-01-01"

    dt = cftime.datetime(2001, 2, 29)

    num_time = cftime.date2num(dt, units, calendar)
    actual = coding.times.decode_cf_datetime(num_time, units, calendar=calendar)

    expected = np.asarray(
        cftime.num2date(num_time, units, calendar, only_use_cftime_datetimes=True)
    )
    assert actual.dtype == np.dtype("O")
    assert expected == actual


@requires_cftime
def test_decode_360_day_calendar():
    import cftime

    calendar = "360_day"
    # ensure leap year doesn't matter
    for year in [2010, 2011, 2012, 2013, 2014]:
        units = f"days since {year}-01-01"
        num_times = np.arange(100)

        expected = cftime.num2date(
            num_times, units, calendar, only_use_cftime_datetimes=True
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            actual = coding.times.decode_cf_datetime(
                num_times, units, calendar=calendar
            )
            assert len(w) == 0

        assert actual.dtype == np.dtype("O")
        assert_array_equal(actual, expected)


@arm_xfail
@requires_cftime
@pytest.mark.parametrize(
    ["num_dates", "units", "expected_list"],
    [
        ([np.nan], "days since 2000-01-01", ["NaT"]),
        ([np.nan, 0], "days since 2000-01-01", ["NaT", "2000-01-01T00:00:00Z"]),
        (
            [np.nan, 0, 1],
            "days since 2000-01-01",
            ["NaT", "2000-01-01T00:00:00Z", "2000-01-02T00:00:00Z"],
        ),
    ],
)
def test_cf_datetime_nan(num_dates, units, expected_list):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN")
        actual = coding.times.decode_cf_datetime(num_dates, units)
    # use pandas because numpy will deprecate timezone-aware conversions
    expected = pd.to_datetime(expected_list).to_numpy(dtype="datetime64[ns]")
    assert_array_equal(expected, actual)


@requires_cftime
def test_decoded_cf_datetime_array_2d():
    # regression test for GH1229
    variable = Variable(
        ("x", "y"), np.array([[0, 1], [2, 3]]), {"units": "days since 2000-01-01"}
    )
    result = coding.times.CFDatetimeCoder().decode(variable)
    assert result.dtype == "datetime64[ns]"
    expected = pd.date_range("2000-01-01", periods=4).values.reshape(2, 2)
    assert_array_equal(np.asarray(result), expected)


@pytest.mark.parametrize(
    ["dates", "expected"],
    [
        (pd.date_range("1900-01-01", periods=5), "days since 1900-01-01 00:00:00"),
        (
            pd.date_range("1900-01-01 12:00:00", freq="H", periods=2),
            "hours since 1900-01-01 12:00:00",
        ),
        (
            pd.to_datetime(["1900-01-01", "1900-01-02", "NaT"]),
            "days since 1900-01-01 00:00:00",
        ),
        (
            pd.to_datetime(["1900-01-01", "1900-01-02T00:00:00.005"]),
            "seconds since 1900-01-01 00:00:00",
        ),
        (pd.to_datetime(["NaT", "1900-01-01"]), "days since 1900-01-01 00:00:00"),
        (pd.to_datetime(["NaT"]), "days since 1970-01-01 00:00:00"),
    ],
)
def test_infer_datetime_units(dates, expected):
    assert expected == coding.times.infer_datetime_units(dates)


_CFTIME_DATETIME_UNITS_TESTS = [
    ([(1900, 1, 1), (1900, 1, 1)], "days since 1900-01-01 00:00:00.000000"),
    (
        [(1900, 1, 1), (1900, 1, 2), (1900, 1, 2, 0, 0, 1)],
        "seconds since 1900-01-01 00:00:00.000000",
    ),
    (
        [(1900, 1, 1), (1900, 1, 8), (1900, 1, 16)],
        "days since 1900-01-01 00:00:00.000000",
    ),
]


@requires_cftime
@pytest.mark.parametrize(
    "calendar", _NON_STANDARD_CALENDARS + ["gregorian", "proleptic_gregorian"]
)
@pytest.mark.parametrize(("date_args", "expected"), _CFTIME_DATETIME_UNITS_TESTS)
def test_infer_cftime_datetime_units(calendar, date_args, expected):
    date_type = _all_cftime_date_types()[calendar]
    dates = [date_type(*args) for args in date_args]
    assert expected == coding.times.infer_datetime_units(dates)


@pytest.mark.parametrize(
    ["timedeltas", "units", "numbers"],
    [
        ("1D", "days", np.int64(1)),
        (["1D", "2D", "3D"], "days", np.array([1, 2, 3], "int64")),
        ("1h", "hours", np.int64(1)),
        ("1ms", "milliseconds", np.int64(1)),
        ("1us", "microseconds", np.int64(1)),
        (["NaT", "0s", "1s"], None, [np.nan, 0, 1]),
        (["30m", "60m"], "hours", [0.5, 1.0]),
        ("NaT", "days", np.nan),
        (["NaT", "NaT"], "days", [np.nan, np.nan]),
    ],
)
def test_cf_timedelta(timedeltas, units, numbers):
    if timedeltas == "NaT":
        timedeltas = np.timedelta64("NaT", "ns")
    else:
        timedeltas = to_timedelta_unboxed(timedeltas)
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

    expected = np.timedelta64("NaT", "ns")
    actual = coding.times.decode_cf_timedelta(np.array(np.nan), "days")
    assert_array_equal(expected, actual)


def test_cf_timedelta_2d():
    timedeltas = ["1D", "2D", "3D"]
    units = "days"
    numbers = np.atleast_2d([1, 2, 3])

    timedeltas = np.atleast_2d(to_timedelta_unboxed(timedeltas))
    expected = timedeltas

    actual = coding.times.decode_cf_timedelta(numbers, units)
    assert_array_equal(expected, actual)
    assert expected.dtype == actual.dtype


@pytest.mark.parametrize(
    ["deltas", "expected"],
    [
        (pd.to_timedelta(["1 day", "2 days"]), "days"),
        (pd.to_timedelta(["1h", "1 day 1 hour"]), "hours"),
        (pd.to_timedelta(["1m", "2m", np.nan]), "minutes"),
        (pd.to_timedelta(["1m3s", "1m4s"]), "seconds"),
    ],
)
def test_infer_timedelta_units(deltas, expected):
    assert expected == coding.times.infer_timedelta_units(deltas)


@requires_cftime
@pytest.mark.parametrize(
    ["date_args", "expected"],
    [
        ((1, 2, 3, 4, 5, 6), "0001-02-03 04:05:06.000000"),
        ((10, 2, 3, 4, 5, 6), "0010-02-03 04:05:06.000000"),
        ((100, 2, 3, 4, 5, 6), "0100-02-03 04:05:06.000000"),
        ((1000, 2, 3, 4, 5, 6), "1000-02-03 04:05:06.000000"),
    ],
)
def test_format_cftime_datetime(date_args, expected):
    date_types = _all_cftime_date_types()
    for date_type in date_types.values():
        result = coding.times.format_cftime_datetime(date_type(*date_args))
        assert result == expected


@pytest.mark.parametrize("calendar", _ALL_CALENDARS)
def test_decode_cf(calendar):
    days = [1.0, 2.0, 3.0]
    da = DataArray(days, coords=[days], dims=["time"], name="test")
    ds = da.to_dataset()

    for v in ["test", "time"]:
        ds[v].attrs["units"] = "days since 2001-01-01"
        ds[v].attrs["calendar"] = calendar

    if not has_cftime and calendar not in _STANDARD_CALENDARS:
        with pytest.raises(ValueError):
            ds = decode_cf(ds)
    else:
        ds = decode_cf(ds)

        if calendar not in _STANDARD_CALENDARS:
            assert ds.test.dtype == np.dtype("O")
        else:
            assert ds.test.dtype == np.dtype("M8[ns]")


def test_decode_cf_time_bounds():

    da = DataArray(
        np.arange(6, dtype="int64").reshape((3, 2)),
        coords={"time": [1, 2, 3]},
        dims=("time", "nbnd"),
        name="time_bnds",
    )

    attrs = {
        "units": "days since 2001-01",
        "calendar": "standard",
        "bounds": "time_bnds",
    }

    ds = da.to_dataset()
    ds["time"].attrs.update(attrs)
    _update_bounds_attributes(ds.variables)
    assert ds.variables["time_bnds"].attrs == {
        "units": "days since 2001-01",
        "calendar": "standard",
    }
    dsc = decode_cf(ds)
    assert dsc.time_bnds.dtype == np.dtype("M8[ns]")
    dsc = decode_cf(ds, decode_times=False)
    assert dsc.time_bnds.dtype == np.dtype("int64")

    # Do not overwrite existing attrs
    ds = da.to_dataset()
    ds["time"].attrs.update(attrs)
    bnd_attr = {"units": "hours since 2001-01", "calendar": "noleap"}
    ds["time_bnds"].attrs.update(bnd_attr)
    _update_bounds_attributes(ds.variables)
    assert ds.variables["time_bnds"].attrs == bnd_attr

    # If bounds variable not available do not complain
    ds = da.to_dataset()
    ds["time"].attrs.update(attrs)
    ds["time"].attrs["bounds"] = "fake_var"
    _update_bounds_attributes(ds.variables)


@requires_cftime
def test_encode_time_bounds():

    time = pd.date_range("2000-01-16", periods=1)
    time_bounds = pd.date_range("2000-01-01", periods=2, freq="MS")
    ds = Dataset(dict(time=time, time_bounds=time_bounds))
    ds.time.attrs = {"bounds": "time_bounds"}
    ds.time.encoding = {"calendar": "noleap", "units": "days since 2000-01-01"}

    expected = {}
    # expected['time'] = Variable(data=np.array([15]), dims=['time'])
    expected["time_bounds"] = Variable(data=np.array([0, 31]), dims=["time_bounds"])

    encoded, _ = cf_encoder(ds.variables, ds.attrs)
    assert_equal(encoded["time_bounds"], expected["time_bounds"])
    assert "calendar" not in encoded["time_bounds"].attrs
    assert "units" not in encoded["time_bounds"].attrs

    # if time_bounds attrs are same as time attrs, it doesn't matter
    ds.time_bounds.encoding = {"calendar": "noleap", "units": "days since 2000-01-01"}
    encoded, _ = cf_encoder({k: ds[k] for k in ds.variables}, ds.attrs)
    assert_equal(encoded["time_bounds"], expected["time_bounds"])
    assert "calendar" not in encoded["time_bounds"].attrs
    assert "units" not in encoded["time_bounds"].attrs

    # for CF-noncompliant case of time_bounds attrs being different from
    # time attrs; preserve them for faithful roundtrip
    ds.time_bounds.encoding = {"calendar": "noleap", "units": "days since 1849-01-01"}
    encoded, _ = cf_encoder({k: ds[k] for k in ds.variables}, ds.attrs)
    with pytest.raises(AssertionError):
        assert_equal(encoded["time_bounds"], expected["time_bounds"])
    assert "calendar" not in encoded["time_bounds"].attrs
    assert encoded["time_bounds"].attrs["units"] == ds.time_bounds.encoding["units"]

    ds.time.encoding = {}
    with pytest.warns(UserWarning):
        cf_encoder(ds.variables, ds.attrs)


@pytest.fixture(params=_ALL_CALENDARS)
def calendar(request):
    return request.param


@pytest.fixture()
def times(calendar):
    import cftime

    return cftime.num2date(
        np.arange(4),
        units="hours since 2000-01-01",
        calendar=calendar,
        only_use_cftime_datetimes=True,
    )


@pytest.fixture()
def data(times):
    data = np.random.rand(2, 2, 4)
    lons = np.linspace(0, 11, 2)
    lats = np.linspace(0, 20, 2)
    return DataArray(
        data, coords=[lons, lats, times], dims=["lon", "lat", "time"], name="data"
    )


@pytest.fixture()
def times_3d(times):
    lons = np.linspace(0, 11, 2)
    lats = np.linspace(0, 20, 2)
    times_arr = np.random.choice(times, size=(2, 2, 4))
    return DataArray(
        times_arr, coords=[lons, lats, times], dims=["lon", "lat", "time"], name="data"
    )


@requires_cftime
def test_contains_cftime_datetimes_1d(data):
    assert contains_cftime_datetimes(data.time)


@requires_cftime
@requires_dask
def test_contains_cftime_datetimes_dask_1d(data):
    assert contains_cftime_datetimes(data.time.chunk())


@requires_cftime
def test_contains_cftime_datetimes_3d(times_3d):
    assert contains_cftime_datetimes(times_3d)


@requires_cftime
@requires_dask
def test_contains_cftime_datetimes_dask_3d(times_3d):
    assert contains_cftime_datetimes(times_3d.chunk())


@pytest.mark.parametrize("non_cftime_data", [DataArray([]), DataArray([1, 2])])
def test_contains_cftime_datetimes_non_cftimes(non_cftime_data):
    assert not contains_cftime_datetimes(non_cftime_data)


@requires_dask
@pytest.mark.parametrize("non_cftime_data", [DataArray([]), DataArray([1, 2])])
def test_contains_cftime_datetimes_non_cftimes_dask(non_cftime_data):
    assert not contains_cftime_datetimes(non_cftime_data.chunk())


@requires_cftime
@pytest.mark.parametrize("shape", [(24,), (8, 3), (2, 4, 3)])
def test_encode_cf_datetime_overflow(shape):
    # Test for fix to GH 2272
    dates = pd.date_range("2100", periods=24).values.reshape(shape)
    units = "days since 1800-01-01"
    calendar = "standard"

    num, _, _ = encode_cf_datetime(dates, units, calendar)
    roundtrip = decode_cf_datetime(num, units, calendar)
    np.testing.assert_array_equal(dates, roundtrip)


def test_encode_cf_datetime_pandas_min():
    # GH 2623
    dates = pd.date_range("2000", periods=3)
    num, units, calendar = encode_cf_datetime(dates)
    expected_num = np.array([0.0, 1.0, 2.0])
    expected_units = "days since 2000-01-01 00:00:00"
    expected_calendar = "proleptic_gregorian"
    np.testing.assert_array_equal(num, expected_num)
    assert units == expected_units
    assert calendar == expected_calendar


@requires_cftime
def test_time_units_with_timezone_roundtrip(calendar):
    # Regression test for GH 2649
    expected_units = "days since 2000-01-01T00:00:00-05:00"
    expected_num_dates = np.array([1, 2, 3])
    dates = decode_cf_datetime(expected_num_dates, expected_units, calendar)

    # Check that dates were decoded to UTC; here the hours should all
    # equal 5.
    result_hours = DataArray(dates).dt.hour
    expected_hours = DataArray([5, 5, 5])
    assert_equal(result_hours, expected_hours)

    # Check that the encoded values are accurately roundtripped.
    result_num_dates, result_units, result_calendar = encode_cf_datetime(
        dates, expected_units, calendar
    )

    if calendar in _STANDARD_CALENDARS:
        np.testing.assert_array_equal(result_num_dates, expected_num_dates)
    else:
        # cftime datetime arithmetic is not quite exact.
        np.testing.assert_allclose(result_num_dates, expected_num_dates)

    assert result_units == expected_units
    assert result_calendar == calendar


@pytest.mark.parametrize("calendar", _STANDARD_CALENDARS)
def test_use_cftime_default_standard_calendar_in_range(calendar):
    numerical_dates = [0, 1]
    units = "days since 2000-01-01"
    expected = pd.date_range("2000", periods=2)

    with pytest.warns(None) as record:
        result = decode_cf_datetime(numerical_dates, units, calendar)
        np.testing.assert_array_equal(result, expected)
        assert not record


@requires_cftime
@pytest.mark.parametrize("calendar", _STANDARD_CALENDARS)
@pytest.mark.parametrize("units_year", [1500, 2500])
def test_use_cftime_default_standard_calendar_out_of_range(calendar, units_year):
    from cftime import num2date

    numerical_dates = [0, 1]
    units = f"days since {units_year}-01-01"
    expected = num2date(
        numerical_dates, units, calendar, only_use_cftime_datetimes=True
    )

    with pytest.warns(SerializationWarning):
        result = decode_cf_datetime(numerical_dates, units, calendar)
        np.testing.assert_array_equal(result, expected)


@requires_cftime
@pytest.mark.parametrize("calendar", _NON_STANDARD_CALENDARS)
@pytest.mark.parametrize("units_year", [1500, 2000, 2500])
def test_use_cftime_default_non_standard_calendar(calendar, units_year):
    from cftime import num2date

    numerical_dates = [0, 1]
    units = f"days since {units_year}-01-01"
    expected = num2date(
        numerical_dates, units, calendar, only_use_cftime_datetimes=True
    )

    with pytest.warns(None) as record:
        result = decode_cf_datetime(numerical_dates, units, calendar)
        np.testing.assert_array_equal(result, expected)
        assert not record


@requires_cftime
@pytest.mark.parametrize("calendar", _ALL_CALENDARS)
@pytest.mark.parametrize("units_year", [1500, 2000, 2500])
def test_use_cftime_true(calendar, units_year):
    from cftime import num2date

    numerical_dates = [0, 1]
    units = f"days since {units_year}-01-01"
    expected = num2date(
        numerical_dates, units, calendar, only_use_cftime_datetimes=True
    )

    with pytest.warns(None) as record:
        result = decode_cf_datetime(numerical_dates, units, calendar, use_cftime=True)
        np.testing.assert_array_equal(result, expected)
        assert not record


@pytest.mark.parametrize("calendar", _STANDARD_CALENDARS)
def test_use_cftime_false_standard_calendar_in_range(calendar):
    numerical_dates = [0, 1]
    units = "days since 2000-01-01"
    expected = pd.date_range("2000", periods=2)

    with pytest.warns(None) as record:
        result = decode_cf_datetime(numerical_dates, units, calendar, use_cftime=False)
        np.testing.assert_array_equal(result, expected)
        assert not record


@pytest.mark.parametrize("calendar", _STANDARD_CALENDARS)
@pytest.mark.parametrize("units_year", [1500, 2500])
def test_use_cftime_false_standard_calendar_out_of_range(calendar, units_year):
    numerical_dates = [0, 1]
    units = f"days since {units_year}-01-01"
    with pytest.raises(OutOfBoundsDatetime):
        decode_cf_datetime(numerical_dates, units, calendar, use_cftime=False)


@pytest.mark.parametrize("calendar", _NON_STANDARD_CALENDARS)
@pytest.mark.parametrize("units_year", [1500, 2000, 2500])
def test_use_cftime_false_non_standard_calendar(calendar, units_year):
    numerical_dates = [0, 1]
    units = f"days since {units_year}-01-01"
    with pytest.raises(OutOfBoundsDatetime):
        decode_cf_datetime(numerical_dates, units, calendar, use_cftime=False)
