from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Hashable
from datetime import datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING, Literal, Union, cast

import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime, OutOfBoundsTimedelta

from xarray.coding.variables import (
    SerializationWarning,
    VariableCoder,
    lazy_elemwise_func,
    pop_to,
    safe_setitem,
    unpack_for_decoding,
    unpack_for_encoding,
)
from xarray.core import indexing
from xarray.core.common import contains_cftime_datetimes, is_np_datetime_like
from xarray.core.duck_array_ops import asarray, ravel, reshape
from xarray.core.formatting import first_n_items, format_timestamp, last_item
from xarray.core.pdcompat import nanosecond_precision_timestamp, timestamp_as_unit
from xarray.core.utils import attempt_import, emit_user_level_warning
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import T_ChunkedArray, get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
from xarray.namedarray.utils import is_duck_dask_array

try:
    import cftime
except ImportError:
    cftime = None

from xarray.core.types import (
    CFCalendar,
    NPDatetimeUnitOptions,
    T_DuckArray,
)

T_Name = Union[Hashable, None]

# standard calendars recognized by cftime
_STANDARD_CALENDARS = {"standard", "gregorian", "proleptic_gregorian"}

_NS_PER_TIME_DELTA = {
    "ns": 1,
    "us": int(1e3),
    "ms": int(1e6),
    "s": int(1e9),
    "m": int(1e9) * 60,
    "h": int(1e9) * 60 * 60,
    "D": int(1e9) * 60 * 60 * 24,
}

_US_PER_TIME_DELTA = {
    "microseconds": 1,
    "milliseconds": 1_000,
    "seconds": 1_000_000,
    "minutes": 60 * 1_000_000,
    "hours": 60 * 60 * 1_000_000,
    "days": 24 * 60 * 60 * 1_000_000,
}

_NETCDF_TIME_UNITS_CFTIME = [
    "days",
    "hours",
    "minutes",
    "seconds",
    "milliseconds",
    "microseconds",
]

_NETCDF_TIME_UNITS_NUMPY = _NETCDF_TIME_UNITS_CFTIME + ["nanoseconds"]

TIME_UNITS = frozenset(
    [
        "days",
        "hours",
        "minutes",
        "seconds",
        "milliseconds",
        "microseconds",
        "nanoseconds",
    ]
)


def _is_standard_calendar(calendar: str) -> bool:
    return calendar.lower() in _STANDARD_CALENDARS


def _is_numpy_compatible_time_range(times):
    if is_np_datetime_like(times.dtype):
        return True
    # times array contains cftime objects
    times = np.asarray(times)
    tmin = times.min()
    tmax = times.max()
    try:
        convert_time_or_go_back(tmin, pd.Timestamp)
        convert_time_or_go_back(tmax, pd.Timestamp)
    except pd.errors.OutOfBoundsDatetime:
        return False
    except ValueError as err:
        if err.args[0] == "year 0 is out of range":
            return False
        raise
    else:
        return True


def _netcdf_to_numpy_timeunit(units: str) -> NPDatetimeUnitOptions:
    units = units.lower()
    if not units.endswith("s"):
        units = f"{units}s"
    return cast(
        NPDatetimeUnitOptions,
        {
            "nanoseconds": "ns",
            "microseconds": "us",
            "milliseconds": "ms",
            "seconds": "s",
            "minutes": "m",
            "hours": "h",
            "days": "D",
        }[units],
    )


def _numpy_to_netcdf_timeunit(units: NPDatetimeUnitOptions) -> str:
    return {
        "ns": "nanoseconds",
        "us": "microseconds",
        "ms": "milliseconds",
        "s": "seconds",
        "m": "minutes",
        "h": "hours",
        "D": "days",
    }[units]


def _ensure_padded_year(ref_date: str) -> str:
    # Reference dates without a padded year (e.g. since 1-1-1 or since 2-3-4)
    # are ambiguous (is it YMD or DMY?). This can lead to some very odd
    # behaviour e.g. pandas (via dateutil) passes '1-1-1 00:00:0.0' as
    # '2001-01-01 00:00:00' (because it assumes a) DMY and b) that year 1 is
    # shorthand for 2001 (like 02 would be shorthand for year 2002)).

    # Here we ensure that there is always a four-digit year, with the
    # assumption being that year comes first if we get something ambiguous.
    matches_year = re.match(r".*\d{4}.*", ref_date)
    if matches_year:
        # all good, return
        return ref_date

    # No four-digit strings, assume the first digits are the year and pad
    # appropriately
    matches_start_digits = re.match(r"(\d+)(.*)", ref_date)
    if not matches_start_digits:
        raise ValueError(f"invalid reference date for time units: {ref_date}")
    ref_year, everything_else = (s for s in matches_start_digits.groups())
    ref_date_padded = f"{int(ref_year):04d}{everything_else}"

    warning_msg = (
        f"Ambiguous reference date string: {ref_date}. The first value is "
        "assumed to be the year hence will be padded with zeros to remove "
        f"the ambiguity (the padded reference date string is: {ref_date_padded}). "
        "To remove this message, remove the ambiguity by padding your reference "
        "date strings with zeros."
    )
    warnings.warn(warning_msg, SerializationWarning, stacklevel=2)

    return ref_date_padded


def _unpack_netcdf_time_units(units: str) -> tuple[str, str]:
    # CF datetime units follow the format: "UNIT since DATE"
    # this parses out the unit and date allowing for extraneous
    # whitespace. It also ensures that the year is padded with zeros
    # so it will be correctly understood by pandas (via dateutil).
    matches = re.match(r"(.+) since (.+)", units)
    if not matches:
        raise ValueError(f"invalid time units: {units}")

    delta_units, ref_date = (s.strip() for s in matches.groups())
    ref_date = _ensure_padded_year(ref_date)

    return delta_units, ref_date


def named(name: str, pattern: str) -> str:
    return "(?P<" + name + ">" + pattern + ")"


def optional(x: str) -> str:
    return "(?:" + x + ")?"


def trailing_optional(xs: list[str]) -> str:
    if not xs:
        return ""
    return xs[0] + optional(trailing_optional(xs[1:]))


def build_pattern(
    date_sep: str = r"\-",
    datetime_sep: str = r"T",
    time_sep: str = r"\:",
    micro_sep: str = r".",
) -> str:
    pieces = [
        (None, "year", r"[+-]?\d{4,5}"),
        (date_sep, "month", r"\d{2}"),
        (date_sep, "day", r"\d{2}"),
        (datetime_sep, "hour", r"\d{2}"),
        (time_sep, "minute", r"\d{2}"),
        (time_sep, "second", r"\d{2}"),
        (micro_sep, "microsecond", r"\d{1,6}"),
    ]
    pattern_list = []
    for sep, name, sub_pattern in pieces:
        pattern_list.append((sep if sep else "") + named(name, sub_pattern))
        # TODO: allow timezone offsets?
    return "^" + trailing_optional(pattern_list) + "$"


_BASIC_PATTERN = build_pattern(date_sep="", time_sep="")
_EXTENDED_PATTERN = build_pattern()
_CFTIME_PATTERN = build_pattern(datetime_sep=" ")
_PATTERNS = [_BASIC_PATTERN, _EXTENDED_PATTERN, _CFTIME_PATTERN]


def parse_iso8601_like(datetime_string: str) -> dict[str, str | None]:
    for pattern in _PATTERNS:
        match = re.match(pattern, datetime_string)
        if match:
            return match.groupdict()
    raise ValueError(
        f"no ISO-8601 or cftime-string-like match for string: {datetime_string}"
    )


def _parse_iso8601(date_type, timestr):
    default = date_type(1, 1, 1)
    result = parse_iso8601_like(timestr)
    replace = {}

    for attr in ["year", "month", "day", "hour", "minute", "second", "microsecond"]:
        value = result.get(attr, None)
        if value is not None:
            resolution = attr
            if attr == "microsecond":
                if len(value) <= 3:
                    resolution = "millisecond"
                # convert match string into valid microsecond value
                value = 10 ** (6 - len(value)) * int(value)
            replace[attr] = int(value)
    return default.replace(**replace), resolution


def _maybe_strip_tz_from_timestamp(date: pd.Timestamp) -> pd.Timestamp:
    # If the ref_date Timestamp is timezone-aware, convert to UTC and
    # make it timezone-naive (GH 2649).
    if date.tz is not None:
        return date.tz_convert("UTC").tz_convert(None)
    return date


def _unpack_time_unit_and_ref_date(
    units: str,
) -> tuple[NPDatetimeUnitOptions, pd.Timestamp]:
    # same us _unpack_netcdf_time_units but finalizes ref_date for
    # processing in encode_cf_datetime
    time_unit, _ref_date = _unpack_netcdf_time_units(units)
    time_unit = _netcdf_to_numpy_timeunit(time_unit)
    # TODO: the strict enforcement of nanosecond precision Timestamps can be
    # relaxed when addressing GitHub issue #7493.
    ref_date = nanosecond_precision_timestamp(_ref_date)
    ref_date = _maybe_strip_tz_from_timestamp(ref_date)
    return time_unit, ref_date


def _decode_cf_datetime_dtype(
    data,
    units: str,
    calendar: str | None,
    use_cftime: bool | None,
) -> np.dtype:
    # Verify that at least the first and last date can be decoded
    # successfully. Otherwise, tracebacks end up swallowed by
    # Dataset.__repr__ when users try to view their lazily decoded array.
    values = indexing.ImplicitToExplicitIndexingAdapter(indexing.as_indexable(data))
    example_value = np.concatenate(
        [first_n_items(values, 1) or [0], last_item(values) or [0]]
    )

    try:
        result = decode_cf_datetime(example_value, units, calendar, use_cftime)
    except Exception as err:
        calendar_msg = (
            "the default calendar" if calendar is None else f"calendar {calendar!r}"
        )
        msg = (
            f"unable to decode time units {units!r} with {calendar_msg!r}. Try "
            "opening your dataset with decode_times=False or installing cftime "
            "if it is not installed."
        )
        raise ValueError(msg) from err
    else:
        dtype = getattr(result, "dtype", np.dtype("object"))

    return dtype


def _decode_datetime_with_cftime(
    num_dates: np.ndarray, units: str, calendar: str
) -> np.ndarray:
    if TYPE_CHECKING:
        import cftime
    else:
        cftime = attempt_import("cftime")
    if num_dates.size > 0:
        return np.asarray(
            cftime.num2date(num_dates, units, calendar, only_use_cftime_datetimes=True)
        )
    else:
        return np.array([], dtype=object)


def _check_date_for_units_since_refdate(
    date, unit: str, ref_date: pd.Timestamp
) -> pd.Timestamp:
    # check for out-of-bounds floats and raise
    if date > np.iinfo("int64").max or date < np.iinfo("int64").min:
        raise OutOfBoundsTimedelta(
            f"Value {date} can't be represented as Datetime/Timedelta."
        )
    delta = date * np.timedelta64(1, unit)
    if not np.isnan(delta):
        # this will raise on dtype overflow for integer dtypes
        if date.dtype.kind in "u" and not np.int64(delta) == date:
            raise OutOfBoundsTimedelta(
                "DType overflow in Datetime/Timedelta calculation."
            )
        # this will raise on overflow if ref_date + delta
        # can't be represented in the current ref_date resolution
        return timestamp_as_unit(ref_date + delta, ref_date.unit)
    else:
        # if date is exactly NaT (np.iinfo("int64").min) return NaT
        # to make follow-up checks work
        return pd.Timestamp("NaT")


def _decode_datetime_with_pandas(
    flat_num_dates: np.ndarray, units: str, calendar: str
) -> np.ndarray:
    if not _is_standard_calendar(calendar):
        raise OutOfBoundsDatetime(
            f"Cannot decode times from a non-standard calendar, {calendar!r}, using "
            "pandas."
        )

    # Work around pandas.to_timedelta issue with dtypes smaller than int64 and
    # NumPy 2.0 by casting all int and uint data to int64 and uint64,
    # respectively. See https://github.com/pandas-dev/pandas/issues/56996 for
    # more details.
    if flat_num_dates.dtype.kind == "i":
        flat_num_dates = flat_num_dates.astype(np.int64)
    elif flat_num_dates.dtype.kind == "u":
        flat_num_dates = flat_num_dates.astype(np.uint64)

    try:
        time_unit, ref_date = _unpack_time_unit_and_ref_date(units)
    except ValueError as err:
        # ValueError is raised by pd.Timestamp for non-ISO timestamp
        # strings, in which case we fall back to using cftime
        raise OutOfBoundsDatetime from err

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value encountered", RuntimeWarning)
        if flat_num_dates.size > 0:
            # avoid size 0 datetimes GH1329
            _check_date_for_units_since_refdate(
                flat_num_dates.min(), time_unit, ref_date
            )
            _check_date_for_units_since_refdate(
                flat_num_dates.max(), time_unit, ref_date
            )

    # To avoid integer overflow when converting to nanosecond units for integer
    # dtypes smaller than np.int64 cast all integer and unsigned integer dtype
    # arrays to np.int64 (GH 2002, GH 6589).  Note this is safe even in the case
    # of np.uint64 values, because any np.uint64 value that would lead to
    # overflow when converting to np.int64 would not be representable with a
    # timedelta64 value, and therefore would raise an error in the lines above.
    if flat_num_dates.dtype.kind in "iu":
        flat_num_dates = flat_num_dates.astype(np.int64)
    elif flat_num_dates.dtype.kind in "f":
        flat_num_dates = flat_num_dates.astype(np.float64)

    # keep NaT/nan mask
    nan = np.isnan(flat_num_dates) | (flat_num_dates == np.iinfo(np.int64).min)
    # in case we need to change the unit, we fix the numbers here
    # this should be safe, as errors would have been raised above
    ns_time_unit = _NS_PER_TIME_DELTA[time_unit]
    ns_ref_date_unit = _NS_PER_TIME_DELTA[ref_date.unit]
    if ns_time_unit > ns_ref_date_unit:
        flat_num_dates *= np.int64(ns_time_unit / ns_ref_date_unit)
        time_unit = ref_date.unit

    # Cast input ordinals to integers and properly handle NaN/NaT
    # to prevent casting NaN to int
    flat_num_dates_int = np.zeros_like(flat_num_dates, dtype=np.int64)
    flat_num_dates_int[nan] = np.iinfo(np.int64).min
    flat_num_dates_int[~nan] = flat_num_dates[~nan].astype(np.int64)

    # cast to timedelta64[time_unit] and add to ref_date
    return ref_date + flat_num_dates_int.astype(f"timedelta64[{time_unit}]")


def decode_cf_datetime(
    num_dates,
    units: str,
    calendar: str | None = None,
    use_cftime: bool | None = None,
) -> np.ndarray:
    """Given an array of numeric dates in netCDF format, convert it into a
    numpy array of date time objects.

    For standard (Gregorian) calendars, this function uses vectorized
    operations, which makes it much faster than cftime.num2date. In such a
    case, the returned array will be of type np.datetime64.

    Note that time unit in `units` must not be smaller than microseconds and
    not larger than days.

    See Also
    --------
    cftime.num2date
    """
    num_dates = np.asarray(num_dates)
    flat_num_dates = ravel(num_dates)
    if calendar is None:
        calendar = "standard"

    if use_cftime is None:
        try:
            dates = _decode_datetime_with_pandas(flat_num_dates, units, calendar)
        except (KeyError, OutOfBoundsDatetime, OutOfBoundsTimedelta, OverflowError):
            dates = _decode_datetime_with_cftime(
                flat_num_dates.astype(float), units, calendar
            )
            # retrieve cftype
            dates_min = dates[np.nanargmin(num_dates)]
            cftype = type(dates_min)
            # "ns" borders
            # between ['1677-09-21T00:12:43.145224193', '2262-04-11T23:47:16.854775807']
            lower = cftype(1677, 9, 21, 0, 12, 43, 145224)
            upper = cftype(2262, 4, 11, 23, 47, 16, 854775)

            if dates_min < lower or dates[np.nanargmax(num_dates)] > upper:
                if _is_standard_calendar(calendar):
                    warnings.warn(
                        "Unable to decode time axis into full "
                        "numpy.datetime64 objects, continuing using "
                        "cftime.datetime objects instead, reason: dates out "
                        "of range",
                        SerializationWarning,
                        stacklevel=3,
                    )
            else:
                if _is_standard_calendar(calendar):
                    dates = cftime_to_nptime(dates)
    elif use_cftime:
        dates = _decode_datetime_with_cftime(flat_num_dates, units, calendar)
    else:
        dates = _decode_datetime_with_pandas(flat_num_dates, units, calendar)

    return reshape(dates, num_dates.shape)


def to_timedelta_unboxed(value, **kwargs):
    result = pd.to_timedelta(value, **kwargs).to_numpy()
    assert result.dtype == "timedelta64[ns]"
    return result


def to_datetime_unboxed(value, **kwargs):
    result = pd.to_datetime(value, **kwargs).to_numpy()
    assert result.dtype == "datetime64[ns]"
    return result


def decode_cf_timedelta(num_timedeltas, units: str) -> np.ndarray:
    """Given an array of numeric timedeltas in netCDF format, convert it into a
    numpy timedelta64[ns] array.
    """
    num_timedeltas = np.asarray(num_timedeltas)
    units = _netcdf_to_numpy_timeunit(units)
    result = to_timedelta_unboxed(ravel(num_timedeltas), unit=units)
    return reshape(result, num_timedeltas.shape)


def _unit_timedelta_cftime(units: str) -> timedelta:
    return timedelta(microseconds=_US_PER_TIME_DELTA[units])


def _unit_timedelta_numpy(units: str) -> np.timedelta64:
    numpy_units = _netcdf_to_numpy_timeunit(units)
    return np.timedelta64(_NS_PER_TIME_DELTA[numpy_units], "ns")


def _infer_time_units_from_diff(unique_timedeltas) -> str:
    unit_timedelta: Callable[[str], timedelta] | Callable[[str], np.timedelta64]
    zero_timedelta: timedelta | np.timedelta64
    if unique_timedeltas.dtype == np.dtype("O"):
        time_units = _NETCDF_TIME_UNITS_CFTIME
        unit_timedelta = _unit_timedelta_cftime
        zero_timedelta = timedelta(microseconds=0)
    else:
        time_units = _NETCDF_TIME_UNITS_NUMPY
        unit_timedelta = _unit_timedelta_numpy
        zero_timedelta = np.timedelta64(0, "ns")
    for time_unit in time_units:
        if np.all(unique_timedeltas % unit_timedelta(time_unit) == zero_timedelta):
            return time_unit
    return "seconds"


def _time_units_to_timedelta64(units: str) -> np.timedelta64:
    return np.timedelta64(1, _netcdf_to_numpy_timeunit(units)).astype("timedelta64[ns]")


def infer_calendar_name(dates) -> CFCalendar:
    """Given an array of datetimes, infer the CF calendar name"""
    if is_np_datetime_like(dates.dtype):
        return "proleptic_gregorian"
    elif dates.dtype == np.dtype("O") and dates.size > 0:
        # Logic copied from core.common.contains_cftime_datetimes.
        if cftime is not None:
            sample = np.asarray(dates).flat[0]
            if is_duck_dask_array(sample):
                sample = sample.compute()
                if isinstance(sample, np.ndarray):
                    sample = sample.item()
            if isinstance(sample, cftime.datetime):
                return sample.calendar

    # Error raise if dtype is neither datetime or "O", if cftime is not importable, and if element of 'O' dtype is not cftime.
    raise ValueError("Array does not contain datetime objects.")


def infer_datetime_units(dates) -> str:
    """Given an array of datetimes, returns a CF compatible time-unit string of
    the form "{time_unit} since {date[0]}", where `time_unit` is 'days',
    'hours', 'minutes' or 'seconds' (the first one that can evenly divide all
    unique time deltas in `dates`)
    """
    dates = ravel(np.asarray(dates))
    if np.asarray(dates).dtype == "datetime64[ns]":
        dates = to_datetime_unboxed(dates)
        dates = dates[pd.notnull(dates)]
        reference_date = dates[0] if len(dates) > 0 else "1970-01-01"
        # TODO: the strict enforcement of nanosecond precision Timestamps can be
        # relaxed when addressing GitHub issue #7493.
        reference_date = nanosecond_precision_timestamp(reference_date)
    else:
        reference_date = dates[0] if len(dates) > 0 else "1970-01-01"
        reference_date = format_cftime_datetime(reference_date)
    unique_timedeltas = np.unique(np.diff(dates))
    units = _infer_time_units_from_diff(unique_timedeltas)
    return f"{units} since {reference_date}"


def format_cftime_datetime(date) -> str:
    """Converts a cftime.datetime object to a string with the format:
    YYYY-MM-DD HH:MM:SS.UUUUUU
    """
    return f"{date.year:04d}-{date.month:02d}-{date.day:02d} {date.hour:02d}:{date.minute:02d}:{date.second:02d}.{date.microsecond:06d}"


def infer_timedelta_units(deltas) -> str:
    """Given an array of timedeltas, returns a CF compatible time-unit from
    {'days', 'hours', 'minutes' 'seconds'} (the first one that can evenly
    divide all unique time deltas in `deltas`)
    """
    deltas = to_timedelta_unboxed(ravel(np.asarray(deltas)))
    unique_timedeltas = np.unique(deltas[pd.notnull(deltas)])
    return _infer_time_units_from_diff(unique_timedeltas)


def cftime_to_nptime(times, raise_on_invalid: bool = True) -> np.ndarray:
    """Given an array of cftime.datetime objects, return an array of
    numpy.datetime64 objects of the same size

    If raise_on_invalid is True (default), invalid dates trigger a ValueError.
    Otherwise, the invalid element is replaced by np.NaT."""
    times = np.asarray(times)
    # TODO: the strict enforcement of nanosecond precision datetime values can
    # be relaxed when addressing GitHub issue #7493.
    new = np.empty(times.shape, dtype="M8[ns]")
    dt: pd.Timestamp | Literal["NaT"]
    for i, t in np.ndenumerate(times):
        try:
            # Use pandas.Timestamp in place of datetime.datetime, because
            # NumPy casts it safely it np.datetime64[ns] for dates outside
            # 1678 to 2262 (this is not currently the case for
            # datetime.datetime).
            dt = nanosecond_precision_timestamp(
                t.year, t.month, t.day, t.hour, t.minute, t.second, t.microsecond
            )
        except ValueError as e:
            if raise_on_invalid:
                raise ValueError(
                    f"Cannot convert date {t} to a date in the "
                    f"standard calendar.  Reason: {e}."
                ) from e
            else:
                dt = "NaT"
        new[i] = np.datetime64(dt)
    return new


def convert_times(times, date_type, raise_on_invalid: bool = True) -> np.ndarray:
    """Given an array of datetimes, return the same dates in another cftime or numpy date type.

    Useful to convert between calendars in numpy and cftime or between cftime calendars.

    If raise_on_valid is True (default), invalid dates trigger a ValueError.
    Otherwise, the invalid element is replaced by np.nan for cftime types and np.NaT for np.datetime64.
    """
    if date_type in (pd.Timestamp, np.datetime64) and not is_np_datetime_like(
        times.dtype
    ):
        return cftime_to_nptime(times, raise_on_invalid=raise_on_invalid)
    if is_np_datetime_like(times.dtype):
        # Convert datetime64 objects to Timestamps since those have year, month, day, etc. attributes
        times = pd.DatetimeIndex(times)
    new = np.empty(times.shape, dtype="O")
    for i, t in enumerate(times):
        try:
            dt = date_type(
                t.year, t.month, t.day, t.hour, t.minute, t.second, t.microsecond
            )
        except ValueError as e:
            if raise_on_invalid:
                raise ValueError(
                    f"Cannot convert date {t} to a date in the "
                    f"{date_type(2000, 1, 1).calendar} calendar.  Reason: {e}."
                ) from e
            else:
                dt = np.nan

        new[i] = dt
    return new


def convert_time_or_go_back(date, date_type):
    """Convert a single date to a new date_type (cftime.datetime or pd.Timestamp).

    If the new date is invalid, it goes back a day and tries again. If it is still
    invalid, goes back a second day.

    This is meant to convert end-of-month dates into a new calendar.
    """
    # TODO: the strict enforcement of nanosecond precision Timestamps can be
    # relaxed when addressing GitHub issue #7493.
    if date_type == pd.Timestamp:
        date_type = nanosecond_precision_timestamp
    try:
        return date_type(
            date.year,
            date.month,
            date.day,
            date.hour,
            date.minute,
            date.second,
            date.microsecond,
        )
    except OutOfBoundsDatetime:
        raise
    except ValueError:
        # Day is invalid, happens at the end of months, try again the day before
        try:
            return date_type(
                date.year,
                date.month,
                date.day - 1,
                date.hour,
                date.minute,
                date.second,
                date.microsecond,
            )
        except ValueError:
            # Still invalid, happens for 360_day to non-leap february. Try again 2 days before date.
            return date_type(
                date.year,
                date.month,
                date.day - 2,
                date.hour,
                date.minute,
                date.second,
                date.microsecond,
            )


def _should_cftime_be_used(
    source, target_calendar: str, use_cftime: bool | None
) -> bool:
    """Return whether conversion of the source to the target calendar should
    result in a cftime-backed array.

    Source is a 1D datetime array, target_cal a string (calendar name) and
    use_cftime is a boolean or None. If use_cftime is None, this returns True
    if the source's range and target calendar are convertible to np.datetime64 objects.
    """
    # Arguments Checks for target
    if use_cftime is not True:
        if _is_standard_calendar(target_calendar):
            if _is_numpy_compatible_time_range(source):
                # Conversion is possible with pandas, force False if it was None
                return False
            elif use_cftime is False:
                raise ValueError(
                    "Source time range is not valid for numpy datetimes. Try using `use_cftime=True`."
                )
        elif use_cftime is False:
            raise ValueError(
                f"Calendar '{target_calendar}' is only valid with cftime. Try using `use_cftime=True`."
            )
    return True


def _cleanup_netcdf_time_units(units: str) -> str:
    time_units, ref_date = _unpack_netcdf_time_units(units)
    time_units = time_units.lower()
    if not time_units.endswith("s"):
        time_units = f"{time_units}s"
    try:
        units = f"{time_units} since {format_timestamp(ref_date)}"
    except (OutOfBoundsDatetime, ValueError):
        # don't worry about reifying the units if they're out of bounds or
        # formatted badly
        pass
    return units


def _encode_datetime_with_cftime(dates, units: str, calendar: str) -> np.ndarray:
    """Fallback method for encoding dates using cftime.

    This method is more flexible than xarray's parsing using datetime64[ns]
    arrays but also slower because it loops over each element.
    """
    if TYPE_CHECKING:
        import cftime
    else:
        cftime = attempt_import("cftime")

    if np.issubdtype(dates.dtype, np.datetime64):
        # numpy's broken datetime conversion only works for us precision
        dates = dates.astype("M8[us]").astype(datetime)

    def encode_datetime(d):
        # Since netCDF files do not support storing float128 values, we ensure
        # that float64 values are used by setting longdouble=False in num2date.
        # This try except logic can be removed when xarray's minimum version of
        # cftime is at least 1.6.2.
        try:
            return (
                np.nan
                if d is None
                else cftime.date2num(d, units, calendar, longdouble=False)
            )
        except TypeError:
            return np.nan if d is None else cftime.date2num(d, units, calendar)

    return reshape(np.array([encode_datetime(d) for d in ravel(dates)]), dates.shape)


def cast_to_int_if_safe(num) -> np.ndarray:
    int_num = np.asarray(num, dtype=np.int64)
    if (num == int_num).all():
        num = int_num
    return num


def _division(deltas, delta, floor):
    if floor:
        # calculate int64 floor division
        # to preserve integer dtype if possible (GH 4045, GH7817).
        num = deltas // delta.astype(np.int64)
        num = num.astype(np.int64, copy=False)
    else:
        num = deltas / delta
    return num


def _cast_to_dtype_if_safe(num: np.ndarray, dtype: np.dtype) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="overflow")
        cast_num = np.asarray(num, dtype=dtype)

    if np.issubdtype(dtype, np.integer):
        if not (num == cast_num).all():
            if np.issubdtype(num.dtype, np.floating):
                raise ValueError(
                    f"Not possible to cast all encoded times from "
                    f"{num.dtype!r} to {dtype!r} without losing precision. "
                    f"Consider modifying the units such that integer values "
                    f"can be used, or removing the units and dtype encoding, "
                    f"at which point xarray will make an appropriate choice."
                )
            else:
                raise OverflowError(
                    f"Not possible to cast encoded times from "
                    f"{num.dtype!r} to {dtype!r} without overflow. Consider "
                    f"removing the dtype encoding, at which point xarray will "
                    f"make an appropriate choice, or explicitly switching to "
                    "a larger integer dtype."
                )
    else:
        if np.isinf(cast_num).any():
            raise OverflowError(
                f"Not possible to cast encoded times from {num.dtype!r} to "
                f"{dtype!r} without overflow.  Consider removing the dtype "
                f"encoding, at which point xarray will make an appropriate "
                f"choice, or explicitly switching to a larger floating point "
                f"dtype."
            )

    return cast_num


def encode_cf_datetime(
    dates: T_DuckArray,  # type: ignore[misc]
    units: str | None = None,
    calendar: str | None = None,
    dtype: np.dtype | None = None,
) -> tuple[T_DuckArray, str, str]:
    """Given an array of datetime objects, returns the tuple `(num, units,
    calendar)` suitable for a CF compliant time variable.

    Unlike `date2num`, this function can handle datetime64 arrays.

    See Also
    --------
    cftime.date2num
    """
    dates = asarray(dates)
    if is_chunked_array(dates):
        return _lazily_encode_cf_datetime(dates, units, calendar, dtype)
    else:
        return _eagerly_encode_cf_datetime(dates, units, calendar, dtype)


def _eagerly_encode_cf_datetime(
    dates: T_DuckArray,  # type: ignore[misc]
    units: str | None = None,
    calendar: str | None = None,
    dtype: np.dtype | None = None,
    allow_units_modification: bool = True,
) -> tuple[T_DuckArray, str, str]:
    dates = asarray(dates)

    data_units = infer_datetime_units(dates)

    if units is None:
        units = data_units
    else:
        units = _cleanup_netcdf_time_units(units)

    if calendar is None:
        calendar = infer_calendar_name(dates)

    try:
        if not _is_standard_calendar(calendar) or dates.dtype.kind == "O":
            # parse with cftime instead
            raise OutOfBoundsDatetime
        assert dates.dtype == "datetime64[ns]"

        time_unit, ref_date = _unpack_time_unit_and_ref_date(units)
        time_delta = np.timedelta64(1, time_unit)

        # Wrap the dates in a DatetimeIndex to do the subtraction to ensure
        # an OverflowError is raised if the ref_date is too far away from
        # dates to be encoded (GH 2272).
        dates_as_index = pd.DatetimeIndex(ravel(dates))
        time_deltas = dates_as_index - ref_date

        # retrieve needed units to faithfully encode to int64
        needed_unit, data_ref_date = _unpack_time_unit_and_ref_date(data_units)
        needed_units = _numpy_to_netcdf_timeunit(needed_unit)
        if data_units != units:
            # this accounts for differences in the reference times
            ref_delta = abs(data_ref_date - ref_date).to_timedelta64()
            data_delta = np.timedelta64(1, needed_unit)
            if (ref_delta % data_delta) > np.timedelta64(0, "ns"):
                needed_units = _infer_time_units_from_diff(ref_delta)

        # needed time delta to encode faithfully to int64
        needed_time_delta = _unit_timedelta_numpy(needed_units)

        floor_division = np.issubdtype(dtype, np.integer) or dtype is None
        if time_delta > needed_time_delta:
            floor_division = False
            if dtype is None:
                emit_user_level_warning(
                    f"Times can't be serialized faithfully to int64 with requested units {units!r}. "
                    f"Resolution of {needed_units!r} needed. Serializing times to floating point instead. "
                    f"Set encoding['dtype'] to integer dtype to serialize to int64. "
                    f"Set encoding['dtype'] to floating point dtype to silence this warning."
                )
            elif np.issubdtype(dtype, np.integer) and allow_units_modification:
                floor_division = True
                new_units = f"{needed_units} since {format_timestamp(ref_date)}"
                emit_user_level_warning(
                    f"Times can't be serialized faithfully to int64 with requested units {units!r}. "
                    f"Serializing with units {new_units!r} instead. "
                    f"Set encoding['dtype'] to floating point dtype to serialize with units {units!r}. "
                    f"Set encoding['units'] to {new_units!r} to silence this warning ."
                )
                units = new_units
                time_delta = needed_time_delta

        # get resolution of TimedeltaIndex and align time_delta
        # todo: check, if this works in any case
        num = _division(
            time_deltas, time_delta.astype(f"=m8[{time_deltas.unit}]"), floor_division
        )
        num = reshape(num.values, dates.shape)

    except (OutOfBoundsDatetime, OverflowError, ValueError):
        num = _encode_datetime_with_cftime(dates, units, calendar)
        # do it now only for cftime-based flow
        # we already covered for this in pandas-based flow
        num = cast_to_int_if_safe(num)

    if dtype is not None:
        num = _cast_to_dtype_if_safe(num, dtype)

    return num, units, calendar


def _encode_cf_datetime_within_map_blocks(
    dates: T_DuckArray,  # type: ignore[misc]
    units: str,
    calendar: str,
    dtype: np.dtype,
) -> T_DuckArray:
    num, *_ = _eagerly_encode_cf_datetime(
        dates, units, calendar, dtype, allow_units_modification=False
    )
    return num


def _lazily_encode_cf_datetime(
    dates: T_ChunkedArray,
    units: str | None = None,
    calendar: str | None = None,
    dtype: np.dtype | None = None,
) -> tuple[T_ChunkedArray, str, str]:
    if calendar is None:
        # This will only trigger minor compute if dates is an object dtype array.
        calendar = infer_calendar_name(dates)

    if units is None and dtype is None:
        if dates.dtype == "O":
            units = "microseconds since 1970-01-01"
            dtype = np.dtype("int64")
        else:
            units = "nanoseconds since 1970-01-01"
            dtype = np.dtype("int64")

    if units is None or dtype is None:
        raise ValueError(
            f"When encoding chunked arrays of datetime values, both the units "
            f"and dtype must be prescribed or both must be unprescribed. "
            f"Prescribing only one or the other is not currently supported. "
            f"Got a units encoding of {units} and a dtype encoding of {dtype}."
        )

    chunkmanager = get_chunked_array_type(dates)
    num = chunkmanager.map_blocks(
        _encode_cf_datetime_within_map_blocks,
        dates,
        units,
        calendar,
        dtype,
        dtype=dtype,
    )
    return num, units, calendar


def encode_cf_timedelta(
    timedeltas: T_DuckArray,  # type: ignore[misc]
    units: str | None = None,
    dtype: np.dtype | None = None,
) -> tuple[T_DuckArray, str]:
    timedeltas = asarray(timedeltas)
    if is_chunked_array(timedeltas):
        return _lazily_encode_cf_timedelta(timedeltas, units, dtype)
    else:
        return _eagerly_encode_cf_timedelta(timedeltas, units, dtype)


def _eagerly_encode_cf_timedelta(
    timedeltas: T_DuckArray,  # type: ignore[misc]
    units: str | None = None,
    dtype: np.dtype | None = None,
    allow_units_modification: bool = True,
) -> tuple[T_DuckArray, str]:
    data_units = infer_timedelta_units(timedeltas)

    if units is None:
        units = data_units

    time_delta = _time_units_to_timedelta64(units)
    time_deltas = pd.TimedeltaIndex(ravel(timedeltas))

    # retrieve needed units to faithfully encode to int64
    needed_units = data_units
    if data_units != units:
        needed_units = _infer_time_units_from_diff(np.unique(time_deltas.dropna()))

    # needed time delta to encode faithfully to int64
    needed_time_delta = _time_units_to_timedelta64(needed_units)

    floor_division = np.issubdtype(dtype, np.integer) or dtype is None
    if time_delta > needed_time_delta:
        floor_division = False
        if dtype is None:
            emit_user_level_warning(
                f"Timedeltas can't be serialized faithfully to int64 with requested units {units!r}. "
                f"Resolution of {needed_units!r} needed. Serializing timeseries to floating point instead. "
                f"Set encoding['dtype'] to integer dtype to serialize to int64. "
                f"Set encoding['dtype'] to floating point dtype to silence this warning."
            )
        elif np.issubdtype(dtype, np.integer) and allow_units_modification:
            emit_user_level_warning(
                f"Timedeltas can't be serialized faithfully with requested units {units!r}. "
                f"Serializing with units {needed_units!r} instead. "
                f"Set encoding['dtype'] to floating point dtype to serialize with units {units!r}. "
                f"Set encoding['units'] to {needed_units!r} to silence this warning ."
            )
            units = needed_units
            time_delta = needed_time_delta
            floor_division = True

    num = _division(time_deltas, time_delta, floor_division)
    num = reshape(num.values, timedeltas.shape)

    if dtype is not None:
        num = _cast_to_dtype_if_safe(num, dtype)

    return num, units


def _encode_cf_timedelta_within_map_blocks(
    timedeltas: T_DuckArray,  # type: ignore[misc]
    units: str,
    dtype: np.dtype,
) -> T_DuckArray:
    num, _ = _eagerly_encode_cf_timedelta(
        timedeltas, units, dtype, allow_units_modification=False
    )
    return num


def _lazily_encode_cf_timedelta(
    timedeltas: T_ChunkedArray, units: str | None = None, dtype: np.dtype | None = None
) -> tuple[T_ChunkedArray, str]:
    if units is None and dtype is None:
        units = "nanoseconds"
        dtype = np.dtype("int64")

    if units is None or dtype is None:
        raise ValueError(
            f"When encoding chunked arrays of timedelta values, both the "
            f"units and dtype must be prescribed or both must be "
            f"unprescribed. Prescribing only one or the other is not "
            f"currently supported. Got a units encoding of {units} and a "
            f"dtype encoding of {dtype}."
        )

    chunkmanager = get_chunked_array_type(timedeltas)
    num = chunkmanager.map_blocks(
        _encode_cf_timedelta_within_map_blocks,
        timedeltas,
        units,
        dtype,
        dtype=dtype,
    )
    return num, units


class CFDatetimeCoder(VariableCoder):
    def __init__(
        self,
        use_cftime: bool | None = None,
    ) -> None:
        self.use_cftime = use_cftime

    def encode(self, variable: Variable, name: T_Name = None) -> Variable:
        if np.issubdtype(
            variable.data.dtype, np.datetime64
        ) or contains_cftime_datetimes(variable):
            dims, data, attrs, encoding = unpack_for_encoding(variable)

            units = encoding.pop("units", None)
            calendar = encoding.pop("calendar", None)
            dtype = encoding.get("dtype", None)
            (data, units, calendar) = encode_cf_datetime(data, units, calendar, dtype)

            safe_setitem(attrs, "units", units, name=name)
            safe_setitem(attrs, "calendar", calendar, name=name)

            return Variable(dims, data, attrs, encoding, fastpath=True)
        else:
            return variable

    def decode(self, variable: Variable, name: T_Name = None) -> Variable:
        units = variable.attrs.get("units", None)
        if isinstance(units, str) and "since" in units:
            dims, data, attrs, encoding = unpack_for_decoding(variable)

            units = pop_to(attrs, encoding, "units")
            calendar = pop_to(attrs, encoding, "calendar")
            dtype = _decode_cf_datetime_dtype(data, units, calendar, self.use_cftime)
            transform = partial(
                decode_cf_datetime,
                units=units,
                calendar=calendar,
                use_cftime=self.use_cftime,
            )
            data = lazy_elemwise_func(data, transform, dtype)

            return Variable(dims, data, attrs, encoding, fastpath=True)
        else:
            return variable


class CFTimedeltaCoder(VariableCoder):
    def encode(self, variable: Variable, name: T_Name = None) -> Variable:
        if np.issubdtype(variable.data.dtype, np.timedelta64):
            dims, data, attrs, encoding = unpack_for_encoding(variable)

            data, units = encode_cf_timedelta(
                data, encoding.pop("units", None), encoding.get("dtype", None)
            )
            safe_setitem(attrs, "units", units, name=name)

            return Variable(dims, data, attrs, encoding, fastpath=True)
        else:
            return variable

    def decode(self, variable: Variable, name: T_Name = None) -> Variable:
        units = variable.attrs.get("units", None)
        if isinstance(units, str) and units in TIME_UNITS:
            dims, data, attrs, encoding = unpack_for_decoding(variable)

            units = pop_to(attrs, encoding, "units")
            transform = partial(decode_cf_timedelta, units=units)
            dtype = np.dtype("timedelta64[ns]")
            data = lazy_elemwise_func(data, transform, dtype=dtype)

            return Variable(dims, data, attrs, encoding, fastpath=True)
        else:
            return variable
