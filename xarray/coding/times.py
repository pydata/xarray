import re
import warnings
from datetime import datetime, timedelta
from distutils.version import LooseVersion
from functools import partial

import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime

from ..core import indexing
from ..core.common import contains_cftime_datetimes
from ..core.formatting import first_n_items, format_timestamp, last_item
from ..core.variable import Variable
from .variables import (
    SerializationWarning,
    VariableCoder,
    lazy_elemwise_func,
    pop_to,
    safe_setitem,
    unpack_for_decoding,
    unpack_for_encoding,
)

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


def _is_standard_calendar(calendar):
    return calendar.lower() in _STANDARD_CALENDARS


def _netcdf_to_numpy_timeunit(units):
    units = units.lower()
    if not units.endswith("s"):
        units = f"{units}s"
    return {
        "nanoseconds": "ns",
        "microseconds": "us",
        "milliseconds": "ms",
        "seconds": "s",
        "minutes": "m",
        "hours": "h",
        "days": "D",
    }[units]


def _ensure_padded_year(ref_date):
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
    ref_year, everything_else = [s for s in matches_start_digits.groups()]
    ref_date_padded = "{:04d}{}".format(int(ref_year), everything_else)

    warning_msg = (
        f"Ambiguous reference date string: {ref_date}. The first value is "
        "assumed to be the year hence will be padded with zeros to remove "
        f"the ambiguity (the padded reference date string is: {ref_date_padded}). "
        "To remove this message, remove the ambiguity by padding your reference "
        "date strings with zeros."
    )
    warnings.warn(warning_msg, SerializationWarning)

    return ref_date_padded


def _unpack_netcdf_time_units(units):
    # CF datetime units follow the format: "UNIT since DATE"
    # this parses out the unit and date allowing for extraneous
    # whitespace. It also ensures that the year is padded with zeros
    # so it will be correctly understood by pandas (via dateutil).
    matches = re.match(r"(.+) since (.+)", units)
    if not matches:
        raise ValueError(f"invalid time units: {units}")

    delta_units, ref_date = [s.strip() for s in matches.groups()]
    ref_date = _ensure_padded_year(ref_date)

    return delta_units, ref_date


def _decode_cf_datetime_dtype(data, units, calendar, use_cftime):
    # Verify that at least the first and last date can be decoded
    # successfully. Otherwise, tracebacks end up swallowed by
    # Dataset.__repr__ when users try to view their lazily decoded array.
    values = indexing.ImplicitToExplicitIndexingAdapter(indexing.as_indexable(data))
    example_value = np.concatenate(
        [first_n_items(values, 1) or [0], last_item(values) or [0]]
    )

    try:
        result = decode_cf_datetime(example_value, units, calendar, use_cftime)
    except Exception:
        calendar_msg = (
            "the default calendar" if calendar is None else f"calendar {calendar!r}"
        )
        msg = (
            f"unable to decode time units {units!r} with {calendar_msg!r}. Try "
            "opening your dataset with decode_times=False or installing cftime "
            "if it is not installed."
        )
        raise ValueError(msg)
    else:
        dtype = getattr(result, "dtype", np.dtype("object"))

    return dtype


def _decode_datetime_with_cftime(num_dates, units, calendar):
    import cftime

    return np.asarray(
        cftime.num2date(num_dates, units, calendar, only_use_cftime_datetimes=True)
    )


def _decode_datetime_with_pandas(flat_num_dates, units, calendar):
    if not _is_standard_calendar(calendar):
        raise OutOfBoundsDatetime(
            "Cannot decode times from a non-standard calendar, {!r}, using "
            "pandas.".format(calendar)
        )

    delta, ref_date = _unpack_netcdf_time_units(units)
    delta = _netcdf_to_numpy_timeunit(delta)
    try:
        ref_date = pd.Timestamp(ref_date)
    except ValueError:
        # ValueError is raised by pd.Timestamp for non-ISO timestamp
        # strings, in which case we fall back to using cftime
        raise OutOfBoundsDatetime

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value encountered", RuntimeWarning)
        pd.to_timedelta(flat_num_dates.min(), delta) + ref_date
        pd.to_timedelta(flat_num_dates.max(), delta) + ref_date

    # To avoid integer overflow when converting to nanosecond units for integer
    # dtypes smaller than np.int64 cast all integer-dtype arrays to np.int64
    # (GH 2002).
    if flat_num_dates.dtype.kind == "i":
        flat_num_dates = flat_num_dates.astype(np.int64)

    # Cast input ordinals to integers of nanoseconds because pd.to_timedelta
    # works much faster when dealing with integers (GH 1399).
    flat_num_dates_ns_int = (flat_num_dates * _NS_PER_TIME_DELTA[delta]).astype(
        np.int64
    )

    # Use pd.to_timedelta to safely cast integer values to timedeltas,
    # and add those to a Timestamp to safely produce a DatetimeIndex.  This
    # ensures that we do not encounter integer overflow at any point in the
    # process without raising OutOfBoundsDatetime.
    return (pd.to_timedelta(flat_num_dates_ns_int, "ns") + ref_date).values


def decode_cf_datetime(num_dates, units, calendar=None, use_cftime=None):
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
    flat_num_dates = num_dates.ravel()
    if calendar is None:
        calendar = "standard"

    if use_cftime is None:
        try:
            dates = _decode_datetime_with_pandas(flat_num_dates, units, calendar)
        except (KeyError, OutOfBoundsDatetime, OverflowError):
            dates = _decode_datetime_with_cftime(
                flat_num_dates.astype(float), units, calendar
            )

            if (
                dates[np.nanargmin(num_dates)].year < 1678
                or dates[np.nanargmax(num_dates)].year >= 2262
            ):
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

    return dates.reshape(num_dates.shape)


def to_timedelta_unboxed(value, **kwargs):
    if LooseVersion(pd.__version__) < "0.25.0":
        result = pd.to_timedelta(value, **kwargs, box=False)
    else:
        result = pd.to_timedelta(value, **kwargs).to_numpy()
    assert result.dtype == "timedelta64[ns]"
    return result


def to_datetime_unboxed(value, **kwargs):
    if LooseVersion(pd.__version__) < "0.25.0":
        result = pd.to_datetime(value, **kwargs, box=False)
    else:
        result = pd.to_datetime(value, **kwargs).to_numpy()
    assert result.dtype == "datetime64[ns]"
    return result


def decode_cf_timedelta(num_timedeltas, units):
    """Given an array of numeric timedeltas in netCDF format, convert it into a
    numpy timedelta64[ns] array.
    """
    num_timedeltas = np.asarray(num_timedeltas)
    units = _netcdf_to_numpy_timeunit(units)
    result = to_timedelta_unboxed(num_timedeltas.ravel(), unit=units)
    return result.reshape(num_timedeltas.shape)


def _unit_timedelta_cftime(units):
    return timedelta(microseconds=_US_PER_TIME_DELTA[units])


def _unit_timedelta_numpy(units):
    numpy_units = _netcdf_to_numpy_timeunit(units)
    return np.timedelta64(_NS_PER_TIME_DELTA[numpy_units], "ns")


def _infer_time_units_from_diff(unique_timedeltas):
    if unique_timedeltas.dtype == np.dtype("O"):
        time_units = _NETCDF_TIME_UNITS_CFTIME
        unit_timedelta = _unit_timedelta_cftime
        zero_timedelta = timedelta(microseconds=0)
        timedeltas = unique_timedeltas
    else:
        time_units = _NETCDF_TIME_UNITS_NUMPY
        unit_timedelta = _unit_timedelta_numpy
        zero_timedelta = np.timedelta64(0, "ns")
        # Note that the modulus operator was only implemented for np.timedelta64
        # arrays as of NumPy version 1.16.0.  Once our minimum version of NumPy
        # supported is greater than or equal to this we will no longer need to cast
        # unique_timedeltas to a TimedeltaIndex.  In the meantime, however, the
        # modulus operator works for TimedeltaIndex objects.
        timedeltas = pd.TimedeltaIndex(unique_timedeltas)
    for time_unit in time_units:
        if np.all(timedeltas % unit_timedelta(time_unit) == zero_timedelta):
            return time_unit
    return "seconds"


def infer_calendar_name(dates):
    """Given an array of datetimes, infer the CF calendar name"""
    if np.asarray(dates).dtype == "datetime64[ns]":
        return "proleptic_gregorian"
    else:
        return np.asarray(dates).ravel()[0].calendar


def infer_datetime_units(dates):
    """Given an array of datetimes, returns a CF compatible time-unit string of
    the form "{time_unit} since {date[0]}", where `time_unit` is 'days',
    'hours', 'minutes' or 'seconds' (the first one that can evenly divide all
    unique time deltas in `dates`)
    """
    dates = np.asarray(dates).ravel()
    if np.asarray(dates).dtype == "datetime64[ns]":
        dates = to_datetime_unboxed(dates)
        dates = dates[pd.notnull(dates)]
        reference_date = dates[0] if len(dates) > 0 else "1970-01-01"
        reference_date = pd.Timestamp(reference_date)
    else:
        reference_date = dates[0] if len(dates) > 0 else "1970-01-01"
        reference_date = format_cftime_datetime(reference_date)
    unique_timedeltas = np.unique(np.diff(dates))
    units = _infer_time_units_from_diff(unique_timedeltas)
    return f"{units} since {reference_date}"


def format_cftime_datetime(date):
    """Converts a cftime.datetime object to a string with the format:
    YYYY-MM-DD HH:MM:SS.UUUUUU
    """
    return "{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}.{:06d}".format(
        date.year,
        date.month,
        date.day,
        date.hour,
        date.minute,
        date.second,
        date.microsecond,
    )


def infer_timedelta_units(deltas):
    """Given an array of timedeltas, returns a CF compatible time-unit from
    {'days', 'hours', 'minutes' 'seconds'} (the first one that can evenly
    divide all unique time deltas in `deltas`)
    """
    deltas = to_timedelta_unboxed(np.asarray(deltas).ravel())
    unique_timedeltas = np.unique(deltas[pd.notnull(deltas)])
    return _infer_time_units_from_diff(unique_timedeltas)


def cftime_to_nptime(times):
    """Given an array of cftime.datetime objects, return an array of
    numpy.datetime64 objects of the same size"""
    times = np.asarray(times)
    new = np.empty(times.shape, dtype="M8[ns]")
    for i, t in np.ndenumerate(times):
        try:
            # Use pandas.Timestamp in place of datetime.datetime, because
            # NumPy casts it safely it np.datetime64[ns] for dates outside
            # 1678 to 2262 (this is not currently the case for
            # datetime.datetime).
            dt = pd.Timestamp(
                t.year, t.month, t.day, t.hour, t.minute, t.second, t.microsecond
            )
        except ValueError as e:
            raise ValueError(
                "Cannot convert date {} to a date in the "
                "standard calendar.  Reason: {}.".format(t, e)
            )
        new[i] = np.datetime64(dt)
    return new


def _cleanup_netcdf_time_units(units):
    delta, ref_date = _unpack_netcdf_time_units(units)
    try:
        units = "{} since {}".format(delta, format_timestamp(ref_date))
    except OutOfBoundsDatetime:
        # don't worry about reifying the units if they're out of bounds
        pass
    return units


def _encode_datetime_with_cftime(dates, units, calendar):
    """Fallback method for encoding dates using cftime.

    This method is more flexible than xarray's parsing using datetime64[ns]
    arrays but also slower because it loops over each element.
    """
    import cftime

    if np.issubdtype(dates.dtype, np.datetime64):
        # numpy's broken datetime conversion only works for us precision
        dates = dates.astype("M8[us]").astype(datetime)

    def encode_datetime(d):
        return np.nan if d is None else cftime.date2num(d, units, calendar)

    return np.array([encode_datetime(d) for d in dates.ravel()]).reshape(dates.shape)


def cast_to_int_if_safe(num):
    int_num = np.array(num, dtype=np.int64)
    if (num == int_num).all():
        num = int_num
    return num


def encode_cf_datetime(dates, units=None, calendar=None):
    """Given an array of datetime objects, returns the tuple `(num, units,
    calendar)` suitable for a CF compliant time variable.

    Unlike `date2num`, this function can handle datetime64 arrays.

    See Also
    --------
    cftime.date2num
    """
    dates = np.asarray(dates)

    if units is None:
        units = infer_datetime_units(dates)
    else:
        units = _cleanup_netcdf_time_units(units)

    if calendar is None:
        calendar = infer_calendar_name(dates)

    delta, ref_date = _unpack_netcdf_time_units(units)
    try:
        if not _is_standard_calendar(calendar) or dates.dtype.kind == "O":
            # parse with cftime instead
            raise OutOfBoundsDatetime
        assert dates.dtype == "datetime64[ns]"

        delta_units = _netcdf_to_numpy_timeunit(delta)
        time_delta = np.timedelta64(1, delta_units).astype("timedelta64[ns]")
        ref_date = pd.Timestamp(ref_date)

        # If the ref_date Timestamp is timezone-aware, convert to UTC and
        # make it timezone-naive (GH 2649).
        if ref_date.tz is not None:
            ref_date = ref_date.tz_convert(None)

        # Wrap the dates in a DatetimeIndex to do the subtraction to ensure
        # an OverflowError is raised if the ref_date is too far away from
        # dates to be encoded (GH 2272).
        dates_as_index = pd.DatetimeIndex(dates.ravel())
        time_deltas = dates_as_index - ref_date

        # Use floor division if time_delta evenly divides all differences
        # to preserve integer dtype if possible (GH 4045).
        if np.all(time_deltas % time_delta == np.timedelta64(0, "ns")):
            num = time_deltas // time_delta
        else:
            num = time_deltas / time_delta
        num = num.values.reshape(dates.shape)

    except (OutOfBoundsDatetime, OverflowError):
        num = _encode_datetime_with_cftime(dates, units, calendar)

    num = cast_to_int_if_safe(num)
    return (num, units, calendar)


def encode_cf_timedelta(timedeltas, units=None):
    if units is None:
        units = infer_timedelta_units(timedeltas)

    np_unit = _netcdf_to_numpy_timeunit(units)
    num = 1.0 * timedeltas / np.timedelta64(1, np_unit)
    num = np.where(pd.isnull(timedeltas), np.nan, num)
    num = cast_to_int_if_safe(num)
    return (num, units)


class CFDatetimeCoder(VariableCoder):
    def __init__(self, use_cftime=None):
        self.use_cftime = use_cftime

    def encode(self, variable, name=None):
        dims, data, attrs, encoding = unpack_for_encoding(variable)
        if np.issubdtype(data.dtype, np.datetime64) or contains_cftime_datetimes(
            variable
        ):
            (data, units, calendar) = encode_cf_datetime(
                data, encoding.pop("units", None), encoding.pop("calendar", None)
            )
            safe_setitem(attrs, "units", units, name=name)
            safe_setitem(attrs, "calendar", calendar, name=name)

        return Variable(dims, data, attrs, encoding)

    def decode(self, variable, name=None):
        dims, data, attrs, encoding = unpack_for_decoding(variable)

        if "units" in attrs and "since" in attrs["units"]:
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

        return Variable(dims, data, attrs, encoding)


class CFTimedeltaCoder(VariableCoder):
    def encode(self, variable, name=None):
        dims, data, attrs, encoding = unpack_for_encoding(variable)

        if np.issubdtype(data.dtype, np.timedelta64):
            data, units = encode_cf_timedelta(data, encoding.pop("units", None))
            safe_setitem(attrs, "units", units, name=name)

        return Variable(dims, data, attrs, encoding)

    def decode(self, variable, name=None):
        dims, data, attrs, encoding = unpack_for_decoding(variable)

        if "units" in attrs and attrs["units"] in TIME_UNITS:
            units = pop_to(attrs, encoding, "units")
            transform = partial(decode_cf_timedelta, units=units)
            dtype = np.dtype("timedelta64[ns]")
            data = lazy_elemwise_func(data, transform, dtype=dtype)

        return Variable(dims, data, attrs, encoding)
