from __future__ import absolute_import, division, print_function

import re
import traceback
import warnings
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd

from ..core.common import contains_cftime_datetimes
from ..core import indexing
from ..core.formatting import first_n_items, format_timestamp, last_item
from ..core.options import OPTIONS
from ..core.pycompat import PY3
from ..core.variable import Variable
from .variables import (
    SerializationWarning, VariableCoder, lazy_elemwise_func, pop_to,
    safe_setitem, unpack_for_decoding, unpack_for_encoding)

try:
    from pandas.errors import OutOfBoundsDatetime
except ImportError:
    # pandas < 0.20
    from pandas.tslib import OutOfBoundsDatetime


# standard calendars recognized by cftime
_STANDARD_CALENDARS = set(['standard', 'gregorian', 'proleptic_gregorian'])

_NS_PER_TIME_DELTA = {'us': int(1e3),
                      'ms': int(1e6),
                      's': int(1e9),
                      'm': int(1e9) * 60,
                      'h': int(1e9) * 60 * 60,
                      'D': int(1e9) * 60 * 60 * 24}

TIME_UNITS = frozenset(['days', 'hours', 'minutes', 'seconds',
                        'milliseconds', 'microseconds'])


def _import_cftime():
    '''
    helper function handle the transition to netcdftime/cftime
    as a stand-alone package
    '''
    try:
        import cftime
    except ImportError:
        # in netCDF4 the num2date/date2num function are top-level api
        try:
            import netCDF4 as cftime
        except ImportError:
            raise ImportError("Failed to import cftime")
    return cftime


def _require_standalone_cftime():
    """Raises an ImportError if the standalone cftime is not found"""
    try:
        import cftime  # noqa: F401
    except ImportError:
        raise ImportError('Using a CFTimeIndex requires the standalone '
                          'version of the cftime library.')


def _netcdf_to_numpy_timeunit(units):
    units = units.lower()
    if not units.endswith('s'):
        units = '%ss' % units
    return {'microseconds': 'us', 'milliseconds': 'ms', 'seconds': 's',
            'minutes': 'm', 'hours': 'h', 'days': 'D'}[units]


def _unpack_netcdf_time_units(units):
    # CF datetime units follow the format: "UNIT since DATE"
    # this parses out the unit and date allowing for extraneous
    # whitespace.
    matches = re.match('(.+) since (.+)', units)
    if not matches:
        raise ValueError('invalid time units: %s' % units)
    delta_units, ref_date = [s.strip() for s in matches.groups()]
    return delta_units, ref_date


def _decode_datetime_with_cftime(num_dates, units, calendar,
                                 enable_cftimeindex):
    cftime = _import_cftime()
    if enable_cftimeindex:
        _require_standalone_cftime()
        dates = np.asarray(cftime.num2date(num_dates, units, calendar,
                                           only_use_cftime_datetimes=True))
    else:
        dates = np.asarray(cftime.num2date(num_dates, units, calendar))

    if (dates[np.nanargmin(num_dates)].year < 1678 or
            dates[np.nanargmax(num_dates)].year >= 2262):
        if not enable_cftimeindex or calendar in _STANDARD_CALENDARS:
            warnings.warn(
                'Unable to decode time axis into full '
                'numpy.datetime64 objects, continuing using dummy '
                'cftime.datetime objects instead, reason: dates out '
                'of range', SerializationWarning, stacklevel=3)
    else:
        if enable_cftimeindex:
            if calendar in _STANDARD_CALENDARS:
                dates = cftime_to_nptime(dates)
        else:
            try:
                dates = cftime_to_nptime(dates)
            except ValueError as e:
                warnings.warn(
                    'Unable to decode time axis into full '
                    'numpy.datetime64 objects, continuing using '
                    'dummy cftime.datetime objects instead, reason:'
                    '{0}'.format(e), SerializationWarning, stacklevel=3)
    return dates


def _decode_cf_datetime_dtype(data, units, calendar, enable_cftimeindex):
    # Verify that at least the first and last date can be decoded
    # successfully. Otherwise, tracebacks end up swallowed by
    # Dataset.__repr__ when users try to view their lazily decoded array.
    values = indexing.ImplicitToExplicitIndexingAdapter(
        indexing.as_indexable(data))
    example_value = np.concatenate([first_n_items(values, 1) or [0],
                                    last_item(values) or [0]])

    try:
        result = decode_cf_datetime(example_value, units, calendar,
                                    enable_cftimeindex)
    except Exception:
        calendar_msg = ('the default calendar' if calendar is None
                        else 'calendar %r' % calendar)
        msg = ('unable to decode time units %r with %s. Try '
               'opening your dataset with decode_times=False.'
               % (units, calendar_msg))
        if not PY3:
            msg += ' Full traceback:\n' + traceback.format_exc()
        raise ValueError(msg)
    else:
        dtype = getattr(result, 'dtype', np.dtype('object'))

    return dtype


def decode_cf_datetime(num_dates, units, calendar=None,
                       enable_cftimeindex=False):
    """Given an array of numeric dates in netCDF format, convert it into a
    numpy array of date time objects.

    For standard (Gregorian) calendars, this function uses vectorized
    operations, which makes it much faster than cftime.num2date. In such a
    case, the returned array will be of type np.datetime64.

    Note that time unit in `units` must not be smaller than microseconds and
    not larger than days.

    See also
    --------
    cftime.num2date
    """
    num_dates = np.asarray(num_dates)
    flat_num_dates = num_dates.ravel()
    if calendar is None:
        calendar = 'standard'

    delta, ref_date = _unpack_netcdf_time_units(units)

    try:
        if calendar not in _STANDARD_CALENDARS:
            raise OutOfBoundsDatetime

        delta = _netcdf_to_numpy_timeunit(delta)
        try:
            ref_date = pd.Timestamp(ref_date)
        except ValueError:
            # ValueError is raised by pd.Timestamp for non-ISO timestamp
            # strings, in which case we fall back to using cftime
            raise OutOfBoundsDatetime

        # fixes: https://github.com/pydata/pandas/issues/14068
        # these lines check if the the lowest or the highest value in dates
        # cause an OutOfBoundsDatetime (Overflow) error
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'invalid value encountered',
                                    RuntimeWarning)
            pd.to_timedelta(flat_num_dates.min(), delta) + ref_date
            pd.to_timedelta(flat_num_dates.max(), delta) + ref_date

        # Cast input dates to integers of nanoseconds because `pd.to_datetime`
        # works much faster when dealing with integers
        # make _NS_PER_TIME_DELTA an array to ensure type upcasting
        flat_num_dates_ns_int = (flat_num_dates.astype(np.float64) *
                                 _NS_PER_TIME_DELTA[delta]).astype(np.int64)

        dates = (pd.to_timedelta(flat_num_dates_ns_int, 'ns') +
                 ref_date).values

    except (OutOfBoundsDatetime, OverflowError):
        dates = _decode_datetime_with_cftime(
            flat_num_dates.astype(np.float), units, calendar,
            enable_cftimeindex)

    return dates.reshape(num_dates.shape)


def decode_cf_timedelta(num_timedeltas, units):
    """Given an array of numeric timedeltas in netCDF format, convert it into a
    numpy timedelta64[ns] array.
    """
    num_timedeltas = np.asarray(num_timedeltas)
    units = _netcdf_to_numpy_timeunit(units)

    shape = num_timedeltas.shape
    num_timedeltas = num_timedeltas.ravel()

    result = pd.to_timedelta(num_timedeltas, unit=units, box=False)
    # NaT is returned unboxed with wrong units; this should be fixed in pandas
    if result.dtype != 'timedelta64[ns]':
        result = result.astype('timedelta64[ns]')
    return result.reshape(shape)


def _infer_time_units_from_diff(unique_timedeltas):
    for time_unit in ['days', 'hours', 'minutes', 'seconds']:
        delta_ns = _NS_PER_TIME_DELTA[_netcdf_to_numpy_timeunit(time_unit)]
        unit_delta = np.timedelta64(delta_ns, 'ns')
        diffs = unique_timedeltas / unit_delta
        if np.all(diffs == diffs.astype(int)):
            return time_unit
    return 'seconds'


def infer_calendar_name(dates):
    """Given an array of datetimes, infer the CF calendar name"""
    if np.asarray(dates).dtype == 'datetime64[ns]':
        return 'proleptic_gregorian'
    else:
        return np.asarray(dates).ravel()[0].calendar


def infer_datetime_units(dates):
    """Given an array of datetimes, returns a CF compatible time-unit string of
    the form "{time_unit} since {date[0]}", where `time_unit` is 'days',
    'hours', 'minutes' or 'seconds' (the first one that can evenly divide all
    unique time deltas in `dates`)
    """
    dates = np.asarray(dates).ravel()
    if np.asarray(dates).dtype == 'datetime64[ns]':
        dates = pd.to_datetime(dates, box=False)
        dates = dates[pd.notnull(dates)]
        reference_date = dates[0] if len(dates) > 0 else '1970-01-01'
        reference_date = pd.Timestamp(reference_date)
    else:
        reference_date = dates[0] if len(dates) > 0 else '1970-01-01'
        reference_date = format_cftime_datetime(reference_date)
    unique_timedeltas = np.unique(np.diff(dates))
    if unique_timedeltas.dtype == np.dtype('O'):
        # Convert to np.timedelta64 objects using pandas to work around a
        # NumPy casting bug: https://github.com/numpy/numpy/issues/11096
        unique_timedeltas = pd.to_timedelta(unique_timedeltas, box=False)
    units = _infer_time_units_from_diff(unique_timedeltas)
    return '%s since %s' % (units, reference_date)


def format_cftime_datetime(date):
    """Converts a cftime.datetime object to a string with the format:
    YYYY-MM-DD HH:MM:SS.UUUUUU
    """
    return '{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}.{:06d}'.format(
        date.year, date.month, date.day, date.hour, date.minute, date.second,
        date.microsecond)


def infer_timedelta_units(deltas):
    """Given an array of timedeltas, returns a CF compatible time-unit from
    {'days', 'hours', 'minutes' 'seconds'} (the first one that can evenly
    divide all unique time deltas in `deltas`)
    """
    deltas = pd.to_timedelta(np.asarray(deltas).ravel(), box=False)
    unique_timedeltas = np.unique(deltas[pd.notnull(deltas)])
    units = _infer_time_units_from_diff(unique_timedeltas)
    return units


def cftime_to_nptime(times):
    """Given an array of cftime.datetime objects, return an array of
    numpy.datetime64 objects of the same size"""
    times = np.asarray(times)
    new = np.empty(times.shape, dtype='M8[ns]')
    for i, t in np.ndenumerate(times):
        dt = datetime(t.year, t.month, t.day, t.hour, t.minute, t.second)
        new[i] = np.datetime64(dt)
    return new


def _cleanup_netcdf_time_units(units):
    delta, ref_date = _unpack_netcdf_time_units(units)
    try:
        units = '%s since %s' % (delta, format_timestamp(ref_date))
    except OutOfBoundsDatetime:
        # don't worry about reifying the units if they're out of bounds
        pass
    return units


def _encode_datetime_with_cftime(dates, units, calendar):
    """Fallback method for encoding dates using cftime.

    This method is more flexible than xarray's parsing using datetime64[ns]
    arrays but also slower because it loops over each element.
    """
    cftime = _import_cftime()

    if np.issubdtype(dates.dtype, np.datetime64):
        # numpy's broken datetime conversion only works for us precision
        dates = dates.astype('M8[us]').astype(datetime)

    def encode_datetime(d):
        return np.nan if d is None else cftime.date2num(d, units, calendar)

    return np.vectorize(encode_datetime)(dates)


def cast_to_int_if_safe(num):
    int_num = np.array(num, dtype=np.int64)
    if (num == int_num).all():
        num = int_num
    return num


def encode_cf_datetime(dates, units=None, calendar=None):
    """Given an array of datetime objects, returns the tuple `(num, units,
    calendar)` suitable for a CF compliant time variable.

    Unlike `date2num`, this function can handle datetime64 arrays.

    See also
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
        if calendar not in _STANDARD_CALENDARS or dates.dtype.kind == 'O':
            # parse with cftime instead
            raise OutOfBoundsDatetime
        assert dates.dtype == 'datetime64[ns]'

        delta_units = _netcdf_to_numpy_timeunit(delta)
        time_delta = np.timedelta64(1, delta_units).astype('timedelta64[ns]')
        ref_date = np.datetime64(pd.Timestamp(ref_date))
        num = (dates - ref_date) / time_delta

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

    def encode(self, variable, name=None):
        dims, data, attrs, encoding = unpack_for_encoding(variable)
        if (np.issubdtype(data.dtype, np.datetime64) or
           contains_cftime_datetimes(variable)):
            (data, units, calendar) = encode_cf_datetime(
                data,
                encoding.pop('units', None),
                encoding.pop('calendar', None))
            safe_setitem(attrs, 'units', units, name=name)
            safe_setitem(attrs, 'calendar', calendar, name=name)

        return Variable(dims, data, attrs, encoding)

    def decode(self, variable, name=None):
        dims, data, attrs, encoding = unpack_for_decoding(variable)

        enable_cftimeindex = OPTIONS['enable_cftimeindex']
        if 'units' in attrs and 'since' in attrs['units']:
            units = pop_to(attrs, encoding, 'units')
            calendar = pop_to(attrs, encoding, 'calendar')
            dtype = _decode_cf_datetime_dtype(
                data, units, calendar, enable_cftimeindex)
            transform = partial(
                decode_cf_datetime, units=units, calendar=calendar,
                enable_cftimeindex=enable_cftimeindex)
            data = lazy_elemwise_func(data, transform, dtype)

        return Variable(dims, data, attrs, encoding)


class CFTimedeltaCoder(VariableCoder):

    def encode(self, variable, name=None):
        dims, data, attrs, encoding = unpack_for_encoding(variable)

        if np.issubdtype(data.dtype, np.timedelta64):
            data, units = encode_cf_timedelta(
                data, encoding.pop('units', None))
            safe_setitem(attrs, 'units', units, name=name)

        return Variable(dims, data, attrs, encoding)

    def decode(self, variable, name=None):
        dims, data, attrs, encoding = unpack_for_decoding(variable)

        if 'units' in attrs and attrs['units'] in TIME_UNITS:
            units = pop_to(attrs, encoding, 'units')
            transform = partial(decode_cf_timedelta, units=units)
            dtype = np.dtype('timedelta64[ns]')
            data = lazy_elemwise_func(data, transform, dtype=dtype)

        return Variable(dims, data, attrs, encoding)
