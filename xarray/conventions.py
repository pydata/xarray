from datetime import datetime
import re
import traceback
import warnings

import numpy as np
import pandas as pd
from collections import defaultdict
from pandas.tslib import OutOfBoundsDatetime

from .core import indexing, ops, utils
from .core.formatting import format_timestamp, first_n_items
from .core.variable import as_variable, Variable
from .core.pycompat import (iteritems, bytes_type, unicode_type, OrderedDict,
                            PY3, basestring)


# standard calendars recognized by netcdftime
_STANDARD_CALENDARS = set(['standard', 'gregorian', 'proleptic_gregorian'])


def mask_and_scale(array, fill_value=None, scale_factor=None, add_offset=None,
                   dtype=float):
    """Scale and mask array values according to CF conventions for packed and
    missing values

    First, values equal to the fill_value are replaced by NaN. Then, new values
    are given by the formula:

        original_values * scale_factor + add_offset

    Parameters
    ----------
    array : array-like
        Original array of values to wrap
    fill_value : number, optional
        All values equal to fill_value in the original array are replaced
        by NaN.  If an array of multiple values is provided a warning will be
        issued and all array elements matching an value in the fill_value array
        will be replaced by NaN.
    scale_factor : number, optional
        Multiply entries in the original array by this number.
    add_offset : number, optional
        After applying scale_factor, add this number to entries in the
        original array.

    Returns
    -------
    scaled : np.ndarray
        Array of masked and scaled values.

    References
    ----------
    http://www.unidata.ucar.edu/software/netcdf/docs/BestPractices.html
    """
    # by default, cast to float to ensure NaN is meaningful
    values = np.array(array, dtype=dtype, copy=True)
    if fill_value is not None and not np.all(pd.isnull(fill_value)):
        if getattr(fill_value, 'size', 1) > 1:
            fill_values = fill_value  # multiple fill values
        else:
            fill_values = [fill_value]
        for f_value in fill_values:
            if values.ndim > 0:
                values[values == f_value] = np.nan
            elif values == f_value:
                values = np.array(np.nan)
    if scale_factor is not None:
        values *= scale_factor
    if add_offset is not None:
        values += add_offset
    return values


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


def _decode_datetime_with_netcdf4(num_dates, units, calendar):
    import netCDF4 as nc4

    dates = np.asarray(nc4.num2date(num_dates, units, calendar))
    if (dates[np.nanargmin(num_dates)].year < 1678 or
            dates[np.nanargmax(num_dates)].year >= 2262):
        warnings.warn('Unable to decode time axis into full '
                      'numpy.datetime64 objects, continuing using dummy '
                      'netCDF4.datetime objects instead, reason: dates out'
                      ' of range', RuntimeWarning, stacklevel=3)
    else:
        try:
            dates = nctime_to_nptime(dates)
        except ValueError as e:
            warnings.warn('Unable to decode time axis into full '
                          'numpy.datetime64 objects, continuing using '
                          'dummy netCDF4.datetime objects instead, reason:'
                          '{0}'.format(e), RuntimeWarning, stacklevel=3)
    return dates


def decode_cf_datetime(num_dates, units, calendar=None):
    """Given an array of numeric dates in netCDF format, convert it into a
    numpy array of date time objects.

    For standard (Gregorian) calendars, this function uses vectorized
    operations, which makes it much faster than netCDF4.num2date. In such a
    case, the returned array will be of type np.datetime64.

    See also
    --------
    netCDF4.num2date
    """
    num_dates = np.asarray(num_dates, dtype=float)
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
            # strings, in which case we fall back to using netCDF4
            raise OutOfBoundsDatetime

        dates = (pd.to_timedelta(flat_num_dates, delta) + ref_date).values

    except (OutOfBoundsDatetime, OverflowError):
        dates = _decode_datetime_with_netcdf4(flat_num_dates, units, calendar)

    return dates.reshape(num_dates.shape)


def _asarray_or_scalar(x):
    x = np.asarray(x)
    if x.ndim > 0:
        return x
    else:
        return x.item()


def decode_cf_timedelta(num_timedeltas, units):
    """Given an array of numeric timedeltas in netCDF format, convert it into a
    numpy timedelta64[ns] array.
    """
    num_timedeltas = _asarray_or_scalar(num_timedeltas)
    units = _netcdf_to_numpy_timeunit(units)
    result = pd.to_timedelta(num_timedeltas, unit=units, box=False)
    # NaT is returned unboxed with wrong units; this should be fixed in pandas
    if result.dtype != 'timedelta64[ns]':
        result = result.astype('timedelta64[ns]')
    return result


TIME_UNITS = frozenset(['days', 'hours', 'minutes', 'seconds',
                        'milliseconds', 'microseconds'])


def _infer_time_units_from_diff(unique_timedeltas):
    for time_unit, delta in [('days', 86400), ('hours', 3600),
                             ('minutes', 60), ('seconds', 1)]:
        unit_delta = np.timedelta64(10 ** 9 * delta, 'ns')
        diffs = unique_timedeltas / unit_delta
        if np.all(diffs == diffs.astype(int)):
            return time_unit
    return 'seconds'


def infer_datetime_units(dates):
    """Given an array of datetimes, returns a CF compatible time-unit string of
    the form "{time_unit} since {date[0]}", where `time_unit` is 'days',
    'hours', 'minutes' or 'seconds' (the first one that can evenly divide all
    unique time deltas in `dates`)
    """
    dates = pd.to_datetime(np.asarray(dates).ravel(), box=False)
    dates = dates[pd.notnull(dates)]
    unique_timedeltas = np.unique(np.diff(dates))
    units = _infer_time_units_from_diff(unique_timedeltas)
    reference_date = dates[0] if len(dates) > 0 else '1970-01-01'
    return '%s since %s' % (units, pd.Timestamp(reference_date))


def infer_timedelta_units(deltas):
    """Given an array of timedeltas, returns a CF compatible time-unit from
    {'days', 'hours', 'minutes' 'seconds'} (the first one that can evenly
    divide all unique time deltas in `deltas`)
    """
    deltas = pd.to_timedelta(np.asarray(deltas).ravel(), box=False)
    unique_timedeltas = np.unique(deltas[pd.notnull(deltas)])
    units = _infer_time_units_from_diff(unique_timedeltas)
    return units


def nctime_to_nptime(times):
    """Given an array of netCDF4.datetime objects, return an array of
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


def _encode_datetime_with_netcdf4(dates, units, calendar):
    """Fallback method for encoding dates using netCDF4-python.

    This method is more flexible than xarray's parsing using datetime64[ns]
    arrays but also slower because it loops over each element.
    """
    import netCDF4 as nc4

    if np.issubdtype(dates.dtype, np.datetime64):
        # numpy's broken datetime conversion only works for us precision
        dates = dates.astype('M8[us]').astype(datetime)

    def encode_datetime(d):
        return np.nan if d is None else nc4.date2num(d, units, calendar)

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
    netCDF4.date2num
    """
    dates = np.asarray(dates)

    if units is None:
        units = infer_datetime_units(dates)
    else:
        units = _cleanup_netcdf_time_units(units)

    if calendar is None:
        calendar = 'proleptic_gregorian'

    delta, ref_date = _unpack_netcdf_time_units(units)
    try:
        if calendar not in _STANDARD_CALENDARS or dates.dtype.kind == 'O':
            # parse with netCDF4 instead
            raise OutOfBoundsDatetime
        assert dates.dtype == 'datetime64[ns]'

        delta_units = _netcdf_to_numpy_timeunit(delta)
        time_delta = np.timedelta64(1, delta_units).astype('timedelta64[ns]')
        ref_date = np.datetime64(pd.Timestamp(ref_date))
        num = (dates - ref_date) / time_delta

    except (OutOfBoundsDatetime, OverflowError):
        num = _encode_datetime_with_netcdf4(dates, units, calendar)

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


class MaskedAndScaledArray(utils.NDArrayMixin):
    """Wrapper around array-like objects to create a new indexable object where
    values, when accessesed, are automatically scaled and masked according to
    CF conventions for packed and missing data values.

    New values are given by the formula:
        original_values * scale_factor + add_offset

    Values can only be accessed via `__getitem__`:

    >>> x = MaskedAndScaledArray(np.array([-99, -1, 0, 1, 2]), -99, 0.01, 1)
    >>> x
    MaskedAndScaledArray(array([-99, -1,  0,  1,  2]), fill_value=-99,
    scale_factor=0.01, add_offset=1)
    >>> x[:]
    array([  nan,  0.99,  1.  ,  1.01,  1.02]

    References
    ----------
    http://www.unidata.ucar.edu/software/netcdf/docs/BestPractices.html
    """
    def __init__(self, array, fill_value=None, scale_factor=None,
                 add_offset=None, dtype=float):
        """
        Parameters
        ----------
        array : array-like
            Original array of values to wrap
        fill_value : number, optional
            All values equal to fill_value in the original array are replaced
            by NaN.
        scale_factor : number, optional
            Multiply entries in the original array by this number.
        add_offset : number, optional
            After applying scale_factor, add this number to entries in the
            original array.
        """
        self.array = array
        self.fill_value = fill_value
        self.scale_factor = scale_factor
        self.add_offset = add_offset
        self._dtype = dtype

    @property
    def dtype(self):
        return np.dtype(self._dtype)

    def __getitem__(self, key):
        return mask_and_scale(self.array[key], self.fill_value,
                              self.scale_factor, self.add_offset, self._dtype)

    def __repr__(self):
        return ("%s(%r, fill_value=%r, scale_factor=%r, add_offset=%r, "
                "dtype=%r)" %
                (type(self).__name__, self.array, self.fill_value,
                 self.scale_factor, self.add_offset, self._dtype))


class DecodedCFDatetimeArray(utils.NDArrayMixin):
    """Wrapper around array-like objects to create a new indexable object where
    values, when accessed, are automatically converted into datetime objects
    using decode_cf_datetime.
    """
    def __init__(self, array, units, calendar=None):
        self.array = array
        self.units = units
        self.calendar = calendar
        # Verify at least one date can be decoded successfully.
        # Otherwise, tracebacks end up swallowed by Dataset.__repr__ when users
        # try to view their lazily decoded array.
        example_value = first_n_items(array, 1) or 0
        try:
            result = decode_cf_datetime(example_value, units, calendar)
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
            self._dtype = getattr(result, 'dtype', np.dtype('object'))

    @property
    def dtype(self):
        return self._dtype

    def __getitem__(self, key):
        return decode_cf_datetime(self.array[key], units=self.units,
                                  calendar=self.calendar)


class DecodedCFTimedeltaArray(utils.NDArrayMixin):
    """Wrapper around array-like objects to create a new indexable object where
    values, when accessed, are automatically converted into timedelta objects
    using decode_cf_timedelta.
    """
    def __init__(self, array, units):
        self.array = array
        self.units = units

    @property
    def dtype(self):
        return np.dtype('timedelta64[ns]')

    def __getitem__(self, key):
        return decode_cf_timedelta(self.array[key], units=self.units)


class CharToStringArray(utils.NDArrayMixin):
    """Wrapper around array-like objects to create a new indexable object where
    values, when accessed, are automatically concatenated along the last
    dimension.

    >>> CharToStringArray(np.array(['a', 'b', 'c']))[:]
    array('abc',
          dtype='|S3')
    """
    def __init__(self, array):
        """
        Parameters
        ----------
        array : array-like
            Original array of values to wrap.
        """
        self.array = array

    @property
    def dtype(self):
        return np.dtype('S' + str(self.array.shape[-1]))

    @property
    def shape(self):
        return self.array.shape[:-1]

    def __str__(self):
        if self.ndim == 0:
            # always return a unicode str if it's a single item for py3 compat
            return self[...].item().decode('utf-8')
        else:
            return repr(self)

    def __repr__(self):
        return '%s(%r)' % (type(self).__name__, self.array)

    def __getitem__(self, key):
        if self.array.ndim == 0:
            values = self.array[key]
        else:
            # require slicing the last dimension completely
            key = indexing.expanded_indexer(key, self.array.ndim)
            if key[-1] != slice(None):
                raise IndexError('too many indices')
            values = char_to_string(self.array[key])
        return values


class NativeEndiannessArray(utils.NDArrayMixin):
    """Decode arrays on the fly from non-native to native endianness

    This is useful for decoding arrays from netCDF3 files (which are all
    big endian) into native endianness, so they can be used with Cython
    functions, such as those found in bottleneck and pandas.

    >>> x = np.arange(5, dtype='>i2')

    >>> x.dtype
    dtype('>i2')

    >>> NativeEndianArray(x).dtype
    dtype('int16')

    >>> NativeEndianArray(x)[:].dtype
    dtype('int16')
    """
    def __init__(self, array):
        self.array = array

    @property
    def dtype(self):
        return np.dtype(self.array.dtype.kind + str(self.array.dtype.itemsize))

    def __getitem__(self, key):
        return np.asarray(self.array[key], dtype=self.dtype)


def string_to_char(arr):
    """Like netCDF4.stringtochar, but faster and more flexible.
    """
    # ensure the array is contiguous
    arr = np.array(arr, copy=False, order='C')
    kind = arr.dtype.kind
    if kind not in ['U', 'S']:
        raise ValueError('argument must be a string')
    return arr.reshape(arr.shape + (1,)).view(kind + '1')


def char_to_string(arr):
    """Like netCDF4.chartostring, but faster and more flexible.
    """
    # based on: http://stackoverflow.com/a/10984878/809705
    arr = np.array(arr, copy=False, order='C')
    kind = arr.dtype.kind
    if kind not in ['U', 'S']:
        raise ValueError('argument must be a string')
    return arr.view(kind + str(arr.shape[-1]))[..., 0]


def safe_setitem(dest, key, value):
    if key in dest:
        raise ValueError('Failed hard to prevent overwriting key %r' % key)
    dest[key] = value


def pop_to(source, dest, key, default=None):
    """
    A convenience function which pops a key k from source to dest.
    None values are not passed on.  If k already exists in dest an
    error is raised.
    """
    value = source.pop(key, None)
    if value is not None:
        safe_setitem(dest, key, value)
    return value


def _var_as_tuple(var):
    return var.dims, var.data, var.attrs.copy(), var.encoding.copy()


def maybe_encode_datetime(var):
    if np.issubdtype(var.dtype, np.datetime64):
        dims, data, attrs, encoding = _var_as_tuple(var)
        (data, units, calendar) = encode_cf_datetime(
            data, encoding.pop('units', None), encoding.pop('calendar', None))
        safe_setitem(attrs, 'units', units)
        safe_setitem(attrs, 'calendar', calendar)
        var = Variable(dims, data, attrs, encoding)
    return var


def maybe_encode_timedelta(var):
    if np.issubdtype(var.dtype, np.timedelta64):
        dims, data, attrs, encoding = _var_as_tuple(var)
        data, units = encode_cf_timedelta(
            data, encoding.pop('units', None))
        safe_setitem(attrs, 'units', units)
        var = Variable(dims, data, attrs, encoding)
    return var


def maybe_encode_offset_and_scale(var, needs_copy=True):
    if any(k in var.encoding for k in ['add_offset', 'scale_factor']):
        dims, data, attrs, encoding = _var_as_tuple(var)
        data = data.astype(dtype=float, copy=needs_copy)
        needs_copy = False
        if 'add_offset' in encoding:
            data -= pop_to(encoding, attrs, 'add_offset')
        if 'scale_factor' in encoding:
            data /= pop_to(encoding, attrs, 'scale_factor')
        var = Variable(dims, data, attrs, encoding)
    return var, needs_copy


def maybe_encode_fill_value(var, needs_copy=True):
    # replace NaN with the fill value
    if '_FillValue' in var.encoding:
        dims, data, attrs, encoding = _var_as_tuple(var)
        fill_value = pop_to(encoding, attrs, '_FillValue')
        if not pd.isnull(fill_value):
            data = ops.fillna(data, fill_value)
            needs_copy = False
        var = Variable(dims, data, attrs, encoding)
    return var, needs_copy


def maybe_encode_dtype(var, name=None):
    if 'dtype' in var.encoding:
        dims, data, attrs, encoding = _var_as_tuple(var)
        dtype = np.dtype(encoding.pop('dtype'))
        if dtype != var.dtype and dtype.kind != 'O':
            if np.issubdtype(dtype, np.integer):
                if (np.issubdtype(var.dtype, np.floating) and
                        '_FillValue' not in var.attrs):
                    warnings.warn('saving variable %s with floating '
                                  'point data as an integer dtype without '
                                  'any _FillValue to use for NaNs' % name,
                                  RuntimeWarning, stacklevel=3)
                data = ops.around(data)[...]
            if dtype == 'S1' and data.dtype != 'S1':
                data = string_to_char(np.asarray(data, 'S'))
                dims = dims + ('string%s' % data.shape[-1],)
            data = data.astype(dtype=dtype)
            var = Variable(dims, data, attrs, encoding)
    return var


def _infer_dtype(array):
    """Given an object array with no missing values, infer its dtype from its
    first element
    """
    if array.size == 0:
        dtype = np.dtype(float)
    else:
        dtype = np.array(array[(0,) * array.ndim]).dtype
        if dtype.kind in ['S', 'U']:
            # don't just use inferred dtype to avoid truncating arrays to
            # the length of their first element
            dtype = np.dtype(dtype.kind)
        elif dtype.kind == 'O':
            raise ValueError('unable to infer dtype; xarray cannot '
                             'serialize arbitrary Python objects')
    return dtype


def ensure_dtype_not_object(var):
    # TODO: move this from conventions to backends? (it's not CF related)
    if var.dtype.kind == 'O':
        dims, data, attrs, encoding = _var_as_tuple(var)
        missing = pd.isnull(data)
        if missing.any():
            # nb. this will fail for dask.array data
            non_missing_values = data[~missing]
            inferred_dtype = _infer_dtype(non_missing_values)

            if inferred_dtype.kind in ['S', 'U']:
                # There is no safe bit-pattern for NA in typical binary string
                # formats, we so can't set a fill_value. Unfortunately, this
                # means we won't be able to restore string arrays with missing
                # values.
                fill_value = ''
            else:
                # insist on using float for numeric values
                if not np.issubdtype(inferred_dtype, float):
                    inferred_dtype = np.dtype(float)
                fill_value = np.nan

            data = np.array(data, dtype=inferred_dtype, copy=True)
            data[missing] = fill_value
        else:
            data = data.astype(dtype=_infer_dtype(data))
        var = Variable(dims, data, attrs, encoding)
    return var


def encode_cf_variable(var, needs_copy=True, name=None):
    """
    Converts an Variable into an Variable which follows some
    of the CF conventions:

        - Nans are masked using _FillValue (or the deprecated missing_value)
        - Rescaling via: scale_factor and add_offset
        - datetimes are converted to the CF 'units since time' format
        - dtype encodings are enforced.

    Parameters
    ----------
    var : xarray.Variable
        A variable holding un-encoded data.

    Returns
    -------
    out : xarray.Variable
        A variable which has been encoded as described above.
    """
    var = maybe_encode_datetime(var)
    var = maybe_encode_timedelta(var)
    var, needs_copy = maybe_encode_offset_and_scale(var, needs_copy)
    var, needs_copy = maybe_encode_fill_value(var, needs_copy)
    var = maybe_encode_dtype(var, name)
    var = ensure_dtype_not_object(var)
    return var


def decode_cf_variable(var, concat_characters=True, mask_and_scale=True,
                       decode_times=True, decode_endianness=True):
    """
    Decodes a variable which may hold CF encoded information.

    This includes variables that have been masked and scaled, which
    hold CF style time variables (this is almost always the case if
    the dataset has been serialized) and which have strings encoded
    as character arrays.

    Parameters
    ----------
    var : Variable
        A variable holding potentially CF encoded information.
    concat_characters : bool
        Should character arrays be concatenated to strings, for
        example: ['h', 'e', 'l', 'l', 'o'] -> 'hello'
    mask_and_scale: bool
        Lazily scale (using scale_factor and add_offset) and mask
        (using _FillValue).
    decode_times : bool
        Decode cf times ('hours since 2000-01-01') to np.datetime64.
    decode_endianness : bool
        Decode arrays from non-native to native endianness.

    Returns
    -------
    out : Variable
        A variable holding the decoded equivalent of var
    """
    # use _data instead of data so as not to trigger loading data
    var = as_variable(var)
    data = var._data
    dimensions = var.dims
    attributes = var.attrs.copy()
    encoding = var.encoding.copy()

    original_dtype = data.dtype

    if concat_characters:
        if data.dtype.kind == 'S' and data.dtype.itemsize == 1:
            dimensions = dimensions[:-1]
            data = CharToStringArray(data)

    if mask_and_scale:
        if 'missing_value' in attributes:
            # missing_value is deprecated, but we still want to support it as
            # an alias for _FillValue.
            if ('_FillValue' in attributes and
                not utils.equivalent(attributes['_FillValue'],
                                     attributes['missing_value'])):
                raise ValueError("Discovered conflicting _FillValue "
                                 "and missing_value.  Considering "
                                 "opening the offending dataset using "
                                 "decode_cf=False, corrected the attributes",
                                 "and decoding explicitly using "
                                 "xarray.conventions.decode_cf(ds)")
            attributes['_FillValue'] = attributes.pop('missing_value')

        fill_value = np.array(pop_to(attributes, encoding, '_FillValue'))
        if fill_value.size > 1:
            warnings.warn("variable has multiple fill values {0}, decoding "
                          "all values to NaN.".format(str(fill_value)),
                          RuntimeWarning, stacklevel=3)
        scale_factor = pop_to(attributes, encoding, 'scale_factor')
        add_offset = pop_to(attributes, encoding, 'add_offset')
        if ((fill_value is not None and not np.any(pd.isnull(fill_value))) or
                scale_factor is not None or add_offset is not None):
            if fill_value.dtype.kind in ['U', 'S']:
                dtype = object
            else:
                dtype = float
            data = MaskedAndScaledArray(data, fill_value, scale_factor,
                                        add_offset, dtype)

    if decode_times and 'units' in attributes:
        if 'since' in attributes['units']:
            # datetime
            units = pop_to(attributes, encoding, 'units')
            calendar = pop_to(attributes, encoding, 'calendar')
            data = DecodedCFDatetimeArray(data, units, calendar)
        elif attributes['units'] in TIME_UNITS:
            # timedelta
            units = pop_to(attributes, encoding, 'units')
            data = DecodedCFTimedeltaArray(data, units)

    if decode_endianness and not data.dtype.isnative:
        # do this last, so it's only done if we didn't already unmask/scale
        data = NativeEndiannessArray(data)
        original_dtype = data.dtype

    if 'dtype' in encoding:
        if original_dtype != encoding['dtype']:
            warnings.warn("CF decoding is overwriting dtype")
    else:
        encoding['dtype'] = original_dtype

    return Variable(dimensions, indexing.LazilyIndexedArray(data),
                    attributes, encoding=encoding)


def decode_cf_variables(variables, attributes, concat_characters=True,
                        mask_and_scale=True, decode_times=True,
                        decode_coords=True, drop_variables=None):
    """
    Decode a several CF encoded variables.

    See: decode_cf_variable
    """
    dimensions_used_by = defaultdict(list)
    for v in variables.values():
        for d in v.dims:
            dimensions_used_by[d].append(v)

    def stackable(dim):
        # figure out if a dimension can be concatenated over
        if dim in variables:
            return False
        for v in dimensions_used_by[dim]:
            if v.dtype.kind != 'S' or dim != v.dims[-1]:
                return False
        return True

    coord_names = set()

    if isinstance(drop_variables, basestring):
        drop_variables = [drop_variables]
    elif drop_variables is None:
        drop_variables = []
    drop_variables = set(drop_variables)

    new_vars = OrderedDict()
    for k, v in iteritems(variables):
        if k in drop_variables:
            continue
        concat = (concat_characters and v.dtype.kind == 'S' and v.ndim > 0 and
                  stackable(v.dims[-1]))
        new_vars[k] = decode_cf_variable(
            v, concat_characters=concat, mask_and_scale=mask_and_scale,
            decode_times=decode_times)
        if decode_coords:
            var_attrs = new_vars[k].attrs
            if 'coordinates' in var_attrs:
                coord_str = var_attrs['coordinates']
                var_coord_names = coord_str.split()
                if all(k in variables for k in var_coord_names):
                    new_vars[k].encoding['coordinates'] = coord_str
                    del var_attrs['coordinates']
                    coord_names.update(var_coord_names)

    if decode_coords and 'coordinates' in attributes:
        attributes = OrderedDict(attributes)
        coord_names.update(attributes.pop('coordinates').split())

    return new_vars, attributes, coord_names


def decode_cf(obj, concat_characters=True, mask_and_scale=True,
              decode_times=True, decode_coords=True, drop_variables=None):
    """Decode the given Dataset or Datastore according to CF conventions into
    a new Dataset.

    Parameters
    ----------
    obj : Dataset or DataStore
        Object to decode.
    concat_characters : bool, optional
        Should character arrays be concatenated to strings, for
        example: ['h', 'e', 'l', 'l', 'o'] -> 'hello'
    mask_and_scale: bool, optional
        Lazily scale (using scale_factor and add_offset) and mask
        (using _FillValue).
    decode_times : bool, optional
        Decode cf times (e.g., integers since 'hours since 2000-01-01') to
        np.datetime64.
    decode_coords : bool, optional
        Use the 'coordinates' attribute on variable (or the dataset itself) to
        identify coordinates.
    drop_variables: string or iterable, optional
        A variable or list of variables to exclude from being parsed from the
        dataset.This may be useful to drop variables with problems or
        inconsistent values.

    Returns
    -------
    decoded : Dataset
    """
    from .core.dataset import Dataset
    from .backends.common import AbstractDataStore

    if isinstance(obj, Dataset):
        vars = obj._variables
        attrs = obj.attrs
        extra_coords = set(obj.coords)
        file_obj = obj._file_obj
    elif isinstance(obj, AbstractDataStore):
        vars, attrs = obj.load()
        extra_coords = set()
        file_obj = obj
    else:
        raise TypeError('can only decode Dataset or DataStore objects')

    vars, attrs, coord_names = decode_cf_variables(
        vars, attrs, concat_characters, mask_and_scale, decode_times,
        decode_coords, drop_variables=drop_variables)
    ds = Dataset(vars, attrs=attrs)
    ds = ds.set_coords(coord_names.union(extra_coords))
    ds._file_obj = file_obj
    return ds


def cf_decoder(variables, attributes,
               concat_characters=True, mask_and_scale=True,
               decode_times=True):
    """
    Decode a set of CF encoded variables and attributes.

    See Also, decode_cf_variable

    Parameters
    ----------
    variables : dict
        A dictionary mapping from variable name to xarray.Variable
    attributes : dict
        A dictionary mapping from attribute name to value
    concat_characters : bool
        Should character arrays be concatenated to strings, for
        example: ['h', 'e', 'l', 'l', 'o'] -> 'hello'
    mask_and_scale: bool
        Lazily scale (using scale_factor and add_offset) and mask
        (using _FillValue).
    decode_times : bool
        Decode cf times ('hours since 2000-01-01') to np.datetime64.

    Returns
    -------
    decoded_variables : dict
        A dictionary mapping from variable name to xarray.Variable objects.
    decoded_attributes : dict
        A dictionary mapping from attribute name to values.
    """
    variables, attributes, _ = decode_cf_variables(
        variables, attributes, concat_characters, mask_and_scale, decode_times)
    return variables, attributes


def _encode_coordinates(variables, attributes, non_dim_coord_names):
    # calculate global and variable specific coordinates
    non_dim_coord_names = set(non_dim_coord_names)
    global_coordinates = non_dim_coord_names.copy()
    variable_coordinates = defaultdict(set)
    for coord_name in non_dim_coord_names:
        target_dims = variables[coord_name].dims
        for k, v in variables.items():
            if (k not in non_dim_coord_names and k not in v.dims and
                    any(d in target_dims for d in v.dims)):
                variable_coordinates[k].add(coord_name)
                global_coordinates.discard(coord_name)

    variables = OrderedDict((k, v.copy(deep=False))
                            for k, v in variables.items())

    # These coordinates are saved according to CF conventions
    for var_name, coord_names in variable_coordinates.items():
        attrs = variables[var_name].attrs
        if 'coordinates' in attrs:
            raise ValueError('cannot serialize coordinates because variable '
                             "%s already has an attribute 'coordinates'"
                             % var_name)
        attrs['coordinates'] = ' '.join(map(str, coord_names))

    # These coordinates are not associated with any particular variables, so we
    # save them under a global 'coordinates' attribute so xarray can roundtrip
    # the dataset faithfully. Because this serialization goes beyond CF
    # conventions, only do it if necessary.
    # Reference discussion:
    # http://mailman.cgd.ucar.edu/pipermail/cf-metadata/2014/057771.html
    if global_coordinates:
        attributes = OrderedDict(attributes)
        if 'coordinates' in attributes:
            raise ValueError('cannot serialize coordinates because the global '
                             "attribute 'coordinates' already exists")
        attributes['coordinates'] = ' '.join(map(str, global_coordinates))

    return variables, attributes


def encode_dataset_coordinates(dataset):
    """Encode coordinates on the given dataset object into variable specific
    and global attributes.

    When possible, this is done according to CF conventions.

    Parameters
    ----------
    dataset : Dataset
        Object to encode.

    Returns
    -------
    variables : dict
    attrs : dict
    """
    non_dim_coord_names = set(dataset.coords) - set(dataset.dims)
    return _encode_coordinates(dataset._variables, dataset.attrs,
                               non_dim_coord_names=non_dim_coord_names)


def cf_encoder(variables, attributes):
    """
    A function which takes a dicts of variables and attributes
    and encodes them to conform to CF conventions as much
    as possible.  This includes masking, scaling, character
    array handling, and CF-time encoding.

    Decode a set of CF encoded variables and attributes.

    See Also, decode_cf_variable

    Parameters
    ----------
    variables : dict
        A dictionary mapping from variable name to xarray.Variable
    attributes : dict
        A dictionary mapping from attribute name to value

    Returns
    -------
    encoded_variables : dict
        A dictionary mapping from variable name to xarray.Variable,
    encoded_attributes : dict
        A dictionary mapping from attribute name to value

    See also: encode_cf_variable
    """
    new_vars = OrderedDict((k, encode_cf_variable(v, name=k))
                           for k, v in iteritems(variables))
    return new_vars, attributes
