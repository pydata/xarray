import numpy as np
import pandas as pd
import warnings
from collections import defaultdict, OrderedDict
from datetime import datetime

from . import indexing
from . import utils
from .pycompat import iteritems
import xray

# standard calendars recognized by netcdftime
_STANDARD_CALENDARS = {'standard', 'gregorian', 'proleptic_gregorian'}


def mask_and_scale(array, fill_value=None, scale_factor=None, add_offset=None):
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
        by NaN.
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
    # cast to float to insure NaN is meaningful
    values = np.array(array, dtype=float, copy=True)
    if fill_value is not None and not np.isnan(fill_value):
        if values.ndim > 0:
            values[values == fill_value] = np.nan
        elif values == fill_value:
            values = np.array(np.nan)
    if scale_factor is not None:
        values *= scale_factor
    if add_offset is not None:
        values += add_offset
    return values


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
    import netCDF4 as nc4
    num_dates = np.asarray(num_dates).astype(float)
    if calendar is None:
        calendar = 'standard'

    def nan_safe_num2date(num):
        return pd.NaT if np.isnan(num) else nc4.num2date(num, units, calendar)

    min_num = np.nanmin(num_dates)
    max_num = np.nanmax(num_dates)
    min_date = nan_safe_num2date(min_num)
    if num_dates.size > 1:
        max_date = nan_safe_num2date(max_num)
    else:
        max_date = min_date

    if ((calendar not in _STANDARD_CALENDARS
            or min_date.year < 1678 or max_date.year >= 2262)
            and min_date is not pd.NaT):

        dates = nc4.num2date(num_dates, units, calendar)

        if min_date.year >= 1678 and max_date.year < 2262:
            try:
                dates = nctime_to_nptime(dates)
            except ValueError as e:
                warnings.warn('Unable to decode time axis into full '
                              'numpy.datetime64 objects, continuing using '
                              'dummy netCDF4.datetime objects instead, reason:'
                              '{0}'.format(e), RuntimeWarning, stacklevel=2)
                dates = np.asarray(dates)
        else:
            warnings.warn('Unable to decode time axis into full '
                          'numpy.datetime64 objects, continuing using dummy '
                          'netCDF4.datetime objects instead, reason: dates out'
                          ' of range', RuntimeWarning, stacklevel=2)
            dates = np.asarray(dates)

    else:
        # we can safely use np.datetime64 with nanosecond precision (pandas
        # likes ns precision so it can directly make DatetimeIndex objects)
        if pd.isnull(min_num):
            # pandas.NaT doesn't cast to numpy.datetime64('NaT'), so handle it
            # separately
            dates = np.repeat(np.datetime64('NaT'), num_dates.size)
        elif min_num == max_num:
            # we can't safely divide by max_num - min_num
            dates = np.repeat(np.datetime64(min_date), num_dates.size)
            if dates.size > 1:
                # don't bother with one element, since it will be fixed at
                # min_date and isn't indexable anyways
                dates[np.isnan(num_dates)] = np.datetime64('NaT')
        else:
            # Calculate the date as a np.datetime64 array from linear scaling
            # of the max and min dates calculated via num2date.
            flat_num_dates = num_dates.reshape(-1)
            # Use second precision for the timedelta to decrease the chance of
            # a numeric overflow
            time_delta = np.timedelta64(max_date - min_date).astype('m8[s]')
            if time_delta != max_date - min_date:
                raise ValueError('unable to exactly represent max_date minus'
                                 'min_date with second precision')
            # apply the numerator and denominator separately so we don't need
            # to cast to floating point numbers under the assumption that all
            # dates can be given exactly with ns precision
            numerator = flat_num_dates - min_num
            denominator = max_num - min_num
            dates = (time_delta * numerator / denominator
                     + np.datetime64(min_date))
        # restore original shape and ensure dates are given in ns
        dates = dates.reshape(num_dates.shape).astype('M8[ns]')

    return dates


def guess_time_units(dates):
    """Given an array of dates suitable for input to `pandas.DatetimeIndex`,
    returns a CF compatible time-unit string of the form "{time_unit} since
    {date[0]}", where `time_unit` is 'days', 'hours', 'minutes' or 'seconds'
    (the first one that can evenly divide all unique time deltas in `dates`)
    """
    dates = pd.DatetimeIndex(np.asarray(dates).reshape(-1))
    unique_timedeltas = np.unique(np.diff(dates.values))
    for time_unit, delta in [('days', '1 days'), ('hours', '3600s'),
                             ('minutes', '60s'), ('seconds', '1s')]:
        unit_delta = pd.to_timedelta(delta)
        diffs = unique_timedeltas / unit_delta
        if np.all(diffs == diffs.astype(int)):
            break
    else:
        raise ValueError('could not automatically determine time units')
    return '%s since %s' % (time_unit, dates[0])


def nctime_to_nptime(times):
    """Given an array of netCDF4.datetime objects, return an array of
    numpy.datetime64 objects of the same size"""
    times = np.asarray(times)
    new = np.empty(times.shape, dtype='M8[ns]')
    for i, t in np.ndenumerate(times):
        new[i] = np.datetime64(datetime(*t.timetuple()[:6]))
    return new


def encode_cf_datetime(dates, units=None, calendar=None):
    """Given an array of datetime objects, returns the tuple `(num, units,
    calendar)` suitable for a CF complient time variable.

    Unlike encode_cf_datetime, this function does not (yet) speedup encoding
    of datetime64 arrays. However, unlike `date2num`, it can handle datetime64
    arrays.

    See also
    --------
    netCDF4.date2num
    """
    import netCDF4 as nc4
    if units is None:
        units = guess_time_units(dates)
    if calendar is None:
        calendar = 'proleptic_gregorian'

    if (isinstance(dates, np.ndarray)
            and np.issubdtype(dates.dtype, np.datetime64)):
        # for now, don't bother doing any trickery like decode_cf_datetime to
        # convert dates to numbers faster
        # note: numpy's broken datetime conversion only works for us precision
        dates = np.asarray(dates).astype('M8[us]').astype(datetime)

    if hasattr(dates, 'ndim') and dates.ndim == 0:
        # unpack dates because date2num doesn't like 0-dimensional arguments
        dates = dates.item()

    num = nc4.date2num(dates, units, calendar)
    return (num, units, calendar)


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
                 add_offset=None):
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

    @property
    def dtype(self):
        return np.dtype('float')

    def __getitem__(self, key):
        return mask_and_scale(self.array[key], self.fill_value,
                              self.scale_factor, self.add_offset)

    def __repr__(self):
        return ("%s(%r, fill_value=%r, scale_factor=%r, add_offset=%r)" %
                (type(self).__name__, self.array, self.fill_value,
                 self.scale_factor, self.add_offset))



class DecodedCFDatetimeArray(utils.NDArrayMixin):
    """Wrapper around array-like objects to create a new indexable object where
    values, when accessesed, are automatically converted into datetime objects
    using decode_cf_datetime.
    """
    def __init__(self, array, units, calendar=None):
        self.array = array
        self.units = units
        self.calendar = calendar

    @property
    def dtype(self):
        return np.dtype('datetime64[ns]')

    def __getitem__(self, key):
        return decode_cf_datetime(self.array, units=self.units,
                                  calendar=self.calendar)


class CharToStringArray(utils.NDArrayMixin):
    """Wrapper around array-like objects to create a new indexable object where
    values, when accessesed, are automatically concatenated along the last
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
            if  key[-1] != slice(None):
                raise IndexError('too many indices')
            values = char_to_string(self.array[key])
        return values


def string_to_char(arr):
    """Like netCDF4.stringtochar, but faster and more flexible.
    """
    arr = np.asarray(arr)
    kind = arr.dtype.kind
    if kind not in ['U', 'S']:
        raise ValueError('argument must be a string')
    return arr.view(kind + '1').reshape(*[arr.shape + (-1,)])


def char_to_string(arr):
    """Like netCDF4.chartostring, but faster and more flexible.
    """
    # based on: http://stackoverflow.com/a/10984878/809705
    arr = np.asarray(arr)
    kind = arr.dtype.kind
    if kind not in ['U', 'S']:
        raise ValueError('argument must be a string')
    return arr.view(kind + str(arr.shape[-1]))[..., 0]


def encode_cf_variable(var):
    """Converts an Variable into an Variable suitable for saving as a netCDF
    variable
    """
    dimensions = var.dimensions
    data = var.values
    attributes = var.attrs.copy()
    encoding = var.encoding.copy()

    if (np.issubdtype(data.dtype, np.datetime64)
            or (data.dtype.kind == 'O'
                and isinstance(data.reshape(-1)[0], datetime))):
        # encode datetime arrays into numeric arrays
        (data, units, calendar) = encode_cf_datetime(
            data, encoding.pop('units', None), encoding.pop('calendar', None))
        attributes['units'] = units
        attributes['calendar'] = calendar
    elif data.dtype.kind == 'O':
        # Occasionally, one will end up with variables with dtype=object
        # (likely because they were created from pandas objects which don't
        # maintain dtype careful). This code makes a best effort attempt to
        # encode them into a dtype that NETCDF can handle by inspecting the
        # dtype of the first element.
        dtype = np.array(data.reshape(-1)[0]).dtype
        # N.B. the "astype" call below will fail if data cannot be cast to the
        # type of its first element (which is probably the only sensible thing
        # to do).
        data = np.asarray(data).astype(dtype)

    def get_to(source, dest, k):
        v = source.get(k)
        dest[k] = v
        return v

    # unscale/mask
    if any(k in encoding for k in ['add_offset', 'scale_factor']):
        data = np.array(data, dtype=float, copy=True)
        if 'add_offset' in encoding:
            data -= get_to(encoding, attributes, 'add_offset')
        if 'scale_factor' in encoding:
            data /= get_to(encoding, attributes, 'scale_factor')

    # replace NaN with the fill value
    if '_FillValue' in encoding:
        if encoding['_FillValue'] is np.nan:
            attributes['_FillValue'] = np.nan
        else:
            nans = np.isnan(data)
            if nans.any():
                data[nans] = get_to(encoding, attributes, '_FillValue')

    # restore original dtype
    if 'dtype' in encoding and encoding['dtype'].kind != 'O':
        if np.issubdtype(encoding['dtype'], int):
            data = data.round()
        data = data.astype(encoding['dtype'])

    return xray.Variable(dimensions, data, attributes, encoding=encoding)


def decode_cf_variable(var, concat_characters=True, mask_and_scale=True,
                       decode_times=True):
    # use _data instead of data so as not to trigger loading data
    data = var._data
    dimensions = var.dimensions
    attributes = var.attrs.copy()
    encoding = var.encoding.copy()

    def pop_to(source, dest, k):
        """
        A convenience function which pops a key k from source to dest.
        None values are not passed on.  If k already exists in dest an
        error is raised.
        """
        v = source.pop(k, None)
        if v is not None:
            if k in dest:
                raise ValueError("Failed hard to prevent overwriting key %s" % k)
            dest[k] = v
        return v

    if 'dtype' in encoding:
        if var.data.dtype != encoding['dtype']:
            raise ValueError("Refused to overwrite dtype")
    encoding['dtype'] = data.dtype

    if concat_characters:
        if data.dtype.kind == 'S' and data.dtype.itemsize == 1:
            dimensions = dimensions[:-1]
            data = CharToStringArray(data)

    if mask_and_scale:
        fill_value = pop_to(attributes, encoding, '_FillValue')
        scale_factor = pop_to(attributes, encoding, 'scale_factor')
        add_offset = pop_to(attributes, encoding, 'add_offset')
        if ((fill_value is not None and not np.isnan(fill_value))
                or scale_factor is not None or add_offset is not None):
            data = MaskedAndScaledArray(data, fill_value, scale_factor,
                                        add_offset)

    if decode_times:
        if 'units' in attributes and 'since' in attributes['units']:
            units = pop_to(attributes, encoding, 'units')
            calendar = pop_to(attributes, encoding, 'calendar')
            data = DecodedCFDatetimeArray(data, units, calendar)

    return xray.Variable(dimensions, indexing.LazilyIndexedArray(data),
                         attributes, encoding=encoding)


def decode_cf_variables(variables, concat_characters=True, mask_and_scale=True,
                        decode_times=True):
    """Decode a bunch of CF variables together.
    """
    dimensions_used_by = defaultdict(list)
    for v in variables.values():
        for d in v.dimensions:
            dimensions_used_by[d].append(v)

    def stackable(dim):
        # figure out if a dimension can be concatenated over
        if dim in variables:
            return False
        for v in dimensions_used_by[dim]:
            if v.dtype.kind != 'S' or dim != v.dimensions[-1]:
                return False
        return True

    new_vars = OrderedDict()
    for k, v in iteritems(variables):
        concat = (concat_characters and v.dtype.kind == 'S' and v.ndim > 0 and
                  stackable(v.dimensions[-1]))
        new_vars[k] = decode_cf_variable(
            v, concat_characters=concat, mask_and_scale=mask_and_scale,
            decode_times=decode_times)
    return new_vars
