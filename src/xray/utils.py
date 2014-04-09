import netCDF4 as nc4
import operator
from collections import OrderedDict, Mapping
from datetime import datetime

import numpy as np
import pandas as pd

import xarray


def expanded_indexer(key, ndim):
    """Given a key for indexing an ndarray, return an equivalent key which is a
    tuple with length equal to the number of dimensions.

    The expansion is done by replacing all `Ellipsis` items with the right
    number of full slices and then padding the key with full slices so that it
    reaches the appropriate dimensionality.
    """
    if not isinstance(key, tuple):
        # numpy treats non-tuple keys equivalent to tuples of length 1
        key = (key,)
    new_key = []
    # handling Ellipsis right is a little tricky, see:
    # http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing
    found_ellipsis = False
    for k in key:
        if k is Ellipsis:
            if not found_ellipsis:
                new_key.extend((ndim + 1 - len(key)) * [slice(None)])
                found_ellipsis = True
            else:
                new_key.append(slice(None))
        else:
            new_key.append(k)
    new_key.extend((ndim - len(new_key)) * [slice(None)])
    return tuple(new_key)


def orthogonal_indexer(key, shape):
    """Given a key for orthogonal array indexing, returns an equivalent key
    suitable for indexing a numpy.ndarray with fancy indexing.
    """
    def expand_array(k, length):
        if isinstance(k, slice):
            return np.arange(k.start or 0, k.stop or length, k.step or 1)
        else:
            arr = np.asarray(k)
            if (not np.issubdtype(arr.dtype, int)
                    and not np.issubdtype(arr.dtype, bool)):
                raise ValueError("invalid subkey '%s' for integer based array "
                                 'indexing; all subkeys must be slices, '
                                 'integers or sequences of integers or '
                                 'Booleans' % k)
            if arr.ndim != 1:
                raise ValueError('orthogonal array indexing only supports '
                                 '1d arrays')
            return arr
    # replace Ellipsis objects with slices
    key = list(expanded_indexer(key, len(shape)))
    # replace 1d arrays and slices with broadcast compatible arrays
    # note: we treat integers separately (instead of turning them into 1d
    # arrays) because integers (and only integers) collapse axes when used with
    # __getitem__
    non_int_keys = [n for n, k in enumerate(key) if not isinstance(k, int)]

    def full_slices_unselected(n_list):
        def all_full_slices(key_index):
            return all(isinstance(key[n], slice) and key[n] == slice(None)
                       for n in key_index)
        if not n_list:
            return n_list
        elif all_full_slices(range(n_list[0] + 1)):
            return full_slices_unselected(n_list[1:])
        elif all_full_slices(range(n_list[-1], len(key))):
            return full_slices_unselected(n_list[:-1])
        else:
            return n_list

    # However, testing suggests it is OK to keep contiguous sequences of full
    # slices at the start or the end of the key. Keeping slices around (when
    # possible) instead of converting slices to arrays significantly speeds up
    # indexing.
    # (Honestly, I don't understand when it's not OK to keep slices even in
    # between integer indices if as array is somewhere in the key, but such are
    # the admittedly mind-boggling ways of numpy's advanced indexing.)
    array_keys = full_slices_unselected(non_int_keys)

    array_indexers = np.ix_(*(expand_array(key[n], shape[n])
                              for n in array_keys))
    for i, n in enumerate(array_keys):
        key[n] = array_indexers[i]
    return tuple(key)


def remap_loc_indexers(indices, indexers):
    """Given mappings of XArray indices and label based indexers, return
    equivalent location based indexers.
    """
    new_indexers = OrderedDict()
    for dim, loc in indexers.iteritems():
        index = indices[dim].index
        if isinstance(loc, slice):
            indexer = index.slice_indexer(loc.start, loc.stop, loc.step)
        else:
            loc = np.asarray(loc)
            if loc.ndim == 0:
                indexer = index.get_loc(np.asscalar(loc))
            else:
                indexer = index.get_indexer(loc)
                if np.any(indexer < 0):
                    raise ValueError('not all values found in index %r' % dim)
        new_indexers[dim] = indexer
    return new_indexers


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
    num_dates = np.asarray(num_dates).astype(float)
    if calendar is None:
        calendar = 'standard'

    min_num = np.min(num_dates)
    max_num = np.max(num_dates)
    min_date = nc4.num2date(min_num, units, calendar)
    if num_dates.size > 1:
        max_date = nc4.num2date(max_num, units, calendar)
    else:
        max_date = min_date

    if (calendar not in ['standard', 'gregorian', 'proleptic_gregorian']
            or min_date < datetime(1678, 1, 1)
            or max_date > datetime(2262, 4, 11)):
        dates = nc4.num2date(num_dates, units, calendar)
    else:
        # we can safely use np.datetime64 with nanosecond precision (pandas
        # likes ns precision so it can directly make DatetimeIndex objects)
        if min_num == max_num:
            # we can't safely divide by max_num - min_num
            dates = np.repeat(np.datetime64(min_date), num_dates.size)
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


def safe_isnan(arr):
    """Like np.isnan for floating point arrays, but returns a vector of the
    repeated value `False` for non-floating point arrays

    This is necessary because numpy does not support NaN or isnan for integer
    valued arrays.
    """
    if np.issubdtype(arr.dtype, float):
        return np.isnan(arr)
    else:
        return np.zeros(arr.shape, dtype=bool)


def allclose_or_equiv(arr1, arr2, rtol=1e-5, atol=1e-8):
    """Like np.allclose, but also allows values to be NaN in both arrays
    """
    if arr1.shape != arr2.shape:
        return False
    nan_indices = safe_isnan(arr1)
    if not (nan_indices == safe_isnan(arr2)).all():
        return False
    if arr1.ndim > 0:
        arr1 = arr1[~nan_indices]
        arr2 = arr2[~nan_indices]
    elif nan_indices:
        # 0-d arrays can't be indexed, so just check if the value is NaN
        return True
    return np.allclose(arr1, arr2, rtol=rtol, atol=atol)


def xarray_equal(v1, v2, rtol=1e-05, atol=1e-08, check_attributes=True):
    """True if two objects have the same dimensions, attributes and data;
    otherwise False.

    This function is necessary because `v1 == v2` for XArrays and DataArrays
    does element-wise comparisions (like numpy.ndarrays).
    """
    def data_equiv(arr1, arr2):
        exact_dtypes = [np.datetime64, np.timedelta64, np.string_]
        if any(any(np.issubdtype(arr.dtype, t) for t in exact_dtypes)
               or arr.dtype == object for arr in [arr1, arr2]):
            return np.array_equal(arr1, arr2)
        else:
            return allclose_or_equiv(arr1, arr2, rtol=rtol, atol=atol)

    v1, v2 = map(xarray.as_xarray, [v1, v2])
    return (v1.dimensions == v2.dimensions
            and (not check_attributes
                 or dict_equal(v1.attributes, v2.attributes))
            and (v1._data is v2._data or data_equiv(v1.data, v2.data)))


def safe_cast_to_index(array):
    """Given an array, safely cast it to a pandas.Index

    Unlike pandas.Index, if the array has dtype=object or dtype=timedelta64,
    this function will not attempt to do automatic type conversion but will
    always return an index with dtype=object.
    """
    kwargs = {}
    if isinstance(array, np.ndarray):
        if array.dtype == object or array.dtype == np.timedelta64:
            kwargs['dtype'] = object
    return pd.Index(array, **kwargs)


def multi_index_from_product(iterables, names=None):
    """Like pandas.MultiIndex.from_product, but not buggy

    Contains work-around for https://github.com/pydata/pandas/issues/6439
    """
    # note: pd.MultiIndex.from_product is new in pandas-0.13.1
    coords = [np.asarray(v) for v in iterables]
    return pd.MultiIndex.from_product(coords, names=names)


def update_safety_check(first_dict, second_dict, compat=operator.eq):
    """Check the safety of updating one dictionary with another.

    Raises ValueError if dictionaries have non-compatible values for any key,
    where compatibility is determined by identity (they are the same item) or
    the `compat` function.

    Parameters
    ----------
    first_dict, second_dict : dict-like
        All items in the second dictionary are checked against for conflicts
        against items in the first dictionary.
    compat : function, optional
        Binary operator to determine if two values are compatible. By default,
        checks for equality.
    """
    for k, v in second_dict.iteritems():
        if (k in first_dict and
                not (v is first_dict[k] or compat(v, first_dict[k]))):
            raise ValueError('unsafe to merge dictionaries without '
                             'overriding values; conflicting key %r' % k)


def remove_incompatible_items(first_dict, second_dict, compat=operator.eq):
    """Remove incompatible items from the first dictionary in-place.

    Items are retained if their keys are found in both dictionaries and the
    values are compatible.

    Parameters
    ----------
    first_dict, second_dict : dict-like
        Mappings to merge.
    compat : function, optional
        Binary operator to determine if two values are compatible. By default,
        checks for equality.
    """
    for k, v in second_dict.iteritems():
        if k in first_dict and not compat(v, first_dict[k]):
            del first_dict[k]


def dict_equal(first, second):
    """Test equality of two dict-like objects.  If any of the values
    are numpy arrays, compare them for equality correctly.

    Parameters
    ----------
    first, second : dict-like
        Dictionaries to compare for equality

    Returns
    -------
    equals : bool
        True if the dictionaries are equal
    """
    k1 = sorted(first.keys())
    k2 = sorted(second.keys())
    if k1 != k2:
        return False
    for k in k1:
        v1 = first[k]
        v2 = second[k]
        if isinstance(v1, np.ndarray) != isinstance(v2, np.ndarray):
            # one is an ndarray, other is not
            return False
        elif (isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray)):
            if not np.array_equal(v1, v2):
                return False
        elif v1 != v2:
            return False
    return True


def ordered_dict_intersection(first_dict, second_dict, compat=operator.eq):
    """Return the intersection of two dictionaries as a new OrderedDict.

    Items are retained if their keys are found in both dictionaries and the
    values are compatible.

    Parameters
    ----------
    first_dict, second_dict : dict-like
        Mappings to merge.
    compat : function, optional
        Binary operator to determine if two values are compatible. By default,
        checks for equality.

    Returns
    -------
    intersection : OrderedDict
        Intersection of the contents.
    """
    new_dict = OrderedDict(first_dict)
    remove_incompatible_items(new_dict, second_dict, compat)
    return new_dict


class Frozen(Mapping):
    """Wrapper around an object implementing the mapping interface to make it
    immutable. If you really want to modify the mapping, the mutable version is
    saved under the `mapping` attribute.
    """
    def __init__(self, mapping):
        self.mapping = mapping

    def __getitem__(self, key):
        return self.mapping[key]

    def __iter__(self):
        return iter(self.mapping)

    def __len__(self):
        return len(self.mapping)

    def __contains__(self, key):
        return key in self.mapping

    def __repr__(self):
        return '%s(%r)' % (type(self).__name__, self.mapping)


def FrozenOrderedDict(*args, **kwargs):
    return Frozen(OrderedDict(*args, **kwargs))
