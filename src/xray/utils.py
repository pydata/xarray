import netCDF4 as nc4
import operator
from collections import OrderedDict, Mapping
from datetime import datetime

import numpy as np
import pandas as pd


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
            k = np.asarray(k)
            if k.ndim != 1:
                raise ValueError('orthogonal array indexing only supports '
                                 '1d arrays')
            return k
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
    """Given mappings of indices and label based indexers, return equivalent
    location based indexers.
    """
    new_indexers = OrderedDict()
    for dim, loc in indexers.iteritems():
        index = indices[dim].data
        if isinstance(loc, slice):
            indexer = index.slice_indexer(loc.start, loc.stop, loc.step)
        else:
            try:
                indexer = index.get_loc(loc)
            except TypeError:
                # value is a list or array
                indexer = index.get_indexer(np.asarray(loc))
                if np.any(indexer < 0):
                    raise ValueError('not all values found in index %r' % dim)
        new_indexers[dim] = indexer
    return new_indexers


def first_value(arr):
    """Helper function for extracting the first value from an n-dimensional
    array (including n=0)
    """
    return np.nditer(arr, ['refs_ok']).next()[()]


def decode_cf_datetime(num_dates, units, calendar=None):
    """Given an array of numeric dates in netCDF format, convert it into a
    numpy array of date time objects.

    For standard (Gregorian) calendars, this function uses vectorized
    operations, which makes it much faster than netCDF4.num2date. In such a
    case, the returned array will be of type np.datetime64.

    Warning
    -------
    This function checks only the first date to determine that it falls within
    the range expressable by the np.datetime64 dtype with nanosecond precision
    (roughly the years 1678 through 2361). If subsequent dates fall outside
    this range, the array may possibly overflow.

    See also
    --------
    netCDF4.num2date
    """
    num_dates = np.asarray(num_dates)
    if calendar is None:
        calendar = 'standard'

    first_date = nc4.num2date(first_value(num_dates), units, calendar)
    if (calendar not in ['standard', 'gregorian', 'proleptic_gregorian']
            or first_date < datetime(1678, 1, 1)
            or first_date > datetime(2262, 4, 11)):
        dates = nc4.num2date(num_dates, units, calendar)
    else:
        # assume we can safely use np.datetime64 arrays
        # TODO: catch the unlikely corner cases described in the warning in the
        # function docstring
        if num_dates.size == 1:
            dates = np.array(first_date)
        else:
            # Calculate the date as a np.datetime64 array from linear scaling
            # of two example dates calculated via num2date.
            # Despite the names "first_date" and "second_date", it is not
            # actually necessary for these dates to be in order.
            flat_num_dates = num_dates.reshape(-1)
            ex_nums = flat_num_dates[:2].astype(float)
            second_date = nc4.num2date(ex_nums[1], units, calendar)
            ex_time_delta = np.timedelta64(second_date - first_date)
            num_delta = ((flat_num_dates - ex_nums[0])
                         / (ex_nums[1] - ex_nums[0]))
            dates = ex_time_delta * num_delta + np.datetime64(first_date)
        # restore original shape
        dates = dates.reshape(num_dates.shape)
        if num_dates.ndim == 0:
            dates = dates[()]
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
        # for now, don't bother doing any trickery like num_to_datetime64 to
        # convert dates to numbers faster
        dates_array = np.asarray(dates)
        date_index = pd.DatetimeIndex(dates_array.reshape(-1))
        dates = date_index.to_pydatetime().reshape(dates_array.shape)
        if dates_array.ndim == 0:
            # unpack dates out of the array
            dates = dates[()]
    num = nc4.date2num(dates, units, calendar)
    return (num, units, calendar)


def allclose_or_equiv(arr1, arr2, rtol=1e-5, atol=1e-8):
    """Like np.allclose, but also allows values to be NaN in both arrays
    """
    if arr1.shape != arr2.shape:
        return False
    nan_indices = np.isnan(arr1)
    if not (nan_indices == np.isnan(arr2)).all():
        return False
    if arr1.ndim > 0:
        arr1 = arr1[~nan_indices]
        arr2 = arr2[~nan_indices]
    elif nan_indices:
        # 0-d arrays can't be indexed, so just check if the value is NaN
        return True
    return np.allclose(arr1, arr2, rtol=rtol, atol=atol)


def xarray_equal(v1, v2, rtol=1e-05, atol=1e-08):
    """True if two objects have the same dimensions, attributes and data;
    otherwise False.

    This function is necessary because `v1 == v2` for XArrays and DatasetArrays
    does element-wise comparisions (like numpy.ndarrays).
    """
    if (v1.dimensions == v2.dimensions
        and dict_equal(v1.attributes, v2.attributes)):
        try:
            # if _data is identical, skip checking arrays by value
            if v1._data is v2._data:
                return True
        except AttributeError:
            # _data is not part of the public interface, so it's okay if its
            # missing
            pass

        def is_floating(arr):
            return np.issubdtype(arr.dtype, float)

        data1 = v1.data
        data2 = v2.data
        if hasattr(data1, 'equals'):
            # handle pandas.Index objects
            return data1.equals(data2)
        elif is_floating(data1) or is_floating(data2):
            return allclose_or_equiv(data1, data2)
        else:
            return np.array_equal(data1, data2)
    else:
        return False


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
            return False # one is an ndarray, other is not
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
