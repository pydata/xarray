import netCDF4 as nc4
import operator
from collections import OrderedDict, Mapping
from datetime import datetime

import numpy as np
import pandas as pd


def expanded_indexer(key, ndim):
    """Given a key for indexing an ndarray, return an equivalent key which is a
    tuple with length equal to the number of dimensions

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
    suitable for indexing a numpy.ndarray with fancy indexing
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
    array_indexers = np.ix_(*(expand_array(key[n], shape[n])
                              for n in non_int_keys))
    for i, n in enumerate(non_int_keys):
        key[n] = array_indexers[i]
    return tuple(key)


def remap_loc_indexers(indices, indexers):
    """Given mappings of indices and label based indexers, return equivalent
    location based indexers
    """
    new_indexers = OrderedDict()
    for dim, loc in indexers.iteritems():
        index = indices[dim]
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


def num2datetimeindex(num_dates, units, calendar=None):
    """Convert an array of numeric dates in netCDF format into a
    pandas.DatetimeIndex

    For standard (Gregorian) calendars, this function uses vectorized
    operations, which makes it much faster than netCDF4.num2date.
    """
    num_dates = np.asarray(num_dates)
    if calendar is None:
        calendar = 'standard'
    start_date = nc4.num2date(num_dates[0], units, calendar)
    if (num_dates.size < 2
            or calendar not in ['standard', 'gregorian', 'proleptic_gregorian']
            or (start_date < datetime(1582, 10, 15)
                and calendar != 'proleptic_gregorian')):
        dates = nc4.num2date(num_dates, units, calendar)
    else:
        first_dates = nc4.num2date(num_dates[:2], units, calendar)
        first_time_delta = np.timedelta64(first_dates[1] - first_dates[0])
        num_delta = (num_dates - num_dates[0]) / (num_dates[1] - num_dates[0])
        dates = first_time_delta * num_delta + np.datetime64(first_dates[0])
    return pd.Index(dates)


def variable_equal(v1, v2):
    """True if two objects have the same dimensions, attributes and data;
    otherwise False

    This function is necessary because `v1 == v2` for variables and dataviews
    does element-wise comparisions (like numpy.ndarrays).
    """
    if (v1.dimensions == v2.dimensions
            and v1.attributes == v2.attributes):
        try:
            # if _data is identical, skip checking arrays by value
            if v1._data is v2._data:
                return True
        except AttributeError:
            # _data is not part of the public interface, so it's okay if its
            # missing
            pass
        # TODO: replace this with a NaN safe version.
        # see: pandas.core.common.array_equivalent
        return np.array_equal(v1.data, v2.data)
    else:
        return False


def update_safety_check(first_dict, second_dict, compat=operator.eq):
    """Check the safety of updating one dictionary with another

    Raises ValueError if dictionaries have non-compatible values for any key,
    where compatibility is determined by the `compat` function.

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
        if k in first_dict and not compat(v, first_dict[k]):
            raise ValueError('unsafe to merge dictionaries without '
                             'overriding values')


def remove_incompatible_items(first_dict, second_dict, compat=operator.eq):
    """Remove incompatible items from the first dictionary in-place

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


def ordered_dict_intersection(first_dict, second_dict, compat=operator.eq):
    """Return the intersection of two dictionaries as a new OrderedDict

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
