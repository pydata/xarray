# TODO: separate out these utilities into different modules based on whether
# they are for internal or external use
import netCDF4 as nc4
import operator
from collections import OrderedDict, Mapping, MutableMapping
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


class SortedKeysDict(MutableMapping):
    """An wrapper for dictionary-like objects that always iterates over its
    items in sorted order by key but is otherwise equivalent to the underlying
    mapping.
    """
    def __init__(self, mapping=None):
        self.mapping = {} if mapping is None else mapping

    def __getitem__(self, key):
        return self.mapping[key]

    def __setitem__(self, key, value):
        self.mapping[key] = value

    def __delitem__(self, key):
        del self.mapping[key]

    def __iter__(self):
        return iter(sorted(self.mapping))

    def __len__(self):
        return len(self.mapping)

    def __contains__(self, key):
        return key in self.mapping

    def __repr__(self):
        return '%s(%r)' % (type(self).__name__, self.mapping)


class ChainMap(MutableMapping):
    """Partial backport of collections.ChainMap from Python>=3.3

    Don't return this from any public APIs, since some of the public methods
    for a MutableMapping are missing (they will raise a NotImplementedError)
    """
    def __init__(self, *maps):
        self.maps = maps

    def __getitem__(self, key):
        for mapping in self.maps:
            try:
                return mapping[key]
            except KeyError:
                pass
        raise KeyError(key)

    def __setitem__(self, key, value):
        self.maps[0][key] = value

    def __delitem__(self, value):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
