# TODO: separate out these utilities into different modules based on whether
# they are for internal or external use
import functools
import operator
import warnings
from collections import OrderedDict, Mapping, MutableMapping
from datetime import datetime

import numpy as np
import pandas as pd

import xray
import variable


def alias_warning(old_name, new_name, stacklevel=2):
    warnings.warn('%s has been renamed to %s; this alias will be removed '
                  "before xray's initial release" % (old_name, new_name),
                  FutureWarning, stacklevel=stacklevel)


def function_alias(obj, old_name):
    @functools.wraps(obj)
    def wrapper(*args, **kwargs):
        alias_warning(old_name, obj.__name__, stacklevel=3)
        return obj(*args, **kwargs)
    return wrapper


def class_alias(obj, old_name):
    class Wrapper(obj):
        def __new__(cls, *args, **kwargs):
            alias_warning(old_name, obj.__name__, stacklevel=3)
            return super(Wrapper, cls).__new__(cls, *args, **kwargs)
    Wrapper.__name__ = obj.__name__
    return Wrapper


def as_array_or_item(values, dtype=None):
    """Return the given values as a numpy array of the indicated dtype, or as
    an individual value if it's a 0-dimensional object array or datetime.
    """
    if isinstance(values, datetime):
        # shortcut because if you try to make a datetime or Timestamp object
        # into an array with the proper dtype, it is liable to be silently
        # converted into an integer instead :(
        return values
    values = np.asarray(values, dtype=dtype)
    if values.ndim == 0 and values.dtype.kind == 'O':
        # unpack 0d object arrays to be consistent with numpy
        values = values.item()
        if isinstance(values, pd.Timestamp):
            # turn Timestamps back into datetime64 objects
            values = np.datetime64(values)
    return values


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
            if arr.dtype.kind not in ('i', 'b'):
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


def convert_label_indexer(index, label, index_name=''):
    """Given a pandas.Index (or xray.CoordVariable) and labels (e.g., from
    __getitem__) for one dimension, return an indexer suitable for indexing an
    ndarray along that dimension
    """
    if isinstance(label, slice):
        indexer = index.slice_indexer(label.start, label.stop, label.step)
    else:
        label = np.asarray(label)
        if label.ndim == 0:
            indexer = index.get_loc(np.asscalar(label))
        else:
            indexer = index.get_indexer(label)
            if np.any(indexer < 0):
                raise ValueError('not all values found in index %r'
                                 % index_name)
    return indexer


def remap_label_indexers(data_obj, indexers):
    """Given an xray data object and label based indexers, return a mapping
    of equivalent location based indexers.
    """
    return {dim: convert_label_indexer(data_obj.coordinates[dim], label, dim)
            for dim, label in indexers.iteritems()}


def squeeze(xray_obj, dimensions, dimension=None):
    """Squeeze the dimensions of an xray object."""
    if dimension is None:
        dimension = [d for d, s in dimensions.iteritems() if s == 1]
    else:
        if isinstance(dimension, basestring):
            dimension = [dimension]
        if any(dimensions[k] > 1 for k in dimension):
            raise ValueError('cannot select a dimension to squeeze out '
                             'which has length greater than one')
    return xray_obj.indexed(**{dim: 0 for dim in dimension})


def allclose_or_equiv(arr1, arr2, rtol=1e-5, atol=1e-8):
    """Like np.allclose, but also allows values to be NaN in both arrays
    """
    if arr1.shape != arr2.shape:
        return False
    return np.isclose(arr1, arr2, rtol=rtol, atol=atol, equal_nan=True).all()


def array_equiv(arr1, arr2):
    """Like np.array_equal, but also allows values to be NaN in both arrays
    """
    arr1, arr2 = np.asarray(arr1), np.asarray(arr2)
    if arr1.shape != arr2.shape:
        return False
    # we could make this faster by not-checking for null values if the dtype
    # does not support them, but the logic would get more convoluted.
    # using pd.isnull lets us defer the NaN handling to pandas (and unlike
    # np.isnan it works on every dtype).
    return ((arr1 == arr2) | (pd.isnull(arr1) & pd.isnull(arr2))).all()


def variable_equal(v1, v2, check_attributes=True):
    """True if two objects have the same dimensions, attributes and data;
    otherwise False.

    This function is necessary because `v1 == v2` for XArrays and DataArrays
    does element-wise comparisions (like numpy.ndarrays).
    """
    warnings.warn('use the `equals` or `identical` methods instead of calling '
                  'utils.variable_equal directly; this alias will be removed '
                  'at some point',
                  FutureWarning, stacklevel=2)
    v1, v2 = map(variable.as_variable, [v1, v2])
    if check_attributes:
        return v1.identical(v2)
    else:
        return v1.equals(v2)

xarray_equal = function_alias(variable_equal, 'xarray_equal')


def safe_cast_to_index(array):
    """Given an array, safely cast it to a pandas.Index.

    If it is already a pandas.Index, return it unchanged.

    Unlike pandas.Index, if the array has dtype=object or dtype=timedelta64,
    this function will not attempt to do automatic type conversion but will
    always return an index with dtype=object.
    """
    if isinstance(array, pd.Index):
        index = array
    elif isinstance(array, xray.Coordinate):
        index = array.as_index
    else:
        kwargs = {}
        if hasattr(array, 'dtype'):
            if array.dtype == object or array.dtype == np.timedelta64:
                kwargs['dtype'] = object
        index = pd.Index(np.asarray(array), **kwargs)
    return index


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
        if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
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


class NDArrayMixin(object):
    """Mixin class for making wrappers of N-dimensional arrays that conform to
    the ndarray interface required for the data argument to Variable objects.

    A subclass should set the `array` property and override one or more of
    `dtype`, `shape` and `__getitem__`.
    """
    @property
    def dtype(self):
        return self.array.dtype

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return np.prod(self.shape)

    def __len__(self):
        try:
            return self.shape[0]
        except IndexError:
            raise TypeError('len() of unsized object')

    def __array__(self):
        return np.asarray(self[...])

    def __getitem__(self, key):
        raise self.array[key]
