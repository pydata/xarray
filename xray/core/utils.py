"""Internal utilties; not for external use
"""
import datetime
import functools
import itertools
import warnings
from collections import Mapping, MutableMapping

import numpy as np
import pandas as pd

from .pycompat import basestring, iteritems, PY3, OrderedDict


def alias_warning(old_name, new_name, stacklevel=3):
    warnings.warn('%s has been deprecated and renamed to %s'
                  % (old_name, new_name),
                  FutureWarning, stacklevel=stacklevel)


def function_alias(obj, old_name):
    @functools.wraps(obj)
    def wrapper(*args, **kwargs):
        alias_warning(old_name, obj.__name__)
        return obj(*args, **kwargs)
    return wrapper


def class_alias(obj, old_name):
    class Wrapper(obj):
        def __new__(cls, *args, **kwargs):
            alias_warning(old_name, obj.__name__)
            return super(Wrapper, cls).__new__(cls, *args, **kwargs)
    Wrapper.__name__ = obj.__name__
    return Wrapper


def allclose_or_equiv(arr1, arr2, rtol=1e-5, atol=1e-8):
    """Like np.allclose, but also allows values to be NaN in both arrays
    """
    arr1, arr2 = np.asarray(arr1), np.asarray(arr2)
    if arr1.shape != arr2.shape:
        return False
    return np.isclose(arr1, arr2, rtol=rtol, atol=atol, equal_nan=True).all()


def isnull(obj):
    """Like pd.isnull, but with a bug-fix for 0d object arrays.
    """
    obj = np.asarray(obj)
    if obj.ndim == 0:
        # work around for pd.isnull not working for 0-dimensional object
        # arrays: https://github.com/pydata/pandas/pull/7176 (should be fixed
        # in pandas 0.14)
        # use .item() instead of keeping around 0-dimensional arrays to avoid
        # the numpy quirk where object arrays are checked as equal by identity
        # (hence NaN in an object array is equal to itself):
        obj = obj.item()
        return obj != obj
    else:
        return pd.isnull(obj)


def array_equiv(arr1, arr2):
    """Like np.array_equal, but also allows values to be NaN in both arrays
    """
    arr1, arr2 = np.asarray(arr1), np.asarray(arr2)
    if arr1.shape != arr2.shape:
        return False
    return ((arr1 == arr2) | (isnull(arr1) & isnull(arr2))).all()


def safe_cast_to_index(array):
    """Given an array, safely cast it to a pandas.Index.

    If it is already a pandas.Index, return it unchanged.

    Unlike pandas.Index, if the array has dtype=object or dtype=timedelta64,
    this function will not attempt to do automatic type conversion but will
    always return an index with dtype=object.
    """
    if isinstance(array, pd.Index):
        index = array
    elif hasattr(array, 'to_index'):
        index = array.to_index()
    else:
        kwargs = {}
        if hasattr(array, 'dtype'):
            if array.dtype == object or array.dtype.kind == 'm':
                kwargs['dtype'] = object
        index = pd.Index(np.asarray(array), **kwargs)
    return index


def multi_index_from_product(iterables, names=None):
    """Like pandas.MultiIndex.from_product, but with a bug fix.
    """
    # fixed in 0.14: https://github.com/pydata/pandas/issues/6439
    # note: pd.MultiIndex.from_product is new in pandas-0.13.1
    coords = [np.asarray(v) for v in iterables]
    return pd.MultiIndex.from_product(coords, names=names)


def equivalent(first, second):
    """Compare two objects for equivalence (identity or equality), using
    array_equiv if either object is an ndarray
    """
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        return array_equiv(first, second)
    else:
        return first is second or first == second


def peek_at(iterable):
    """Returns the first value from iterable, as well as a new iterable with
    the same content as the original iterable
    """
    gen = iter(iterable)
    peek = next(gen)
    return peek, itertools.chain([peek], gen)


def update_safety_check(first_dict, second_dict, compat=equivalent):
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
        checks for equivalence.
    """
    for k, v in iteritems(second_dict):
        if k in first_dict and not compat(v, first_dict[k]):
            raise ValueError('unsafe to merge dictionaries without '
                             'overriding values; conflicting key %r' % k)


def remove_incompatible_items(first_dict, second_dict, compat=equivalent):
    """Remove incompatible items from the first dictionary in-place.

    Items are retained if their keys are found in both dictionaries and the
    values are compatible.

    Parameters
    ----------
    first_dict, second_dict : dict-like
        Mappings to merge.
    compat : function, optional
        Binary operator to determine if two values are compatible. By default,
        checks for equivalence.
    """
    for k, v in iteritems(first_dict):
        if (k not in second_dict
                or (k in second_dict and not compat(v, second_dict[k]))):
            del first_dict[k]


def is_dict_like(value):
    return hasattr(value, '__getitem__') and hasattr(value, 'keys')


def is_scalar(value):
    """np.isscalar only work on primitive numeric types and (bizarrely)
    excludes 0-d ndarrays; this version does more comprehensive checks
    """
    if np.isscalar(value):
        return True
    if hasattr(value, 'ndim'):
        return value.ndim == 0
    if isinstance(value, datetime.datetime) or value is None:
        return True
    return False


def dict_equiv(first, second, compat=equivalent):
    """Test equivalence of two dict-like objects. If any of the values are
    numpy arrays, compare them correctly.

    Parameters
    ----------
    first, second : dict-like
        Dictionaries to compare for equality
    compat : function, optional
        Binary operator to determine if two values are compatible. By default,
        checks for equivalence.

    Returns
    -------
    equals : bool
        True if the dictionaries are equal
    """
    for k in first:
        if k not in second or not compat(first[k], second[k]):
            return False
    for k in second:
        if k not in first:
            return False
    return True


def ordered_dict_intersection(first_dict, second_dict, compat=equivalent):
    """Return the intersection of two dictionaries as a new OrderedDict.

    Items are retained if their keys are found in both dictionaries and the
    values are compatible.

    Parameters
    ----------
    first_dict, second_dict : dict-like
        Mappings to merge.
    compat : function, optional
        Binary operator to determine if two values are compatible. By default,
        checks for equivalence.

    Returns
    -------
    intersection : OrderedDict
        Intersection of the contents.
    """
    new_dict = OrderedDict(first_dict)
    remove_incompatible_items(new_dict, second_dict, compat)
    return new_dict


class SingleSlotPickleMixin(object):
    """Mixin class to add the ability to pickle objects whose state is defined
    by a single __slots__ attribute. Only necessary under Python 2.
    """
    def __getstate__(self):
        return getattr(self, self.__slots__[0])

    def __setstate__(self, state):
        setattr(self, self.__slots__[0], state)


class Frozen(Mapping, SingleSlotPickleMixin):
    """Wrapper around an object implementing the mapping interface to make it
    immutable. If you really want to modify the mapping, the mutable version is
    saved under the `mapping` attribute.
    """
    __slots__ = ['mapping']

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


class SortedKeysDict(MutableMapping, SingleSlotPickleMixin):
    """An wrapper for dictionary-like objects that always iterates over its
    items in sorted order by key but is otherwise equivalent to the underlying
    mapping.
    """
    __slots__ = ['mapping']

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

    def copy(self):
        return type(self)(self.mapping.copy())


class ChainMap(MutableMapping, SingleSlotPickleMixin):
    """Partial backport of collections.ChainMap from Python>=3.3

    Don't return this from any public APIs, since some of the public methods
    for a MutableMapping are missing (they will raise a NotImplementedError)
    """
    __slots__ = ['maps']

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
        seen = set()
        for mapping in self.maps:
            for item in mapping:
                if item not in seen:
                    yield item
                    seen.add(item)

    def __len__(self):
        raise len(iter(self))


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
        # cast to int so that shape = () gives size = 1
        return int(np.prod(self.shape))

    def __len__(self):
        try:
            return self.shape[0]
        except IndexError:
            raise TypeError('len() of unsized object')

    def __array__(self, dtype=None):
        return np.asarray(self[...], dtype=dtype)

    def __getitem__(self, key):
        return self.array[key]

    def __repr__(self):
        return '%s(array=%r)' % (type(self).__name__, self.array)
