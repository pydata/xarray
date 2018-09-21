"""Internal utilties; not for external use
"""
from __future__ import absolute_import, division, print_function

import contextlib
import functools
import itertools
import re
import warnings
from collections import Iterable, Mapping, MutableMapping, MutableSet

import numpy as np
import pandas as pd

from .options import OPTIONS
from .pycompat import (
    OrderedDict, basestring, bytes_type, dask_array_type, iteritems)


def alias_message(old_name, new_name):
    return '%s has been deprecated. Use %s instead.' % (old_name, new_name)


def alias_warning(old_name, new_name, stacklevel=3):
    warnings.warn(alias_message(old_name, new_name), FutureWarning,
                  stacklevel=stacklevel)


def alias(obj, old_name):
    assert isinstance(old_name, basestring)

    @functools.wraps(obj)
    def wrapper(*args, **kwargs):
        alias_warning(old_name, obj.__name__)
        return obj(*args, **kwargs)
    wrapper.__doc__ = alias_message(old_name, obj.__name__)
    return wrapper


def _maybe_cast_to_cftimeindex(index):
    from ..coding.cftimeindex import CFTimeIndex

    if not OPTIONS['enable_cftimeindex']:
        return index
    else:
        if index.dtype == 'O':
            try:
                return CFTimeIndex(index)
            except (ImportError, TypeError):
                return index
        else:
            return index


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
        if hasattr(array, 'dtype') and array.dtype.kind == 'O':
            kwargs['dtype'] = object
        index = pd.Index(np.asarray(array), **kwargs)
    return _maybe_cast_to_cftimeindex(index)


def multiindex_from_product_levels(levels, names=None):
    """Creating a MultiIndex from a product without refactorizing levels.

    Keeping levels the same gives back the original labels when we unstack.

    Parameters
    ----------
    levels : sequence of pd.Index
        Values for each MultiIndex level.
    names : optional sequence of objects
        Names for each level.

    Returns
    -------
    pandas.MultiIndex
    """
    if any(not isinstance(lev, pd.Index) for lev in levels):
        raise TypeError('levels must be a list of pd.Index objects')

    split_labels, levels = zip(*[lev.factorize() for lev in levels])
    labels_mesh = np.meshgrid(*split_labels, indexing='ij')
    labels = [x.ravel() for x in labels_mesh]
    return pd.MultiIndex(levels, labels, sortorder=0, names=names)


def maybe_wrap_array(original, new_array):
    """Wrap a transformed array with __array_wrap__ is it can be done safely.

    This lets us treat arbitrary functions that take and return ndarray objects
    like ufuncs, as long as they return an array with the same shape.
    """
    # in case func lost array's metadata
    if isinstance(new_array, np.ndarray) and new_array.shape == original.shape:
        return original.__array_wrap__(new_array)
    else:
        return new_array


def equivalent(first, second):
    """Compare two objects for equivalence (identity or equality), using
    array_equiv if either object is an ndarray
    """
    # TODO: refactor to avoid circular import
    from . import duck_array_ops
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        return duck_array_ops.array_equiv(first, second)
    else:
        return ((first is second) or
                (first == second) or
                (pd.isnull(first) and pd.isnull(second)))


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
    for k in list(first_dict):
        if (k not in second_dict or
            (k in second_dict and
                not compat(first_dict[k], second_dict[k]))):
            del first_dict[k]


def is_dict_like(value):
    return hasattr(value, 'keys') and hasattr(value, '__getitem__')


def is_full_slice(value):
    return isinstance(value, slice) and value == slice(None)


def either_dict_or_kwargs(pos_kwargs, kw_kwargs, func_name):
    if pos_kwargs is not None:
        if not is_dict_like(pos_kwargs):
            raise ValueError('the first argument to .%s must be a dictionary'
                             % func_name)
        if kw_kwargs:
            raise ValueError('cannot specify both keyword and positional '
                             'arguments to .%s' % func_name)
        return pos_kwargs
    else:
        return kw_kwargs


def is_scalar(value):
    """Whether to treat a value as a scalar.

    Any non-iterable, string, or 0-D array
    """
    return (
        getattr(value, 'ndim', None) == 0 or
        isinstance(value, (basestring, bytes_type)) or not
        isinstance(value, (Iterable, ) + dask_array_type))


def is_valid_numpy_dtype(dtype):
    try:
        np.dtype(dtype)
    except (TypeError, ValueError):
        return False
    else:
        return True


def to_0d_object_array(value):
    """Given a value, wrap it in a 0-D numpy.ndarray with dtype=object."""
    result = np.empty((), dtype=object)
    result[()] = value
    return result


def to_0d_array(value):
    """Given a value, wrap it in a 0-D numpy.ndarray."""
    if np.isscalar(value) or (isinstance(value, np.ndarray) and
                              value.ndim == 0):
        return np.array(value)
    else:
        return to_0d_object_array(value)


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

    def __delitem__(self, value):  # pragma: no cover
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


class OrderedSet(MutableSet):
    """A simple ordered set.

    The API matches the builtin set, but it preserves insertion order of
    elements, like an OrderedDict.
    """

    def __init__(self, values=None):
        self._ordered_dict = OrderedDict()
        if values is not None:
            self |= values

    # Required methods for MutableSet

    def __contains__(self, value):
        return value in self._ordered_dict

    def __iter__(self):
        return iter(self._ordered_dict)

    def __len__(self):
        return len(self._ordered_dict)

    def add(self, value):
        self._ordered_dict[value] = None

    def discard(self, value):
        del self._ordered_dict[value]

    # Additional methods

    def update(self, values):
        self |= values

    def __repr__(self):
        return '%s(%r)' % (type(self).__name__, list(self))


class NdimSizeLenMixin(object):
    """Mixin class that extends a class that defines a ``shape`` property to
    one that also defines ``ndim``, ``size`` and ``__len__``.
    """
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


class NDArrayMixin(NdimSizeLenMixin):
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

    def __getitem__(self, key):
        return self.array[key]

    def __repr__(self):
        return '%s(array=%r)' % (type(self).__name__, self.array)


class ReprObject(object):
    """Object that prints as the given value, for use with sentinel values."""

    def __init__(self, value):  # type: str
        self._value = value

    def __repr__(self):
        return self._value


@contextlib.contextmanager
def close_on_error(f):
    """Context manager to ensure that a file opened by xarray is closed if an
    exception is raised before the user sees the file object.
    """
    try:
        yield
    except Exception:
        f.close()
        raise


def is_remote_uri(path):
    return bool(re.search('^https?\://', path))


def is_uniform_spaced(arr, **kwargs):
    """Return True if values of an array are uniformly spaced and sorted.

    >>> is_uniform_spaced(range(5))
    True
    >>> is_uniform_spaced([-4, 0, 100])
    False

    kwargs are additional arguments to ``np.isclose``
    """
    arr = np.array(arr, dtype=float)
    diffs = np.diff(arr)
    return np.isclose(diffs.min(), diffs.max(), **kwargs)


def hashable(v):
    """Determine whether `v` can be hashed."""
    try:
        hash(v)
    except TypeError:
        return False
    return True


def not_implemented(*args, **kwargs):
    return NotImplemented


def decode_numpy_dict_values(attrs):
    """Convert attribute values from numpy objects to native Python objects,
    for use in to_dict"""
    attrs = dict(attrs)
    for k, v in attrs.items():
        if isinstance(v, np.ndarray):
            attrs[k] = v.tolist()
        elif isinstance(v, np.generic):
            attrs[k] = np.asscalar(v)
    return attrs


def ensure_us_time_resolution(val):
    """Convert val out of numpy time, for use in to_dict.
    Needed because of numpy bug GH#7619"""
    if np.issubdtype(val.dtype, np.datetime64):
        val = val.astype('datetime64[us]')
    elif np.issubdtype(val.dtype, np.timedelta64):
        val = val.astype('timedelta64[us]')
    return val


class HiddenKeyDict(MutableMapping):
    '''
    Acts like a normal dictionary, but hides certain keys.
    '''
    # ``__init__`` method required to create instance from class.

    def __init__(self, data, hidden_keys):
        self._data = data
        if type(hidden_keys) not in (list, tuple):
            raise TypeError("hidden_keys must be a list or tuple")
        self._hidden_keys = hidden_keys

    def _raise_if_hidden(self, key):
        if key in self._hidden_keys:
            raise KeyError('Key `%r` is hidden.' % key)

    # The next five methods are requirements of the ABC.
    def __setitem__(self, key, value):
        self._raise_if_hidden(key)
        self._data[key] = value

    def __getitem__(self, key):
        self._raise_if_hidden(key)
        return self._data[key]

    def __delitem__(self, key):
        self._raise_if_hidden(key)
        del self._data[key]

    def __iter__(self):
        for k in self._data:
            if k not in self._hidden_keys:
                yield k

    def __len__(self):
        num_hidden = sum([k in self._hidden_keys for k in self._data])
        return len(self._data) - num_hidden


def to_numeric(array, offset=None, datetime_unit=None, dtype=float):
    """
    Make datetime array float

    offset: Scalar with the same type of array or None
        If None, subtract minimum values to reduce round off error
    datetime_unit: None or any of {'Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms',
        'us', 'ns', 'ps', 'fs', 'as'}
    dtype: target dtype
    """
    if array.dtype.kind not in ['m', 'M']:
        return array.astype(dtype)
    if offset is None:
        offset = np.min(array)
    array = array - offset

    if datetime_unit:
        return (array / np.timedelta64(1, datetime_unit)).astype(dtype)
    return array.astype(dtype)
