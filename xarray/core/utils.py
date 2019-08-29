"""Internal utilties; not for external use
"""
import contextlib
import functools
import itertools
import os.path
import re
import warnings
from collections import OrderedDict
from typing import (
    AbstractSet,
    Any,
    Callable,
    Container,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSet,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    cast,
)

import numpy as np
import pandas as pd

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")


def _check_inplace(inplace: Optional[bool]) -> None:
    if inplace is not None:
        raise TypeError(
            "The `inplace` argument has been removed from xarray. "
            "You can achieve an identical effect with python's standard assignment."
        )


def alias_message(old_name: str, new_name: str) -> str:
    return "%s has been deprecated. Use %s instead." % (old_name, new_name)


def alias_warning(old_name: str, new_name: str, stacklevel: int = 3) -> None:
    warnings.warn(
        alias_message(old_name, new_name), FutureWarning, stacklevel=stacklevel
    )


def alias(obj: Callable[..., T], old_name: str) -> Callable[..., T]:
    assert isinstance(old_name, str)

    @functools.wraps(obj)
    def wrapper(*args, **kwargs):
        alias_warning(old_name, obj.__name__)
        return obj(*args, **kwargs)

    wrapper.__doc__ = alias_message(old_name, obj.__name__)
    return wrapper


def _maybe_cast_to_cftimeindex(index: pd.Index) -> pd.Index:
    from ..coding.cftimeindex import CFTimeIndex

    if len(index) > 0 and index.dtype == "O":
        try:
            return CFTimeIndex(index)
        except (ImportError, TypeError):
            return index
    else:
        return index


def maybe_cast_to_coords_dtype(label, coords_dtype):
    if coords_dtype.kind == "f" and not isinstance(label, slice):
        label = np.asarray(label, dtype=coords_dtype)
    return label


def safe_cast_to_index(array: Any) -> pd.Index:
    """Given an array, safely cast it to a pandas.Index.

    If it is already a pandas.Index, return it unchanged.

    Unlike pandas.Index, if the array has dtype=object or dtype=timedelta64,
    this function will not attempt to do automatic type conversion but will
    always return an index with dtype=object.
    """
    if isinstance(array, pd.Index):
        index = array
    elif hasattr(array, "to_index"):
        index = array.to_index()
    else:
        kwargs = {}
        if hasattr(array, "dtype") and array.dtype.kind == "O":
            kwargs["dtype"] = object
        index = pd.Index(np.asarray(array), **kwargs)
    return _maybe_cast_to_cftimeindex(index)


def multiindex_from_product_levels(
    levels: Sequence[pd.Index], names: Sequence[str] = None
) -> pd.MultiIndex:
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
        raise TypeError("levels must be a list of pd.Index objects")

    split_labels, levels = zip(*[lev.factorize() for lev in levels])
    labels_mesh = np.meshgrid(*split_labels, indexing="ij")
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


def equivalent(first: T, second: T) -> bool:
    """Compare two objects for equivalence (identity or equality), using
    array_equiv if either object is an ndarray. If both objects are lists,
    equivalent is sequentially called on all the elements.
    """
    # TODO: refactor to avoid circular import
    from . import duck_array_ops

    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        return duck_array_ops.array_equiv(first, second)
    elif isinstance(first, list) or isinstance(second, list):
        return list_equiv(first, second)
    else:
        return (
            (first is second)
            or (first == second)
            or (pd.isnull(first) and pd.isnull(second))
        )


def list_equiv(first, second):
    equiv = True
    if len(first) != len(second):
        return False
    else:
        for f, s in zip(first, second):
            equiv = equiv and equivalent(f, s)
    return equiv


def peek_at(iterable: Iterable[T]) -> Tuple[T, Iterator[T]]:
    """Returns the first value from iterable, as well as a new iterator with
    the same content as the original iterable
    """
    gen = iter(iterable)
    peek = next(gen)
    return peek, itertools.chain([peek], gen)


def update_safety_check(
    first_dict: MutableMapping[K, V],
    second_dict: Mapping[K, V],
    compat: Callable[[V, V], bool] = equivalent,
) -> None:
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
    for k, v in second_dict.items():
        if k in first_dict and not compat(v, first_dict[k]):
            raise ValueError(
                "unsafe to merge dictionaries without "
                "overriding values; conflicting key %r" % k
            )


def remove_incompatible_items(
    first_dict: MutableMapping[K, V],
    second_dict: Mapping[K, V],
    compat: Callable[[V, V], bool] = equivalent,
) -> None:
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
        if k not in second_dict or not compat(first_dict[k], second_dict[k]):
            del first_dict[k]


def is_dict_like(value: Any) -> bool:
    return hasattr(value, "keys") and hasattr(value, "__getitem__")


def is_full_slice(value: Any) -> bool:
    return isinstance(value, slice) and value == slice(None)


def is_list_like(value: Any) -> bool:
    return isinstance(value, list) or isinstance(value, tuple)


def either_dict_or_kwargs(
    pos_kwargs: Optional[Mapping[Hashable, T]],
    kw_kwargs: Mapping[str, T],
    func_name: str,
) -> Mapping[Hashable, T]:
    if pos_kwargs is not None:
        if not is_dict_like(pos_kwargs):
            raise ValueError(
                "the first argument to .%s must be a dictionary" % func_name
            )
        if kw_kwargs:
            raise ValueError(
                "cannot specify both keyword and positional "
                "arguments to .%s" % func_name
            )
        return pos_kwargs
    else:
        # Need an explicit cast to appease mypy due to invariance; see
        # https://github.com/python/mypy/issues/6228
        return cast(Mapping[Hashable, T], kw_kwargs)


def is_scalar(value: Any, include_0d: bool = True) -> bool:
    """Whether to treat a value as a scalar.

    Any non-iterable, string, or 0-D array
    """
    from .variable import NON_NUMPY_SUPPORTED_ARRAY_TYPES

    if include_0d:
        include_0d = getattr(value, "ndim", None) == 0
    return (
        include_0d
        or isinstance(value, (str, bytes))
        or not (
            isinstance(value, (Iterable,) + NON_NUMPY_SUPPORTED_ARRAY_TYPES)
            or hasattr(value, "__array_function__")
        )
    )


def is_valid_numpy_dtype(dtype: Any) -> bool:
    try:
        np.dtype(dtype)
    except (TypeError, ValueError):
        return False
    else:
        return True


def to_0d_object_array(value: Any) -> np.ndarray:
    """Given a value, wrap it in a 0-D numpy.ndarray with dtype=object.
    """
    result = np.empty((), dtype=object)
    result[()] = value
    return result


def to_0d_array(value: Any) -> np.ndarray:
    """Given a value, wrap it in a 0-D numpy.ndarray.
    """
    if np.isscalar(value) or (isinstance(value, np.ndarray) and value.ndim == 0):
        return np.array(value)
    else:
        return to_0d_object_array(value)


def dict_equiv(
    first: Mapping[K, V],
    second: Mapping[K, V],
    compat: Callable[[V, V], bool] = equivalent,
) -> bool:
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


def ordered_dict_intersection(
    first_dict: Mapping[K, V],
    second_dict: Mapping[K, V],
    compat: Callable[[V, V], bool] = equivalent,
) -> MutableMapping[K, V]:
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


class Frozen(Mapping[K, V]):
    """Wrapper around an object implementing the mapping interface to make it
    immutable. If you really want to modify the mapping, the mutable version is
    saved under the `mapping` attribute.
    """

    __slots__ = ("mapping",)

    def __init__(self, mapping: Mapping[K, V]):
        self.mapping = mapping

    def __getitem__(self, key: K) -> V:
        return self.mapping[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self.mapping)

    def __len__(self) -> int:
        return len(self.mapping)

    def __contains__(self, key: object) -> bool:
        return key in self.mapping

    def __repr__(self) -> str:
        return "%s(%r)" % (type(self).__name__, self.mapping)


def FrozenOrderedDict(*args, **kwargs) -> Frozen:
    return Frozen(OrderedDict(*args, **kwargs))


class SortedKeysDict(MutableMapping[K, V]):
    """An wrapper for dictionary-like objects that always iterates over its
    items in sorted order by key but is otherwise equivalent to the underlying
    mapping.
    """

    __slots__ = ("mapping",)

    def __init__(self, mapping: MutableMapping[K, V] = None):
        self.mapping = {} if mapping is None else mapping

    def __getitem__(self, key: K) -> V:
        return self.mapping[key]

    def __setitem__(self, key: K, value: V) -> None:
        self.mapping[key] = value

    def __delitem__(self, key: K) -> None:
        del self.mapping[key]

    def __iter__(self) -> Iterator[K]:
        return iter(sorted(self.mapping))

    def __len__(self) -> int:
        return len(self.mapping)

    def __contains__(self, key: object) -> bool:
        return key in self.mapping

    def __repr__(self) -> str:
        return "%s(%r)" % (type(self).__name__, self.mapping)


class OrderedSet(MutableSet[T]):
    """A simple ordered set.

    The API matches the builtin set, but it preserves insertion order of
    elements, like an OrderedDict.
    """

    __slots__ = ("_ordered_dict",)

    def __init__(self, values: AbstractSet[T] = None):
        self._ordered_dict = OrderedDict()  # type: MutableMapping[T, None]
        if values is not None:
            # Disable type checking - both mypy and PyCharm believes that
            # we're altering the type of self in place (see signature of
            # MutableSet.__ior__)
            self |= values  # type: ignore

    # Required methods for MutableSet

    def __contains__(self, value: object) -> bool:
        return value in self._ordered_dict

    def __iter__(self) -> Iterator[T]:
        return iter(self._ordered_dict)

    def __len__(self) -> int:
        return len(self._ordered_dict)

    def add(self, value: T) -> None:
        self._ordered_dict[value] = None

    def discard(self, value: T) -> None:
        del self._ordered_dict[value]

    # Additional methods

    def update(self, values: AbstractSet[T]) -> None:
        # See comment on __init__ re. type checking
        self |= values  # type: ignore

    def __repr__(self) -> str:
        return "%s(%r)" % (type(self).__name__, list(self))


class NdimSizeLenMixin:
    """Mixin class that extends a class that defines a ``shape`` property to
    one that also defines ``ndim``, ``size`` and ``__len__``.
    """

    __slots__ = ()

    @property
    def ndim(self: Any) -> int:
        return len(self.shape)

    @property
    def size(self: Any) -> int:
        # cast to int so that shape = () gives size = 1
        return int(np.prod(self.shape))

    def __len__(self: Any) -> int:
        try:
            return self.shape[0]
        except IndexError:
            raise TypeError("len() of unsized object")


class NDArrayMixin(NdimSizeLenMixin):
    """Mixin class for making wrappers of N-dimensional arrays that conform to
    the ndarray interface required for the data argument to Variable objects.

    A subclass should set the `array` property and override one or more of
    `dtype`, `shape` and `__getitem__`.
    """

    __slots__ = ()

    @property
    def dtype(self: Any) -> np.dtype:
        return self.array.dtype

    @property
    def shape(self: Any) -> Tuple[int]:
        return self.array.shape

    def __getitem__(self: Any, key):
        return self.array[key]

    def __repr__(self: Any) -> str:
        return "%s(array=%r)" % (type(self).__name__, self.array)


class ReprObject:
    """Object that prints as the given value, for use with sentinel values.
    """

    __slots__ = ("_value",)

    def __init__(self, value: str):
        self._value = value

    def __repr__(self) -> str:
        return self._value

    def __eq__(self, other) -> bool:
        if isinstance(other, ReprObject):
            return self._value == other._value
        return False

    def __hash__(self) -> int:
        return hash((ReprObject, self._value))


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


def is_remote_uri(path: str) -> bool:
    return bool(re.search(r"^https?\://", path))


def is_grib_path(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext in [".grib", ".grb", ".grib2", ".grb2"]


def is_uniform_spaced(arr, **kwargs) -> bool:
    """Return True if values of an array are uniformly spaced and sorted.

    >>> is_uniform_spaced(range(5))
    True
    >>> is_uniform_spaced([-4, 0, 100])
    False

    kwargs are additional arguments to ``np.isclose``
    """
    arr = np.array(arr, dtype=float)
    diffs = np.diff(arr)
    return bool(np.isclose(diffs.min(), diffs.max(), **kwargs))


def hashable(v: Any) -> bool:
    """Determine whether `v` can be hashed.
    """
    try:
        hash(v)
    except TypeError:
        return False
    return True


def not_implemented(*args, **kwargs):
    return NotImplemented


def decode_numpy_dict_values(attrs: Mapping[K, V]) -> Dict[K, V]:
    """Convert attribute values from numpy objects to native Python objects,
    for use in to_dict
    """
    attrs = dict(attrs)
    for k, v in attrs.items():
        if isinstance(v, np.ndarray):
            attrs[k] = v.tolist()
        elif isinstance(v, np.generic):
            attrs[k] = v.item()
    return attrs


def ensure_us_time_resolution(val):
    """Convert val out of numpy time, for use in to_dict.
    Needed because of numpy bug GH#7619"""
    if np.issubdtype(val.dtype, np.datetime64):
        val = val.astype("datetime64[us]")
    elif np.issubdtype(val.dtype, np.timedelta64):
        val = val.astype("timedelta64[us]")
    return val


class HiddenKeyDict(MutableMapping[K, V]):
    """Acts like a normal dictionary, but hides certain keys.
    """

    __slots__ = ("_data", "_hidden_keys")

    # ``__init__`` method required to create instance from class.

    def __init__(self, data: MutableMapping[K, V], hidden_keys: Iterable[K]):
        self._data = data
        self._hidden_keys = frozenset(hidden_keys)

    def _raise_if_hidden(self, key: K) -> None:
        if key in self._hidden_keys:
            raise KeyError("Key `%r` is hidden." % key)

    # The next five methods are requirements of the ABC.
    def __setitem__(self, key: K, value: V) -> None:
        self._raise_if_hidden(key)
        self._data[key] = value

    def __getitem__(self, key: K) -> V:
        self._raise_if_hidden(key)
        return self._data[key]

    def __delitem__(self, key: K) -> None:
        self._raise_if_hidden(key)
        del self._data[key]

    def __iter__(self) -> Iterator[K]:
        for k in self._data:
            if k not in self._hidden_keys:
                yield k

    def __len__(self) -> int:
        num_hidden = len(self._hidden_keys & self._data.keys())
        return len(self._data) - num_hidden


def get_temp_dimname(dims: Container[Hashable], new_dim: Hashable) -> Hashable:
    """ Get an new dimension name based on new_dim, that is not used in dims.
    If the same name exists, we add an underscore(s) in the head.

    Example1:
        dims: ['a', 'b', 'c']
        new_dim: ['_rolling']
        -> ['_rolling']
    Example2:
        dims: ['a', 'b', 'c', '_rolling']
        new_dim: ['_rolling']
        -> ['__rolling']
    """
    while new_dim in dims:
        new_dim = "_" + str(new_dim)
    return new_dim
