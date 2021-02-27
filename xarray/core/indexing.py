import enum
import functools
import operator
from collections import defaultdict
from contextlib import suppress
from datetime import timedelta
from typing import Any, Callable, Iterable, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from . import duck_array_ops, nputils, utils
from .npcompat import DTypeLike
from .pycompat import (
    dask_array_type,
    integer_types,
    is_duck_dask_array,
    sparse_array_type,
)
from .utils import is_dict_like, maybe_cast_to_coords_dtype


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
    if len(new_key) > ndim:
        raise IndexError("too many indices")
    new_key.extend((ndim - len(new_key)) * [slice(None)])
    return tuple(new_key)


def _expand_slice(slice_, size):
    return np.arange(*slice_.indices(size))


def _sanitize_slice_element(x):
    from .dataarray import DataArray
    from .variable import Variable

    if isinstance(x, (Variable, DataArray)):
        x = x.values

    if isinstance(x, np.ndarray):
        if x.ndim != 0:
            raise ValueError(
                f"cannot use non-scalar arrays in a slice for xarray indexing: {x}"
            )
        x = x[()]

    return x


def _asarray_tuplesafe(values):
    """
    Convert values into a numpy array of at most 1-dimension, while preserving
    tuples.

    Adapted from pandas.core.common._asarray_tuplesafe
    """
    if isinstance(values, tuple):
        result = utils.to_0d_object_array(values)
    else:
        result = np.asarray(values)
        if result.ndim == 2:
            result = np.empty(len(values), dtype=object)
            result[:] = values

    return result


def _is_nested_tuple(possible_tuple):
    return isinstance(possible_tuple, tuple) and any(
        isinstance(value, (tuple, list, slice)) for value in possible_tuple
    )


def get_indexer_nd(index, labels, method=None, tolerance=None):
    """Wrapper around :meth:`pandas.Index.get_indexer` supporting n-dimensional
    labels
    """
    flat_labels = np.ravel(labels)
    flat_indexer = index.get_indexer(flat_labels, method=method, tolerance=tolerance)
    indexer = flat_indexer.reshape(labels.shape)
    return indexer


def convert_label_indexer(index, label, index_name="", method=None, tolerance=None):
    """Given a pandas.Index and labels (e.g., from __getitem__) for one
    dimension, return an indexer suitable for indexing an ndarray along that
    dimension. If `index` is a pandas.MultiIndex and depending on `label`,
    return a new pandas.Index or pandas.MultiIndex (otherwise return None).
    """
    new_index = None

    if isinstance(label, slice):
        if method is not None or tolerance is not None:
            raise NotImplementedError(
                "cannot use ``method`` argument if any indexers are slice objects"
            )
        indexer = index.slice_indexer(
            _sanitize_slice_element(label.start),
            _sanitize_slice_element(label.stop),
            _sanitize_slice_element(label.step),
        )
        if not isinstance(indexer, slice):
            # unlike pandas, in xarray we never want to silently convert a
            # slice indexer into an array indexer
            raise KeyError(
                "cannot represent labeled-based slice indexer for dimension "
                f"{index_name!r} with a slice over integer positions; the index is "
                "unsorted or non-unique"
            )

    elif is_dict_like(label):
        is_nested_vals = _is_nested_tuple(tuple(label.values()))
        if not isinstance(index, pd.MultiIndex):
            raise ValueError(
                "cannot use a dict-like object for selection on "
                "a dimension that does not have a MultiIndex"
            )
        elif len(label) == index.nlevels and not is_nested_vals:
            indexer = index.get_loc(tuple(label[k] for k in index.names))
        else:
            for k, v in label.items():
                # index should be an item (i.e. Hashable) not an array-like
                if isinstance(v, Sequence) and not isinstance(v, str):
                    raise ValueError(
                        "Vectorized selection is not "
                        "available along level variable: " + k
                    )
            indexer, new_index = index.get_loc_level(
                tuple(label.values()), level=tuple(label.keys())
            )

            # GH2619. Raise a KeyError if nothing is chosen
            if indexer.dtype.kind == "b" and indexer.sum() == 0:
                raise KeyError(f"{label} not found")

    elif isinstance(label, tuple) and isinstance(index, pd.MultiIndex):
        if _is_nested_tuple(label):
            indexer = index.get_locs(label)
        elif len(label) == index.nlevels:
            indexer = index.get_loc(label)
        else:
            indexer, new_index = index.get_loc_level(
                label, level=list(range(len(label)))
            )
    else:
        label = (
            label
            if getattr(label, "ndim", 1) > 1  # vectorized-indexing
            else _asarray_tuplesafe(label)
        )
        if label.ndim == 0:
            # see https://github.com/pydata/xarray/pull/4292 for details
            label_value = label[()] if label.dtype.kind in "mM" else label.item()
            if isinstance(index, pd.MultiIndex):
                indexer, new_index = index.get_loc_level(label_value, level=0)
            elif isinstance(index, pd.CategoricalIndex):
                if method is not None:
                    raise ValueError(
                        "'method' is not a valid kwarg when indexing using a CategoricalIndex."
                    )
                if tolerance is not None:
                    raise ValueError(
                        "'tolerance' is not a valid kwarg when indexing using a CategoricalIndex."
                    )
                indexer = index.get_loc(label_value)
            else:
                indexer = index.get_loc(label_value, method=method, tolerance=tolerance)
        elif label.dtype.kind == "b":
            indexer = label
        else:
            if isinstance(index, pd.MultiIndex) and label.ndim > 1:
                raise ValueError(
                    "Vectorized selection is not available along "
                    "MultiIndex variable: " + index_name
                )
            indexer = get_indexer_nd(index, label, method, tolerance)
            if np.any(indexer < 0):
                raise KeyError(f"not all values found in index {index_name!r}")
    return indexer, new_index


def get_dim_indexers(data_obj, indexers):
    """Given a xarray data object and label based indexers, return a mapping
    of label indexers with only dimension names as keys.

    It groups multiple level indexers given on a multi-index dimension
    into a single, dictionary indexer for that dimension (Raise a ValueError
    if it is not possible).
    """
    invalid = [
        k
        for k in indexers
        if k not in data_obj.dims and k not in data_obj._level_coords
    ]
    if invalid:
        raise ValueError(f"dimensions or multi-index levels {invalid!r} do not exist")

    level_indexers = defaultdict(dict)
    dim_indexers = {}
    for key, label in indexers.items():
        (dim,) = data_obj[key].dims
        if key != dim:
            # assume here multi-index level indexer
            level_indexers[dim][key] = label
        else:
            dim_indexers[key] = label

    for dim, level_labels in level_indexers.items():
        if dim_indexers.get(dim, False):
            raise ValueError(
                "cannot combine multi-index level indexers with an indexer for "
                f"dimension {dim}"
            )
        dim_indexers[dim] = level_labels

    return dim_indexers


def remap_label_indexers(data_obj, indexers, method=None, tolerance=None):
    """Given an xarray data object and label based indexers, return a mapping
    of equivalent location based indexers. Also return a mapping of updated
    pandas index objects (in case of multi-index level drop).
    """
    if method is not None and not isinstance(method, str):
        raise TypeError("``method`` must be a string")

    pos_indexers = {}
    new_indexes = {}

    dim_indexers = get_dim_indexers(data_obj, indexers)
    for dim, label in dim_indexers.items():
        try:
            index = data_obj.indexes[dim]
        except KeyError:
            # no index for this dimension: reuse the provided labels
            if method is not None or tolerance is not None:
                raise ValueError(
                    "cannot supply ``method`` or ``tolerance`` "
                    "when the indexed dimension does not have "
                    "an associated coordinate."
                )
            pos_indexers[dim] = label
        else:
            coords_dtype = data_obj.coords[dim].dtype
            label = maybe_cast_to_coords_dtype(label, coords_dtype)
            idxr, new_idx = convert_label_indexer(index, label, dim, method, tolerance)
            pos_indexers[dim] = idxr
            if new_idx is not None:
                new_indexes[dim] = new_idx

    return pos_indexers, new_indexes


def _normalize_slice(sl, size):
    """Ensure that given slice only contains positive start and stop values
    (stop can be -1 for full-size slices with negative steps, e.g. [-10::-1])"""
    return slice(*sl.indices(size))


def slice_slice(old_slice, applied_slice, size):
    """Given a slice and the size of the dimension to which it will be applied,
    index it with another slice to return a new slice equivalent to applying
    the slices sequentially
    """
    old_slice = _normalize_slice(old_slice, size)

    size_after_old_slice = len(range(old_slice.start, old_slice.stop, old_slice.step))
    if size_after_old_slice == 0:
        # nothing left after applying first slice
        return slice(0)

    applied_slice = _normalize_slice(applied_slice, size_after_old_slice)

    start = old_slice.start + applied_slice.start * old_slice.step
    if start < 0:
        # nothing left after applying second slice
        # (can only happen for old_slice.step < 0, e.g. [10::-1], [20:])
        return slice(0)

    stop = old_slice.start + applied_slice.stop * old_slice.step
    if stop < 0:
        stop = None

    step = old_slice.step * applied_slice.step

    return slice(start, stop, step)


def _index_indexer_1d(old_indexer, applied_indexer, size):
    assert isinstance(applied_indexer, integer_types + (slice, np.ndarray))
    if isinstance(applied_indexer, slice) and applied_indexer == slice(None):
        # shortcut for the usual case
        return old_indexer
    if isinstance(old_indexer, slice):
        if isinstance(applied_indexer, slice):
            indexer = slice_slice(old_indexer, applied_indexer, size)
        else:
            indexer = _expand_slice(old_indexer, size)[applied_indexer]
    else:
        indexer = old_indexer[applied_indexer]
    return indexer


class ExplicitIndexer:
    """Base class for explicit indexer objects.

    ExplicitIndexer objects wrap a tuple of values given by their ``tuple``
    property. These tuples should always have length equal to the number of
    dimensions on the indexed array.

    Do not instantiate BaseIndexer objects directly: instead, use one of the
    sub-classes BasicIndexer, OuterIndexer or VectorizedIndexer.
    """

    __slots__ = ("_key",)

    def __init__(self, key):
        if type(self) is ExplicitIndexer:
            raise TypeError("cannot instantiate base ExplicitIndexer objects")
        self._key = tuple(key)

    @property
    def tuple(self):
        return self._key

    def __repr__(self):
        return f"{type(self).__name__}({self.tuple})"


def as_integer_or_none(value):
    return None if value is None else operator.index(value)


def as_integer_slice(value):
    start = as_integer_or_none(value.start)
    stop = as_integer_or_none(value.stop)
    step = as_integer_or_none(value.step)
    return slice(start, stop, step)


class BasicIndexer(ExplicitIndexer):
    """Tuple for basic indexing.

    All elements should be int or slice objects. Indexing follows NumPy's
    rules for basic indexing: each axis is independently sliced and axes
    indexed with an integer are dropped from the result.
    """

    __slots__ = ()

    def __init__(self, key):
        if not isinstance(key, tuple):
            raise TypeError(f"key must be a tuple: {key!r}")

        new_key = []
        for k in key:
            if isinstance(k, integer_types):
                k = int(k)
            elif isinstance(k, slice):
                k = as_integer_slice(k)
            else:
                raise TypeError(
                    f"unexpected indexer type for {type(self).__name__}: {k!r}"
                )
            new_key.append(k)

        super().__init__(new_key)


class OuterIndexer(ExplicitIndexer):
    """Tuple for outer/orthogonal indexing.

    All elements should be int, slice or 1-dimensional np.ndarray objects with
    an integer dtype. Indexing is applied independently along each axis, and
    axes indexed with an integer are dropped from the result. This type of
    indexing works like MATLAB/Fortran.
    """

    __slots__ = ()

    def __init__(self, key):
        if not isinstance(key, tuple):
            raise TypeError(f"key must be a tuple: {key!r}")

        new_key = []
        for k in key:
            if isinstance(k, integer_types):
                k = int(k)
            elif isinstance(k, slice):
                k = as_integer_slice(k)
            elif isinstance(k, np.ndarray):
                if not np.issubdtype(k.dtype, np.integer):
                    raise TypeError(
                        f"invalid indexer array, does not have integer dtype: {k!r}"
                    )
                if k.ndim != 1:
                    raise TypeError(
                        f"invalid indexer array for {type(self).__name__}; must have "
                        f"exactly 1 dimension: {k!r}"
                    )
                k = np.asarray(k, dtype=np.int64)
            else:
                raise TypeError(
                    f"unexpected indexer type for {type(self).__name__}: {k!r}"
                )
            new_key.append(k)

        super().__init__(new_key)


class VectorizedIndexer(ExplicitIndexer):
    """Tuple for vectorized indexing.

    All elements should be slice or N-dimensional np.ndarray objects with an
    integer dtype and the same number of dimensions. Indexing follows proposed
    rules for np.ndarray.vindex, which matches NumPy's advanced indexing rules
    (including broadcasting) except sliced axes are always moved to the end:
    https://github.com/numpy/numpy/pull/6256
    """

    __slots__ = ()

    def __init__(self, key):
        if not isinstance(key, tuple):
            raise TypeError(f"key must be a tuple: {key!r}")

        new_key = []
        ndim = None
        for k in key:
            if isinstance(k, slice):
                k = as_integer_slice(k)
            elif isinstance(k, np.ndarray):
                if not np.issubdtype(k.dtype, np.integer):
                    raise TypeError(
                        f"invalid indexer array, does not have integer dtype: {k!r}"
                    )
                if ndim is None:
                    ndim = k.ndim
                elif ndim != k.ndim:
                    ndims = [k.ndim for k in key if isinstance(k, np.ndarray)]
                    raise ValueError(
                        "invalid indexer key: ndarray arguments "
                        f"have different numbers of dimensions: {ndims}"
                    )
                k = np.asarray(k, dtype=np.int64)
            else:
                raise TypeError(
                    f"unexpected indexer type for {type(self).__name__}: {k!r}"
                )
            new_key.append(k)

        super().__init__(new_key)


class ExplicitlyIndexed:
    """Mixin to mark support for Indexer subclasses in indexing."""

    __slots__ = ()


class ExplicitlyIndexedNDArrayMixin(utils.NDArrayMixin, ExplicitlyIndexed):
    __slots__ = ()

    def __array__(self, dtype=None):
        key = BasicIndexer((slice(None),) * self.ndim)
        return np.asarray(self[key], dtype=dtype)


class ImplicitToExplicitIndexingAdapter(utils.NDArrayMixin):
    """Wrap an array, converting tuples into the indicated explicit indexer."""

    __slots__ = ("array", "indexer_cls")

    def __init__(self, array, indexer_cls=BasicIndexer):
        self.array = as_indexable(array)
        self.indexer_cls = indexer_cls

    def __array__(self, dtype=None):
        return np.asarray(self.array, dtype=dtype)

    def __getitem__(self, key):
        key = expanded_indexer(key, self.ndim)
        result = self.array[self.indexer_cls(key)]
        if isinstance(result, ExplicitlyIndexed):
            return type(self)(result, self.indexer_cls)
        else:
            # Sometimes explicitly indexed arrays return NumPy arrays or
            # scalars.
            return result


class LazilyOuterIndexedArray(ExplicitlyIndexedNDArrayMixin):
    """Wrap an array to make basic and outer indexing lazy."""

    __slots__ = ("array", "key")

    def __init__(self, array, key=None):
        """
        Parameters
        ----------
        array : array_like
            Array like object to index.
        key : ExplicitIndexer, optional
            Array indexer. If provided, it is assumed to already be in
            canonical expanded form.
        """
        if isinstance(array, type(self)) and key is None:
            # unwrap
            key = array.key
            array = array.array

        if key is None:
            key = BasicIndexer((slice(None),) * array.ndim)

        self.array = as_indexable(array)
        self.key = key

    def _updated_key(self, new_key):
        iter_new_key = iter(expanded_indexer(new_key.tuple, self.ndim))
        full_key = []
        for size, k in zip(self.array.shape, self.key.tuple):
            if isinstance(k, integer_types):
                full_key.append(k)
            else:
                full_key.append(_index_indexer_1d(k, next(iter_new_key), size))
        full_key = tuple(full_key)

        if all(isinstance(k, integer_types + (slice,)) for k in full_key):
            return BasicIndexer(full_key)
        return OuterIndexer(full_key)

    @property
    def shape(self):
        shape = []
        for size, k in zip(self.array.shape, self.key.tuple):
            if isinstance(k, slice):
                shape.append(len(range(*k.indices(size))))
            elif isinstance(k, np.ndarray):
                shape.append(k.size)
        return tuple(shape)

    def __array__(self, dtype=None):
        array = as_indexable(self.array)
        return np.asarray(array[self.key], dtype=None)

    def transpose(self, order):
        return LazilyVectorizedIndexedArray(self.array, self.key).transpose(order)

    def __getitem__(self, indexer):
        if isinstance(indexer, VectorizedIndexer):
            array = LazilyVectorizedIndexedArray(self.array, self.key)
            return array[indexer]
        return type(self)(self.array, self._updated_key(indexer))

    def __setitem__(self, key, value):
        if isinstance(key, VectorizedIndexer):
            raise NotImplementedError(
                "Lazy item assignment with the vectorized indexer is not yet "
                "implemented. Load your data first by .load() or compute()."
            )
        full_key = self._updated_key(key)
        self.array[full_key] = value

    def __repr__(self):
        return f"{type(self).__name__}(array={self.array!r}, key={self.key!r})"


class LazilyVectorizedIndexedArray(ExplicitlyIndexedNDArrayMixin):
    """Wrap an array to make vectorized indexing lazy."""

    __slots__ = ("array", "key")

    def __init__(self, array, key):
        """
        Parameters
        ----------
        array : array_like
            Array like object to index.
        key : VectorizedIndexer
        """
        if isinstance(key, (BasicIndexer, OuterIndexer)):
            self.key = _outer_to_vectorized_indexer(key, array.shape)
        else:
            self.key = _arrayize_vectorized_indexer(key, array.shape)
        self.array = as_indexable(array)

    @property
    def shape(self):
        return np.broadcast(*self.key.tuple).shape

    def __array__(self, dtype=None):
        return np.asarray(self.array[self.key], dtype=None)

    def _updated_key(self, new_key):
        return _combine_indexers(self.key, self.shape, new_key)

    def __getitem__(self, indexer):
        # If the indexed array becomes a scalar, return LazilyOuterIndexedArray
        if all(isinstance(ind, integer_types) for ind in indexer.tuple):
            key = BasicIndexer(tuple(k[indexer.tuple] for k in self.key.tuple))
            return LazilyOuterIndexedArray(self.array, key)
        return type(self)(self.array, self._updated_key(indexer))

    def transpose(self, order):
        key = VectorizedIndexer(tuple(k.transpose(order) for k in self.key.tuple))
        return type(self)(self.array, key)

    def __setitem__(self, key, value):
        raise NotImplementedError(
            "Lazy item assignment with the vectorized indexer is not yet "
            "implemented. Load your data first by .load() or compute()."
        )

    def __repr__(self):
        return f"{type(self).__name__}(array={self.array!r}, key={self.key!r})"


def _wrap_numpy_scalars(array):
    """Wrap NumPy scalars in 0d arrays."""
    if np.isscalar(array):
        return np.array(array)
    else:
        return array


class CopyOnWriteArray(ExplicitlyIndexedNDArrayMixin):
    __slots__ = ("array", "_copied")

    def __init__(self, array):
        self.array = as_indexable(array)
        self._copied = False

    def _ensure_copied(self):
        if not self._copied:
            self.array = as_indexable(np.array(self.array))
            self._copied = True

    def __array__(self, dtype=None):
        return np.asarray(self.array, dtype=dtype)

    def __getitem__(self, key):
        return type(self)(_wrap_numpy_scalars(self.array[key]))

    def transpose(self, order):
        return self.array.transpose(order)

    def __setitem__(self, key, value):
        self._ensure_copied()
        self.array[key] = value

    def __deepcopy__(self, memo):
        # CopyOnWriteArray is used to wrap backend array objects, which might
        # point to files on disk, so we can't rely on the default deepcopy
        # implementation.
        return type(self)(self.array)


class MemoryCachedArray(ExplicitlyIndexedNDArrayMixin):
    __slots__ = ("array",)

    def __init__(self, array):
        self.array = _wrap_numpy_scalars(as_indexable(array))

    def _ensure_cached(self):
        if not isinstance(self.array, NumpyIndexingAdapter):
            self.array = NumpyIndexingAdapter(np.asarray(self.array))

    def __array__(self, dtype=None):
        self._ensure_cached()
        return np.asarray(self.array, dtype=dtype)

    def __getitem__(self, key):
        return type(self)(_wrap_numpy_scalars(self.array[key]))

    def transpose(self, order):
        return self.array.transpose(order)

    def __setitem__(self, key, value):
        self.array[key] = value


def as_indexable(array):
    """
    This function always returns a ExplicitlyIndexed subclass,
    so that the vectorized indexing is always possible with the returned
    object.
    """
    if isinstance(array, ExplicitlyIndexed):
        return array
    if isinstance(array, np.ndarray):
        return NumpyIndexingAdapter(array)
    if isinstance(array, pd.Index):
        return PandasIndexAdapter(array)
    if isinstance(array, dask_array_type):
        return DaskIndexingAdapter(array)
    if hasattr(array, "__array_function__"):
        return NdArrayLikeIndexingAdapter(array)

    raise TypeError("Invalid array type: {}".format(type(array)))


def _outer_to_vectorized_indexer(key, shape):
    """Convert an OuterIndexer into an vectorized indexer.

    Parameters
    ----------
    key : Outer/Basic Indexer
        An indexer to convert.
    shape : tuple
        Shape of the array subject to the indexing.

    Returns
    -------
    VectorizedIndexer
        Tuple suitable for use to index a NumPy array with vectorized indexing.
        Each element is an array: broadcasting them together gives the shape
        of the result.
    """
    key = key.tuple

    n_dim = len([k for k in key if not isinstance(k, integer_types)])
    i_dim = 0
    new_key = []
    for k, size in zip(key, shape):
        if isinstance(k, integer_types):
            new_key.append(np.array(k).reshape((1,) * n_dim))
        else:  # np.ndarray or slice
            if isinstance(k, slice):
                k = np.arange(*k.indices(size))
            assert k.dtype.kind in {"i", "u"}
            shape = [(1,) * i_dim + (k.size,) + (1,) * (n_dim - i_dim - 1)]
            new_key.append(k.reshape(*shape))
            i_dim += 1
    return VectorizedIndexer(tuple(new_key))


def _outer_to_numpy_indexer(key, shape):
    """Convert an OuterIndexer into an indexer for NumPy.

    Parameters
    ----------
    key : Basic/OuterIndexer
        An indexer to convert.
    shape : tuple
        Shape of the array subject to the indexing.

    Returns
    -------
    tuple
        Tuple suitable for use to index a NumPy array.
    """
    if len([k for k in key.tuple if not isinstance(k, slice)]) <= 1:
        # If there is only one vector and all others are slice,
        # it can be safely used in mixed basic/advanced indexing.
        # Boolean index should already be converted to integer array.
        return key.tuple
    else:
        return _outer_to_vectorized_indexer(key, shape).tuple


def _combine_indexers(old_key, shape, new_key):
    """Combine two indexers.

    Parameters
    ----------
    old_key : ExplicitIndexer
        The first indexer for the original array
    shape : tuple of ints
        Shape of the original array to be indexed by old_key
    new_key
        The second indexer for indexing original[old_key]
    """
    if not isinstance(old_key, VectorizedIndexer):
        old_key = _outer_to_vectorized_indexer(old_key, shape)
    if len(old_key.tuple) == 0:
        return new_key

    new_shape = np.broadcast(*old_key.tuple).shape
    if isinstance(new_key, VectorizedIndexer):
        new_key = _arrayize_vectorized_indexer(new_key, new_shape)
    else:
        new_key = _outer_to_vectorized_indexer(new_key, new_shape)

    return VectorizedIndexer(
        tuple(o[new_key.tuple] for o in np.broadcast_arrays(*old_key.tuple))
    )


@enum.unique
class IndexingSupport(enum.Enum):
    # for backends that support only basic indexer
    BASIC = 0
    # for backends that support basic / outer indexer
    OUTER = 1
    # for backends that support outer indexer including at most 1 vector.
    OUTER_1VECTOR = 2
    # for backends that support full vectorized indexer.
    VECTORIZED = 3


def explicit_indexing_adapter(
    key: ExplicitIndexer,
    shape: Tuple[int, ...],
    indexing_support: IndexingSupport,
    raw_indexing_method: Callable,
) -> Any:
    """Support explicit indexing by delegating to a raw indexing method.

    Outer and/or vectorized indexers are supported by indexing a second time
    with a NumPy array.

    Parameters
    ----------
    key : ExplicitIndexer
        Explicit indexing object.
    shape : Tuple[int, ...]
        Shape of the indexed array.
    indexing_support : IndexingSupport enum
        Form of indexing supported by raw_indexing_method.
    raw_indexing_method : callable
        Function (like ndarray.__getitem__) that when called with indexing key
        in the form of a tuple returns an indexed array.

    Returns
    -------
    Indexing result, in the form of a duck numpy-array.
    """
    raw_key, numpy_indices = decompose_indexer(key, shape, indexing_support)
    result = raw_indexing_method(raw_key.tuple)
    if numpy_indices.tuple:
        # index the loaded np.ndarray
        result = NumpyIndexingAdapter(np.asarray(result))[numpy_indices]
    return result


def decompose_indexer(
    indexer: ExplicitIndexer, shape: Tuple[int, ...], indexing_support: IndexingSupport
) -> Tuple[ExplicitIndexer, ExplicitIndexer]:
    if isinstance(indexer, VectorizedIndexer):
        return _decompose_vectorized_indexer(indexer, shape, indexing_support)
    if isinstance(indexer, (BasicIndexer, OuterIndexer)):
        return _decompose_outer_indexer(indexer, shape, indexing_support)
    raise TypeError(f"unexpected key type: {indexer}")


def _decompose_slice(key, size):
    """convert a slice to successive two slices. The first slice always has
    a positive step.
    """
    start, stop, step = key.indices(size)
    if step > 0:
        # If key already has a positive step, use it as is in the backend
        return key, slice(None)
    else:
        # determine stop precisely for step > 1 case
        # e.g. [98:2:-2] -> [98:3:-2]
        stop = start + int((stop - start - 1) / step) * step + 1
        start, stop = stop + 1, start + 1
        return slice(start, stop, -step), slice(None, None, -1)


def _decompose_vectorized_indexer(
    indexer: VectorizedIndexer,
    shape: Tuple[int, ...],
    indexing_support: IndexingSupport,
) -> Tuple[ExplicitIndexer, ExplicitIndexer]:
    """
    Decompose vectorized indexer to the successive two indexers, where the
    first indexer will be used to index backend arrays, while the second one
    is used to index loaded on-memory np.ndarray.

    Parameters
    ----------
    indexer : VectorizedIndexer
    indexing_support : one of IndexerSupport entries

    Returns
    -------
    backend_indexer: OuterIndexer or BasicIndexer
    np_indexers: an ExplicitIndexer (VectorizedIndexer / BasicIndexer)

    Notes
    -----
    This function is used to realize the vectorized indexing for the backend
    arrays that only support basic or outer indexing.

    As an example, let us consider to index a few elements from a backend array
    with a vectorized indexer ([0, 3, 1], [2, 3, 2]).
    Even if the backend array only supports outer indexing, it is more
    efficient to load a subslice of the array than loading the entire array,

    >>> array = np.arange(36).reshape(6, 6)
    >>> backend_indexer = OuterIndexer((np.array([0, 1, 3]), np.array([2, 3])))
    >>> # load subslice of the array
    ... array = NumpyIndexingAdapter(array)[backend_indexer]
    >>> np_indexer = VectorizedIndexer((np.array([0, 2, 1]), np.array([0, 1, 0])))
    >>> # vectorized indexing for on-memory np.ndarray.
    ... NumpyIndexingAdapter(array)[np_indexer]
    array([ 2, 21,  8])
    """
    assert isinstance(indexer, VectorizedIndexer)

    if indexing_support is IndexingSupport.VECTORIZED:
        return indexer, BasicIndexer(())

    backend_indexer_elems = []
    np_indexer_elems = []
    # convert negative indices
    indexer_elems = [
        np.where(k < 0, k + s, k) if isinstance(k, np.ndarray) else k
        for k, s in zip(indexer.tuple, shape)
    ]

    for k, s in zip(indexer_elems, shape):
        if isinstance(k, slice):
            # If it is a slice, then we will slice it as-is
            # (but make its step positive) in the backend,
            # and then use all of it (slice(None)) for the in-memory portion.
            bk_slice, np_slice = _decompose_slice(k, s)
            backend_indexer_elems.append(bk_slice)
            np_indexer_elems.append(np_slice)
        else:
            # If it is a (multidimensional) np.ndarray, just pickup the used
            # keys without duplication and store them as a 1d-np.ndarray.
            oind, vind = np.unique(k, return_inverse=True)
            backend_indexer_elems.append(oind)
            np_indexer_elems.append(vind.reshape(*k.shape))

    backend_indexer = OuterIndexer(tuple(backend_indexer_elems))
    np_indexer = VectorizedIndexer(tuple(np_indexer_elems))

    if indexing_support is IndexingSupport.OUTER:
        return backend_indexer, np_indexer

    # If the backend does not support outer indexing,
    # backend_indexer (OuterIndexer) is also decomposed.
    backend_indexer1, np_indexer1 = _decompose_outer_indexer(
        backend_indexer, shape, indexing_support
    )
    np_indexer = _combine_indexers(np_indexer1, shape, np_indexer)
    return backend_indexer1, np_indexer


def _decompose_outer_indexer(
    indexer: Union[BasicIndexer, OuterIndexer],
    shape: Tuple[int, ...],
    indexing_support: IndexingSupport,
) -> Tuple[ExplicitIndexer, ExplicitIndexer]:
    """
    Decompose outer indexer to the successive two indexers, where the
    first indexer will be used to index backend arrays, while the second one
    is used to index the loaded on-memory np.ndarray.

    Parameters
    ----------
    indexer : OuterIndexer or BasicIndexer
    indexing_support : One of the entries of IndexingSupport

    Returns
    -------
    backend_indexer: OuterIndexer or BasicIndexer
    np_indexers: an ExplicitIndexer (OuterIndexer / BasicIndexer)

    Notes
    -----
    This function is used to realize the vectorized indexing for the backend
    arrays that only support basic or outer indexing.

    As an example, let us consider to index a few elements from a backend array
    with a orthogonal indexer ([0, 3, 1], [2, 3, 2]).
    Even if the backend array only supports basic indexing, it is more
    efficient to load a subslice of the array than loading the entire array,

    >>> array = np.arange(36).reshape(6, 6)
    >>> backend_indexer = BasicIndexer((slice(0, 3), slice(2, 4)))
    >>> # load subslice of the array
    ... array = NumpyIndexingAdapter(array)[backend_indexer]
    >>> np_indexer = OuterIndexer((np.array([0, 2, 1]), np.array([0, 1, 0])))
    >>> # outer indexing for on-memory np.ndarray.
    ... NumpyIndexingAdapter(array)[np_indexer]
    array([[ 2,  3,  2],
           [14, 15, 14],
           [ 8,  9,  8]])
    """
    if indexing_support == IndexingSupport.VECTORIZED:
        return indexer, BasicIndexer(())
    assert isinstance(indexer, (OuterIndexer, BasicIndexer))

    backend_indexer: List[Any] = []
    np_indexer = []
    # make indexer positive
    pos_indexer = []
    for k, s in zip(indexer.tuple, shape):
        if isinstance(k, np.ndarray):
            pos_indexer.append(np.where(k < 0, k + s, k))
        elif isinstance(k, integer_types) and k < 0:
            pos_indexer.append(k + s)
        else:
            pos_indexer.append(k)
    indexer_elems = pos_indexer

    if indexing_support is IndexingSupport.OUTER_1VECTOR:
        # some backends such as h5py supports only 1 vector in indexers
        # We choose the most efficient axis
        gains = [
            (np.max(k) - np.min(k) + 1.0) / len(np.unique(k))
            if isinstance(k, np.ndarray)
            else 0
            for k in indexer_elems
        ]
        array_index = np.argmax(np.array(gains)) if len(gains) > 0 else None

        for i, (k, s) in enumerate(zip(indexer_elems, shape)):
            if isinstance(k, np.ndarray) and i != array_index:
                # np.ndarray key is converted to slice that covers the entire
                # entries of this key.
                backend_indexer.append(slice(np.min(k), np.max(k) + 1))
                np_indexer.append(k - np.min(k))
            elif isinstance(k, np.ndarray):
                # Remove duplicates and sort them in the increasing order
                pkey, ekey = np.unique(k, return_inverse=True)
                backend_indexer.append(pkey)
                np_indexer.append(ekey)
            elif isinstance(k, integer_types):
                backend_indexer.append(k)
            else:  # slice:  convert positive step slice for backend
                bk_slice, np_slice = _decompose_slice(k, s)
                backend_indexer.append(bk_slice)
                np_indexer.append(np_slice)

        return (OuterIndexer(tuple(backend_indexer)), OuterIndexer(tuple(np_indexer)))

    if indexing_support == IndexingSupport.OUTER:
        for k, s in zip(indexer_elems, shape):
            if isinstance(k, slice):
                # slice:  convert positive step slice for backend
                bk_slice, np_slice = _decompose_slice(k, s)
                backend_indexer.append(bk_slice)
                np_indexer.append(np_slice)
            elif isinstance(k, integer_types):
                backend_indexer.append(k)
            elif isinstance(k, np.ndarray) and (np.diff(k) >= 0).all():
                backend_indexer.append(k)
                np_indexer.append(slice(None))
            else:
                # Remove duplicates and sort them in the increasing order
                oind, vind = np.unique(k, return_inverse=True)
                backend_indexer.append(oind)
                np_indexer.append(vind.reshape(*k.shape))

        return (OuterIndexer(tuple(backend_indexer)), OuterIndexer(tuple(np_indexer)))

    # basic indexer
    assert indexing_support == IndexingSupport.BASIC

    for k, s in zip(indexer_elems, shape):
        if isinstance(k, np.ndarray):
            # np.ndarray key is converted to slice that covers the entire
            # entries of this key.
            backend_indexer.append(slice(np.min(k), np.max(k) + 1))
            np_indexer.append(k - np.min(k))
        elif isinstance(k, integer_types):
            backend_indexer.append(k)
        else:  # slice:  convert positive step slice for backend
            bk_slice, np_slice = _decompose_slice(k, s)
            backend_indexer.append(bk_slice)
            np_indexer.append(np_slice)

    return (BasicIndexer(tuple(backend_indexer)), OuterIndexer(tuple(np_indexer)))


def _arrayize_vectorized_indexer(indexer, shape):
    """ Return an identical vindex but slices are replaced by arrays """
    slices = [v for v in indexer.tuple if isinstance(v, slice)]
    if len(slices) == 0:
        return indexer

    arrays = [v for v in indexer.tuple if isinstance(v, np.ndarray)]
    n_dim = arrays[0].ndim if len(arrays) > 0 else 0
    i_dim = 0
    new_key = []
    for v, size in zip(indexer.tuple, shape):
        if isinstance(v, np.ndarray):
            new_key.append(np.reshape(v, v.shape + (1,) * len(slices)))
        else:  # slice
            shape = (1,) * (n_dim + i_dim) + (-1,) + (1,) * (len(slices) - i_dim - 1)
            new_key.append(np.arange(*v.indices(size)).reshape(shape))
            i_dim += 1
    return VectorizedIndexer(tuple(new_key))


def _dask_array_with_chunks_hint(array, chunks):
    """Create a dask array using the chunks hint for dimensions of size > 1."""
    import dask.array as da

    if len(chunks) < array.ndim:
        raise ValueError("not enough chunks in hint")
    new_chunks = []
    for chunk, size in zip(chunks, array.shape):
        new_chunks.append(chunk if size > 1 else (1,))
    return da.from_array(array, new_chunks)


def _logical_any(args):
    return functools.reduce(operator.or_, args)


def _masked_result_drop_slice(key, data=None):

    key = (k for k in key if not isinstance(k, slice))
    chunks_hint = getattr(data, "chunks", None)

    new_keys = []
    for k in key:
        if isinstance(k, np.ndarray):
            if is_duck_dask_array(data):
                new_keys.append(_dask_array_with_chunks_hint(k, chunks_hint))
            elif isinstance(data, sparse_array_type):
                import sparse

                new_keys.append(sparse.COO.from_numpy(k))
            else:
                new_keys.append(k)
        else:
            new_keys.append(k)

    mask = _logical_any(k == -1 for k in new_keys)
    return mask


def create_mask(indexer, shape, data=None):
    """Create a mask for indexing with a fill-value.

    Parameters
    ----------
    indexer : ExplicitIndexer
        Indexer with -1 in integer or ndarray value to indicate locations in
        the result that should be masked.
    shape : tuple
        Shape of the array being indexed.
    data : optional
        Data for which mask is being created. If data is a dask arrays, its chunks
        are used as a hint for chunks on the resulting mask. If data is a sparse
        array, the returned mask is also a sparse array.

    Returns
    -------
    mask : bool, np.ndarray, SparseArray or dask.array.Array with dtype=bool
        Same type as data. Has the same shape as the indexing result.
    """
    if isinstance(indexer, OuterIndexer):
        key = _outer_to_vectorized_indexer(indexer, shape).tuple
        assert not any(isinstance(k, slice) for k in key)
        mask = _masked_result_drop_slice(key, data)

    elif isinstance(indexer, VectorizedIndexer):
        key = indexer.tuple
        base_mask = _masked_result_drop_slice(key, data)
        slice_shape = tuple(
            np.arange(*k.indices(size)).size
            for k, size in zip(key, shape)
            if isinstance(k, slice)
        )
        expanded_mask = base_mask[(Ellipsis,) + (np.newaxis,) * len(slice_shape)]
        mask = duck_array_ops.broadcast_to(expanded_mask, base_mask.shape + slice_shape)

    elif isinstance(indexer, BasicIndexer):
        mask = any(k == -1 for k in indexer.tuple)

    else:
        raise TypeError("unexpected key type: {}".format(type(indexer)))

    return mask


def _posify_mask_subindexer(index):
    """Convert masked indices in a flat array to the nearest unmasked index.

    Parameters
    ----------
    index : np.ndarray
        One dimensional ndarray with dtype=int.

    Returns
    -------
    np.ndarray
        One dimensional ndarray with all values equal to -1 replaced by an
        adjacent non-masked element.
    """
    masked = index == -1
    unmasked_locs = np.flatnonzero(~masked)
    if not unmasked_locs.size:
        # indexing unmasked_locs is invalid
        return np.zeros_like(index)
    masked_locs = np.flatnonzero(masked)
    prev_value = np.maximum(0, np.searchsorted(unmasked_locs, masked_locs) - 1)
    new_index = index.copy()
    new_index[masked_locs] = index[unmasked_locs[prev_value]]
    return new_index


def posify_mask_indexer(indexer):
    """Convert masked values (-1) in an indexer to nearest unmasked values.

    This routine is useful for dask, where it can be much faster to index
    adjacent points than arbitrary points from the end of an array.

    Parameters
    ----------
    indexer : ExplicitIndexer
        Input indexer.

    Returns
    -------
    ExplicitIndexer
        Same type of input, with all values in ndarray keys equal to -1
        replaced by an adjacent non-masked element.
    """
    key = tuple(
        _posify_mask_subindexer(k.ravel()).reshape(k.shape)
        if isinstance(k, np.ndarray)
        else k
        for k in indexer.tuple
    )
    return type(indexer)(key)


def is_fancy_indexer(indexer: Any) -> bool:
    """Return False if indexer is a int, slice, a 1-dimensional list, or a 0 or
    1-dimensional ndarray; in all other cases return True
    """
    if isinstance(indexer, (int, slice)):
        return False
    if isinstance(indexer, np.ndarray):
        return indexer.ndim > 1
    if isinstance(indexer, list):
        return bool(indexer) and not isinstance(indexer[0], int)
    return True


class NumpyIndexingAdapter(ExplicitlyIndexedNDArrayMixin):
    """Wrap a NumPy array to use explicit indexing."""

    __slots__ = ("array",)

    def __init__(self, array):
        # In NumpyIndexingAdapter we only allow to store bare np.ndarray
        if not isinstance(array, np.ndarray):
            raise TypeError(
                "NumpyIndexingAdapter only wraps np.ndarray. "
                "Trying to wrap {}".format(type(array))
            )
        self.array = array

    def _indexing_array_and_key(self, key):
        if isinstance(key, OuterIndexer):
            array = self.array
            key = _outer_to_numpy_indexer(key, self.array.shape)
        elif isinstance(key, VectorizedIndexer):
            array = nputils.NumpyVIndexAdapter(self.array)
            key = key.tuple
        elif isinstance(key, BasicIndexer):
            array = self.array
            # We want 0d slices rather than scalars. This is achieved by
            # appending an ellipsis (see
            # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#detailed-notes).
            key = key.tuple + (Ellipsis,)
        else:
            raise TypeError("unexpected key type: {}".format(type(key)))

        return array, key

    def transpose(self, order):
        return self.array.transpose(order)

    def __getitem__(self, key):
        array, key = self._indexing_array_and_key(key)
        return array[key]

    def __setitem__(self, key, value):
        array, key = self._indexing_array_and_key(key)
        try:
            array[key] = value
        except ValueError:
            # More informative exception if read-only view
            if not array.flags.writeable and not array.flags.owndata:
                raise ValueError(
                    "Assignment destination is a view.  "
                    "Do you want to .copy() array first?"
                )
            else:
                raise


class NdArrayLikeIndexingAdapter(NumpyIndexingAdapter):
    __slots__ = ("array",)

    def __init__(self, array):
        if not hasattr(array, "__array_function__"):
            raise TypeError(
                "NdArrayLikeIndexingAdapter must wrap an object that "
                "implements the __array_function__ protocol"
            )
        self.array = array


class DaskIndexingAdapter(ExplicitlyIndexedNDArrayMixin):
    """Wrap a dask array to support explicit indexing."""

    __slots__ = ("array",)

    def __init__(self, array):
        """This adapter is created in Variable.__getitem__ in
        Variable._broadcast_indexes.
        """
        self.array = array

    def __getitem__(self, key):

        if not isinstance(key, VectorizedIndexer):
            # if possible, short-circuit when keys are effectively slice(None)
            # This preserves dask name and passes lazy array equivalence checks
            # (see duck_array_ops.lazy_array_equiv)
            rewritten_indexer = False
            new_indexer = []
            for idim, k in enumerate(key.tuple):
                if isinstance(k, Iterable) and duck_array_ops.array_equiv(
                    k, np.arange(self.array.shape[idim])
                ):
                    new_indexer.append(slice(None))
                    rewritten_indexer = True
                else:
                    new_indexer.append(k)
            if rewritten_indexer:
                key = type(key)(tuple(new_indexer))

        if isinstance(key, BasicIndexer):
            return self.array[key.tuple]
        elif isinstance(key, VectorizedIndexer):
            return self.array.vindex[key.tuple]
        else:
            assert isinstance(key, OuterIndexer)
            key = key.tuple
            try:
                return self.array[key]
            except NotImplementedError:
                # manual orthogonal indexing.
                # TODO: port this upstream into dask in a saner way.
                value = self.array
                for axis, subkey in reversed(list(enumerate(key))):
                    value = value[(slice(None),) * axis + (subkey,)]
                return value

    def __setitem__(self, key, value):
        raise TypeError(
            "this variable's data is stored in a dask array, "
            "which does not support item assignment. To "
            "assign to this variable, you must first load it "
            "into memory explicitly using the .load() "
            "method or accessing its .values attribute."
        )

    def transpose(self, order):
        return self.array.transpose(order)


class PandasIndexAdapter(ExplicitlyIndexedNDArrayMixin):
    """Wrap a pandas.Index to preserve dtypes and handle explicit indexing."""

    __slots__ = ("array", "_dtype")

    def __init__(self, array: Any, dtype: DTypeLike = None):
        self.array = utils.safe_cast_to_index(array)
        if dtype is None:
            if isinstance(array, pd.PeriodIndex):
                dtype_ = np.dtype("O")
            elif hasattr(array, "categories"):
                # category isn't a real numpy dtype
                dtype_ = array.categories.dtype
            elif not utils.is_valid_numpy_dtype(array.dtype):
                dtype_ = np.dtype("O")
            else:
                dtype_ = array.dtype
        else:
            dtype_ = np.dtype(dtype)
        self._dtype = dtype_

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def __array__(self, dtype: DTypeLike = None) -> np.ndarray:
        if dtype is None:
            dtype = self.dtype
        array = self.array
        if isinstance(array, pd.PeriodIndex):
            with suppress(AttributeError):
                # this might not be public API
                array = array.astype("object")
        return np.asarray(array.values, dtype=dtype)

    @property
    def shape(self) -> Tuple[int]:
        return (len(self.array),)

    def __getitem__(
        self, indexer
    ) -> Union[NumpyIndexingAdapter, np.ndarray, np.datetime64, np.timedelta64]:
        key = indexer.tuple
        if isinstance(key, tuple) and len(key) == 1:
            # unpack key so it can index a pandas.Index object (pandas.Index
            # objects don't like tuples)
            (key,) = key

        if getattr(key, "ndim", 0) > 1:  # Return np-array if multidimensional
            return NumpyIndexingAdapter(self.array.values)[indexer]

        result = self.array[key]

        if isinstance(result, pd.Index):
            result = PandasIndexAdapter(result, dtype=self.dtype)
        else:
            # result is a scalar
            if result is pd.NaT:
                # work around the impossibility of casting NaT with asarray
                # note: it probably would be better in general to return
                # pd.Timestamp rather np.than datetime64 but this is easier
                # (for now)
                result = np.datetime64("NaT", "ns")
            elif isinstance(result, timedelta):
                result = np.timedelta64(getattr(result, "value", result), "ns")
            elif isinstance(result, pd.Timestamp):
                # Work around for GH: pydata/xarray#1932 and numpy/numpy#10668
                # numpy fails to convert pd.Timestamp to np.datetime64[ns]
                result = np.asarray(result.to_datetime64())
            elif self.dtype != object:
                result = np.asarray(result, dtype=self.dtype)

            # as for numpy.ndarray indexing, we always want the result to be
            # a NumPy array.
            result = utils.to_0d_array(result)

        return result

    def transpose(self, order) -> pd.Index:
        return self.array  # self.array should be always one-dimensional

    def __repr__(self) -> str:
        return "{}(array={!r}, dtype={!r})".format(
            type(self).__name__, self.array, self.dtype
        )

    def copy(self, deep: bool = True) -> "PandasIndexAdapter":
        # Not the same as just writing `self.array.copy(deep=deep)`, as
        # shallow copies of the underlying numpy.ndarrays become deep ones
        # upon pickling
        # >>> len(pickle.dumps((self.array, self.array)))
        # 4000281
        # >>> len(pickle.dumps((self.array, self.array.copy(deep=False))))
        # 8000341
        array = self.array.copy(deep=True) if deep else self.array
        return PandasIndexAdapter(array, self._dtype)
