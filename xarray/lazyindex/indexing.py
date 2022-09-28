import numpy as np
from typing import cast, Tuple, Union
from dataclasses import dataclass

from .core import (
    Array,
    DuckArray,
    BasicIndexer,
    ExplicitIndexer,
    OuterIndexer,
    VectorizedIndexer,
)


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


def _normalize_slice(sl, size):
    """Ensure that given slice only contains positive start and stop values
    (stop can be -1 for full-size slices with negative steps, e.g. [-10::-1])"""
    return slice(*sl.indices(size))


def _slice_slice(old_slice: slice, applied_slice: slice, size: int) -> slice:
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


def _expand_slice(slice_, size):
    return np.arange(*slice_.indices(size))


def _index_indexer_1d(old_indexer, applied_indexer, size):
    assert isinstance(applied_indexer, integer_types + (slice, np.ndarray))
    if isinstance(applied_indexer, slice) and applied_indexer == slice(None):
        # shortcut for the usual case
        return old_indexer
    if isinstance(old_indexer, slice):
        if isinstance(applied_indexer, slice):
            indexer = _slice_slice(old_indexer, applied_indexer, size)
        else:
            indexer = _expand_slice(old_indexer, size)[applied_indexer]
    else:
        indexer = old_indexer[applied_indexer]
    return indexer


BasicOrOuterIndexer = Union[BasicIndexer, OuterIndexer]


@dataclass(frozen=True)
class OIndexArray(Array):
    array: DuckArray
    key: BasicOrOuterIndexer

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def shape(self):
        shape = []
        for size, k in zip(self.array.shape, self.key.value):
            if isinstance(k, slice):
                shape.append(len(range(*k.indices(size))))
            elif isinstance(k, np.ndarray):
                shape.extend(k.shape)
        return tuple(shape)

    def _updated_key(self, new_key: BasicOrOuterIndexer) -> BasicOrOuterIndexer:
        iter_new_key = iter(expanded_indexer(new_key.value, self.ndim))
        full_key = []
        for size, k in zip(self.array.shape, self.key.value):
            if isinstance(k, int):
                full_key.append(k)
            else:
                full_key.append(_index_indexer_1d(k, next(iter_new_key), size))
        full_key = tuple(full_key)

        if all(isinstance(k, (int, slice)) for k in full_key):
            return BasicIndexer(full_key)
        return OuterIndexer(full_key)

    def _getitem_explicit_(self, indexer: ExplicitIndexer) -> Array:
        if isinstance(indexer, VectorizedIndexer):
            return VIndexArray(self, indexer)
        indexer = cast(BasicOrOuterIndexer, indexer)
        return type(self)(self.array, self._updated_key(indexer))

    def _setitem_explicit_(self, indexer: ExplicitIndexer, value):
        if isinstance(indexer, VectorizedIndexer):
            raise NotImplementedError(
                "Lazy item assignment with the vectorized indexer is not yet "
                "implemented. Load your data first by .load() or compute()."
            )
        indexer = cast(BasicOrOuterIndexer, indexer)
        full_key = self._updated_key(indexer)
        self.array[full_key] = value


@dataclass(frozen=True)
class VIndexArray(Array):
    array: DuckArray
    key: VectorizedIndexer

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def shape(self):
        slice_shape = []
        for size, k in zip(self.array.shape, self.key.value):
            if isinstance(k, slice):
                slice_shape.append(len(range(*k.indices(size))))

        return self.key.ndarray_shape + tuple(slice_shape)

    def _updated_key(self, new_key):
        if isinstance(new_key, VectorizedIndexer):
            # TODO: use slicing and transposing rather than converting into
            # ndarrays
            new_key = arrayize_vectorized_indexer(new_key, self.shape)
        else:
            new_key = _outer_to_vectorized_indexer(new_key, self.shape)
        # TODO: handle slices rather than converting entirely into ndarrays.
        old_key = arrayize_vectorized_indexer(self.key, self.array.shape)
        return VectorizedIndexer(
            tuple(o[new_key.value] for o in np.broadcast_arrays(*old_key.value))
        )

    def _getitem_explicit_(self, indexer: ExplicitIndexer):
        # TODO: lower into OIndexArray when possible
        return type(self)(self.array, self._updated_key(indexer))

    def _setitem_explicit_(self, indexer: ExplicitIndexer, value):
        raise NotImplementedError(
            "Lazy item assignment with the vectorized indexer is not yet "
            "implemented. Load your data first by .load() or compute()."
        )


def arrayize_vectorized_indexer(
    indexer: VectorizedIndexer, shape: Tuple[int, ...]
) -> VectorizedIndexer:
    """Return an identical vindex but slices are replaced by arrays.

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
    slices = [v for v in indexer.value if isinstance(v, slice)]
    if not slices:
        return indexer

    arrays = [v for v in indexer.value if isinstance(v, np.ndarray)]
    n_dim = arrays[0].ndim if arrays else 0
    i_dim = 0
    new_key = []
    for v, size in zip(indexer.value, shape):
        if isinstance(v, np.ndarray):
            new_key.append(np.reshape(v, v.shape + (1,) * len(slices)))
        else:
            v = cast(slice, v)
            index_shape = (
                (1,) * (n_dim + i_dim) + (-1,) + (1,) * (len(slices) - i_dim - 1)
            )
            new_key.append(np.arange(*v.indices(size)).reshape(index_shape))
            i_dim += 1
    return VectorizedIndexer(tuple(new_key))


def _outer_to_vectorized_indexer(
    key: BasicOrOuterIndexer, shape: Tuple[int, ...]
) -> VectorizedIndexer:
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
    n_dim = len([k for k in key.value if not isinstance(k, int)])
    i_dim = 0
    new_key = []
    for k, size in zip(key.value, shape):
        if isinstance(k, int):
            new_key.append(np.array(k).reshape((1,) * n_dim))
        else:  # np.ndarray or slice
            if isinstance(k, slice):
                k = np.arange(*k.indices(size))
            assert k.dtype.kind == "i"
            index_shape = [(1,) * i_dim + (k.size,) + (1,) * (n_dim - i_dim - 1)]
            new_key.append(k.reshape(*index_shape))
            i_dim += 1
    return VectorizedIndexer(tuple(new_key))
