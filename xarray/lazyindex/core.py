"""Core data structures for lazy indexing."""
import operator
from dataclasses import dataclass
from typing import Any, Callable, Tuple
import numpy as np

from xarray.core.pycompat import integer_types


# TODO: make this a typing.Protocol
DuckArray = Any


class Array:
    shape: Tuple[int, ...]
    dtype: np.dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        out = 1
        for s in self.shape:
            out *= s
        return out

    def __len__(self) -> int:
        try:
            return self.shape[0]
        except IndexError:
            raise TypeError("len() of unsized object") from None

    def __getitem_explicit__(self, indexer: "ExplicitIndexer") -> "Array":
        raise NotImplementedError

    def __setitem_explicit__(self, indexer: "ExplicitIndexer", value: DuckArray):
        raise NotImplementedError

    def transpose(self, order):
        return TransposeArray(self, order)


class ExplicitIndexer:
    """Base class for explicit indexer objects.

    ExplicitIndexer objects wrap a tuple of values given by their ``tuple``
    property. These tuples should always have length equal to the number of
    dimensions on the indexed array.

    Do not instantiate BaseIndexer objects directly: instead, use one of the
    sub-classes BasicIndexer, OuterIndexer or VectorizedIndexer.
    """

    __slots__ = ("_value",)

    def __init__(self, key):
        if type(self) is ExplicitIndexer:
            raise TypeError("cannot instantiate base ExplicitIndexer objects")
        self._value = tuple(key)

    @property
    def value(self):
        return self._value

    def __repr__(self):
        return f"{type(self).__name__}({self.value})"


def _as_integer_or_none(value):
    return None if value is None else operator.index(value)


def as_integer_slice(value):
    start = _as_integer_or_none(value.start)
    stop = _as_integer_or_none(value.stop)
    step = _as_integer_or_none(value.step)
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

        found_ndarray = False

        new_key = []
        for k in key:
            if isinstance(k, integer_types):
                k = int(k)
            elif isinstance(k, slice):
                k = as_integer_slice(k)
            elif isinstance(k, np.ndarray):
                found_ndarray = True
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

        if not found_ndarray:
            raise ValueError("no ndarray key found: lower to BasicIndexer instead")

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

        if ndim is None:
            raise ValueError("no ndarray key found: lower to BasicIndexer instead")

        super().__init__(new_key)

    @property
    def ndarray_shape(self) -> Tuple[int, ...]:
        arrays = [k for k in self.value if isinstance(k, np.ndarray)]
        return np.broadcast(*arrays).shape


@dataclass(frozen=True)
class TransposeArray(Array):
    array: Array
    order: Tuple[int, ...]

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def shape(self):
        return tuple(self.array.shape[axis] for axis in self.order)

    def _updated_key_and_order(self, key):
        from .indexing import arrayize_vectorized_indexer

        if isinstance(key, (BasicIndexer, OuterIndexer)):
            new_key = type(key)(tuple(key.value[axis] for axis in self.order))
            new_order = tuple(
                axis
                for axis, k in zip(self.order, new_key.value)
                if not isinstance(k, int)
            )
        else:
            assert isinstance(key, VectorizedIndexer)
            key = arrayize_vectorized_indexer(key, self.shape)
            new_key = type(key)(
                tuple(key.value[axis].transpose(self.order) for axis in self.order)
            )
            new_order = None  # check this!

        return new_key, new_order

    def __getitem_explicit__(self, indexer: ExplicitIndexer):
        new_indexer, new_order = self._updated_key_and_order(indexer)
        new_array = self.array.__getitem_explicit__(new_indexer)
        if new_order is None:
            return new_array
        else:
            return type(self)(new_array, new_order)

    def __setitem_explicit__(self, indexer: ExplicitIndexer, value):
        new_indexer, new_order = self._updated_key_and_order(indexer)
        if new_order is not None:
            value = np.transpose(value, new_order)
        self.array.__setitem_explicit__(new_indexer, value)
