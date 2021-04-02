import collections.abc
from contextlib import suppress
from datetime import timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Hashable,
    Iterable,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

from . import formatting, utils
from .indexing import ExplicitlyIndexedNDArrayMixin, NumpyIndexingAdapter
from .npcompat import DTypeLike
from .utils import is_scalar

if TYPE_CHECKING:
    from .variable import Variable


class IndexAdapter:
    """Base class inherited by all xarray-compatible indexes."""

    __slots__ = "coord_names"

    def __init__(self, coord_names: Union[Hashable, Iterable[Hashable]]):
        if isinstance(coord_names, Iterable) and not isinstance(coord_names, str):
            self.coord_names = tuple(coord_names)
        else:
            self.coord_names = tuple([coord_names])

        # TODO (benbovy - flexible indexes): remove
        # temporarly avoid mypy errors: the `array` attribute is used in many places
        # to access the underlying pandas.Index objects from xarray_obj.indexes
        self.array = pd.Index([])

    @classmethod
    def from_variables(
        cls, variables: Dict[Hashable, "Variable"], **kwargs
    ):  # pragma: no cover
        raise NotImplementedError()

    def equals(self, other):  # pragma: no cover
        raise NotImplementedError()

    def union(self, other):  # pragma: no cover
        raise NotImplementedError()

    def intersection(self, other):  # pragma: no cover
        raise NotImplementedError()


class PandasIndexAdapter(IndexAdapter, ExplicitlyIndexedNDArrayMixin):
    """Wrap a pandas.Index to preserve dtypes and handle explicit indexing."""

    __slots__ = ("array", "_dtype")

    def __init__(
        self, array: Any, dtype: DTypeLike = None, coord_name: Optional[Hashable] = None
    ):
        if coord_name is None:
            coord_name = tuple()
        super().__init__(coord_name)

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

    @classmethod
    def from_variables(cls, variables: Dict[Hashable, "Variable"], **kwargs):
        if len(variables) > 1:
            raise ValueError("Cannot set a pandas.Index from more than one variable")

        varname, var = list(variables.items())[0]
        return cls(var.data, dtype=var.dtype, coord_name=varname)

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

    def equals(self, other):
        if isinstance(other, pd.Index):
            return self.array.equals(other)
        else:
            return self.array.equals(other.array)

    def union(self, other):
        if isinstance(other, pd.Index):
            return self.array.union(other)
        else:
            return self.array.union(other.array)

    def intersection(self, other):
        if isinstance(other, pd.Index):
            return self.array.intersection(other)
        else:
            return self.array.intersection(other.array)

    def __getitem__(
        self, indexer
    ) -> Union[
        "PandasIndexAdapter",
        NumpyIndexingAdapter,
        np.ndarray,
        np.datetime64,
        np.timedelta64,
    ]:
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


def remove_unused_levels_categories(index: pd.Index) -> pd.Index:
    """
    Remove unused levels from MultiIndex and unused categories from CategoricalIndex
    """
    if isinstance(index, pd.MultiIndex):
        index = index.remove_unused_levels()
        # if it contains CategoricalIndex, we need to remove unused categories
        # manually. See https://github.com/pandas-dev/pandas/issues/30846
        if any(isinstance(lev, pd.CategoricalIndex) for lev in index.levels):
            levels = []
            for i, level in enumerate(index.levels):
                if isinstance(level, pd.CategoricalIndex):
                    level = level[index.codes[i]].remove_unused_categories()
                else:
                    level = level[index.codes[i]]
                levels.append(level)
            # TODO: calling from_array() reorders MultiIndex levels. It would
            # be best to avoid this, if possible, e.g., by using
            # MultiIndex.remove_unused_levels() (which does not reorder) on the
            # part of the MultiIndex that is not categorical, or by fixing this
            # upstream in pandas.
            index = pd.MultiIndex.from_arrays(levels, names=index.names)
    elif isinstance(index, pd.CategoricalIndex):
        index = index.remove_unused_categories()
    return index


class Indexes(collections.abc.Mapping):
    """Immutable proxy for Dataset or DataArrary indexes."""

    __slots__ = ("_indexes",)

    def __init__(self, indexes):
        """Not for public consumption.

        Parameters
        ----------
        indexes : Dict[Any, pandas.Index]
            Indexes held by this object.
        """
        self._indexes = indexes

    def __iter__(self):
        return iter(self._indexes)

    def __len__(self):
        return len(self._indexes)

    def __contains__(self, key):
        return key in self._indexes

    def __getitem__(self, key):
        return self._indexes[key]

    def __repr__(self):
        return formatting.indexes_repr(self)


def default_indexes(
    coords: Mapping[Any, "Variable"], dims: Iterable
) -> Dict[Hashable, IndexAdapter]:
    """Default indexes for a Dataset/DataArray.

    Parameters
    ----------
    coords : Mapping[Any, xarray.Variable]
        Coordinate variables from which to draw default indexes.
    dims : iterable
        Iterable of dimension names.

    Returns
    -------
    Mapping from indexing keys (levels/dimension names) to indexes used for
    indexing along that dimension.
    """
    return {key: coords[key]._to_index_adpater() for key in dims if key in coords}


def isel_variable_and_index(
    name: Hashable,
    variable: "Variable",
    index: IndexAdapter,
    indexers: Mapping[Hashable, Union[int, slice, np.ndarray, "Variable"]],
) -> Tuple["Variable", Optional[IndexAdapter]]:
    """Index a Variable and pandas.Index together."""
    from .variable import Variable

    if not indexers:
        # nothing to index
        return variable.copy(deep=False), index

    if len(variable.dims) > 1:
        raise NotImplementedError(
            "indexing multi-dimensional variable with indexes is not supported yet"
        )

    new_variable = variable.isel(indexers)

    if new_variable.dims != (name,):
        # can't preserve a index if result has new dimensions
        return new_variable, None

    # we need to compute the new index
    (dim,) = variable.dims
    indexer = indexers[dim]
    if isinstance(indexer, Variable):
        indexer = indexer.data
    new_index = PandasIndexAdapter(index.array[indexer])
    return new_variable, new_index


def roll_index(index: pd.Index, count: int, axis: int = 0) -> pd.Index:
    """Roll an pandas.Index."""
    count %= index.shape[0]
    if count != 0:
        return index[-count:].append(index[:-count])
    else:
        return index[:]


def propagate_indexes(
    indexes: Optional[Dict[Hashable, IndexAdapter]], exclude: Optional[Any] = None
) -> Optional[Dict[Hashable, IndexAdapter]]:
    """Creates new indexes dict from existing dict optionally excluding some dimensions."""
    if exclude is None:
        exclude = ()

    if is_scalar(exclude):
        exclude = (exclude,)

    if indexes is not None:
        new_indexes = {k: v for k, v in indexes.items() if k not in exclude}
    else:
        new_indexes = None  # type: ignore

    return new_indexes
