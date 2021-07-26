import collections.abc
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Hashable,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

from . import formatting, utils
from .indexing import (
    LazilyIndexedArray,
    PandasIndexingAdapter,
    PandasLevelIndexingAdapter,
    PandasMultiIndexingAdapter,
)
from .utils import is_dict_like, is_scalar

if TYPE_CHECKING:
    from .variable import Variable


class Index:
    """Base class inherited by all xarray-compatible indexes."""

    __slots__ = ("coords",)

    def __init__(self, variables: Mapping[Hashable, "Variable"]):
        self.coords = {k: v.to_index_variable() for k, v in variables.items()}

    def to_pandas_index(self) -> pd.Index:
        """Cast this xarray index to a pandas.Index object or raise a TypeError
        if this is not supported.

        This method is used by all xarray operations that expect/require a
        pandas.Index object.

        """
        raise TypeError(f"{type(self)} cannot be cast to a pandas.Index object.")

    def query(self, labels: Dict[Hashable, Any]):  # pragma: no cover
        raise NotImplementedError

    def equals(self, other):  # pragma: no cover
        raise NotImplementedError()

    def union(self, other):  # pragma: no cover
        raise NotImplementedError()

    def intersection(self, other):  # pragma: no cover
        raise NotImplementedError()


def _sanitize_slice_element(x):
    from .dataarray import DataArray
    from .variable import Variable

    if not isinstance(x, tuple) and len(np.shape(x)) != 0:
        raise ValueError(
            f"cannot use non-scalar arrays in a slice for xarray indexing: {x}"
        )

    if isinstance(x, (Variable, DataArray)):
        x = x.values

    if isinstance(x, np.ndarray):
        x = x[()]

    return x


def _query_slice(index, label, coord_name="", method=None, tolerance=None):
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
            "cannot represent labeled-based slice indexer for coordinate "
            f"{coord_name!r} with a slice over integer positions; the index is "
            "unsorted or non-unique"
        )
    return indexer


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


class PandasIndex(Index):
    """Wrap a pandas.Index as an xarray compatible index."""

    __slots__ = ("index", "coords")

    def __init__(self, variables):
        from .variable import IndexVariable

        if len(variables) != 1:
            raise ValueError(
                f"PandasIndex only accepts one variable, found {len(variables)} variables"
            )

        name, var = next(iter(variables.items()))

        if var.ndim != 1:
            raise ValueError(
                "PandasIndex only accepts a 1-dimensional variable, "
                f"variable {name!r} has {var.ndim} dimensions"
            )

        # TODO: fastpath when ``var.data`` is already a PandasIndexingAdapter

        self.index = utils.safe_cast_to_index(var.data)

        data = LazilyIndexedArray(PandasIndexingAdapter(self.index))
        new_var = IndexVariable(
            var.dims, data, attrs=var.attrs, encoding=var.encoding, fastpath=True
        )
        self.coords = {name: new_var}

    def to_pandas_index(self) -> pd.Index:
        return self.index

    def query(
        self, labels, method=None, tolerance=None
    ) -> Tuple[Any, Union["PandasIndex", None]]:
        assert len(labels) == 1
        coord_name, label = next(iter(labels.items()))

        if isinstance(label, slice):
            indexer = _query_slice(self.index, label, coord_name, method, tolerance)
        elif is_dict_like(label):
            raise ValueError(
                "cannot use a dict-like object for selection on "
                "a dimension that does not have a MultiIndex"
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
                if isinstance(self.index, pd.CategoricalIndex):
                    if method is not None:
                        raise ValueError(
                            "'method' is not a valid kwarg when indexing using a CategoricalIndex."
                        )
                    if tolerance is not None:
                        raise ValueError(
                            "'tolerance' is not a valid kwarg when indexing using a CategoricalIndex."
                        )
                    indexer = self.index.get_loc(label_value)
                else:
                    indexer = self.index.get_loc(
                        label_value, method=method, tolerance=tolerance
                    )
            elif label.dtype.kind == "b":
                indexer = label
            else:
                indexer = get_indexer_nd(self.index, label, method, tolerance)
                if np.any(indexer < 0):
                    raise KeyError(f"not all values found in index {coord_name!r}")

        return indexer, None

    def equals(self, other):
        if isinstance(other, PandasIndex):
            other = other.index
        return self.index.equals(other)

    def union(self, other):
        if isinstance(other, PandasIndex):
            other = other.index
        return type(self)(self.index.union(other))

    def intersection(self, other):
        if isinstance(other, PandasIndex):
            other = other.index
        return type(self)(self.index.intersection(other))


class PandasMultiIndex(PandasIndex):
    def __init__(self, variables):
        from .variable import IndexVariable

        # TODO: check that all variables have the same dimension (only 1D)
        dim = "todo"

        # TODO: fastpath when dim_var is already a PandasIndexingAdapter (and level vars hold valid refs)

        self.index = pd.MultiIndex.from_arrays(
            [var.values for var in variables.values()], names=variables.keys()
        )

        mindex_adapter = PandasMultiIndexingAdapter(self.index)
        dim_var = IndexVariable(dim, LazilyIndexedArray(mindex_adapter), fastpath=True)

        self.coords = {dim: dim_var}

        for name, var in variables.items():
            data = PandasLevelIndexingAdapter(mindex_adapter, name, var.dtype)
            self.coords[name] = IndexVariable(
                dim, LazilyIndexedArray(data), fastpath=True
            )

    def query(
        self, labels, method=None, tolerance=None
    ) -> Tuple[Any, Union["PandasIndex", None]]:
        if method is not None or tolerance is not None:
            raise ValueError(
                "multi-index does not support ``method`` and ``tolerance``"
            )

        new_index = None

        # label(s) given for multi-index level(s)
        if all([lbl in self.index.names for lbl in labels]):
            is_nested_vals = _is_nested_tuple(tuple(labels.values()))
            if len(labels) == self.index.nlevels and not is_nested_vals:
                indexer = self.index.get_loc(tuple(labels[k] for k in self.index.names))
            else:
                for k, v in labels.items():
                    # index should be an item (i.e. Hashable) not an array-like
                    if isinstance(v, Sequence) and not isinstance(v, str):
                        raise ValueError(
                            "Vectorized selection is not "
                            f"available along coordinate {k!r} (multi-index level)"
                        )
                indexer, new_index = self.index.get_loc_level(
                    tuple(labels.values()), level=tuple(labels.keys())
                )
                # GH2619. Raise a KeyError if nothing is chosen
                if indexer.dtype.kind == "b" and indexer.sum() == 0:
                    raise KeyError(f"{labels} not found")

        # assume one label value given for the multi-index "array" (dimension)
        else:
            if len(labels) > 1:
                coord_name = next(iter(set(labels) - set(self.index.names)))
                raise ValueError(
                    f"cannot provide labels for both coordinate {coord_name!r} (multi-index array) "
                    f"and one or more coordinates among {self.index.names!r} (multi-index levels)"
                )

            coord_name, label = next(iter(labels.items()))

            if is_dict_like(label):
                invalid_levels = [
                    name for name in label if name not in self.index.names
                ]
                if invalid_levels:
                    raise ValueError(
                        f"invalid multi-index level names {invalid_levels}"
                    )
                return self.query(label)

            elif isinstance(label, slice):
                indexer = _query_slice(self.index, label, coord_name)

            elif isinstance(label, tuple):
                if _is_nested_tuple(label):
                    indexer = self.index.get_locs(label)
                elif len(label) == self.index.nlevels:
                    indexer = self.index.get_loc(label)
                else:
                    indexer, new_index = self.index.get_loc_level(
                        label, level=list(range(len(label)))
                    )

            else:
                label = (
                    label
                    if getattr(label, "ndim", 1) > 1  # vectorized-indexing
                    else _asarray_tuplesafe(label)
                )
                if label.ndim == 0:
                    indexer, new_index = self.index.get_loc_level(label.item(), level=0)
                elif label.dtype.kind == "b":
                    indexer = label
                else:
                    if label.ndim > 1:
                        raise ValueError(
                            "Vectorized selection is not available along "
                            f"coordinate {coord_name!r} with a multi-index"
                        )
                    indexer = get_indexer_nd(self.index, label)
                    if np.any(indexer < 0):
                        raise KeyError(f"not all values found in index {coord_name!r}")

        if new_index is not None:
            new_index = PandasIndex(new_index)

        return indexer, new_index


def wrap_pandas_index(index):
    if isinstance(index, pd.MultiIndex):
        return PandasMultiIndex(index)
    else:
        return PandasIndex(index)


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
) -> Dict[Hashable, Index]:
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
    return {key: coords[key]._to_xindex() for key in dims if key in coords}


def isel_variable_and_index(
    name: Hashable,
    variable: "Variable",
    index: Index,
    indexers: Mapping[Hashable, Union[int, slice, np.ndarray, "Variable"]],
) -> Tuple["Variable", Optional[Index]]:
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
    pd_index = index.to_pandas_index()
    new_index = wrap_pandas_index(pd_index[indexer])
    return new_variable, new_index


def roll_index(index: PandasIndex, count: int, axis: int = 0) -> PandasIndex:
    """Roll an pandas.Index."""
    pd_index = index.to_pandas_index()
    count %= pd_index.shape[0]
    if count != 0:
        new_idx = pd_index[-count:].append(pd_index[:-count])
    else:
        new_idx = pd_index[:]
    return PandasIndex(new_idx)


def propagate_indexes(
    indexes: Optional[Dict[Hashable, Index]], exclude: Optional[Any] = None
) -> Optional[Dict[Hashable, Index]]:
    """Creates new indexes dict from existing dict optionally excluding some dimensions."""
    if exclude is None:
        exclude = ()

    if is_scalar(exclude):
        exclude = (exclude,)

    if indexes is not None:
        new_indexes = {k: v for k, v in indexes.items() if k not in exclude}
    else:
        new_indexes = None  # type: ignore[assignment]

    return new_indexes
