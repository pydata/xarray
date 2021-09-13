import collections.abc
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

from . import formatting, utils
from .indexing import PandasIndexingAdapter, PandasMultiIndexingAdapter, QueryResult
from .utils import is_dict_like, is_scalar

if TYPE_CHECKING:
    from .variable import IndexVariable, Variable

IndexVars = Dict[Any, "IndexVariable"]


class Index:
    """Base class inherited by all xarray-compatible indexes."""

    @classmethod
    def from_variables(
        cls, variables: Mapping[Any, "Variable"]
    ) -> Tuple["Index", IndexVars]:
        raise NotImplementedError()

    def to_pandas_index(self) -> pd.Index:
        """Cast this xarray index to a pandas.Index object or raise a TypeError
        if this is not supported.

        This method is used by all xarray operations that expect/require a
        pandas.Index object.

        """
        raise TypeError(f"{type(self)} cannot be cast to a pandas.Index object.")

    def query(self, labels: Dict[Any, Any]) -> QueryResult:
        raise NotImplementedError()

    def equals(self, other):  # pragma: no cover
        raise NotImplementedError()

    def union(self, other):  # pragma: no cover
        raise NotImplementedError()

    def intersection(self, other):  # pragma: no cover
        raise NotImplementedError()

    def rename(
        self, name_dict: Mapping[Any, Hashable], dims_dict: Mapping[Any, Hashable]
    ) -> Tuple["Index", IndexVars]:
        return self, {}

    def copy(self, deep: bool = True):  # pragma: no cover
        raise NotImplementedError()

    def __getitem__(self, indexer: Any):
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


def normalize_label(value, dtype=None) -> np.ndarray:
    if getattr(value, "ndim", 1) <= 1:
        value = _asarray_tuplesafe(value)
    if dtype is not None and dtype.kind == "f":
        # pd.Index built from coordinate with float precision != 64
        # see https://github.com/pydata/xarray/pull/3153 for details
        value = np.asarray(value, dtype=dtype)
    return value


def as_scalar(value: np.ndarray):
    # see https://github.com/pydata/xarray/pull/4292 for details
    return value[()] if value.dtype.kind in "mM" else value.item()


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

    index: pd.Index
    dim: Hashable
    coord_dtype: Any

    __slots__ = ("index", "dim", "coord_dtype")

    def __init__(self, array: Any, dim: Hashable, coord_dtype: Any = None):
        self.index = utils.safe_cast_to_index(array)
        self.dim = dim

        if coord_dtype is None:
            coord_dtype = self.index.dtype
        self.coord_dtype = coord_dtype

    def _replace(self, index, dim=None, coord_dtype=None):
        if dim is None:
            dim = self.dim
        if coord_dtype is None:
            coord_dtype = self.coord_dtype
        return type(self)(index, dim, coord_dtype)

    @classmethod
    def from_variables(
        cls, variables: Mapping[Any, "Variable"]
    ) -> Tuple["PandasIndex", IndexVars]:
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

        dim = var.dims[0]
        obj = cls(var.data, dim, coord_dtype=var.dtype)
        obj.index.name = name
        data = PandasIndexingAdapter(obj.index, dtype=var.dtype)
        index_var = IndexVariable(
            dim, data, attrs=var.attrs, encoding=var.encoding, fastpath=True
        )

        return obj, {name: index_var}

    @classmethod
    def from_pandas_index(
        cls, index: pd.Index, dim: Hashable
    ) -> Tuple["PandasIndex", IndexVars]:
        from .variable import IndexVariable

        if index.name is None:
            name = dim
            index = index.copy()
            index.name = dim
        else:
            name = index.name

        data = PandasIndexingAdapter(index)
        index_var = IndexVariable(dim, data, fastpath=True)

        return cls(index, dim), {name: index_var}

    def to_pandas_index(self) -> pd.Index:
        return self.index

    def query(self, labels: Dict[Any, Any], method=None, tolerance=None) -> QueryResult:
        from .dataarray import DataArray
        from .variable import Variable

        if method is not None and not isinstance(method, str):
            raise TypeError("``method`` must be a string")

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
            label_array = normalize_label(label, dtype=self.coord_dtype)
            if label_array.ndim == 0:
                label_value = as_scalar(label_array)
                if isinstance(self.index, pd.CategoricalIndex):
                    if method is not None:
                        raise ValueError(
                            "'method' is not supported when indexing using a CategoricalIndex."
                        )
                    if tolerance is not None:
                        raise ValueError(
                            "'tolerance' is not supported when indexing using a CategoricalIndex."
                        )
                    indexer = self.index.get_loc(label_value)
                else:
                    indexer = self.index.get_loc(
                        label_value, method=method, tolerance=tolerance
                    )
            elif label_array.dtype.kind == "b":
                indexer = label_array
            else:
                indexer = get_indexer_nd(self.index, label_array, method, tolerance)
                if np.any(indexer < 0):
                    raise KeyError(f"not all values found in index {coord_name!r}")

            # attach dimension names and/or coordinates to positional indexer
            if isinstance(label, Variable):
                indexer = Variable(label.dims, indexer)
            elif isinstance(label, DataArray):
                indexer = DataArray(indexer, coords=label._coords, dims=label.dims)

        return QueryResult({self.dim: indexer})

    def equals(self, other):
        return self.index.equals(other.index)

    def union(self, other):
        new_index = self.index.union(other.index)
        return type(self)(new_index, self.dim)

    def intersection(self, other):
        new_index = self.index.intersection(other.index)
        return type(self)(new_index, self.dim)

    def rename(self, name_dict, dims_dict):
        if self.index.name not in name_dict and self.dim not in dims_dict:
            return self, {}

        new_name = name_dict.get(self.index.name, self.index.name)
        pd_idx = self.index.rename(new_name)
        new_dim = dims_dict.get(self.dim, self.dim)

        index, index_vars = self.from_pandas_index(pd_idx, dim=new_dim)
        index.coord_dtype = self.coord_dtype

        return index, index_vars

    def copy(self, deep=True):
        return self._replace(self.index.copy(deep=deep))

    def __getitem__(self, indexer: Any):
        return self._replace(self.index[indexer])


def _check_dim_compat(variables: Mapping[Any, "Variable"]) -> Hashable:
    """Check that all multi-index variable candidates share the same (single) dimension
    and return the name of that dimension.

    """
    if any([var.ndim != 1 for var in variables.values()]):
        raise ValueError("PandasMultiIndex only accepts 1-dimensional variables")

    dims = set([var.dims for var in variables.values()])

    if len(dims) > 1:
        raise ValueError(
            "unmatched dimensions for variables "
            + ", ".join([f"{k!r} {v.dims}" for k, v in variables.items()])
        )

    return next(iter(dims))[0]


def _create_variables_from_multiindex(index, dim, var_meta=None):
    from .variable import IndexVariable

    if var_meta is None:
        var_meta = {}

    def create_variable(name):
        if name == dim:
            level = None
        else:
            level = name
        meta = var_meta.get(name, {})
        data = PandasMultiIndexingAdapter(index, dtype=meta.get("dtype"), level=level)
        return IndexVariable(
            dim,
            data,
            attrs=meta.get("attrs"),
            encoding=meta.get("encoding"),
            fastpath=True,
        )

    variables = {}
    variables[dim] = create_variable(dim)
    for level in index.names:
        variables[level] = create_variable(level)

    return variables


class PandasMultiIndex(PandasIndex):
    """Wrap a pandas.MultiIndex as an xarray compatible index."""

    level_coords_dtype: Dict[str, Any]

    __slots__ = ("index", "dim", "coord_dtype", "level_coords_dtype")

    def __init__(self, array: Any, dim: Hashable, level_coords_dtype: Any = None):
        super().__init__(array, dim)

        if level_coords_dtype is None:
            level_coords_dtype = {idx.name: idx.dtype for idx in self.index.levels}
        self.level_coords_dtype = level_coords_dtype

    def _replace(self, index, dim=None, level_coords_dtype=None) -> "PandasMultiIndex":
        if dim is None:
            dim = self.dim
        if level_coords_dtype is None:
            level_coords_dtype = self.level_coords_dtype
        return type(self)(index, dim, level_coords_dtype)

    @classmethod
    def from_variables(
        cls, variables: Mapping[Any, "Variable"]
    ) -> Tuple["PandasMultiIndex", IndexVars]:
        dim = _check_dim_compat(variables)

        index = pd.MultiIndex.from_arrays(
            [var.values for var in variables.values()], names=variables.keys()
        )
        level_coords_dtype = {name: var.dtype for name, var in variables.items()}
        obj = cls(index, dim, level_coords_dtype=level_coords_dtype)

        var_meta = {
            name: {"dtype": var.dtype, "attrs": var.attrs, "encoding": var.encoding}
            for name, var in variables.items()
        }
        index_vars = _create_variables_from_multiindex(index, dim, var_meta=var_meta)

        return obj, index_vars

    @classmethod
    def from_variables_maybe_expand(
        cls,
        dim: Hashable,
        current_variables: Mapping[Any, "Variable"],
        variables: Mapping[Any, "Variable"],
    ) -> Tuple["PandasMultiIndex", IndexVars]:
        """Create a new multi-index maybe by expanding an existing one with
        new variables as index levels.

        the index might be created along a new dimension.
        """
        names: List[Hashable] = []
        codes: List[List[int]] = []
        levels: List[List[int]] = []
        var_meta: Dict[str, Dict] = {}
        level_coords_dtype: Dict[Hashable, Any] = {}

        _check_dim_compat({**current_variables, **variables})

        def add_level_var(name, var):
            var_meta[name] = {
                "dtype": var.dtype,
                "attrs": var.attrs,
                "encoding": var.encoding,
            }
            level_coords_dtype[name] = var.dtype

        if len(current_variables) > 1:
            current_index: pd.MultiIndex = next(
                iter(current_variables.values())
            )._data.array
            names.extend(current_index.names)
            codes.extend(current_index.codes)
            levels.extend(current_index.levels)
            for name in current_index.names:
                add_level_var(name, current_variables[name])

        elif len(current_variables) == 1:
            # one 1D variable (no multi-index): convert it to an index level
            var = next(iter(current_variables.values()))
            new_var_name = f"{dim}_level_0"
            names.append(new_var_name)
            cat = pd.Categorical(var.values, ordered=True)
            codes.append(cat.codes)
            levels.append(cat.categories)
            add_level_var(new_var_name, var)

        for name, var in variables.items():
            names.append(name)
            cat = pd.Categorical(var.values, ordered=True)
            codes.append(cat.codes)
            levels.append(cat.categories)
            add_level_var(name, var)

        index = pd.MultiIndex(levels, codes, names=names)
        obj = cls(index, dim, level_coords_dtype=level_coords_dtype)
        index_vars = _create_variables_from_multiindex(index, dim, var_meta=var_meta)

        return obj, index_vars

    @classmethod
    def from_pandas_index(
        cls, index: pd.MultiIndex, dim: Hashable
    ) -> Tuple["PandasMultiIndex", IndexVars]:
        var_meta = {}
        for i, idx in enumerate(index.levels):
            name = idx.name or f"{dim}_level_{i}"
            if name == dim:
                raise ValueError(
                    f"conflicting multi-index level name {name!r} with dimension {dim!r}"
                )
            var_meta[name] = {"dtype": idx.dtype}

        index = index.rename(var_meta.keys())
        index_vars = _create_variables_from_multiindex(index, dim, var_meta=var_meta)
        return cls(index, dim), index_vars

    def query(self, labels, method=None, tolerance=None) -> QueryResult:
        if method is not None or tolerance is not None:
            raise ValueError(
                "multi-index does not support ``method`` and ``tolerance``"
            )

        new_index = None

        # label(s) given for multi-index level(s)
        if all([lbl in self.index.names for lbl in labels]):
            label_values = {}
            for k, v in labels.items():
                label_array = normalize_label(v, dtype=self.level_coords_dtype[k])
                try:
                    label_values[k] = as_scalar(label_array)
                except ValueError:
                    # label should be an item not an array-like
                    raise ValueError(
                        "Vectorized selection is not "
                        f"available along coordinate {k!r} (multi-index level)"
                    )

            has_slice = any([isinstance(v, slice) for v in label_values.values()])

            if len(label_values) == self.index.nlevels and not has_slice:
                indexer = self.index.get_loc(
                    tuple(label_values[k] for k in self.index.names)
                )
            else:
                indexer, new_index = self.index.get_loc_level(
                    tuple(label_values.values()), level=tuple(label_values.keys())
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
            if isinstance(new_index, pd.MultiIndex):
                new_index, new_vars = PandasMultiIndex.from_pandas_index(
                    new_index, self.dim
                )
                dims_dict = {}
                drop_coords = set(self.index.names) - set(new_index.index.names)
            else:
                new_index, new_vars = PandasIndex.from_pandas_index(
                    new_index, new_index.name
                )
                dims_dict = {self.dim: new_index.index.name}
                drop_coords = set(self.index.names) - {new_index.index.name} | {
                    self.dim
                }

            return QueryResult(
                {self.dim: indexer},
                indexes={k: new_index for k in new_vars},
                index_vars=new_vars,
                drop_coords=list(drop_coords),
                rename_dims=dims_dict,
            )

        else:
            return QueryResult({self.dim: indexer})

    def rename(self, name_dict, dims_dict):
        if not set(self.index.names) & set(name_dict) and self.dim not in dims_dict:
            return self, {}

        # pandas 1.3.0: could simply do `self.index.rename(names_dict)`
        new_names = [name_dict.get(k, k) for k in self.index.names]
        pd_idx = self.index.rename(new_names)
        new_dim = dims_dict.get(self.dim, self.dim)

        index, index_vars = self.from_pandas_index(pd_idx, new_dim)
        index.level_coords_dtype = {
            k: v for k, v in zip(new_names, self.level_coords_dtype.values())
        }

        return index, index_vars


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


def group_coords_by_index(
    indexes: Mapping[Any, Index]
) -> List[Tuple[Index, List[Hashable]]]:
    """Returns a list of unique indexes and their corresponding coordinate names."""
    unique_indexes: Dict[int, Index] = {}
    grouped_coord_names: Mapping[int, List[Hashable]] = defaultdict(list)

    for coord_name, index_obj in indexes.items():
        index_id = id(index_obj)
        unique_indexes[index_id] = index_obj
        grouped_coord_names[index_id].append(coord_name)

    return [(unique_indexes[k], grouped_coord_names[k]) for k in unique_indexes]


def unique_indexes(indexes: Mapping[Any, Index]) -> List[Index]:
    """Returns a list of unique indexes, preserving order."""
    unique_indexes = []
    seen = []

    for index in indexes.values():
        if index not in seen:
            unique_indexes.append(index)
            seen.append(index)

    return unique_indexes


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
    indexers: Mapping[Any, Union[int, slice, np.ndarray, "Variable"]],
) -> Tuple["Variable", Optional[Index]]:
    """Index a Variable and an Index together.

    If the index cannot be indexed, return None (it will be dropped).

    (note: not compatible yet with xarray flexible indexes).

    """
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
    try:
        new_index = index[indexer]
    except NotImplementedError:
        new_index = None

    return new_variable, new_index


def roll_index(index: PandasIndex, count: int, axis: int = 0) -> PandasIndex:
    """Roll an pandas.Index."""
    pd_index = index.to_pandas_index()
    count %= pd_index.shape[0]
    if count != 0:
        new_idx = pd_index[-count:].append(pd_index[:-count])
    else:
        new_idx = pd_index[:]
    return PandasIndex(new_idx, index.dim)


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
