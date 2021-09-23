import collections.abc
import functools
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
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
        cls,
        index: pd.Index,
        dim: Hashable,
        var_meta: Optional[Dict[Any, Dict]] = None,
    ) -> Tuple["PandasIndex", IndexVars]:
        from .variable import IndexVariable

        if index.name is None:
            name = dim
            index = index.copy()
            index.name = dim
        else:
            name = index.name

        if var_meta is None:
            var_meta = {name: {}}

        data = PandasIndexingAdapter(index, dtype=var_meta[name].get("dtype"))
        index_var = IndexVariable(
            dim,
            data,
            fastpath=True,
            attrs=var_meta[name].get("attrs"),
            encoding=var_meta[name].get("encoding"),
        )

        return cls(index, dim, coord_dtype=var_meta[name].get("dtype")), {
            name: index_var
        }

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
        index = self.index.rename(new_name)
        new_dim = dims_dict.get(self.dim, self.dim)
        var_meta = {new_name: {"dtype": self.coord_dtype}}

        return self.from_pandas_index(index, dim=new_dim, var_meta=var_meta)

    def copy(self, deep=True):
        return self._replace(self.index.copy(deep=deep))

    def __getitem__(self, indexer: Any):
        return self._replace(self.index[indexer])


def _check_dim_compat(variables: Mapping[Any, "Variable"], all_dims: str = "equal"):
    """Check that all multi-index variable candidates are 1-dimensional and
    either share the same (single) dimension or each have a different dimension.

    """
    if any([var.ndim != 1 for var in variables.values()]):
        raise ValueError("PandasMultiIndex only accepts 1-dimensional variables")

    dims = set([var.dims for var in variables.values()])

    if all_dims == "equal" and len(dims) > 1:
        raise ValueError(
            "unmatched dimensions for multi-index variables "
            + ", ".join([f"{k!r} {v.dims}" for k, v in variables.items()])
        )

    if all_dims == "different" and len(dims) < len(variables):
        raise ValueError(
            "conflicting dimensions for multi-index product variables "
            + ", ".join([f"{k!r} {v.dims}" for k, v in variables.items()])
        )


def _get_var_metadata(variables: Mapping[Any, "Variable"]) -> Dict[Any, Dict[str, Any]]:
    return {
        name: {"dtype": var.dtype, "attrs": var.attrs, "encoding": var.encoding}
        for name, var in variables.items()
    }


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
        index.name = dim
        if level_coords_dtype is None:
            level_coords_dtype = self.level_coords_dtype
        return type(self)(index, dim, level_coords_dtype)

    @classmethod
    def from_variables(
        cls, variables: Mapping[Any, "Variable"]
    ) -> Tuple["PandasMultiIndex", IndexVars]:
        _check_dim_compat(variables)
        dim = next(iter(variables.values())).dims[0]

        index = pd.MultiIndex.from_arrays(
            [var.values for var in variables.values()], names=variables.keys()
        )
        index.name = dim
        level_coords_dtype = {name: var.dtype for name, var in variables.items()}
        obj = cls(index, dim, level_coords_dtype=level_coords_dtype)

        index_vars = _create_variables_from_multiindex(
            index, dim, var_meta=_get_var_metadata(variables)
        )

        return obj, index_vars

    @classmethod
    def from_product_variables(
        cls, variables: Mapping[Any, "Variable"], dim: Hashable
    ) -> Tuple["PandasMultiIndex", IndexVars]:
        """Create a new Pandas MultiIndex from the product of 1-d variables (levels) along a
        new dimension.

        Level variables must have a dimension distinct from each other.

        Keeps levels the same (doesn't refactorize them) so that it gives back the original
        labels after a stack/unstack roundtrip.

        """
        _check_dim_compat(variables, all_dims="different")

        level_indexes = [utils.safe_cast_to_index(var) for var in variables.values()]

        split_labels, levels = zip(*[lev.factorize() for lev in level_indexes])
        labels_mesh = np.meshgrid(*split_labels, indexing="ij")
        labels = [x.ravel() for x in labels_mesh]

        index = pd.MultiIndex(levels, labels, sortorder=0, names=variables.keys())

        return cls.from_pandas_index(index, dim, var_meta=_get_var_metadata(variables))

    @classmethod
    def from_variables_maybe_expand(
        cls,
        dim: Hashable,
        current_variables: Mapping[Any, "Variable"],
        variables: Mapping[Any, "Variable"],
    ) -> Tuple["PandasMultiIndex", IndexVars]:
        """Create a new multi-index maybe by expanding an existing one with
        new variables as index levels.

        The index and its corresponding coordinates may be created along a new dimension.
        """
        names: List[Hashable] = []
        codes: List[List[int]] = []
        levels: List[List[int]] = []
        level_variables: Dict[Any, "Variable"] = {}

        _check_dim_compat({**current_variables, **variables})

        if len(current_variables) > 1:
            # expand from an existing multi-index
            data = cast(
                PandasMultiIndexingAdapter, next(iter(current_variables.values()))._data
            )
            current_index = data.array
            names.extend(current_index.names)
            codes.extend(current_index.codes)
            levels.extend(current_index.levels)
            for name in current_index.names:
                level_variables[name] = current_variables[name]

        elif len(current_variables) == 1:
            # expand from one 1D variable (no multi-index): convert it to an index level
            var = next(iter(current_variables.values()))
            new_var_name = f"{dim}_level_0"
            names.append(new_var_name)
            cat = pd.Categorical(var.values, ordered=True)
            codes.append(cat.codes)
            levels.append(cat.categories)
            level_variables[new_var_name] = var

        for name, var in variables.items():
            names.append(name)
            cat = pd.Categorical(var.values, ordered=True)
            codes.append(cat.codes)
            levels.append(cat.categories)
            level_variables[name] = var

        index = pd.MultiIndex(levels, codes, names=names)

        return cls.from_pandas_index(
            index, dim, var_meta=_get_var_metadata(level_variables)
        )

    def keep_levels(
        self, level_variables: Mapping[Any, "Variable"]
    ) -> Tuple[Union["PandasMultiIndex", PandasIndex], IndexVars]:
        """Keep only the provided levels and return a new multi-index with its
        corresponding coordinates.

        """
        var_meta = _get_var_metadata(level_variables)
        index = self.index.droplevel(
            [k for k in self.index.names if k not in level_variables]
        )

        if isinstance(index, pd.MultiIndex):
            return self.from_pandas_index(index, self.dim, var_meta=var_meta)
        else:
            return PandasIndex.from_pandas_index(index, self.dim, var_meta=var_meta)

    def reorder_levels(
        self, level_variables: Mapping[Any, "Variable"]
    ) -> Tuple["PandasMultiIndex", IndexVars]:
        """Re-arrange index levels using input order and return a new multi-index with
        its corresponding coordinates.

        """
        index = self.index.reorder_levels(level_variables.keys())
        return self.from_pandas_index(
            index, self.dim, var_meta=_get_var_metadata(level_variables)
        )

    @classmethod
    def from_pandas_index(
        cls,
        index: pd.MultiIndex,
        dim: Hashable,
        var_meta: Optional[Dict[Any, Dict]] = None,
    ) -> Tuple["PandasMultiIndex", IndexVars]:

        names = []
        idx_dtypes = {}
        for i, idx in enumerate(index.levels):
            name = idx.name or f"{dim}_level_{i}"
            if name == dim:
                raise ValueError(
                    f"conflicting multi-index level name {name!r} with dimension {dim!r}"
                )
            names.append(name)
            idx_dtypes[name] = idx.dtype

        if var_meta is None:
            var_meta = {k: {} for k in names}
        for name, dtype in idx_dtypes.items():
            var_meta[name]["dtype"] = var_meta[name].get("dtype", dtype)

        level_coords_dtype = {k: var_meta[k]["dtype"] for k in names}

        index = index.rename(names)
        index.name = dim
        index_vars = _create_variables_from_multiindex(index, dim, var_meta=var_meta)
        return cls(index, dim, level_coords_dtype=level_coords_dtype), index_vars

    def query(self, labels, method=None, tolerance=None) -> QueryResult:
        from .variable import Variable

        if method is not None or tolerance is not None:
            raise ValueError(
                "multi-index does not support ``method`` and ``tolerance``"
            )

        new_index = None
        scalar_coord_values = {}

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
                scalar_coord_values.update(label_values)
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
                    levels = [self.index.names[i] for i in range(len(label))]
                    indexer, new_index = self.index.get_loc_level(label, level=levels)
                    scalar_coord_values.update({k: v for k, v in zip(levels, label)})

            else:
                label = (
                    label
                    if getattr(label, "ndim", 1) > 1  # vectorized-indexing
                    else _asarray_tuplesafe(label)
                )
                if label.ndim == 0:
                    indexer, new_index = self.index.get_loc_level(label.item(), level=0)
                    scalar_coord_values[self.index.names[0]] = label.item()
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
            # variable(s) attrs and encoding metadata are propagated
            # when replacing the indexes in the resulting xarray object
            var_meta = {k: {"dtype": v} for k, v in self.level_coords_dtype.items()}

            if isinstance(new_index, pd.MultiIndex):
                new_index, new_vars = self.from_pandas_index(
                    new_index, self.dim, var_meta=var_meta
                )
                dims_dict = {}
                drop_coords = []
            else:
                new_index, new_vars = PandasIndex.from_pandas_index(
                    new_index, new_index.name, var_meta=var_meta
                )
                dims_dict = {self.dim: new_index.index.name}
                drop_coords = [self.dim]

            indexes = cast(Dict[Any, Index], {k: new_index for k in new_vars})

            # add scalar variable for each dropped level
            variables = cast(
                Dict[Hashable, Union["Variable", "IndexVariable"]], new_vars
            )
            for name, val in scalar_coord_values.items():
                variables[name] = Variable([], val)

            return QueryResult(
                {self.dim: indexer},
                indexes=indexes,
                variables=variables,
                drop_indexes=list(scalar_coord_values),
                drop_coords=drop_coords,
                rename_dims=dims_dict,
            )

        else:
            return QueryResult({self.dim: indexer})

    def rename(self, name_dict, dims_dict):
        if not set(self.index.names) & set(name_dict) and self.dim not in dims_dict:
            return self, {}

        # pandas 1.3.0: could simply do `self.index.rename(names_dict)`
        new_names = [name_dict.get(k, k) for k in self.index.names]
        index = self.index.rename(new_names)

        new_dim = dims_dict.get(self.dim, self.dim)
        var_meta = {
            k: {"dtype": v} for k, v in zip(new_names, self.level_coords_dtype.values())
        }

        return self.from_pandas_index(index, new_dim, var_meta=var_meta)


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


# generic type that represents either pandas or xarray indexes
T_Index = TypeVar("T_Index")


class Indexes(collections.abc.Mapping, Generic[T_Index]):
    """Immutable proxy for Dataset or DataArrary indexes.

    Keys are coordinate names and values may correspond to either pandas or
    xarray indexes.

    Also provides some utility methods.

    """

    _indexes: Dict[Any, T_Index]

    def __init__(self, indexes: Dict[Any, T_Index]):
        """Constructor not for public consumption.

        Parameters
        ----------
        indexes : dict
            Indexes held by this object.
        """
        self._indexes = indexes

    @functools.cached_property
    def _coord_name_id(self) -> Dict[Any, int]:
        return {k: id(idx) for k, idx in self._indexes.items()}

    @functools.cached_property
    def _id_index(self) -> Dict[int, T_Index]:
        return {id(idx): idx for idx in self.get_unique()}

    @functools.cached_property
    def _id_coord_names(self) -> Dict[int, Tuple[Hashable, ...]]:
        id_coord_names: Mapping[int, List[Hashable]] = defaultdict(list)

        for k, v in self._coord_name_id.items():
            id_coord_names[v].append(k)

        return {k: tuple(v) for k, v in id_coord_names.items()}

    def get_unique(self) -> List[T_Index]:
        """Returns a list of unique indexes, preserving order."""

        unique_indexes: List[T_Index] = []
        seen: Set[T_Index] = set()

        for index in self._indexes.values():
            if index not in seen:
                unique_indexes.append(index)
                seen.add(index)

        return unique_indexes

    def get_all_coords(
        self, coord_name: Hashable, errors: str = "raise"
    ) -> Tuple[Hashable, ...]:
        """Return the names of all coordinates having the same index.

        Parameters
        ----------
        coord_name : hashable
            Name of an indexed coordinate.
        errors : {"raise", "ignore"}, optional
            If "raise", raises a ValueError if `coord_name` is not in indexes.
            If "ignore", an empty tuple is returned instead.

        Returns
        -------
        names : tuple
            The names of all coordinates having the same index.

        """
        if errors not in ["raise", "ignore"]:
            raise ValueError('errors must be either "raise" or "ignore"')

        if coord_name not in self._indexes:
            if errors == "raise":
                raise ValueError(f"no index found for {coord_name!r} coordinate")
            else:
                return tuple()

        return self._id_coord_names[self._coord_name_id[coord_name]]

    def group_by_index(self) -> List[Tuple[T_Index, Tuple[Hashable, ...]]]:
        """Returns a list of unique indexes and their corresponding coordinate names."""

        return [(self._id_index[i], self._id_coord_names[i]) for i in self._id_index]

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
