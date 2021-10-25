import collections.abc
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
    Sequence,
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
from .types import T_Index
from .utils import Frozen, is_dict_like, is_scalar

if TYPE_CHECKING:
    from .variable import Variable

IndexVars = Dict[Any, "Variable"]


class Index:
    """Base class inherited by all xarray-compatible indexes."""

    @classmethod
    def from_variables(
        cls, variables: Mapping[Any, "Variable"]
    ) -> Tuple["Index", IndexVars]:
        raise NotImplementedError()

    def create_variables(
        self, variables: Optional[Mapping[Any, "Variable"]] = None
    ) -> IndexVars:
        return {}

    def to_pandas_index(self) -> pd.Index:
        """Cast this xarray index to a pandas.Index object or raise a TypeError
        if this is not supported.

        This method is used by all xarray operations that expect/require a
        pandas.Index object.

        """
        raise TypeError(f"{self!r} cannot be cast to a pandas.Index object")

    def isel(
        self, indexers: Mapping[Any, Union[int, slice, np.ndarray, "Variable"]]
    ) -> Union["Index", None]:
        return None

    def query(self, labels: Dict[Any, Any]) -> QueryResult:
        raise NotImplementedError(f"{self!r} doesn't support label-based selection")

    def join(self: T_Index, other: T_Index, how: str = "inner") -> T_Index:
        raise NotImplementedError(
            f"{self!r} doesn't support alignment with inner/outer join method"
        )

    def reindex_like(self: T_Index, other: T_Index) -> Dict[Hashable, Any]:
        raise NotImplementedError(f"{self!r} doesn't support re-indexing labels")

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
    if dtype is not None and dtype.kind == "f" and value.dtype.kind != "b":
        # pd.Index built from coordinate with float precision != 64
        # see https://github.com/pydata/xarray/pull/3153 for details
        # bypass coercing dtype for boolean indexers (ignore index)
        # see https://github.com/pydata/xarray/issues/5727
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

    def create_variables(
        self, variables: Optional[Mapping[Any, "Variable"]] = None
    ) -> IndexVars:
        from .variable import IndexVariable

        name = self.index.name
        attrs: Union[Mapping[Hashable, Any], None]
        encoding: Union[Mapping[Hashable, Any], None]

        if variables is not None and name in variables:
            var = variables[name]
            attrs = var.attrs
            encoding = var.encoding
        else:
            attrs = None
            encoding = None

        name = self.index.name
        data = PandasIndexingAdapter(self.index, dtype=self.coord_dtype)
        var = IndexVariable(self.dim, data, attrs=attrs, encoding=encoding)
        return {name: var}

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

    def isel(
        self, indexers: Mapping[Any, Union[int, slice, np.ndarray, "Variable"]]
    ) -> Optional["PandasIndex"]:
        from .variable import Variable

        indxr = indexers[self.dim]
        if isinstance(indxr, int):
            # can't preserve index with single value
            return None
        elif isinstance(indxr, Variable):
            if indxr.dims != (self.dim,):
                # can't preserve a index if result has new dimensions
                return None
            else:
                indxr = indxr.data

        return self._replace(self.index[indxr])

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

    def equals(self, other: Index):
        if not isinstance(other, PandasIndex):
            return False
        return self.index.equals(other.index) and self.dim == other.dim

    def join(
        self: "PandasIndex", other: "PandasIndex", how: str = "inner"
    ) -> "PandasIndex":
        if how == "outer":
            index = self.index.union(other.index)
        else:
            # how = "inner"
            index = self.index.intersection(other.index)

        coord_dtype = np.result_type(self.coord_dtype, other.coord_dtype)
        return type(self)(index, self.dim, coord_dtype=coord_dtype)

    def union(self, other):
        new_index = self.index.union(other.index)
        return type(self)(new_index, self.dim)

    def intersection(self, other):
        new_index = self.index.intersection(other.index)
        return type(self)(new_index, self.dim)

    def reindex_like(
        self, other: "PandasIndex", method=None, tolerance=None
    ) -> Dict[Hashable, Any]:
        if not self.index.is_unique:
            raise ValueError(
                f"cannot reindex or align along dimension {self.dim!r} because the "
                "(pandas) index has duplicate values"
            )

        return {self.dim: get_indexer_nd(self.index, other.index, method, tolerance)}

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

    def create_variables(
        self, variables: Optional[Mapping[Any, "Variable"]] = None
    ) -> IndexVars:
        var_meta = {}
        if variables is not None:
            for name in self.index.names:
                var = variables[name]
                var_meta[name] = {
                    "dtype": self.level_coords_dtype[name],
                    "attrs": var.attrs,
                    "encoding": var.encoding,
                }

        return _create_variables_from_multiindex(
            self.index, self.dim, var_meta=var_meta
        )

    def query(self, labels, method=None, tolerance=None) -> QueryResult:
        from .dataarray import DataArray
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
                label_array = normalize_label(label)
                if label_array.ndim == 0:
                    label_value = as_scalar(label_array)
                    indexer, new_index = self.index.get_loc_level(label_value, level=0)
                    scalar_coord_values[self.index.names[0]] = label_value
                elif label_array.dtype.kind == "b":
                    indexer = label_array
                else:
                    if label_array.ndim > 1:
                        raise ValueError(
                            "Vectorized selection is not available along "
                            f"coordinate {coord_name!r} with a multi-index"
                        )
                    indexer = get_indexer_nd(self.index, label_array)
                    if np.any(indexer < 0):
                        raise KeyError(f"not all values found in index {coord_name!r}")

                # attach dimension names and/or coordinates to positional indexer
                if isinstance(label, Variable):
                    indexer = Variable(label.dims, indexer)
                elif isinstance(label, DataArray):
                    # do not include label-indexer DataArray coordinates that conflict
                    # with the level names of this index
                    coords = {
                        k: v
                        for k, v in label._coords.items()
                        if k not in self.index.names
                    }
                    indexer = DataArray(indexer, coords=coords, dims=label.dims)

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
            variables = new_vars
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

    def join(self, other, how: str = "inner"):
        if how == "outer":
            # bug in pandas? need to reset index.name
            other_index = other.index.copy()
            other_index.name = None
            index = self.index.union(other_index)
            index.name = self.dim
        else:
            # how = "inner"
            index = self.index.intersection(other.index)

        level_coords_dtype = {
            k: np.result_type(lvl_dtype, other.level_coords_dtype[k])
            for k, lvl_dtype in self.level_coords_dtype.items()
        }

        return type(self)(index, self.dim, level_coords_dtype=level_coords_dtype)

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


def create_default_index_implicit(
    dim_variable: "Variable",
    all_variables: Optional[Union[Mapping, Iterable[Hashable]]] = None,
) -> Tuple[Index, IndexVars]:
    """Create a default index from a dimension variable.

    Create a PandasMultiIndex if the given variable wraps a pandas.MultiIndex,
    otherwise create a PandasIndex.

    This function will become obsolete once we depreciate
    implcitly passing a pandas.MultiIndex as a coordinate.

    """
    if all_variables is None:
        all_variables = {}

    name = dim_variable.dims[0]
    array = getattr(dim_variable._data, "array", None)
    index: PandasIndex

    if isinstance(array, pd.MultiIndex):
        index, index_vars = PandasMultiIndex.from_pandas_index(array, name)
        # check for conflict between level names and variable names
        duplicate_names = [k for k in index_vars if k in all_variables and k != name]
        if duplicate_names:
            conflict_str = "\n".join(duplicate_names)
            raise ValueError(
                f"conflicting MultiIndex level / variable name(s):\n{conflict_str}"
            )
    else:
        index, index_vars = PandasIndex.from_variables({name: dim_variable})

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


# generic type that represents either a pandas or an xarray index
T_PandasOrXarrayIndex = TypeVar("T_PandasOrXarrayIndex")


class Indexes(collections.abc.Mapping, Generic[T_PandasOrXarrayIndex]):
    """Immutable proxy for Dataset or DataArrary indexes.

    Keys are coordinate names and values may correspond to either pandas or
    xarray indexes.

    Also provides some utility methods.

    """

    _indexes: Dict[Any, T_PandasOrXarrayIndex]
    _variables: Dict[Any, "Variable"]

    __slots__ = (
        "_indexes",
        "_variables",
        "_dims",
        "__coord_name_id",
        "__id_index",
        "__id_coord_names",
    )

    def __init__(
        self,
        indexes: Dict[Any, T_PandasOrXarrayIndex],
        variables: Dict[Any, "Variable"],
    ):
        """Constructor not for public consumption.

        Parameters
        ----------
        indexes : dict
            Indexes held by this object.
        variables : dict
            Indexed coordinate variables in this object.

        """
        self._indexes = indexes
        self._variables = variables

        self._dims: Optional[Mapping[Hashable, int]] = None
        self.__coord_name_id: Optional[Dict[Any, int]] = None
        self.__id_index: Optional[Dict[int, T_PandasOrXarrayIndex]] = None
        self.__id_coord_names: Optional[Dict[int, Tuple[Hashable, ...]]] = None

    @property
    def _coord_name_id(self) -> Dict[Any, int]:
        if self.__coord_name_id is None:
            self.__coord_name_id = {k: id(idx) for k, idx in self._indexes.items()}
        return self.__coord_name_id

    @property
    def _id_index(self) -> Dict[int, T_PandasOrXarrayIndex]:
        if self.__id_index is None:
            self.__id_index = {id(idx): idx for idx in self.get_unique()}
        return self.__id_index

    @property
    def _id_coord_names(self) -> Dict[int, Tuple[Hashable, ...]]:
        if self.__id_coord_names is None:
            id_coord_names: Mapping[int, List[Hashable]] = defaultdict(list)
            for k, v in self._coord_name_id.items():
                id_coord_names[v].append(k)
            self.__id_coord_names = {k: tuple(v) for k, v in id_coord_names.items()}

        return self.__id_coord_names

    @property
    def variables(self) -> Mapping[Hashable, "Variable"]:
        return Frozen(self._variables)

    @property
    def dims(self) -> Mapping[Hashable, int]:
        from .variable import calculate_dimensions

        if self._dims is None:
            self._dims = calculate_dimensions(self._variables)

        return Frozen(self._dims)

    def get_unique(self) -> List[T_PandasOrXarrayIndex]:
        """Return a list of unique indexes, preserving order."""

        unique_indexes: List[T_PandasOrXarrayIndex] = []
        seen: Set[T_PandasOrXarrayIndex] = set()

        for index in self._indexes.values():
            if index not in seen:
                unique_indexes.append(index)
                seen.add(index)

        return unique_indexes

    def get_all_coords(
        self, coord_name: Hashable, errors: str = "raise"
    ) -> Dict[Hashable, "Variable"]:
        """Return all coordinates having the same index.

        Parameters
        ----------
        coord_name : hashable
            Name of an indexed coordinate.
        errors : {"raise", "ignore"}, optional
            If "raise", raises a ValueError if `coord_name` is not in indexes.
            If "ignore", an empty tuple is returned instead.

        Returns
        -------
        coords : dict
            A dictionary of all coordinate variables having the same index.

        """
        if errors not in ["raise", "ignore"]:
            raise ValueError('errors must be either "raise" or "ignore"')

        if coord_name not in self._indexes:
            if errors == "raise":
                raise ValueError(f"no index found for {coord_name!r} coordinate")
            else:
                return {}

        all_coord_names = self._id_coord_names[self._coord_name_id[coord_name]]
        return {k: self._variables[k] for k in all_coord_names}

    def group_by_index(
        self,
    ) -> List[Tuple[T_PandasOrXarrayIndex, Dict[Hashable, "Variable"]]]:
        """Returns a list of unique indexes and their corresponding coordinates."""

        index_coords = []

        for i in self._id_index:
            index = self._id_index[i]
            coords = {k: self._variables[k] for k in self._id_coord_names[i]}
            index_coords.append((index, coords))

        return index_coords

    def to_pandas_indexes(self) -> "Indexes[pd.Index]":
        """Returns an immutable proxy for Dataset or DataArrary pandas indexes.

        Raises an error if this proxy contains indexes that cannot be coerced to
        pandas.Index objects.

        """
        indexes: Dict[Hashable, pd.Index] = {}

        for k, idx in self._indexes.items():
            if isinstance(idx, pd.Index):
                indexes[k] = idx
            elif isinstance(idx, Index):
                indexes[k] = idx.to_pandas_index()

        return Indexes(indexes, self._variables)

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


def indexes_equal(elements: Sequence[Tuple[Index, Dict[Hashable, "Variable"]]]) -> bool:
    """Check if indexes are all equal.

    If they are not of the same type or they do not implement this check, check
    if their coordinate variables are all equal instead.

    """

    def check_variables():
        variables = [e[1] for e in elements]
        return any(
            not variables[0][k].equals(other_vars[k])
            for other_vars in variables[1:]
            for k in variables[0]
        )

    indexes = [e[0] for e in elements]
    same_type = all(type(indexes[0]) is type(other_idx) for other_idx in indexes[1:])
    if same_type:
        try:
            not_equal = any(
                not indexes[0].equals(other_idx) for other_idx in indexes[1:]
            )
        except NotImplementedError:
            not_equal = check_variables()
    else:
        not_equal = check_variables()

    return not not_equal
