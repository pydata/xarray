from __future__ import annotations

import collections.abc
import copy
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    TypeVar,
    cast,
)

import numpy as np
import pandas as pd

from xarray.core import formatting, nputils, utils
from xarray.core.indexing import (
    IndexSelResult,
    PandasIndexingAdapter,
    PandasMultiIndexingAdapter,
)
from xarray.core.utils import Frozen, get_valid_numpy_dtype, is_dict_like, is_scalar

if TYPE_CHECKING:
    from xarray.core.types import ErrorOptions, T_Index
    from xarray.core.variable import Variable

IndexVars = Dict[Any, "Variable"]


class Index:
    """Base class inherited by all xarray-compatible indexes.

    Do not use this class directly for creating index objects.

    """

    @classmethod
    def from_variables(
        cls,
        variables: Mapping[Any, Variable],
        *,
        options: Mapping[str, Any],
    ) -> Index:
        raise NotImplementedError()

    @classmethod
    def concat(
        cls: type[T_Index],
        indexes: Sequence[T_Index],
        dim: Hashable,
        positions: Iterable[Iterable[int]] | None = None,
    ) -> T_Index:
        raise NotImplementedError()

    @classmethod
    def stack(cls, variables: Mapping[Any, Variable], dim: Hashable) -> Index:
        raise NotImplementedError(
            f"{cls!r} cannot be used for creating an index of stacked coordinates"
        )

    def unstack(self) -> tuple[dict[Hashable, Index], pd.MultiIndex]:
        raise NotImplementedError()

    def create_variables(
        self, variables: Mapping[Any, Variable] | None = None
    ) -> IndexVars:
        if variables is not None:
            # pass through
            return dict(**variables)
        else:
            return {}

    def to_pandas_index(self) -> pd.Index:
        """Cast this xarray index to a pandas.Index object or raise a TypeError
        if this is not supported.

        This method is used by all xarray operations that expect/require a
        pandas.Index object.

        """
        raise TypeError(f"{self!r} cannot be cast to a pandas.Index object")

    def isel(
        self, indexers: Mapping[Any, int | slice | np.ndarray | Variable]
    ) -> Index | None:
        return None

    def sel(self, labels: dict[Any, Any]) -> IndexSelResult:
        raise NotImplementedError(f"{self!r} doesn't support label-based selection")

    def join(self: T_Index, other: T_Index, how: str = "inner") -> T_Index:
        raise NotImplementedError(
            f"{self!r} doesn't support alignment with inner/outer join method"
        )

    def reindex_like(self: T_Index, other: T_Index) -> dict[Hashable, Any]:
        raise NotImplementedError(f"{self!r} doesn't support re-indexing labels")

    def equals(self, other):  # pragma: no cover
        raise NotImplementedError()

    def roll(self, shifts: Mapping[Any, int]) -> Index | None:
        return None

    def rename(
        self, name_dict: Mapping[Any, Hashable], dims_dict: Mapping[Any, Hashable]
    ) -> Index:
        return self

    def __copy__(self) -> Index:
        return self._copy(deep=False)

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Index:
        return self._copy(deep=True, memo=memo)

    def copy(self: T_Index, deep: bool = True) -> T_Index:
        return self._copy(deep=deep)

    def _copy(
        self: T_Index, deep: bool = True, memo: dict[int, Any] | None = None
    ) -> T_Index:
        cls = self.__class__
        copied = cls.__new__(cls)
        if deep:
            for k, v in self.__dict__.items():
                setattr(copied, k, copy.deepcopy(v, memo))
        else:
            copied.__dict__.update(self.__dict__)
        return copied

    def __getitem__(self, indexer: Any):
        raise NotImplementedError()

    def _repr_inline_(self, max_width):
        return self.__class__.__name__


def _maybe_cast_to_cftimeindex(index: pd.Index) -> pd.Index:
    from xarray.coding.cftimeindex import CFTimeIndex

    if len(index) > 0 and index.dtype == "O":
        try:
            return CFTimeIndex(index)
        except (ImportError, TypeError):
            return index
    else:
        return index


def safe_cast_to_index(array: Any) -> pd.Index:
    """Given an array, safely cast it to a pandas.Index.

    If it is already a pandas.Index, return it unchanged.

    Unlike pandas.Index, if the array has dtype=object or dtype=timedelta64,
    this function will not attempt to do automatic type conversion but will
    always return an index with dtype=object.
    """
    from xarray.core.dataarray import DataArray
    from xarray.core.variable import Variable

    if isinstance(array, pd.Index):
        index = array
    elif isinstance(array, (DataArray, Variable)):
        # returns the original multi-index for pandas.MultiIndex level coordinates
        index = array._to_index()
    elif isinstance(array, Index):
        index = array.to_pandas_index()
    elif isinstance(array, PandasIndexingAdapter):
        index = array.array
    else:
        kwargs = {}
        if hasattr(array, "dtype") and array.dtype.kind == "O":
            kwargs["dtype"] = object
        index = pd.Index(np.asarray(array), **kwargs)

    return _maybe_cast_to_cftimeindex(index)


def _sanitize_slice_element(x):
    from xarray.core.dataarray import DataArray
    from xarray.core.variable import Variable

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


T_PandasIndex = TypeVar("T_PandasIndex", bound="PandasIndex")


class PandasIndex(Index):
    """Wrap a pandas.Index as an xarray compatible index."""

    index: pd.Index
    dim: Hashable
    coord_dtype: Any

    __slots__ = ("index", "dim", "coord_dtype")

    def __init__(self, array: Any, dim: Hashable, coord_dtype: Any = None):
        # make a shallow copy: cheap and because the index name may be updated
        # here or in other constructors (cannot use pd.Index.rename as this
        # constructor is also called from PandasMultiIndex)
        index = safe_cast_to_index(array).copy()

        if index.name is None:
            index.name = dim

        self.index = index
        self.dim = dim

        if coord_dtype is None:
            coord_dtype = get_valid_numpy_dtype(index)
        self.coord_dtype = coord_dtype

    def _replace(self, index, dim=None, coord_dtype=None):
        if dim is None:
            dim = self.dim
        if coord_dtype is None:
            coord_dtype = self.coord_dtype
        return type(self)(index, dim, coord_dtype)

    @classmethod
    def from_variables(
        cls,
        variables: Mapping[Any, Variable],
        *,
        options: Mapping[str, Any],
    ) -> PandasIndex:
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

        # TODO: (benbovy - explicit indexes): add __index__ to ExplicitlyIndexesNDArrayMixin?
        # this could be eventually used by Variable.to_index() and would remove the need to perform
        # the checks below.

        # preserve wrapped pd.Index (if any)
        data = getattr(var._data, "array", var.data)
        # multi-index level variable: get level index
        if isinstance(var._data, PandasMultiIndexingAdapter):
            level = var._data.level
            if level is not None:
                data = var._data.array.get_level_values(level)

        obj = cls(data, dim, coord_dtype=var.dtype)
        assert not isinstance(obj.index, pd.MultiIndex)
        obj.index.name = name

        return obj

    @staticmethod
    def _concat_indexes(indexes, dim, positions=None) -> pd.Index:
        new_pd_index: pd.Index

        if not indexes:
            new_pd_index = pd.Index([])
        else:
            if not all(idx.dim == dim for idx in indexes):
                dims = ",".join({f"{idx.dim!r}" for idx in indexes})
                raise ValueError(
                    f"Cannot concatenate along dimension {dim!r} indexes with "
                    f"dimensions: {dims}"
                )
            pd_indexes = [idx.index for idx in indexes]
            new_pd_index = pd_indexes[0].append(pd_indexes[1:])

            if positions is not None:
                indices = nputils.inverse_permutation(np.concatenate(positions))
                new_pd_index = new_pd_index.take(indices)

        return new_pd_index

    @classmethod
    def concat(
        cls,
        indexes: Sequence[PandasIndex],
        dim: Hashable,
        positions: Iterable[Iterable[int]] | None = None,
    ) -> PandasIndex:
        new_pd_index = cls._concat_indexes(indexes, dim, positions)

        if not indexes:
            coord_dtype = None
        else:
            coord_dtype = np.result_type(*[idx.coord_dtype for idx in indexes])

        return cls(new_pd_index, dim=dim, coord_dtype=coord_dtype)

    def create_variables(
        self, variables: Mapping[Any, Variable] | None = None
    ) -> IndexVars:
        from xarray.core.variable import IndexVariable

        name = self.index.name
        attrs: Mapping[Hashable, Any] | None
        encoding: Mapping[Hashable, Any] | None

        if variables is not None and name in variables:
            var = variables[name]
            attrs = var.attrs
            encoding = var.encoding
        else:
            attrs = None
            encoding = None

        data = PandasIndexingAdapter(self.index, dtype=self.coord_dtype)
        var = IndexVariable(self.dim, data, attrs=attrs, encoding=encoding)
        return {name: var}

    def to_pandas_index(self) -> pd.Index:
        return self.index

    def isel(
        self, indexers: Mapping[Any, int | slice | np.ndarray | Variable]
    ) -> PandasIndex | None:
        from xarray.core.variable import Variable

        indxr = indexers[self.dim]
        if isinstance(indxr, Variable):
            if indxr.dims != (self.dim,):
                # can't preserve a index if result has new dimensions
                return None
            else:
                indxr = indxr.data
        if not isinstance(indxr, slice) and is_scalar(indxr):
            # scalar indexer: drop index
            return None

        return self._replace(self.index[indxr])

    def sel(
        self, labels: dict[Any, Any], method=None, tolerance=None
    ) -> IndexSelResult:
        from xarray.core.dataarray import DataArray
        from xarray.core.variable import Variable

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
                    if method is not None:
                        indexer = get_indexer_nd(
                            self.index, label_array, method, tolerance
                        )
                        if np.any(indexer < 0):
                            raise KeyError(
                                f"not all values found in index {coord_name!r}"
                            )
                    else:
                        try:
                            indexer = self.index.get_loc(label_value)
                        except KeyError as e:
                            raise KeyError(
                                f"not all values found in index {coord_name!r}. "
                                "Try setting the `method` keyword argument (example: method='nearest')."
                            ) from e

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

        return IndexSelResult({self.dim: indexer})

    def equals(self, other: Index):
        if not isinstance(other, PandasIndex):
            return False
        return self.index.equals(other.index) and self.dim == other.dim

    def join(self: PandasIndex, other: PandasIndex, how: str = "inner") -> PandasIndex:
        if how == "outer":
            index = self.index.union(other.index)
        else:
            # how = "inner"
            index = self.index.intersection(other.index)

        coord_dtype = np.result_type(self.coord_dtype, other.coord_dtype)
        return type(self)(index, self.dim, coord_dtype=coord_dtype)

    def reindex_like(
        self, other: PandasIndex, method=None, tolerance=None
    ) -> dict[Hashable, Any]:
        if not self.index.is_unique:
            raise ValueError(
                f"cannot reindex or align along dimension {self.dim!r} because the "
                "(pandas) index has duplicate values"
            )

        return {self.dim: get_indexer_nd(self.index, other.index, method, tolerance)}

    def roll(self, shifts: Mapping[Any, int]) -> PandasIndex:
        shift = shifts[self.dim] % self.index.shape[0]

        if shift != 0:
            new_pd_idx = self.index[-shift:].append(self.index[:-shift])
        else:
            new_pd_idx = self.index[:]

        return self._replace(new_pd_idx)

    def rename(self, name_dict, dims_dict):
        if self.index.name not in name_dict and self.dim not in dims_dict:
            return self

        new_name = name_dict.get(self.index.name, self.index.name)
        index = self.index.rename(new_name)
        new_dim = dims_dict.get(self.dim, self.dim)
        return self._replace(index, dim=new_dim)

    def _copy(
        self: T_PandasIndex, deep: bool = True, memo: dict[int, Any] | None = None
    ) -> T_PandasIndex:
        if deep:
            # pandas is not using the memo
            index = self.index.copy(deep=True)
        else:
            # index will be copied in constructor
            index = self.index
        return self._replace(index)

    def __getitem__(self, indexer: Any):
        return self._replace(self.index[indexer])

    def __repr__(self):
        return f"PandasIndex({repr(self.index)})"


def _check_dim_compat(variables: Mapping[Any, Variable], all_dims: str = "equal"):
    """Check that all multi-index variable candidates are 1-dimensional and
    either share the same (single) dimension or each have a different dimension.

    """
    if any([var.ndim != 1 for var in variables.values()]):
        raise ValueError("PandasMultiIndex only accepts 1-dimensional variables")

    dims = {var.dims for var in variables.values()}

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


class PandasMultiIndex(PandasIndex):
    """Wrap a pandas.MultiIndex as an xarray compatible index."""

    level_coords_dtype: dict[str, Any]

    __slots__ = ("index", "dim", "coord_dtype", "level_coords_dtype")

    def __init__(self, array: Any, dim: Hashable, level_coords_dtype: Any = None):
        super().__init__(array, dim)

        # default index level names
        names = []
        for i, idx in enumerate(self.index.levels):
            name = idx.name or f"{dim}_level_{i}"
            if name == dim:
                raise ValueError(
                    f"conflicting multi-index level name {name!r} with dimension {dim!r}"
                )
            names.append(name)
        self.index.names = names

        if level_coords_dtype is None:
            level_coords_dtype = {
                idx.name: get_valid_numpy_dtype(idx) for idx in self.index.levels
            }
        self.level_coords_dtype = level_coords_dtype

    def _replace(self, index, dim=None, level_coords_dtype=None) -> PandasMultiIndex:
        if dim is None:
            dim = self.dim
        index.name = dim
        if level_coords_dtype is None:
            level_coords_dtype = self.level_coords_dtype
        return type(self)(index, dim, level_coords_dtype)

    @classmethod
    def from_variables(
        cls,
        variables: Mapping[Any, Variable],
        *,
        options: Mapping[str, Any],
    ) -> PandasMultiIndex:
        _check_dim_compat(variables)
        dim = next(iter(variables.values())).dims[0]

        index = pd.MultiIndex.from_arrays(
            [var.values for var in variables.values()], names=variables.keys()
        )
        index.name = dim
        level_coords_dtype = {name: var.dtype for name, var in variables.items()}
        obj = cls(index, dim, level_coords_dtype=level_coords_dtype)

        return obj

    @classmethod
    def concat(  # type: ignore[override]
        cls,
        indexes: Sequence[PandasMultiIndex],
        dim: Hashable,
        positions: Iterable[Iterable[int]] | None = None,
    ) -> PandasMultiIndex:
        new_pd_index = cls._concat_indexes(indexes, dim, positions)

        if not indexes:
            level_coords_dtype = None
        else:
            level_coords_dtype = {}
            for name in indexes[0].level_coords_dtype:
                level_coords_dtype[name] = np.result_type(
                    *[idx.level_coords_dtype[name] for idx in indexes]
                )

        return cls(new_pd_index, dim=dim, level_coords_dtype=level_coords_dtype)

    @classmethod
    def stack(
        cls, variables: Mapping[Any, Variable], dim: Hashable
    ) -> PandasMultiIndex:
        """Create a new Pandas MultiIndex from the product of 1-d variables (levels) along a
        new dimension.

        Level variables must have a dimension distinct from each other.

        Keeps levels the same (doesn't refactorize them) so that it gives back the original
        labels after a stack/unstack roundtrip.

        """
        _check_dim_compat(variables, all_dims="different")

        level_indexes = [safe_cast_to_index(var) for var in variables.values()]
        for name, idx in zip(variables, level_indexes):
            if isinstance(idx, pd.MultiIndex):
                raise ValueError(
                    f"cannot create a multi-index along stacked dimension {dim!r} "
                    f"from variable {name!r} that wraps a multi-index"
                )

        split_labels, levels = zip(*[lev.factorize() for lev in level_indexes])
        labels_mesh = np.meshgrid(*split_labels, indexing="ij")
        labels = [x.ravel() for x in labels_mesh]

        index = pd.MultiIndex(levels, labels, sortorder=0, names=variables.keys())
        level_coords_dtype = {k: var.dtype for k, var in variables.items()}

        return cls(index, dim, level_coords_dtype=level_coords_dtype)

    def unstack(self) -> tuple[dict[Hashable, Index], pd.MultiIndex]:
        clean_index = remove_unused_levels_categories(self.index)

        new_indexes: dict[Hashable, Index] = {}
        for name, lev in zip(clean_index.names, clean_index.levels):
            idx = PandasIndex(
                lev.copy(), name, coord_dtype=self.level_coords_dtype[name]
            )
            new_indexes[name] = idx

        return new_indexes, clean_index

    @classmethod
    def from_variables_maybe_expand(
        cls,
        dim: Hashable,
        current_variables: Mapping[Any, Variable],
        variables: Mapping[Any, Variable],
    ) -> tuple[PandasMultiIndex, IndexVars]:
        """Create a new multi-index maybe by expanding an existing one with
        new variables as index levels.

        The index and its corresponding coordinates may be created along a new dimension.
        """
        names: list[Hashable] = []
        codes: list[list[int]] = []
        levels: list[list[int]] = []
        level_variables: dict[Any, Variable] = {}

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
        level_coords_dtype = {k: var.dtype for k, var in level_variables.items()}
        obj = cls(index, dim, level_coords_dtype=level_coords_dtype)
        index_vars = obj.create_variables(level_variables)

        return obj, index_vars

    def keep_levels(
        self, level_variables: Mapping[Any, Variable]
    ) -> PandasMultiIndex | PandasIndex:
        """Keep only the provided levels and return a new multi-index with its
        corresponding coordinates.

        """
        index = self.index.droplevel(
            [k for k in self.index.names if k not in level_variables]
        )

        if isinstance(index, pd.MultiIndex):
            level_coords_dtype = {k: self.level_coords_dtype[k] for k in index.names}
            return self._replace(index, level_coords_dtype=level_coords_dtype)
        else:
            # backward compatibility: rename the level coordinate to the dimension name
            return PandasIndex(
                index.rename(self.dim),
                self.dim,
                coord_dtype=self.level_coords_dtype[index.name],
            )

    def reorder_levels(
        self, level_variables: Mapping[Any, Variable]
    ) -> PandasMultiIndex:
        """Re-arrange index levels using input order and return a new multi-index with
        its corresponding coordinates.

        """
        index = self.index.reorder_levels(level_variables.keys())
        level_coords_dtype = {k: self.level_coords_dtype[k] for k in index.names}
        return self._replace(index, level_coords_dtype=level_coords_dtype)

    def create_variables(
        self, variables: Mapping[Any, Variable] | None = None
    ) -> IndexVars:
        from xarray.core.variable import IndexVariable

        if variables is None:
            variables = {}

        index_vars: IndexVars = {}
        for name in (self.dim,) + self.index.names:
            if name == self.dim:
                level = None
                dtype = None
            else:
                level = name
                dtype = self.level_coords_dtype[name]

            var = variables.get(name, None)
            if var is not None:
                attrs = var.attrs
                encoding = var.encoding
            else:
                attrs = {}
                encoding = {}

            data = PandasMultiIndexingAdapter(self.index, dtype=dtype, level=level)
            index_vars[name] = IndexVariable(
                self.dim,
                data,
                attrs=attrs,
                encoding=encoding,
                fastpath=True,
            )

        return index_vars

    def sel(self, labels, method=None, tolerance=None) -> IndexSelResult:
        from xarray.core.dataarray import DataArray
        from xarray.core.variable import Variable

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
                return self.sel(label)

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
            if isinstance(new_index, pd.MultiIndex):
                level_coords_dtype = {
                    k: self.level_coords_dtype[k] for k in new_index.names
                }
                new_index = self._replace(
                    new_index, level_coords_dtype=level_coords_dtype
                )
                dims_dict = {}
                drop_coords = []
            else:
                new_index = PandasIndex(
                    new_index,
                    new_index.name,
                    coord_dtype=self.level_coords_dtype[new_index.name],
                )
                dims_dict = {self.dim: new_index.index.name}
                drop_coords = [self.dim]

            # variable(s) attrs and encoding metadata are propagated
            # when replacing the indexes in the resulting xarray object
            new_vars = new_index.create_variables()
            indexes = cast(Dict[Any, Index], {k: new_index for k in new_vars})

            # add scalar variable for each dropped level
            variables = new_vars
            for name, val in scalar_coord_values.items():
                variables[name] = Variable([], val)

            return IndexSelResult(
                {self.dim: indexer},
                indexes=indexes,
                variables=variables,
                drop_indexes=list(scalar_coord_values),
                drop_coords=drop_coords,
                rename_dims=dims_dict,
            )

        else:
            return IndexSelResult({self.dim: indexer})

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
            return self

        # pandas 1.3.0: could simply do `self.index.rename(names_dict)`
        new_names = [name_dict.get(k, k) for k in self.index.names]
        index = self.index.rename(new_names)

        new_dim = dims_dict.get(self.dim, self.dim)
        new_level_coords_dtype = {
            k: v for k, v in zip(new_names, self.level_coords_dtype.values())
        }
        return self._replace(
            index, dim=new_dim, level_coords_dtype=new_level_coords_dtype
        )


def create_default_index_implicit(
    dim_variable: Variable,
    all_variables: Mapping | Iterable[Hashable] | None = None,
) -> tuple[PandasIndex, IndexVars]:
    """Create a default index from a dimension variable.

    Create a PandasMultiIndex if the given variable wraps a pandas.MultiIndex,
    otherwise create a PandasIndex (note that this will become obsolete once we
    depreciate implicitly passing a pandas.MultiIndex as a coordinate).

    """
    if all_variables is None:
        all_variables = {}
    if not isinstance(all_variables, Mapping):
        all_variables = {k: None for k in all_variables}

    name = dim_variable.dims[0]
    array = getattr(dim_variable._data, "array", None)
    index: PandasIndex

    if isinstance(array, pd.MultiIndex):
        index = PandasMultiIndex(array, name)
        index_vars = index.create_variables()
        # check for conflict between level names and variable names
        duplicate_names = [k for k in index_vars if k in all_variables and k != name]
        if duplicate_names:
            # dirty workaround for an edge case where both the dimension
            # coordinate and the level coordinates are given for the same
            # multi-index object => do not raise an error
            # TODO: remove this check when removing the multi-index dimension coordinate
            if len(duplicate_names) < len(index.index.names):
                conflict = True
            else:
                duplicate_vars = [all_variables[k] for k in duplicate_names]
                conflict = any(
                    v is None or not dim_variable.equals(v) for v in duplicate_vars
                )

            if conflict:
                conflict_str = "\n".join(duplicate_names)
                raise ValueError(
                    f"conflicting MultiIndex level / variable name(s):\n{conflict_str}"
                )
    else:
        dim_var = {name: dim_variable}
        index = PandasIndex.from_variables(dim_var, options={})
        index_vars = index.create_variables(dim_var)

    return index, index_vars


# generic type that represents either a pandas or an xarray index
T_PandasOrXarrayIndex = TypeVar("T_PandasOrXarrayIndex", Index, pd.Index)


class Indexes(collections.abc.Mapping, Generic[T_PandasOrXarrayIndex]):
    """Immutable proxy for Dataset or DataArrary indexes.

    Keys are coordinate names and values may correspond to either pandas or
    xarray indexes.

    Also provides some utility methods.

    """

    _indexes: dict[Any, T_PandasOrXarrayIndex]
    _variables: dict[Any, Variable]

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
        indexes: dict[Any, T_PandasOrXarrayIndex],
        variables: dict[Any, Variable],
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

        self._dims: Mapping[Hashable, int] | None = None
        self.__coord_name_id: dict[Any, int] | None = None
        self.__id_index: dict[int, T_PandasOrXarrayIndex] | None = None
        self.__id_coord_names: dict[int, tuple[Hashable, ...]] | None = None

    @property
    def _coord_name_id(self) -> dict[Any, int]:
        if self.__coord_name_id is None:
            self.__coord_name_id = {k: id(idx) for k, idx in self._indexes.items()}
        return self.__coord_name_id

    @property
    def _id_index(self) -> dict[int, T_PandasOrXarrayIndex]:
        if self.__id_index is None:
            self.__id_index = {id(idx): idx for idx in self.get_unique()}
        return self.__id_index

    @property
    def _id_coord_names(self) -> dict[int, tuple[Hashable, ...]]:
        if self.__id_coord_names is None:
            id_coord_names: Mapping[int, list[Hashable]] = defaultdict(list)
            for k, v in self._coord_name_id.items():
                id_coord_names[v].append(k)
            self.__id_coord_names = {k: tuple(v) for k, v in id_coord_names.items()}

        return self.__id_coord_names

    @property
    def variables(self) -> Mapping[Hashable, Variable]:
        return Frozen(self._variables)

    @property
    def dims(self) -> Mapping[Hashable, int]:
        from xarray.core.variable import calculate_dimensions

        if self._dims is None:
            self._dims = calculate_dimensions(self._variables)

        return Frozen(self._dims)

    def copy(self) -> Indexes:
        return type(self)(dict(self._indexes), dict(self._variables))

    def get_unique(self) -> list[T_PandasOrXarrayIndex]:
        """Return a list of unique indexes, preserving order."""

        unique_indexes: list[T_PandasOrXarrayIndex] = []
        seen: set[int] = set()

        for index in self._indexes.values():
            index_id = id(index)
            if index_id not in seen:
                unique_indexes.append(index)
                seen.add(index_id)

        return unique_indexes

    def is_multi(self, key: Hashable) -> bool:
        """Return True if ``key`` maps to a multi-coordinate index,
        False otherwise.
        """
        return len(self._id_coord_names[self._coord_name_id[key]]) > 1

    def get_all_coords(
        self, key: Hashable, errors: ErrorOptions = "raise"
    ) -> dict[Hashable, Variable]:
        """Return all coordinates having the same index.

        Parameters
        ----------
        key : hashable
            Index key.
        errors : {"raise", "ignore"}, default: "raise"
            If "raise", raises a ValueError if `key` is not in indexes.
            If "ignore", an empty tuple is returned instead.

        Returns
        -------
        coords : dict
            A dictionary of all coordinate variables having the same index.

        """
        if errors not in ["raise", "ignore"]:
            raise ValueError('errors must be either "raise" or "ignore"')

        if key not in self._indexes:
            if errors == "raise":
                raise ValueError(f"no index found for {key!r} coordinate")
            else:
                return {}

        all_coord_names = self._id_coord_names[self._coord_name_id[key]]
        return {k: self._variables[k] for k in all_coord_names}

    def get_all_dims(
        self, key: Hashable, errors: ErrorOptions = "raise"
    ) -> Mapping[Hashable, int]:
        """Return all dimensions shared by an index.

        Parameters
        ----------
        key : hashable
            Index key.
        errors : {"raise", "ignore"}, default: "raise"
            If "raise", raises a ValueError if `key` is not in indexes.
            If "ignore", an empty tuple is returned instead.

        Returns
        -------
        dims : dict
            A dictionary of all dimensions shared by an index.

        """
        from xarray.core.variable import calculate_dimensions

        return calculate_dimensions(self.get_all_coords(key, errors=errors))

    def group_by_index(
        self,
    ) -> list[tuple[T_PandasOrXarrayIndex, dict[Hashable, Variable]]]:
        """Returns a list of unique indexes and their corresponding coordinates."""

        index_coords = []

        for i in self._id_index:
            index = self._id_index[i]
            coords = {k: self._variables[k] for k in self._id_coord_names[i]}
            index_coords.append((index, coords))

        return index_coords

    def to_pandas_indexes(self) -> Indexes[pd.Index]:
        """Returns an immutable proxy for Dataset or DataArrary pandas indexes.

        Raises an error if this proxy contains indexes that cannot be coerced to
        pandas.Index objects.

        """
        indexes: dict[Hashable, pd.Index] = {}

        for k, idx in self._indexes.items():
            if isinstance(idx, pd.Index):
                indexes[k] = idx
            elif isinstance(idx, Index):
                indexes[k] = idx.to_pandas_index()

        return Indexes(indexes, self._variables)

    def copy_indexes(
        self, deep: bool = True, memo: dict[int, Any] | None = None
    ) -> tuple[dict[Hashable, T_PandasOrXarrayIndex], dict[Hashable, Variable]]:
        """Return a new dictionary with copies of indexes, preserving
        unique indexes.

        Parameters
        ----------
        deep : bool, default: True
            Whether the indexes are deep or shallow copied onto the new object.
        memo : dict if object id to copied objects or None, optional
            To prevent infinite recursion deepcopy stores all copied elements
            in this dict.

        """
        new_indexes = {}
        new_index_vars = {}

        for idx, coords in self.group_by_index():
            if isinstance(idx, pd.Index):
                convert_new_idx = True
                dim = next(iter(coords.values())).dims[0]
                if isinstance(idx, pd.MultiIndex):
                    idx = PandasMultiIndex(idx, dim)
                else:
                    idx = PandasIndex(idx, dim)
            else:
                convert_new_idx = False

            new_idx = idx._copy(deep=deep, memo=memo)
            idx_vars = idx.create_variables(coords)

            if convert_new_idx:
                new_idx = cast(PandasIndex, new_idx).index

            new_indexes.update({k: new_idx for k in coords})
            new_index_vars.update(idx_vars)

        return new_indexes, new_index_vars

    def __iter__(self) -> Iterator[T_PandasOrXarrayIndex]:
        return iter(self._indexes)

    def __len__(self) -> int:
        return len(self._indexes)

    def __contains__(self, key) -> bool:
        return key in self._indexes

    def __getitem__(self, key) -> T_PandasOrXarrayIndex:
        return self._indexes[key]

    def __repr__(self):
        return formatting.indexes_repr(self)


def default_indexes(
    coords: Mapping[Any, Variable], dims: Iterable
) -> dict[Hashable, Index]:
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
    indexes: dict[Hashable, Index] = {}
    coord_names = set(coords)

    for name, var in coords.items():
        if name in dims:
            index, index_vars = create_default_index_implicit(var, coords)
            if set(index_vars) <= coord_names:
                indexes.update({k: index for k in index_vars})

    return indexes


def indexes_equal(
    index: Index,
    other_index: Index,
    variable: Variable,
    other_variable: Variable,
    cache: dict[tuple[int, int], bool | None] | None = None,
) -> bool:
    """Check if two indexes are equal, possibly with cached results.

    If the two indexes are not of the same type or they do not implement
    equality, fallback to coordinate labels equality check.

    """
    if cache is None:
        # dummy cache
        cache = {}

    key = (id(index), id(other_index))
    equal: bool | None = None

    if key not in cache:
        if type(index) is type(other_index):
            try:
                equal = index.equals(other_index)
            except NotImplementedError:
                equal = None
            else:
                cache[key] = equal
        else:
            equal = None
    else:
        equal = cache[key]

    if equal is None:
        equal = variable.equals(other_variable)

    return cast(bool, equal)


def indexes_all_equal(
    elements: Sequence[tuple[Index, dict[Hashable, Variable]]]
) -> bool:
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

    same_objects = all(indexes[0] is other_idx for other_idx in indexes[1:])
    if same_objects:
        return True

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


def _apply_indexes(
    indexes: Indexes[Index],
    args: Mapping[Any, Any],
    func: str,
) -> tuple[dict[Hashable, Index], dict[Hashable, Variable]]:
    new_indexes: dict[Hashable, Index] = {k: v for k, v in indexes.items()}
    new_index_variables: dict[Hashable, Variable] = {}

    for index, index_vars in indexes.group_by_index():
        index_dims = {d for var in index_vars.values() for d in var.dims}
        index_args = {k: v for k, v in args.items() if k in index_dims}
        if index_args:
            new_index = getattr(index, func)(index_args)
            if new_index is not None:
                new_indexes.update({k: new_index for k in index_vars})
                new_index_vars = new_index.create_variables(index_vars)
                new_index_variables.update(new_index_vars)
            else:
                for k in index_vars:
                    new_indexes.pop(k, None)

    return new_indexes, new_index_variables


def isel_indexes(
    indexes: Indexes[Index],
    indexers: Mapping[Any, Any],
) -> tuple[dict[Hashable, Index], dict[Hashable, Variable]]:
    return _apply_indexes(indexes, indexers, "isel")


def roll_indexes(
    indexes: Indexes[Index],
    shifts: Mapping[Any, int],
) -> tuple[dict[Hashable, Index], dict[Hashable, Variable]]:
    return _apply_indexes(indexes, shifts, "roll")


def filter_indexes_from_coords(
    indexes: Mapping[Any, Index],
    filtered_coord_names: set,
) -> dict[Hashable, Index]:
    """Filter index items given a (sub)set of coordinate names.

    Drop all multi-coordinate related index items for any key missing in the set
    of coordinate names.

    """
    filtered_indexes: dict[Any, Index] = dict(**indexes)

    index_coord_names: dict[Hashable, set[Hashable]] = defaultdict(set)
    for name, idx in indexes.items():
        index_coord_names[id(idx)].add(name)

    for idx_coord_names in index_coord_names.values():
        if not idx_coord_names <= filtered_coord_names:
            for k in idx_coord_names:
                del filtered_indexes[k]

    return filtered_indexes


def assert_no_index_corrupted(
    indexes: Indexes[Index],
    coord_names: set[Hashable],
    action: str = "remove coordinate(s)",
) -> None:
    """Assert removing coordinates or indexes will not corrupt indexes."""

    # An index may be corrupted when the set of its corresponding coordinate name(s)
    # partially overlaps the set of coordinate names to remove
    for index, index_coords in indexes.group_by_index():
        common_names = set(index_coords) & coord_names
        if common_names and len(common_names) != len(index_coords):
            common_names_str = ", ".join(f"{k!r}" for k in common_names)
            index_names_str = ", ".join(f"{k!r}" for k in index_coords)
            raise ValueError(
                f"cannot {action} {common_names_str}, which would corrupt "
                f"the following index built from coordinates {index_names_str}:\n"
                f"{index}"
            )
