from __future__ import annotations

import datetime
import warnings
from collections.abc import Hashable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, TypeVar, Union, cast

import numpy as np
import pandas as pd

from xarray.core import dtypes, duck_array_ops, nputils, ops
from xarray.core._aggregations import (
    DataArrayGroupByAggregations,
    DatasetGroupByAggregations,
)
from xarray.core.alignment import align
from xarray.core.arithmetic import DataArrayGroupbyArithmetic, DatasetGroupbyArithmetic
from xarray.core.common import ImplementsArrayReduce, ImplementsDatasetReduce
from xarray.core.concat import concat
from xarray.core.formatting import format_array_flat
from xarray.core.indexes import (
    create_default_index_implicit,
    filter_indexes_from_coords,
    safe_cast_to_index,
)
from xarray.core.options import _get_keep_attrs
from xarray.core.pycompat import integer_types
from xarray.core.types import Dims, QuantileMethods, T_Xarray
from xarray.core.utils import (
    either_dict_or_kwargs,
    hashable,
    is_scalar,
    maybe_wrap_array,
    peek_at,
)
from xarray.core.variable import IndexVariable, Variable

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset
    from xarray.core.types import DatetimeLike, SideOptions
    from xarray.core.utils import Frozen

    GroupKey = Any

    T_GroupIndicesListInt = list[list[int]]
    T_GroupIndices = Union[T_GroupIndicesListInt, list[slice], np.ndarray]


def check_reduce_dims(reduce_dims, dimensions):
    if reduce_dims is not ...:
        if is_scalar(reduce_dims):
            reduce_dims = [reduce_dims]
        if any(dim not in dimensions for dim in reduce_dims):
            raise ValueError(
                f"cannot reduce over dimensions {reduce_dims!r}. expected either '...' "
                f"to reduce over all dimensions or one or more of {dimensions!r}."
            )


def unique_value_groups(
    ar, sort: bool = True
) -> tuple[np.ndarray | pd.Index, T_GroupIndices, np.ndarray]:
    """Group an array by its unique values.

    Parameters
    ----------
    ar : array-like
        Input array. This will be flattened if it is not already 1-D.
    sort : bool, default: True
        Whether or not to sort unique values.

    Returns
    -------
    values : np.ndarray
        Sorted, unique values as returned by `np.unique`.
    indices : list of lists of int
        Each element provides the integer indices in `ar` with values given by
        the corresponding value in `unique_values`.
    """
    inverse, values = pd.factorize(ar, sort=sort)
    if isinstance(values, pd.MultiIndex):
        values.names = ar.names
    groups = _codes_to_groups(inverse, len(values))
    return values, groups, inverse


def _codes_to_groups(inverse: np.ndarray, N: int) -> T_GroupIndicesListInt:
    groups: T_GroupIndicesListInt = [[] for _ in range(N)]
    for n, g in enumerate(inverse):
        if g >= 0:
            groups[g].append(n)
    return groups


def _dummy_copy(xarray_obj):
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset

    if isinstance(xarray_obj, Dataset):
        res = Dataset(
            {
                k: dtypes.get_fill_value(v.dtype)
                for k, v in xarray_obj.data_vars.items()
            },
            {
                k: dtypes.get_fill_value(v.dtype)
                for k, v in xarray_obj.coords.items()
                if k not in xarray_obj.dims
            },
            xarray_obj.attrs,
        )
    elif isinstance(xarray_obj, DataArray):
        res = DataArray(
            dtypes.get_fill_value(xarray_obj.dtype),
            {
                k: dtypes.get_fill_value(v.dtype)
                for k, v in xarray_obj.coords.items()
                if k not in xarray_obj.dims
            },
            dims=[],
            name=xarray_obj.name,
            attrs=xarray_obj.attrs,
        )
    else:  # pragma: no cover
        raise AssertionError
    return res


def _is_one_or_none(obj):
    return obj == 1 or obj is None


def _consolidate_slices(slices):
    """Consolidate adjacent slices in a list of slices."""
    result = []
    last_slice = slice(None)
    for slice_ in slices:
        if not isinstance(slice_, slice):
            raise ValueError(f"list element is not a slice: {slice_!r}")
        if (
            result
            and last_slice.stop == slice_.start
            and _is_one_or_none(last_slice.step)
            and _is_one_or_none(slice_.step)
        ):
            last_slice = slice(last_slice.start, slice_.stop, slice_.step)
            result[-1] = last_slice
        else:
            result.append(slice_)
            last_slice = slice_
    return result


def _inverse_permutation_indices(positions, N: int | None = None) -> np.ndarray | None:
    """Like inverse_permutation, but also handles slices.

    Parameters
    ----------
    positions : list of ndarray or slice
        If slice objects, all are assumed to be slices.

    Returns
    -------
    np.ndarray of indices or None, if no permutation is necessary.
    """
    if not positions:
        return None

    if isinstance(positions[0], slice):
        positions = _consolidate_slices(positions)
        if positions == slice(None):
            return None
        positions = [np.arange(sl.start, sl.stop, sl.step) for sl in positions]

    newpositions = nputils.inverse_permutation(np.concatenate(positions), N)
    return newpositions[newpositions != -1]


class _DummyGroup:
    """Class for keeping track of grouped dimensions without coordinates.

    Should not be user visible.
    """

    __slots__ = ("name", "coords", "size")

    def __init__(self, obj: T_Xarray, name: Hashable, coords) -> None:
        self.name = name
        self.coords = coords
        self.size = obj.sizes[name]

    @property
    def dims(self) -> tuple[Hashable]:
        return (self.name,)

    @property
    def ndim(self) -> Literal[1]:
        return 1

    @property
    def values(self) -> range:
        return range(self.size)

    @property
    def data(self) -> range:
        return range(self.size)

    @property
    def shape(self) -> tuple[int]:
        return (self.size,)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            key = key[0]
        return self.values[key]

    def copy(self, deep: bool = True, data: Any = None):
        raise NotImplementedError


T_Group = TypeVar("T_Group", bound=Union["DataArray", "IndexVariable", _DummyGroup])


def _ensure_1d(
    group: T_Group, obj: T_Xarray
) -> tuple[T_Group, T_Xarray, Hashable | None, list[Hashable]]:
    # 1D cases: do nothing
    from xarray.core.dataarray import DataArray

    if isinstance(group, (IndexVariable, _DummyGroup)) or group.ndim == 1:
        return group, obj, None, []

    if isinstance(group, DataArray):
        # try to stack the dims of the group into a single dim
        orig_dims = group.dims
        stacked_dim = "stacked_" + "_".join(map(str, orig_dims))
        # these dimensions get created by the stack operation
        inserted_dims = [dim for dim in group.dims if dim not in group.coords]
        # the copy is necessary here, otherwise read only array raises error
        # in pandas: https://github.com/pydata/pandas/issues/12813
        newgroup = group.stack({stacked_dim: orig_dims}).copy()
        newobj = obj.stack({stacked_dim: orig_dims})
        return cast(T_Group, newgroup), newobj, stacked_dim, inserted_dims

    raise TypeError(
        f"group must be DataArray, IndexVariable or _DummyGroup, got {type(group)!r}."
    )


def _unique_and_monotonic(group: T_Group) -> bool:
    if isinstance(group, _DummyGroup):
        return True
    index = safe_cast_to_index(group)
    return index.is_unique and index.is_monotonic_increasing


def _apply_loffset(
    loffset: str | pd.DateOffset | datetime.timedelta | pd.Timedelta,
    result: pd.Series | pd.DataFrame,
):
    """
    (copied from pandas)
    if loffset is set, offset the result index

    This is NOT an idempotent routine, it will be applied
    exactly once to the result.

    Parameters
    ----------
    result : Series or DataFrame
        the result of resample
    """
    # pd.Timedelta is a subclass of datetime.timedelta so we do not need to
    # include it in instance checks.
    if not isinstance(loffset, (str, pd.DateOffset, datetime.timedelta)):
        raise ValueError(
            f"`loffset` must be a str, pd.DateOffset, datetime.timedelta, or pandas.Timedelta object. "
            f"Got {loffset}."
        )

    if isinstance(loffset, str):
        loffset = pd.tseries.frequencies.to_offset(loffset)

    needs_offset = (
        isinstance(loffset, (pd.DateOffset, datetime.timedelta))
        and isinstance(result.index, pd.DatetimeIndex)
        and len(result.index) > 0
    )

    if needs_offset:
        result.index = result.index + loffset


def _get_index_and_items(index, grouper):
    first_items, codes = grouper.first_items(index)
    full_index = first_items.index
    if first_items.isnull().any():
        first_items = first_items.dropna()
    return full_index, first_items, codes


def _factorize_grouper(
    group, grouper
) -> tuple[
    DataArray | IndexVariable | _DummyGroup,
    T_GroupIndices,
    np.ndarray,
    pd.Index,
]:
    index = safe_cast_to_index(group)
    if not index.is_monotonic_increasing:
        # TODO: sort instead of raising an error
        raise ValueError("index must be monotonic for resampling")
    full_index, first_items, codes = _get_index_and_items(index, grouper)
    sbins = first_items.values.astype(np.int64)
    group_indices: T_GroupIndices = [
        slice(i, j) for i, j in zip(sbins[:-1], sbins[1:])
    ] + [slice(sbins[-1], None)]
    unique_coord = IndexVariable(group.name, first_items.index)
    return unique_coord, group_indices, codes, full_index


def _factorize_bins(
    group, bins, cut_kwargs: Mapping | None
) -> tuple[IndexVariable, T_GroupIndices, np.ndarray, pd.IntervalIndex, DataArray]:
    from xarray.core.dataarray import DataArray

    if cut_kwargs is None:
        cut_kwargs = {}

    if duck_array_ops.isnull(bins).all():
        raise ValueError("All bin edges are NaN.")
    binned, bins = pd.cut(group.values, bins, **cut_kwargs, retbins=True)
    codes = binned.codes
    if (codes == -1).all():
        raise ValueError(f"None of the data falls within bins with edges {bins!r}")
    full_index = binned.categories
    uniques = np.sort(pd.unique(codes))
    unique_values = full_index[uniques[uniques != -1]]
    group_indices = [g for g in _codes_to_groups(codes, len(full_index)) if g]

    if len(group_indices) == 0:
        raise ValueError(f"None of the data falls within bins with edges {bins!r}")

    new_dim_name = str(group.name) + "_bins"
    group_ = DataArray(binned, getattr(group, "coords", None), name=new_dim_name)
    unique_coord = IndexVariable(new_dim_name, unique_values)
    return unique_coord, group_indices, codes, full_index, group_


def _factorize_rest(
    group,
) -> tuple[IndexVariable, T_GroupIndices, np.ndarray]:
    # look through group to find the unique values
    group_as_index = safe_cast_to_index(group)
    sort = not isinstance(group_as_index, pd.MultiIndex)
    unique_values, group_indices, codes = unique_value_groups(group_as_index, sort=sort)
    if len(group_indices) == 0:
        raise ValueError(
            "Failed to group data. Are you grouping by a variable that is all NaN?"
        )
    unique_coord = IndexVariable(group.name, unique_values)
    return unique_coord, group_indices, codes


def _factorize_dummy(
    group, squeeze: bool
) -> tuple[IndexVariable, T_GroupIndices, np.ndarray]:
    # no need to factorize
    group_indices: T_GroupIndices
    if not squeeze:
        # use slices to do views instead of fancy indexing
        # equivalent to: group_indices = group_indices.reshape(-1, 1)
        group_indices = [slice(i, i + 1) for i in range(group.size)]
    else:
        group_indices = np.arange(group.size)
    codes = np.arange(group.size)
    unique_coord = group
    return unique_coord, group_indices, codes


class GroupBy(Generic[T_Xarray]):
    """A object that implements the split-apply-combine pattern.

    Modeled after `pandas.GroupBy`. The `GroupBy` object can be iterated over
    (unique_value, grouped_array) pairs, but the main way to interact with a
    groupby object are with the `apply` or `reduce` methods. You can also
    directly call numpy methods like `mean` or `std`.

    You should create a GroupBy object by using the `DataArray.groupby` or
    `Dataset.groupby` methods.

    See Also
    --------
    Dataset.groupby
    DataArray.groupby
    """

    __slots__ = (
        "_full_index",
        "_inserted_dims",
        "_group",
        "_group_dim",
        "_group_indices",
        "_groups",
        "_obj",
        "_restore_coord_dims",
        "_stacked_dim",
        "_unique_coord",
        "_dims",
        "_sizes",
        "_squeeze",
        # Save unstacked object for flox
        "_original_obj",
        "_original_group",
        "_bins",
        "_codes",
    )
    _obj: T_Xarray

    def __init__(
        self,
        obj: T_Xarray,
        group: Hashable | DataArray | IndexVariable,
        squeeze: bool = False,
        grouper: pd.Grouper | None = None,
        bins: ArrayLike | None = None,
        restore_coord_dims: bool = True,
        cut_kwargs: Mapping[Any, Any] | None = None,
    ) -> None:
        """Create a GroupBy object

        Parameters
        ----------
        obj : Dataset or DataArray
            Object to group.
        group : Hashable, DataArray or Index
            Array with the group values or name of the variable.
        squeeze : bool, default: False
            If "group" is a coordinate of object, `squeeze` controls whether
            the subarrays have a dimension of length 1 along that coordinate or
            if the dimension is squeezed out.
        grouper : pandas.Grouper, optional
            Used for grouping values along the `group` array.
        bins : array-like, optional
            If `bins` is specified, the groups will be discretized into the
            specified bins by `pandas.cut`.
        restore_coord_dims : bool, default: True
            If True, also restore the dimension order of multi-dimensional
            coordinates.
        cut_kwargs : dict-like, optional
            Extra keyword arguments to pass to `pandas.cut`

        """
        from xarray.core.dataarray import DataArray

        if grouper is not None and bins is not None:
            raise TypeError("can't specify both `grouper` and `bins`")

        if not isinstance(group, (DataArray, IndexVariable)):
            if not hashable(group):
                raise TypeError(
                    "`group` must be an xarray.DataArray or the "
                    "name of an xarray variable or dimension. "
                    f"Received {group!r} instead."
                )
            group = obj[group]
            if len(group) == 0:
                raise ValueError(f"{group.name} must not be empty")

            if group.name not in obj.coords and group.name in obj.dims:
                # DummyGroups should not appear on groupby results
                group = _DummyGroup(obj, group.name, group.coords)

        if getattr(group, "name", None) is None:
            group.name = "group"

        self._original_obj: T_Xarray = obj
        self._original_group = group
        self._bins = bins

        group, obj, stacked_dim, inserted_dims = _ensure_1d(group, obj)
        (group_dim,) = group.dims

        expected_size = obj.sizes[group_dim]
        if group.size != expected_size:
            raise ValueError(
                "the group variable's length does not "
                "match the length of this variable along its "
                "dimension"
            )

        self._codes: DataArray
        if grouper is not None:
            unique_coord, group_indices, codes, full_index = _factorize_grouper(
                group, grouper
            )
            self._codes = group.copy(data=codes)
        elif bins is not None:
            unique_coord, group_indices, codes, full_index, group = _factorize_bins(
                group, bins, cut_kwargs
            )
            self._codes = group.copy(data=codes)
        elif group.dims == (group.name,) and _unique_and_monotonic(group):
            unique_coord, group_indices, codes = _factorize_dummy(group, squeeze)
            full_index = None
            self._codes = obj[group.name].copy(data=codes)
        else:
            unique_coord, group_indices, codes = _factorize_rest(group)
            full_index = None
            self._codes = group.copy(data=codes)

        # specification for the groupby operation
        self._obj: T_Xarray = obj
        self._group = group
        self._group_dim = group_dim
        self._group_indices = group_indices
        self._unique_coord = unique_coord
        self._stacked_dim = stacked_dim
        self._inserted_dims = inserted_dims
        self._full_index = full_index
        self._restore_coord_dims = restore_coord_dims
        self._bins = bins
        self._squeeze = squeeze
        self._codes = self._maybe_unstack(self._codes)

        # cached attributes
        self._groups: dict[GroupKey, slice | int | list[int]] | None = None
        self._dims: tuple[Hashable, ...] | Frozen[Hashable, int] | None = None
        self._sizes: Frozen[Hashable, int] | None = None

    @property
    def sizes(self) -> Frozen[Hashable, int]:
        """Ordered mapping from dimension names to lengths.

        Immutable.

        See Also
        --------
        DataArray.sizes
        Dataset.sizes
        """
        if self._sizes is None:
            self._sizes = self._obj.isel(
                {self._group_dim: self._group_indices[0]}
            ).sizes

        return self._sizes

    def map(
        self,
        func: Callable,
        args: tuple[Any, ...] = (),
        shortcut: bool | None = None,
        **kwargs: Any,
    ) -> T_Xarray:
        raise NotImplementedError()

    def reduce(
        self,
        func: Callable[..., Any],
        dim: Dims = None,
        *,
        axis: int | Sequence[int] | None = None,
        keep_attrs: bool | None = None,
        keepdims: bool = False,
        shortcut: bool = True,
        **kwargs: Any,
    ) -> T_Xarray:
        raise NotImplementedError()

    @property
    def groups(self) -> dict[GroupKey, slice | int | list[int]]:
        """
        Mapping from group labels to indices. The indices can be used to index the underlying object.
        """
        # provided to mimic pandas.groupby
        if self._groups is None:
            self._groups = dict(zip(self._unique_coord.values, self._group_indices))
        return self._groups

    def __getitem__(self, key: GroupKey) -> T_Xarray:
        """
        Get DataArray or Dataset corresponding to a particular group label.
        """
        return self._obj.isel({self._group_dim: self.groups[key]})

    def __len__(self) -> int:
        return self._unique_coord.size

    def __iter__(self) -> Iterator[tuple[GroupKey, T_Xarray]]:
        return zip(self._unique_coord.values, self._iter_grouped())

    def __repr__(self) -> str:
        return "{}, grouped over {!r}\n{!r} groups with labels {}.".format(
            self.__class__.__name__,
            self._unique_coord.name,
            self._unique_coord.size,
            ", ".join(format_array_flat(self._unique_coord, 30).split()),
        )

    def _iter_grouped(self) -> Iterator[T_Xarray]:
        """Iterate over each element in this group"""
        for indices in self._group_indices:
            yield self._obj.isel({self._group_dim: indices})

    def _infer_concat_args(self, applied_example):
        if self._group_dim in applied_example.dims:
            coord = self._group
            positions = self._group_indices
        else:
            coord = self._unique_coord
            positions = None
        (dim,) = coord.dims
        if isinstance(coord, _DummyGroup):
            coord = None
        coord = getattr(coord, "variable", coord)
        return coord, dim, positions

    def _binary_op(self, other, f, reflexive=False):
        from xarray.core.dataarray import DataArray
        from xarray.core.dataset import Dataset

        g = f if not reflexive else lambda x, y: f(y, x)

        obj = self._original_obj
        group = self._original_group
        codes = self._codes
        dims = group.dims

        if isinstance(group, _DummyGroup):
            group = obj[group.name]
            coord = group
        else:
            coord = self._unique_coord
            if not isinstance(coord, DataArray):
                coord = DataArray(self._unique_coord)
        name = self._group.name

        if not isinstance(other, (Dataset, DataArray)):
            raise TypeError(
                "GroupBy objects only support binary ops "
                "when the other argument is a Dataset or "
                "DataArray"
            )

        if name not in other.dims:
            raise ValueError(
                "incompatible dimensions for a grouped "
                f"binary operation: the group variable {name!r} "
                "is not a dimension on the other argument "
                f"with dimensions {other.dims!r}"
            )

        # Broadcast out scalars for backwards compatibility
        # TODO: get rid of this when fixing GH2145
        for var in other.coords:
            if other[var].ndim == 0:
                other[var] = (
                    other[var].drop_vars(var).expand_dims({name: other.sizes[name]})
                )

        # need to handle NaNs in group or elements that don't belong to any bins
        mask = codes == -1
        if mask.any():
            obj = obj.where(~mask, drop=True)
            codes = codes.where(~mask, drop=True).astype(int)

        other, _ = align(other, coord, join="outer")
        expanded = other.isel({name: codes})

        result = g(obj, expanded)

        if group.ndim > 1:
            # backcompat:
            # TODO: get rid of this when fixing GH2145
            for var in set(obj.coords) - set(obj.xindexes):
                if set(obj[var].dims) < set(group.dims):
                    result[var] = obj[var].reset_coords(drop=True).broadcast_like(group)

        if isinstance(result, Dataset) and isinstance(obj, Dataset):
            for var in set(result):
                for d in dims:
                    if d not in obj[var].dims:
                        result[var] = result[var].transpose(d, ...)
        return result

    def _maybe_restore_empty_groups(self, combined):
        """Our index contained empty groups (e.g., from a resampling). If we
        reduced on that dimension, we want to restore the full index.
        """
        if self._full_index is not None and self._group.name in combined.dims:
            indexers = {self._group.name: self._full_index}
            combined = combined.reindex(**indexers)
        return combined

    def _maybe_unstack(self, obj):
        """This gets called if we are applying on an array with a
        multidimensional group."""
        if self._stacked_dim is not None and self._stacked_dim in obj.dims:
            obj = obj.unstack(self._stacked_dim)
            for dim in self._inserted_dims:
                if dim in obj.coords:
                    del obj.coords[dim]
            obj._indexes = filter_indexes_from_coords(obj._indexes, set(obj.coords))
        return obj

    def _flox_reduce(
        self,
        dim: Dims,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ):
        """Adaptor function that translates our groupby API to that of flox."""
        from flox.xarray import xarray_reduce

        from xarray.core.dataset import Dataset

        obj = self._original_obj
        group = self._original_group

        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=True)

        # preserve current strategy (approximately) for dask groupby.
        # We want to control the default anyway to prevent surprises
        # if flox decides to change its default
        kwargs.setdefault("method", "cohorts")

        numeric_only = kwargs.pop("numeric_only", None)
        if numeric_only:
            non_numeric = {
                name: var
                for name, var in obj.data_vars.items()
                if not (np.issubdtype(var.dtype, np.number) or (var.dtype == np.bool_))
            }
        else:
            non_numeric = {}

        # weird backcompat
        # reducing along a unique indexed dimension with squeeze=True
        # should raise an error
        if (
            dim is None or dim == self._group.name
        ) and self._group.name in obj.xindexes:
            index = obj.indexes[self._group.name]
            if index.is_unique and self._squeeze:
                raise ValueError(f"cannot reduce over dimensions {self._group.name!r}")

        unindexed_dims: tuple[Hashable, ...] = tuple()
        if isinstance(group, _DummyGroup) and self._bins is None:
            unindexed_dims = (group.name,)

        parsed_dim: tuple[Hashable, ...]
        if isinstance(dim, str):
            parsed_dim = (dim,)
        elif dim is None:
            parsed_dim = group.dims
        elif dim is ...:
            parsed_dim = tuple(obj.dims)
        else:
            parsed_dim = tuple(dim)

        # Do this so we raise the same error message whether flox is present or not.
        # Better to control it here than in flox.
        if any(d not in group.dims and d not in obj.dims for d in parsed_dim):
            raise ValueError(f"cannot reduce over dimensions {dim}.")

        if kwargs["func"] not in ["all", "any", "count"]:
            kwargs.setdefault("fill_value", np.nan)
        if self._bins is not None and kwargs["func"] == "count":
            # This is an annoying hack. Xarray returns np.nan
            # when there are no observations in a bin, instead of 0.
            # We can fake that here by forcing min_count=1.
            # note min_count makes no sense in the xarray world
            # as a kwarg for count, so this should be OK
            kwargs.setdefault("fill_value", np.nan)
            kwargs.setdefault("min_count", 1)

        output_index = self._get_output_index()
        result = xarray_reduce(
            obj.drop_vars(non_numeric.keys()),
            self._codes,
            dim=parsed_dim,
            # pass RangeIndex as a hint to flox that `by` is already factorized
            expected_groups=(pd.RangeIndex(len(output_index)),),
            isbin=False,
            keep_attrs=keep_attrs,
            **kwargs,
        )

        # we did end up reducing over dimension(s) that are
        # in the grouped variable
        if set(self._codes.dims).issubset(set(parsed_dim)):
            result[self._unique_coord.name] = output_index
            result = result.drop_vars(unindexed_dims)

        # broadcast and restore non-numeric data variables (backcompat)
        for name, var in non_numeric.items():
            if all(d not in var.dims for d in parsed_dim):
                result[name] = var.variable.set_dims(
                    (group.name,) + var.dims, (result.sizes[group.name],) + var.shape
                )

        if self._bins is not None:
            # Fix dimension order when binning a dimension coordinate
            # Needed as long as we do a separate code path for pint;
            # For some reason Datasets and DataArrays behave differently!
            if isinstance(self._obj, Dataset) and self._group_dim in self._obj.dims:
                result = result.transpose(self._group.name, ...)

        return result

    def _get_output_index(self) -> pd.Index:
        """Return pandas.Index object for the output array."""
        if self._full_index is not None:
            # binning and resample
            return self._full_index.rename(self._unique_coord.name)
        if isinstance(self._unique_coord, _DummyGroup):
            return IndexVariable(self._group.name, self._unique_coord.values)
        return self._unique_coord

    def fillna(self, value: Any) -> T_Xarray:
        """Fill missing values in this object by group.

        This operation follows the normal broadcasting and alignment rules that
        xarray uses for binary arithmetic, except the result is aligned to this
        object (``join='left'``) instead of aligned to the intersection of
        index coordinates (``join='inner'``).

        Parameters
        ----------
        value
            Used to fill all matching missing values by group. Needs
            to be of a valid type for the wrapped object's fillna
            method.

        Returns
        -------
        same type as the grouped object

        See Also
        --------
        Dataset.fillna
        DataArray.fillna
        """
        return ops.fillna(self, value)

    def quantile(
        self,
        q: ArrayLike,
        dim: Dims = None,
        method: QuantileMethods = "linear",
        keep_attrs: bool | None = None,
        skipna: bool | None = None,
        interpolation: QuantileMethods | None = None,
    ) -> T_Xarray:
        """Compute the qth quantile over each array in the groups and
        concatenate them together into a new array.

        Parameters
        ----------
        q : float or sequence of float
            Quantile to compute, which must be between 0 and 1
            inclusive.
        dim : str or Iterable of Hashable, optional
            Dimension(s) over which to apply quantile.
            Defaults to the grouped dimension.
        method : str, default: "linear"
            This optional parameter specifies the interpolation method to use when the
            desired quantile lies between two data points. The options sorted by their R
            type as summarized in the H&F paper [1]_ are:

                1. "inverted_cdf" (*)
                2. "averaged_inverted_cdf" (*)
                3. "closest_observation" (*)
                4. "interpolated_inverted_cdf" (*)
                5. "hazen" (*)
                6. "weibull" (*)
                7. "linear"  (default)
                8. "median_unbiased" (*)
                9. "normal_unbiased" (*)

            The first three methods are discontiuous.  The following discontinuous
            variations of the default "linear" (7.) option are also available:

                * "lower"
                * "higher"
                * "midpoint"
                * "nearest"

            See :py:func:`numpy.quantile` or [1]_ for details. Methods marked with
            an asterisk require numpy version 1.22 or newer. The "method" argument was
            previously called "interpolation", renamed in accordance with numpy
            version 1.22.0.
        keep_attrs : bool or None, default: None
            If True, the dataarray's attributes (`attrs`) will be copied from
            the original object to the new one.  If False, the new
            object will be returned without attributes.
        skipna : bool or None, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).

        Returns
        -------
        quantiles : Variable
            If `q` is a single quantile, then the result is a
            scalar. If multiple percentiles are given, first axis of
            the result corresponds to the quantile. In either case a
            quantile dimension is added to the return array. The other
            dimensions are the dimensions that remain after the
            reduction of the array.

        See Also
        --------
        numpy.nanquantile, numpy.quantile, pandas.Series.quantile, Dataset.quantile
        DataArray.quantile

        Examples
        --------
        >>> da = xr.DataArray(
        ...     [[1.3, 8.4, 0.7, 6.9], [0.7, 4.2, 9.4, 1.5], [6.5, 7.3, 2.6, 1.9]],
        ...     coords={"x": [0, 0, 1], "y": [1, 1, 2, 2]},
        ...     dims=("x", "y"),
        ... )
        >>> ds = xr.Dataset({"a": da})
        >>> da.groupby("x").quantile(0)
        <xarray.DataArray (x: 2, y: 4)>
        array([[0.7, 4.2, 0.7, 1.5],
               [6.5, 7.3, 2.6, 1.9]])
        Coordinates:
          * y         (y) int64 1 1 2 2
            quantile  float64 0.0
          * x         (x) int64 0 1
        >>> ds.groupby("y").quantile(0, dim=...)
        <xarray.Dataset>
        Dimensions:   (y: 2)
        Coordinates:
            quantile  float64 0.0
          * y         (y) int64 1 2
        Data variables:
            a         (y) float64 0.7 0.7
        >>> da.groupby("x").quantile([0, 0.5, 1])
        <xarray.DataArray (x: 2, y: 4, quantile: 3)>
        array([[[0.7 , 1.  , 1.3 ],
                [4.2 , 6.3 , 8.4 ],
                [0.7 , 5.05, 9.4 ],
                [1.5 , 4.2 , 6.9 ]],
        <BLANKLINE>
               [[6.5 , 6.5 , 6.5 ],
                [7.3 , 7.3 , 7.3 ],
                [2.6 , 2.6 , 2.6 ],
                [1.9 , 1.9 , 1.9 ]]])
        Coordinates:
          * y         (y) int64 1 1 2 2
          * quantile  (quantile) float64 0.0 0.5 1.0
          * x         (x) int64 0 1
        >>> ds.groupby("y").quantile([0, 0.5, 1], dim=...)
        <xarray.Dataset>
        Dimensions:   (y: 2, quantile: 3)
        Coordinates:
          * quantile  (quantile) float64 0.0 0.5 1.0
          * y         (y) int64 1 2
        Data variables:
            a         (y, quantile) float64 0.7 5.35 8.4 0.7 2.25 9.4

        References
        ----------
        .. [1] R. J. Hyndman and Y. Fan,
           "Sample quantiles in statistical packages,"
           The American Statistician, 50(4), pp. 361-365, 1996
        """
        if dim is None:
            dim = (self._group_dim,)

        return self.map(
            self._obj.__class__.quantile,
            shortcut=False,
            q=q,
            dim=dim,
            method=method,
            keep_attrs=keep_attrs,
            skipna=skipna,
            interpolation=interpolation,
        )

    def where(self, cond, other=dtypes.NA) -> T_Xarray:
        """Return elements from `self` or `other` depending on `cond`.

        Parameters
        ----------
        cond : DataArray or Dataset
            Locations at which to preserve this objects values. dtypes have to be `bool`
        other : scalar, DataArray or Dataset, optional
            Value to use for locations in this object where ``cond`` is False.
            By default, inserts missing values.

        Returns
        -------
        same type as the grouped object

        See Also
        --------
        Dataset.where
        """
        return ops.where_method(self, cond, other)

    def _first_or_last(self, op, skipna, keep_attrs):
        if isinstance(self._group_indices[0], integer_types):
            # NB. this is currently only used for reductions along an existing
            # dimension
            return self._obj
        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=True)
        return self.reduce(
            op, dim=[self._group_dim], skipna=skipna, keep_attrs=keep_attrs
        )

    def first(self, skipna: bool | None = None, keep_attrs: bool | None = None):
        """Return the first element of each group along the group dimension"""
        return self._first_or_last(duck_array_ops.first, skipna, keep_attrs)

    def last(self, skipna: bool | None = None, keep_attrs: bool | None = None):
        """Return the last element of each group along the group dimension"""
        return self._first_or_last(duck_array_ops.last, skipna, keep_attrs)

    def assign_coords(self, coords=None, **coords_kwargs):
        """Assign coordinates by group.

        See Also
        --------
        Dataset.assign_coords
        Dataset.swap_dims
        """
        coords_kwargs = either_dict_or_kwargs(coords, coords_kwargs, "assign_coords")
        return self.map(lambda ds: ds.assign_coords(**coords_kwargs))


def _maybe_reorder(xarray_obj, dim, positions, N: int | None):
    order = _inverse_permutation_indices(positions, N)

    if order is None or len(order) != xarray_obj.sizes[dim]:
        return xarray_obj
    else:
        return xarray_obj[{dim: order}]


class DataArrayGroupByBase(GroupBy["DataArray"], DataArrayGroupbyArithmetic):
    """GroupBy object specialized to grouping DataArray objects"""

    __slots__ = ()
    _dims: tuple[Hashable, ...] | None

    @property
    def dims(self) -> tuple[Hashable, ...]:
        if self._dims is None:
            self._dims = self._obj.isel({self._group_dim: self._group_indices[0]}).dims

        return self._dims

    def _iter_grouped_shortcut(self):
        """Fast version of `_iter_grouped` that yields Variables without
        metadata
        """
        var = self._obj.variable
        for indices in self._group_indices:
            yield var[{self._group_dim: indices}]

    def _concat_shortcut(self, applied, dim, positions=None):
        # nb. don't worry too much about maintaining this method -- it does
        # speed things up, but it's not very interpretable and there are much
        # faster alternatives (e.g., doing the grouped aggregation in a
        # compiled language)
        # TODO: benbovy - explicit indexes: this fast implementation doesn't
        # create an explicit index for the stacked dim coordinate
        stacked = Variable.concat(applied, dim, shortcut=True)
        reordered = _maybe_reorder(stacked, dim, positions, N=self._group.size)
        return self._obj._replace_maybe_drop_dims(reordered)

    def _restore_dim_order(self, stacked: DataArray) -> DataArray:
        def lookup_order(dimension):
            if dimension == self._group.name:
                (dimension,) = self._group.dims
            if dimension in self._obj.dims:
                axis = self._obj.get_axis_num(dimension)
            else:
                axis = 1e6  # some arbitrarily high value
            return axis

        new_order = sorted(stacked.dims, key=lookup_order)
        return stacked.transpose(*new_order, transpose_coords=self._restore_coord_dims)

    def map(
        self,
        func: Callable[..., DataArray],
        args: tuple[Any, ...] = (),
        shortcut: bool | None = None,
        **kwargs: Any,
    ) -> DataArray:
        """Apply a function to each array in the group and concatenate them
        together into a new array.

        `func` is called like `func(ar, *args, **kwargs)` for each array `ar`
        in this group.

        Apply uses heuristics (like `pandas.GroupBy.apply`) to figure out how
        to stack together the array. The rule is:

        1. If the dimension along which the group coordinate is defined is
           still in the first grouped array after applying `func`, then stack
           over this dimension.
        2. Otherwise, stack over the new dimension given by name of this
           grouping (the argument to the `groupby` function).

        Parameters
        ----------
        func : callable
            Callable to apply to each array.
        shortcut : bool, optional
            Whether or not to shortcut evaluation under the assumptions that:

            (1) The action of `func` does not depend on any of the array
                metadata (attributes or coordinates) but only on the data and
                dimensions.
            (2) The action of `func` creates arrays with homogeneous metadata,
                that is, with the same dimensions and attributes.

            If these conditions are satisfied `shortcut` provides significant
            speedup. This should be the case for many common groupby operations
            (e.g., applying numpy ufuncs).
        *args : tuple, optional
            Positional arguments passed to `func`.
        **kwargs
            Used to call `func(ar, **kwargs)` for each array `ar`.

        Returns
        -------
        applied : DataArray
            The result of splitting, applying and combining this array.
        """
        grouped = self._iter_grouped_shortcut() if shortcut else self._iter_grouped()
        applied = (maybe_wrap_array(arr, func(arr, *args, **kwargs)) for arr in grouped)
        return self._combine(applied, shortcut=shortcut)

    def apply(self, func, shortcut=False, args=(), **kwargs):
        """
        Backward compatible implementation of ``map``

        See Also
        --------
        DataArrayGroupBy.map
        """
        warnings.warn(
            "GroupBy.apply may be deprecated in the future. Using GroupBy.map is encouraged",
            PendingDeprecationWarning,
            stacklevel=2,
        )
        return self.map(func, shortcut=shortcut, args=args, **kwargs)

    def _combine(self, applied, shortcut=False):
        """Recombine the applied objects like the original."""
        applied_example, applied = peek_at(applied)
        coord, dim, positions = self._infer_concat_args(applied_example)
        if shortcut:
            combined = self._concat_shortcut(applied, dim, positions)
        else:
            combined = concat(applied, dim)
            combined = _maybe_reorder(combined, dim, positions, N=self._group.size)

        if isinstance(combined, type(self._obj)):
            # only restore dimension order for arrays
            combined = self._restore_dim_order(combined)
        # assign coord and index when the applied function does not return that coord
        if coord is not None and dim not in applied_example.dims:
            index, index_vars = create_default_index_implicit(coord)
            indexes = {k: index for k in index_vars}
            combined = combined._overwrite_indexes(indexes, index_vars)
        combined = self._maybe_restore_empty_groups(combined)
        combined = self._maybe_unstack(combined)
        return combined

    def reduce(
        self,
        func: Callable[..., Any],
        dim: Dims = None,
        *,
        axis: int | Sequence[int] | None = None,
        keep_attrs: bool | None = None,
        keepdims: bool = False,
        shortcut: bool = True,
        **kwargs: Any,
    ) -> DataArray:
        """Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : callable
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of collapsing
            an np.ndarray over an integer valued axis.
        dim : "...", str, Iterable of Hashable or None, optional
            Dimension(s) over which to apply `func`. If None, apply over the
            groupby dimension, if "..." apply over all dimensions.
        axis : int or sequence of int, optional
            Axis(es) over which to apply `func`. Only one of the 'dimension'
            and 'axis' arguments can be supplied. If neither are supplied, then
            `func` is calculated over all dimension for each group item.
        keep_attrs : bool, optional
            If True, the datasets's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : Array
            Array with summarized data and the indicated dimension(s)
            removed.
        """
        if dim is None:
            dim = [self._group_dim]

        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=True)

        def reduce_array(ar: DataArray) -> DataArray:
            return ar.reduce(
                func=func,
                dim=dim,
                axis=axis,
                keep_attrs=keep_attrs,
                keepdims=keepdims,
                **kwargs,
            )

        check_reduce_dims(dim, self.dims)

        return self.map(reduce_array, shortcut=shortcut)


# https://github.com/python/mypy/issues/9031
class DataArrayGroupBy(  # type: ignore[misc]
    DataArrayGroupByBase,
    DataArrayGroupByAggregations,
    ImplementsArrayReduce,
):
    __slots__ = ()


class DatasetGroupByBase(GroupBy["Dataset"], DatasetGroupbyArithmetic):
    __slots__ = ()
    _dims: Frozen[Hashable, int] | None

    @property
    def dims(self) -> Frozen[Hashable, int]:
        if self._dims is None:
            self._dims = self._obj.isel({self._group_dim: self._group_indices[0]}).dims

        return self._dims

    def map(
        self,
        func: Callable[..., Dataset],
        args: tuple[Any, ...] = (),
        shortcut: bool | None = None,
        **kwargs: Any,
    ) -> Dataset:
        """Apply a function to each Dataset in the group and concatenate them
        together into a new Dataset.

        `func` is called like `func(ds, *args, **kwargs)` for each dataset `ds`
        in this group.

        Apply uses heuristics (like `pandas.GroupBy.apply`) to figure out how
        to stack together the datasets. The rule is:

        1. If the dimension along which the group coordinate is defined is
           still in the first grouped item after applying `func`, then stack
           over this dimension.
        2. Otherwise, stack over the new dimension given by name of this
           grouping (the argument to the `groupby` function).

        Parameters
        ----------
        func : callable
            Callable to apply to each sub-dataset.
        args : tuple, optional
            Positional arguments to pass to `func`.
        **kwargs
            Used to call `func(ds, **kwargs)` for each sub-dataset `ar`.

        Returns
        -------
        applied : Dataset
            The result of splitting, applying and combining this dataset.
        """
        # ignore shortcut if set (for now)
        applied = (func(ds, *args, **kwargs) for ds in self._iter_grouped())
        return self._combine(applied)

    def apply(self, func, args=(), shortcut=None, **kwargs):
        """
        Backward compatible implementation of ``map``

        See Also
        --------
        DatasetGroupBy.map
        """

        warnings.warn(
            "GroupBy.apply may be deprecated in the future. Using GroupBy.map is encouraged",
            PendingDeprecationWarning,
            stacklevel=2,
        )
        return self.map(func, shortcut=shortcut, args=args, **kwargs)

    def _combine(self, applied):
        """Recombine the applied objects like the original."""
        applied_example, applied = peek_at(applied)
        coord, dim, positions = self._infer_concat_args(applied_example)
        combined = concat(applied, dim)
        combined = _maybe_reorder(combined, dim, positions, N=self._group.size)
        # assign coord when the applied function does not return that coord
        if coord is not None and dim not in applied_example.dims:
            index, index_vars = create_default_index_implicit(coord)
            indexes = {k: index for k in index_vars}
            combined = combined._overwrite_indexes(indexes, index_vars)
        combined = self._maybe_restore_empty_groups(combined)
        combined = self._maybe_unstack(combined)
        return combined

    def reduce(
        self,
        func: Callable[..., Any],
        dim: Dims = None,
        *,
        axis: int | Sequence[int] | None = None,
        keep_attrs: bool | None = None,
        keepdims: bool = False,
        shortcut: bool = True,
        **kwargs: Any,
    ) -> Dataset:
        """Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : callable
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of collapsing
            an np.ndarray over an integer valued axis.
        dim : ..., str, Iterable of Hashable or None, optional
            Dimension(s) over which to apply `func`. By default apply over the
            groupby dimension, with "..." apply over all dimensions.
        axis : int or sequence of int, optional
            Axis(es) over which to apply `func`. Only one of the 'dimension'
            and 'axis' arguments can be supplied. If neither are supplied, then
            `func` is calculated over all dimension for each group item.
        keep_attrs : bool, optional
            If True, the datasets's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : Dataset
            Array with summarized data and the indicated dimension(s)
            removed.
        """
        if dim is None:
            dim = [self._group_dim]

        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=True)

        def reduce_dataset(ds: Dataset) -> Dataset:
            return ds.reduce(
                func=func,
                dim=dim,
                axis=axis,
                keep_attrs=keep_attrs,
                keepdims=keepdims,
                **kwargs,
            )

        check_reduce_dims(dim, self.dims)

        return self.map(reduce_dataset)

    def assign(self, **kwargs: Any) -> Dataset:
        """Assign data variables by group.

        See Also
        --------
        Dataset.assign
        """
        return self.map(lambda ds: ds.assign(**kwargs))


# https://github.com/python/mypy/issues/9031
class DatasetGroupBy(  # type: ignore[misc]
    DatasetGroupByBase,
    DatasetGroupByAggregations,
    ImplementsDatasetReduce,
):
    __slots__ = ()


class TimeResampleGrouper:
    def __init__(
        self,
        freq: str,
        closed: SideOptions | None,
        label: SideOptions | None,
        origin: str | DatetimeLike,
        offset: pd.Timedelta | datetime.timedelta | str | None,
        loffset: datetime.timedelta | str | None,
    ):
        self.freq = freq
        self.closed = closed
        self.label = label
        self.origin = origin
        self.offset = offset
        self.loffset = loffset

    def first_items(self, index):
        from xarray import CFTimeIndex
        from xarray.core.resample_cftime import CFTimeGrouper

        if isinstance(index, CFTimeIndex):
            grouper = CFTimeGrouper(
                freq=self.freq,
                closed=self.closed,
                label=self.label,
                origin=self.origin,
                offset=self.offset,
                loffset=self.loffset,
            )
            return grouper.first_items(index)
        else:
            s = pd.Series(np.arange(index.size), index, copy=False)
            grouper = pd.Grouper(
                freq=self.freq,
                closed=self.closed,
                label=self.label,
                origin=self.origin,
                offset=self.offset,
            )

            grouped = s.groupby(grouper)
            first_items = grouped.first()
            counts = grouped.count()
            # This way we generate codes for the final output index: full_index.
            # So for _flox_reduce we avoid one reindex and copy by avoiding
            # _maybe_restore_empty_groups
            codes = np.repeat(np.arange(len(first_items)), counts)
            if self.loffset is not None:
                _apply_loffset(self.loffset, first_items)
            return first_items, codes
