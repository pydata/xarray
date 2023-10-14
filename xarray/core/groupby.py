from __future__ import annotations

import datetime
import warnings
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Literal,
    Union,
)

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
from xarray.core.types import Dims, QuantileMethods, T_DataArray, T_Xarray
from xarray.core.utils import (
    either_dict_or_kwargs,
    hashable,
    is_scalar,
    maybe_wrap_array,
    peek_at,
)
from xarray.core.variable import IndexVariable, Variable
from xarray.util.deprecation_helpers import _deprecate_positional_args

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset
    from xarray.core.resample_cftime import CFTimeGrouper
    from xarray.core.types import DatetimeLike, SideOptions
    from xarray.core.utils import Frozen

    GroupKey = Any
    GroupIndex = Union[int, slice, list[int]]
    T_GroupIndices = list[GroupIndex]
    T_FactorizeOut = tuple[
        DataArray, T_GroupIndices, Union[IndexVariable, "_DummyGroup"], pd.Index
    ]


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


def _codes_to_groups(inverse: np.ndarray, N: int) -> T_GroupIndices:
    groups: T_GroupIndices = [[] for _ in range(N)]
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


def _is_one_or_none(obj) -> bool:
    return obj == 1 or obj is None


def _consolidate_slices(slices: list[slice]) -> list[slice]:
    """Consolidate adjacent slices in a list of slices."""
    result: list[slice] = []
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


class _DummyGroup(Generic[T_Xarray]):
    """Class for keeping track of grouped dimensions without coordinates.

    Should not be user visible.
    """

    __slots__ = ("name", "coords", "size", "dataarray")

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

    def __array__(self) -> np.ndarray:
        return np.arange(self.size)

    @property
    def shape(self) -> tuple[int]:
        return (self.size,)

    @property
    def attrs(self) -> dict:
        return {}

    def __getitem__(self, key):
        if isinstance(key, tuple):
            key = key[0]
        return self.values[key]

    def to_index(self) -> pd.Index:
        # could be pd.RangeIndex?
        return pd.Index(np.arange(self.size))

    def copy(self, deep: bool = True, data: Any = None):
        raise NotImplementedError

    def to_dataarray(self) -> DataArray:
        from xarray.core.dataarray import DataArray

        return DataArray(
            data=self.data, dims=(self.name,), coords=self.coords, name=self.name
        )


T_Group = Union["T_DataArray", "IndexVariable", _DummyGroup]


def _ensure_1d(
    group: T_Group, obj: T_Xarray
) -> tuple[T_Group, T_Xarray, Hashable | None, list[Hashable],]:
    # 1D cases: do nothing
    if isinstance(group, (IndexVariable, _DummyGroup)) or group.ndim == 1:
        return group, obj, None, []

    from xarray.core.dataarray import DataArray

    if isinstance(group, DataArray):
        # try to stack the dims of the group into a single dim
        orig_dims = group.dims
        stacked_dim = "stacked_" + "_".join(map(str, orig_dims))
        # these dimensions get created by the stack operation
        inserted_dims = [dim for dim in group.dims if dim not in group.coords]
        newgroup = group.stack({stacked_dim: orig_dims})
        newobj = obj.stack({stacked_dim: orig_dims})
        return newgroup, newobj, stacked_dim, inserted_dims

    raise TypeError(
        f"group must be DataArray, IndexVariable or _DummyGroup, got {type(group)!r}."
    )


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


@dataclass
class ResolvedGrouper(ABC, Generic[T_Xarray]):
    grouper: Grouper
    group: T_Group
    obj: T_Xarray

    _group_as_index: pd.Index | None = field(default=None, init=False)

    # Defined by factorize:
    codes: DataArray = field(init=False)
    group_indices: T_GroupIndices = field(init=False)
    unique_coord: IndexVariable | _DummyGroup = field(init=False)
    full_index: pd.Index = field(init=False)

    # _ensure_1d:
    group1d: T_Group = field(init=False)
    stacked_obj: T_Xarray = field(init=False)
    stacked_dim: Hashable | None = field(init=False)
    inserted_dims: list[Hashable] = field(init=False)

    def __post_init__(self) -> None:
        self.group: T_Group = _resolve_group(self.obj, self.group)

        (
            self.group1d,
            self.stacked_obj,
            self.stacked_dim,
            self.inserted_dims,
        ) = _ensure_1d(group=self.group, obj=self.obj)

    @property
    def name(self) -> Hashable:
        return self.group1d.name

    @property
    def size(self) -> int:
        return len(self)

    def __len__(self) -> int:
        return len(self.full_index)  # TODO: full_index not def, abstractmethod?

    @property
    def dims(self):
        return self.group1d.dims

    @abstractmethod
    def _factorize(self, squeeze: bool) -> T_FactorizeOut:
        raise NotImplementedError

    def factorize(self, squeeze: bool) -> None:
        # This design makes it clear  to mypy that
        # codes, group_indices, unique_coord, and full_index
        # are set by the factorize  method on the derived class.
        (
            self.codes,
            self.group_indices,
            self.unique_coord,
            self.full_index,
        ) = self._factorize(squeeze)

    @property
    def is_unique_and_monotonic(self) -> bool:
        if isinstance(self.group, _DummyGroup):
            return True
        index = self.group_as_index
        return index.is_unique and index.is_monotonic_increasing

    @property
    def group_as_index(self) -> pd.Index:
        if self._group_as_index is None:
            self._group_as_index = self.group1d.to_index()
        return self._group_as_index


@dataclass
class ResolvedUniqueGrouper(ResolvedGrouper):
    grouper: UniqueGrouper

    def _factorize(self, squeeze) -> T_FactorizeOut:
        is_dimension = self.group.dims == (self.group.name,)
        if is_dimension and self.is_unique_and_monotonic:
            return self._factorize_dummy(squeeze)
        else:
            return self._factorize_unique()

    def _factorize_unique(self) -> T_FactorizeOut:
        # look through group to find the unique values
        sort = not isinstance(self.group_as_index, pd.MultiIndex)
        unique_values, group_indices, codes_ = unique_value_groups(
            self.group_as_index, sort=sort
        )
        if len(group_indices) == 0:
            raise ValueError(
                "Failed to group data. Are you grouping by a variable that is all NaN?"
            )
        codes = self.group1d.copy(data=codes_)
        group_indices = group_indices
        unique_coord = IndexVariable(
            self.group.name, unique_values, attrs=self.group.attrs
        )
        full_index = unique_coord

        return codes, group_indices, unique_coord, full_index

    def _factorize_dummy(self, squeeze) -> T_FactorizeOut:
        size = self.group.size
        # no need to factorize
        if not squeeze:
            # use slices to do views instead of fancy indexing
            # equivalent to: group_indices = group_indices.reshape(-1, 1)
            group_indices: T_GroupIndices = [slice(i, i + 1) for i in range(size)]
        else:
            group_indices = list(range(size))
        size_range = np.arange(size)
        if isinstance(self.group, _DummyGroup):
            codes = self.group.to_dataarray().copy(data=size_range)
        else:
            codes = self.group.copy(data=size_range)
        unique_coord = self.group
        full_index = IndexVariable(self.name, unique_coord.values, self.group.attrs)

        return codes, group_indices, unique_coord, full_index


@dataclass
class ResolvedBinGrouper(ResolvedGrouper):
    grouper: BinGrouper

    def _factorize(self, squeeze: bool) -> T_FactorizeOut:
        from xarray.core.dataarray import DataArray

        data = self.group1d.values
        binned, bins = pd.cut(
            data, self.grouper.bins, **self.grouper.cut_kwargs, retbins=True
        )
        binned_codes = binned.codes
        if (binned_codes == -1).all():
            raise ValueError(f"None of the data falls within bins with edges {bins!r}")

        full_index = binned.categories
        uniques = np.sort(pd.unique(binned_codes))
        unique_values = full_index[uniques[uniques != -1]]
        group_indices = [
            g for g in _codes_to_groups(binned_codes, len(full_index)) if g
        ]

        if len(group_indices) == 0:
            raise ValueError(f"None of the data falls within bins with edges {bins!r}")

        new_dim_name = str(self.group.name) + "_bins"
        self.group1d = DataArray(
            binned, getattr(self.group1d, "coords", None), name=new_dim_name
        )
        unique_coord = IndexVariable(new_dim_name, unique_values, self.group.attrs)
        codes = self.group1d.copy(data=binned_codes)
        # TODO: support IntervalIndex in IndexVariable

        return codes, group_indices, unique_coord, full_index


@dataclass
class ResolvedTimeResampleGrouper(ResolvedGrouper):
    grouper: TimeResampleGrouper
    index_grouper: CFTimeGrouper | pd.Grouper = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()

        from xarray import CFTimeIndex

        group_as_index = safe_cast_to_index(self.group)
        self._group_as_index = group_as_index

        if not group_as_index.is_monotonic_increasing:
            # TODO: sort instead of raising an error
            raise ValueError("index must be monotonic for resampling")

        grouper = self.grouper
        if isinstance(group_as_index, CFTimeIndex):
            from xarray.core.resample_cftime import CFTimeGrouper

            index_grouper = CFTimeGrouper(
                freq=grouper.freq,
                closed=grouper.closed,
                label=grouper.label,
                origin=grouper.origin,
                offset=grouper.offset,
                loffset=grouper.loffset,
            )
        else:
            index_grouper = pd.Grouper(
                freq=grouper.freq,
                closed=grouper.closed,
                label=grouper.label,
                origin=grouper.origin,
                offset=grouper.offset,
            )
        self.index_grouper = index_grouper

    def _get_index_and_items(self) -> tuple[pd.Index, pd.Series, np.ndarray]:
        first_items, codes = self.first_items()
        full_index = first_items.index
        if first_items.isnull().any():
            first_items = first_items.dropna()

        full_index = full_index.rename("__resample_dim__")
        return full_index, first_items, codes

    def first_items(self) -> tuple[pd.Series, np.ndarray]:
        from xarray import CFTimeIndex

        if isinstance(self.group_as_index, CFTimeIndex):
            return self.index_grouper.first_items(self.group_as_index)
        else:
            s = pd.Series(np.arange(self.group_as_index.size), self.group_as_index)
            grouped = s.groupby(self.index_grouper)
            first_items = grouped.first()
            counts = grouped.count()
            # This way we generate codes for the final output index: full_index.
            # So for _flox_reduce we avoid one reindex and copy by avoiding
            # _maybe_restore_empty_groups
            codes = np.repeat(np.arange(len(first_items)), counts)
            if self.grouper.loffset is not None:
                _apply_loffset(self.grouper.loffset, first_items)
            return first_items, codes

    def _factorize(self, squeeze: bool) -> T_FactorizeOut:
        full_index, first_items, codes_ = self._get_index_and_items()
        sbins = first_items.values.astype(np.int64)
        group_indices: T_GroupIndices = [
            slice(i, j) for i, j in zip(sbins[:-1], sbins[1:])
        ]
        group_indices += [slice(sbins[-1], None)]

        unique_coord = IndexVariable(
            self.group.name, first_items.index, self.group.attrs
        )
        codes = self.group.copy(data=codes_)

        return codes, group_indices, unique_coord, full_index


class Grouper(ABC):
    pass


@dataclass
class UniqueGrouper(Grouper):
    pass


@dataclass
class BinGrouper(Grouper):
    bins: Any  # TODO: What is the typing?
    cut_kwargs: Mapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        if duck_array_ops.isnull(self.bins).all():
            raise ValueError("All bin edges are NaN.")


@dataclass
class TimeResampleGrouper(Grouper):
    freq: str
    closed: SideOptions | None
    label: SideOptions | None
    origin: str | DatetimeLike | None
    offset: pd.Timedelta | datetime.timedelta | str | None
    loffset: datetime.timedelta | str | None


def _validate_groupby_squeeze(squeeze: bool) -> None:
    # While we don't generally check the type of every arg, passing
    # multiple dimensions as multiple arguments is common enough, and the
    # consequences hidden enough (strings evaluate as true) to warrant
    # checking here.
    # A future version could make squeeze kwarg only, but would face
    # backward-compat issues.
    if not isinstance(squeeze, bool):
        raise TypeError(f"`squeeze` must be True or False, but {squeeze} was supplied")


def _resolve_group(obj: T_Xarray, group: T_Group | Hashable) -> T_Group:
    from xarray.core.dataarray import DataArray

    error_msg = (
        "the group variable's length does not "
        "match the length of this variable along its "
        "dimensions"
    )

    newgroup: T_Group
    if isinstance(group, DataArray):
        try:
            align(obj, group, join="exact", copy=False)
        except ValueError:
            raise ValueError(error_msg)

        newgroup = group.copy(deep=False)
        newgroup.name = group.name or "group"

    elif isinstance(group, IndexVariable):
        # This assumption is built in to _ensure_1d.
        if group.ndim != 1:
            raise ValueError(
                "Grouping by multi-dimensional IndexVariables is not allowed."
                "Convert to and pass a DataArray instead."
            )
        (group_dim,) = group.dims
        if len(group) != obj.sizes[group_dim]:
            raise ValueError(error_msg)
        newgroup = DataArray(group)

    else:
        if not hashable(group):
            raise TypeError(
                "`group` must be an xarray.DataArray or the "
                "name of an xarray variable or dimension. "
                f"Received {group!r} instead."
            )
        group = obj[group]
        if group.name not in obj._indexes and group.name in obj.dims:
            # DummyGroups should not appear on groupby results
            newgroup = _DummyGroup(obj, group.name, group.coords)
        else:
            newgroup = group

    if newgroup.size == 0:
        raise ValueError(f"{newgroup.name} must not be empty")

    return newgroup


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
        "groupers",
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
    groupers: tuple[ResolvedGrouper]
    _squeeze: bool
    _restore_coord_dims: bool

    _original_obj: T_Xarray
    _original_group: T_Group
    _group_indices: T_GroupIndices
    _codes: DataArray
    _group_dim: Hashable

    _groups: dict[GroupKey, GroupIndex] | None
    _dims: tuple[Hashable, ...] | Frozen[Hashable, int] | None
    _sizes: Mapping[Hashable, int] | None

    def __init__(
        self,
        obj: T_Xarray,
        groupers: tuple[ResolvedGrouper],
        squeeze: bool = False,
        restore_coord_dims: bool = True,
    ) -> None:
        """Create a GroupBy object

        Parameters
        ----------
        obj : Dataset or DataArray
            Object to group.
        grouper : Grouper
            Grouper object
        restore_coord_dims : bool, default: True
            If True, also restore the dimension order of multi-dimensional
            coordinates.
        """
        self.groupers = groupers

        self._original_obj = obj

        for grouper_ in self.groupers:
            grouper_.factorize(squeeze)

        (grouper,) = self.groupers
        self._original_group = grouper.group

        # specification for the groupby operation
        self._obj = grouper.stacked_obj
        self._restore_coord_dims = restore_coord_dims
        self._squeeze = squeeze

        # These should generalize to multiple groupers
        self._group_indices = grouper.group_indices
        self._codes = self._maybe_unstack(grouper.codes)

        (self._group_dim,) = grouper.group1d.dims
        # cached attributes
        self._groups = None
        self._dims = None
        self._sizes = None

    @property
    def sizes(self) -> Mapping[Hashable, int]:
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
    def groups(self) -> dict[GroupKey, GroupIndex]:
        """
        Mapping from group labels to indices. The indices can be used to index the underlying object.
        """
        # provided to mimic pandas.groupby
        if self._groups is None:
            (grouper,) = self.groupers
            self._groups = dict(zip(grouper.unique_coord.values, self._group_indices))
        return self._groups

    def __getitem__(self, key: GroupKey) -> T_Xarray:
        """
        Get DataArray or Dataset corresponding to a particular group label.
        """
        return self._obj.isel({self._group_dim: self.groups[key]})

    def __len__(self) -> int:
        (grouper,) = self.groupers
        return grouper.size

    def __iter__(self) -> Iterator[tuple[GroupKey, T_Xarray]]:
        (grouper,) = self.groupers
        return zip(grouper.unique_coord.data, self._iter_grouped())

    def __repr__(self) -> str:
        (grouper,) = self.groupers
        return "{}, grouped over {!r}\n{!r} groups with labels {}.".format(
            self.__class__.__name__,
            grouper.name,
            grouper.full_index.size,
            ", ".join(format_array_flat(grouper.full_index, 30).split()),
        )

    def _iter_grouped(self) -> Iterator[T_Xarray]:
        """Iterate over each element in this group"""
        for indices in self._group_indices:
            yield self._obj.isel({self._group_dim: indices})

    def _infer_concat_args(self, applied_example):
        (grouper,) = self.groupers
        if self._group_dim in applied_example.dims:
            coord = grouper.group1d
            positions = self._group_indices
        else:
            coord = grouper.unique_coord
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

        (grouper,) = self.groupers
        obj = self._original_obj
        group = grouper.group
        codes = self._codes
        dims = group.dims

        if isinstance(group, _DummyGroup):
            group = coord = group.to_dataarray()
        else:
            coord = grouper.unique_coord
            if not isinstance(coord, DataArray):
                coord = DataArray(grouper.unique_coord)
        name = grouper.name

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
            group = group.where(~mask, drop=True)
            codes = codes.where(~mask, drop=True).astype(int)

        # if other is dask-backed, that's a hint that the
        # "expanded" dataset is too big to hold in memory.
        # this can be the case when `other` was read from disk
        # and contains our lazy indexing classes
        # We need to check for dask-backed Datasets
        # so utils.is_duck_dask_array does not work for this check
        if obj.chunks and not other.chunks:
            # TODO: What about datasets with some dask vars, and others not?
            # This handles dims other than `name``
            chunks = {k: v for k, v in obj.chunksizes.items() if k in other.dims}
            # a chunk size of 1 seems reasonable since we expect individual elements of
            # other to be repeated multiple times across the reduced dimension(s)
            chunks[name] = 1
            other = other.chunk(chunks)

        # codes are defined for coord, so we align `other` with `coord`
        # before indexing
        other, _ = align(other, coord, join="right", copy=False)
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
        """Our index contained empty groups (e.g., from a resampling or binning). If we
        reduced on that dimension, we want to restore the full index.
        """
        (grouper,) = self.groupers
        if (
            isinstance(grouper, (ResolvedBinGrouper, ResolvedTimeResampleGrouper))
            and grouper.name in combined.dims
        ):
            indexers = {grouper.name: grouper.full_index}
            combined = combined.reindex(**indexers)
        return combined

    def _maybe_unstack(self, obj):
        """This gets called if we are applying on an array with a
        multidimensional group."""
        (grouper,) = self.groupers
        stacked_dim = grouper.stacked_dim
        inserted_dims = grouper.inserted_dims
        if stacked_dim is not None and stacked_dim in obj.dims:
            obj = obj.unstack(stacked_dim)
            for dim in inserted_dims:
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
        (grouper,) = self.groupers
        isbin = isinstance(grouper, ResolvedBinGrouper)

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

        if "min_count" in kwargs:
            if kwargs["func"] not in ["sum", "prod"]:
                raise TypeError("Received an unexpected keyword argument 'min_count'")
            elif kwargs["min_count"] is None:
                # set explicitly to avoid unncessarily accumulating count
                kwargs["min_count"] = 0

        # weird backcompat
        # reducing along a unique indexed dimension with squeeze=True
        # should raise an error
        if (dim is None or dim == grouper.name) and grouper.name in obj.xindexes:
            index = obj.indexes[grouper.name]
            if index.is_unique and self._squeeze:
                raise ValueError(f"cannot reduce over dimensions {grouper.name!r}")

        unindexed_dims: tuple[Hashable, ...] = tuple()
        if isinstance(grouper.group, _DummyGroup) and not isbin:
            unindexed_dims = (grouper.name,)

        parsed_dim: tuple[Hashable, ...]
        if isinstance(dim, str):
            parsed_dim = (dim,)
        elif dim is None:
            parsed_dim = grouper.group.dims
        elif dim is ...:
            parsed_dim = tuple(obj.dims)
        else:
            parsed_dim = tuple(dim)

        # Do this so we raise the same error message whether flox is present or not.
        # Better to control it here than in flox.
        if any(d not in grouper.group.dims and d not in obj.dims for d in parsed_dim):
            raise ValueError(f"cannot reduce over dimensions {dim}.")

        if kwargs["func"] not in ["all", "any", "count"]:
            kwargs.setdefault("fill_value", np.nan)
        if isbin and kwargs["func"] == "count":
            # This is an annoying hack. Xarray returns np.nan
            # when there are no observations in a bin, instead of 0.
            # We can fake that here by forcing min_count=1.
            # note min_count makes no sense in the xarray world
            # as a kwarg for count, so this should be OK
            kwargs.setdefault("fill_value", np.nan)
            kwargs.setdefault("min_count", 1)

        output_index = grouper.full_index
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
        group_dims = grouper.group.dims
        if set(group_dims).issubset(set(parsed_dim)):
            result[grouper.name] = output_index
            result = result.drop_vars(unindexed_dims)

        # broadcast and restore non-numeric data variables (backcompat)
        for name, var in non_numeric.items():
            if all(d not in var.dims for d in parsed_dim):
                result[name] = var.variable.set_dims(
                    (grouper.name,) + var.dims,
                    (result.sizes[grouper.name],) + var.shape,
                )

        if isbin:
            # Fix dimension order when binning a dimension coordinate
            # Needed as long as we do a separate code path for pint;
            # For some reason Datasets and DataArrays behave differently!
            (group_dim,) = grouper.dims
            if isinstance(self._obj, Dataset) and group_dim in self._obj.dims:
                result = result.transpose(grouper.name, ...)

        return result

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

    @_deprecate_positional_args("v2023.10.0")
    def quantile(
        self,
        q: ArrayLike,
        dim: Dims = None,
        *,
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

                1. "inverted_cdf"
                2. "averaged_inverted_cdf"
                3. "closest_observation"
                4. "interpolated_inverted_cdf"
                5. "hazen"
                6. "weibull"
                7. "linear"  (default)
                8. "median_unbiased"
                9. "normal_unbiased"

            The first three methods are discontiuous.  The following discontinuous
            variations of the default "linear" (7.) option are also available:

                * "lower"
                * "higher"
                * "midpoint"
                * "nearest"

            See :py:func:`numpy.quantile` or [1]_ for details. The "method" argument
            was previously called "interpolation", renamed in accordance with numpy
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
            (grouper,) = self.groupers
            dim = grouper.group1d.dims

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

        (grouper,) = self.groupers
        reordered = _maybe_reorder(stacked, dim, positions, N=grouper.group.size)
        return self._obj._replace_maybe_drop_dims(reordered)

    def _restore_dim_order(self, stacked: DataArray) -> DataArray:
        (grouper,) = self.groupers
        group = grouper.group1d

        def lookup_order(dimension):
            if dimension == group.name:
                (dimension,) = group.dims
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
            (grouper,) = self.groupers
            combined = _maybe_reorder(combined, dim, positions, N=grouper.group.size)

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
        (grouper,) = self.groupers
        combined = _maybe_reorder(combined, dim, positions, N=grouper.group.size)
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
