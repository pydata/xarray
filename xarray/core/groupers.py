"""
This module provides Grouper objects that encapsulate the
"factorization" process - conversion of value we are grouping by
to integer codes (one per group).
"""

from __future__ import annotations

import datetime
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from xarray.coding.cftime_offsets import _new_to_legacy_freq
from xarray.core import duck_array_ops
from xarray.core.dataarray import DataArray
from xarray.core.groupby import T_Group, _DummyGroup
from xarray.core.indexes import safe_cast_to_index
from xarray.core.resample_cftime import CFTimeGrouper
from xarray.core.types import DatetimeLike, SideOptions, T_GroupIndices
from xarray.core.utils import emit_user_level_warning
from xarray.core.variable import Variable

__all__ = [
    "EncodedGroups",
    "Grouper",
    "Resampler",
    "UniqueGrouper",
    "BinGrouper",
    "TimeResampler",
]

RESAMPLE_DIM = "__resample_dim__"


@dataclass
class EncodedGroups:
    """
    Dataclass for storing intermediate values for GroupBy operation.
    Returned by factorize method on Grouper objects.

    Parameters
    ----------
    codes: integer codes for each group
    full_index: pandas Index for the group coordinate
    group_indices: optional, List of indices of array elements belonging
                   to each group. Inferred if not provided.
    unique_coord: Unique group values present in dataset. Inferred if not provided
    """

    codes: DataArray
    full_index: pd.Index
    group_indices: T_GroupIndices | None = field(default=None)
    unique_coord: Variable | _DummyGroup | None = field(default=None)

    def __post_init__(self):
        assert isinstance(self.codes, DataArray)
        if self.codes.name is None:
            raise ValueError("Please set a name on the array you are grouping by.")
        assert isinstance(self.full_index, pd.Index)
        assert (
            isinstance(self.unique_coord, (Variable, _DummyGroup))
            or self.unique_coord is None
        )


class Grouper(ABC):
    """Base class for Grouper objects that allow specializing GroupBy instructions."""

    @property
    def can_squeeze(self) -> bool:
        """TODO: delete this when the `squeeze` kwarg is deprecated. Only `UniqueGrouper`
        should override it."""
        return False

    @abstractmethod
    def factorize(self, group) -> EncodedGroups:
        """
        Takes the group, and creates intermediates necessary for GroupBy.
        These intermediates are
        1. codes - Same shape as `group` containing a unique integer code for each group.
        2. group_indices - Indexes that let us index out the members of each group.
        3. unique_coord - Unique groups present in the dataset.
        4. full_index - Unique groups in the output. This differs from `unique_coord` in the
           case of resampling and binning, where certain groups in the output are not present in
           the input.

        Returns an instance of EncodedGroups.
        """
        pass


class Resampler(Grouper):
    """Base class for Grouper objects that allow specializing resampling-type GroupBy instructions.
    Currently only used for TimeResampler, but could be used for SpaceResampler in the future.
    """

    pass


@dataclass
class UniqueGrouper(Grouper):
    """Grouper object for grouping by a categorical variable."""

    _group_as_index: pd.Index | None = None

    @property
    def is_unique_and_monotonic(self) -> bool:
        if isinstance(self.group, _DummyGroup):
            return True
        index = self.group_as_index
        return index.is_unique and index.is_monotonic_increasing

    @property
    def group_as_index(self) -> pd.Index:
        if self._group_as_index is None:
            self._group_as_index = self.group.to_index()
        return self._group_as_index

    @property
    def can_squeeze(self) -> bool:
        """This is a deprecated method and will be removed eventually."""
        is_dimension = self.group.dims == (self.group.name,)
        return is_dimension and self.is_unique_and_monotonic

    def factorize(self, group1d) -> EncodedGroups:
        self.group = group1d

        if self.can_squeeze:
            return self._factorize_dummy()
        else:
            return self._factorize_unique()

    def _factorize_unique(self) -> EncodedGroups:
        # look through group to find the unique values
        sort = not isinstance(self.group_as_index, pd.MultiIndex)
        unique_values, codes_ = unique_value_groups(self.group_as_index, sort=sort)
        if (codes_ == -1).all():
            raise ValueError(
                "Failed to group data. Are you grouping by a variable that is all NaN?"
            )
        codes = self.group.copy(data=codes_)
        unique_coord = Variable(
            dims=codes.name, data=unique_values, attrs=self.group.attrs
        )
        full_index = pd.Index(unique_values)

        return EncodedGroups(
            codes=codes, full_index=full_index, unique_coord=unique_coord
        )

    def _factorize_dummy(self) -> EncodedGroups:
        size = self.group.size
        # no need to factorize
        # use slices to do views instead of fancy indexing
        # equivalent to: group_indices = group_indices.reshape(-1, 1)
        group_indices: T_GroupIndices = [slice(i, i + 1) for i in range(size)]
        size_range = np.arange(size)
        if isinstance(self.group, _DummyGroup):
            codes = self.group.to_dataarray().copy(data=size_range)
            unique_coord = self.group
            full_index = pd.RangeIndex(self.group.size)
        else:
            codes = self.group.copy(data=size_range)
            unique_coord = self.group.variable.to_base_variable()
            full_index = pd.Index(unique_coord.data)

        return EncodedGroups(
            codes=codes,
            group_indices=group_indices,
            full_index=full_index,
            unique_coord=unique_coord,
        )


@dataclass
class BinGrouper(Grouper):
    """Grouper object for binning numeric data."""

    bins: int | Sequence | pd.IntervalIndex
    cut_kwargs: Mapping = field(default_factory=dict)
    binned: Any = None
    name: Any = None

    def __post_init__(self) -> None:
        if duck_array_ops.isnull(self.bins).all():
            raise ValueError("All bin edges are NaN.")

    def factorize(self, group) -> EncodedGroups:
        from xarray.core.dataarray import DataArray

        data = group.data

        binned, self.bins = pd.cut(data, self.bins, **self.cut_kwargs, retbins=True)

        binned_codes = binned.codes
        if (binned_codes == -1).all():
            raise ValueError(
                f"None of the data falls within bins with edges {self.bins!r}"
            )

        new_dim_name = f"{group.name}_bins"

        full_index = binned.categories
        uniques = np.sort(pd.unique(binned_codes))
        unique_values = full_index[uniques[uniques != -1]]

        codes = DataArray(
            binned_codes, getattr(group, "coords", None), name=new_dim_name
        )
        unique_coord = Variable(
            dims=new_dim_name, data=unique_values, attrs=group.attrs
        )
        return EncodedGroups(
            codes=codes, full_index=full_index, unique_coord=unique_coord
        )


@dataclass
class TimeResampler(Resampler):
    """Grouper object specialized to resampling the time coordinate."""

    freq: str
    closed: SideOptions | None = field(default=None)
    label: SideOptions | None = field(default=None)
    origin: str | DatetimeLike = field(default="start_day")
    offset: pd.Timedelta | datetime.timedelta | str | None = field(default=None)
    loffset: datetime.timedelta | str | None = field(default=None)
    base: int | None = field(default=None)

    index_grouper: CFTimeGrouper | pd.Grouper = field(init=False)
    group_as_index: pd.Index = field(init=False)

    def __post_init__(self):
        if self.loffset is not None:
            emit_user_level_warning(
                "Following pandas, the `loffset` parameter to resample is deprecated.  "
                "Switch to updating the resampled dataset time coordinate using "
                "time offset arithmetic.  For example:\n"
                "    >>> offset = pd.tseries.frequencies.to_offset(freq) / 2\n"
                '    >>> resampled_ds["time"] = resampled_ds.get_index("time") + offset',
                FutureWarning,
            )

        if self.base is not None:
            emit_user_level_warning(
                "Following pandas, the `base` parameter to resample will be deprecated in "
                "a future version of xarray.  Switch to using `origin` or `offset` instead.",
                FutureWarning,
            )

        if self.base is not None and self.offset is not None:
            raise ValueError("base and offset cannot be present at the same time")

    def _init_properties(self, group: T_Group) -> None:
        from xarray import CFTimeIndex
        from xarray.core.pdcompat import _convert_base_to_offset

        group_as_index = safe_cast_to_index(group)

        if self.base is not None:
            # grouper constructor verifies that grouper.offset is None at this point
            offset = _convert_base_to_offset(self.base, self.freq, group_as_index)
        else:
            offset = self.offset

        if not group_as_index.is_monotonic_increasing:
            # TODO: sort instead of raising an error
            raise ValueError("index must be monotonic for resampling")

        if isinstance(group_as_index, CFTimeIndex):
            from xarray.core.resample_cftime import CFTimeGrouper

            index_grouper = CFTimeGrouper(
                freq=self.freq,
                closed=self.closed,
                label=self.label,
                origin=self.origin,
                offset=offset,
                loffset=self.loffset,
            )
        else:
            index_grouper = pd.Grouper(
                # TODO remove once requiring pandas >= 2.2
                freq=_new_to_legacy_freq(self.freq),
                closed=self.closed,
                label=self.label,
                origin=self.origin,
                offset=offset,
            )
        self.index_grouper = index_grouper
        self.group_as_index = group_as_index

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
            if self.loffset is not None:
                _apply_loffset(self.loffset, first_items)
            return first_items, codes

    def factorize(self, group) -> EncodedGroups:
        self._init_properties(group)
        full_index, first_items, codes_ = self._get_index_and_items()
        sbins = first_items.values.astype(np.int64)
        group_indices: T_GroupIndices = [
            slice(i, j) for i, j in zip(sbins[:-1], sbins[1:])
        ]
        group_indices += [slice(sbins[-1], None)]

        unique_coord = Variable(
            dims=group.name, data=first_items.index, attrs=group.attrs
        )
        codes = group.copy(data=codes_)

        return EncodedGroups(
            codes=codes,
            group_indices=group_indices,
            full_index=full_index,
            unique_coord=unique_coord,
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


def unique_value_groups(
    ar, sort: bool = True
) -> tuple[np.ndarray | pd.Index, np.ndarray]:
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
    return values, inverse
