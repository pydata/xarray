from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from xarray import Variable
from xarray.core.indexes import Index, PandasIndex
from xarray.core.indexing import IndexSelResult

if TYPE_CHECKING:
    from xarray.core.types import Self


class IntervalIndex(Index):
    """Xarray index of 1-dimensional intervals.

    This index is associated with two coordinate variables:

    - a 1-dimensional coordinate where each label represents an interval that is
      materialized by a central value (commonly the average of its left and right
      boundaries)

    - a 2-dimensional coordinate that represents the left and right boundaries
      of each interval. One of the two dimensions is shared with the
      aforementioned coordinate and the other one has length 2

    Interval boundaries are wrapped in a :py:class:`pandas.IntervalIndex` and
    central values are wrapped in a separate :py:class:`pandas.Index`.

    """

    _mid_index: PandasIndex
    _bounds_index: PandasIndex
    _bounds_dim: str

    def __init__(
        self,
        mid_index: PandasIndex,
        bounds_index: PandasIndex,
        bounds_dim: str | None = None,
    ):
        assert isinstance(bounds_index.index, pd.IntervalIndex)
        self._mid_index = mid_index
        self._bounds_index = bounds_index

        if bounds_dim is None:
            bounds_dim = "bounds"
        self._bounds_dim = bounds_dim

    @classmethod
    def from_variables(
        cls,
        variables: Mapping[Any, Variable],
        *,
        options: Mapping[str, Any],
    ) -> Self:
        if len(variables) == 2:
            mid_var: Variable | None = None
            bounds_var: Variable | None = None

            for name, var in variables.items():
                if var.ndim == 1:
                    mid_name = name
                    mid_var = var
                elif var.ndim == 2:
                    bounds_name = name
                    bounds_var = var

            if mid_var is None or bounds_var is None:
                raise ValueError(
                    "invalid coordinates given to IntervalIndex. When two coordinates are given, "
                    "one must be 1-dimensional (central values) and the other must be "
                    "2-dimensional (boundaries). Actual coordinate variables:\n"
                    + "\n".join(variables.values())
                )

            if mid_var.dims[0] == bounds_var.dims[0]:
                dim, bounds_dim = bounds_var.dims
            elif mid_var.dims[0] == bounds_var.dims[1]:
                bounds_dim, dim = bounds_var.dims
            else:
                raise ValueError(
                    "dimension names mismatch between "
                    f"the central coordinate {mid_name!r} {mid_var.dims!r} and "
                    f"the boundary coordinate {bounds_name!r} {bounds_var.dims!r} "
                    "given to IntervalIndex"
                )

            if bounds_var.sizes[bounds_dim] != 2:
                raise ValueError(
                    f"invalid shape for the boundary coordinate given to IntervalIndex (expected dimension {bounds_dim!r} of size 2)"
                )

            pd_mid_index = pd.Index(mid_var.values, name=mid_name)
            mid_index = PandasIndex(pd_mid_index, dim, coord_dtype=mid_var.dtype)

            left, right = bounds_var.transpose(..., dim).values.tolist()
            # TODO: make closed configurable
            pd_bounds_index = pd.IntervalIndex.from_arrays(
                left, right, name=bounds_name
            )
            bounds_index = PandasIndex(
                pd_bounds_index, dim, coord_dtype=bounds_var.dtype
            )

            actual_indexer = pd_bounds_index.get_indexer(pd_mid_index)
            expected_indexer = np.arange(pd_mid_index.size)
            if not np.array_equal(actual_indexer, expected_indexer):
                raise ValueError(
                    "invalid coordinates given to IntervalIndex. Not all central values are "
                    "in their corresponding interval"
                )

        elif len(variables) == 1:
            # TODO: allow setting the index from one variable? Perhaps in this fallback order:
            # - check if the coordinate wraps a pd.IntervalIndex
            # - look after the CF `bounds` attribute
            # - guess bounds like cf_xarray's add_bounds
            raise ValueError(
                "Setting an IntervalIndex from one coordinate is not yet supported"
            )
        else:
            raise ValueError("Too many coordinate variables given to IntervalIndex")

        return cls(mid_index, bounds_index, bounds_dim=str(bounds_dim))

    @classmethod
    def concat(
        cls,
        indexes: Sequence[IntervalIndex],
        dim: Hashable,
        positions: Iterable[Iterable[int]] | None = None,
    ) -> IntervalIndex:
        new_mid_index = PandasIndex.concat(
            [idx._mid_index for idx in indexes], dim, positions=positions
        )
        new_bounds_index = PandasIndex.concat(
            [idx._bounds_index for idx in indexes], dim, positions=positions
        )

        if indexes:
            bounds_dim = indexes[0]._bounds_dim
            # TODO: check whether this may actually happen or concat fails early during alignment
            if any(idx._bounds_dim != bounds_dim for idx in indexes):
                raise ValueError(
                    f"Cannot concatenate along dimension {dim!r} indexes with different "
                    "boundary coordinate or dimension names"
                )
        else:
            bounds_dim = "bounds"

        return cls(new_mid_index, new_bounds_index, bounds_dim)

    @property
    def dim(self) -> Hashable:
        return self._bounds_index.dim

    @property
    def bounds_dim(self) -> Hashable:
        return self._bounds_dim

    def create_variables(
        self, variables: Mapping[Any, Variable] | None = None
    ) -> dict[Any, Variable]:
        new_variables = self._mid_index.create_variables(variables)

        # boundary variable (we cannot just defer to self._bounds_index.create_variables())
        bounds_pd_index = cast(pd.IntervalIndex, self._bounds_index.index)
        bounds_varname = bounds_pd_index.name
        attrs: Mapping[Hashable, Any] | None
        encoding: Mapping[Hashable, Any] | None

        if variables is not None and bounds_varname in variables:
            var = variables[bounds_varname]
            attrs = var.attrs
            encoding = var.encoding
        else:
            attrs = None
            encoding = None

        # TODO: wrap index data in a PandasIndexingAdapter subclass instead
        # TODO: do we want to preserve the original dimension order for the boundary coordinate?
        # (using CF-compliant order below)
        data = np.stack([bounds_pd_index.left, bounds_pd_index.right], axis=-1)
        new_variables[bounds_varname] = Variable(
            (self.dim, self.bounds_dim), data, attrs=attrs, encoding=encoding
        )

        return new_variables

    def should_add_coord_to_array(
        self,
        name: Hashable,
        var: Variable,
        dims: set[Hashable],
    ) -> bool:
        # add both the central and boundary coordinates if the dimension
        # that they both share is present in the array dimensions
        return self.dim in dims

    def equals(self, other: Index) -> bool:
        if not isinstance(other, IntervalIndex):
            return False
        return self._mid_index.equals(other._mid_index) and self._bounds_index.equals(
            other._bounds_index
        )

    def sel(self, labels: dict[Any, Any], **kwargs) -> IndexSelResult:
        return self._bounds_index.sel(labels, **kwargs)

    def isel(
        self, indexers: Mapping[Any, int | slice | np.ndarray | Variable]
    ) -> Self | None:
        new_mid_index = self._mid_index.isel(indexers)
        new_bounds_index = self._bounds_index.isel(indexers)

        if new_mid_index is None or new_bounds_index is None:
            return None
        else:
            return type(self)(new_mid_index, new_bounds_index, self.bounds_dim)

    def roll(self, shifts: Mapping[Any, int]) -> Self | None:
        new_mid_index = self._mid_index.roll(shifts)
        new_bounds_index = self._bounds_index.roll(shifts)

        return type(self)(new_mid_index, new_bounds_index, self._bounds_dim)

    def rename(
        self,
        name_dict: Mapping[Any, Hashable],
        dims_dict: Mapping[Any, Hashable],
    ) -> Self:
        new_mid_index = self._mid_index.rename(name_dict, dims_dict)
        new_bounds_index = self._bounds_index.rename(name_dict, dims_dict)

        bounds_dim = dims_dict.get(self.bounds_dim, self.bounds_dim)

        return type(self)(new_mid_index, new_bounds_index, str(bounds_dim))

    def __repr__(self) -> str:
        text = "IntervalIndex\n"
        text += f"- central values:\n{self._mid_index!r}\n"
        text += f"- boundaries:\n{self._bounds_index!r}\n"
        return text
