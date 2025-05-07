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

    This index is built on top of :py:class:`~xarray.indexes.PandasIndex` and
    wraps a :py:class:`pandas.IntervalIndex`. It is associated with two
    coordinate variables:

    - a 1-dimensional coordinate where each label represents an interval that is
      materialized by its midpoint (i.e., the average of its left and right
      boundaries)

    - a 2-dimensional coordinate that represents the left and right boundaries
      of each interval. One of the two dimensions is shared with the
      aforementioned coordinate and the other one has length 2.

    """

    _index: PandasIndex
    _bounds_name: Hashable
    _bounds_dim: str

    def __init__(self, index: PandasIndex, bounds_name: Hashable, bounds_dim: str):
        assert isinstance(index.index, pd.IntervalIndex)
        self._index = index
        self._bounds_name = bounds_name
        self._bounds_dim = bounds_dim

    @classmethod
    def from_variables(
        cls,
        variables: Mapping[Any, Variable],
        *,
        options: Mapping[str, Any],
    ) -> Self:
        assert len(variables) == 2

        for k, v in variables.items():
            if v.ndim == 2:
                bounds_name, bounds = k, v
            elif v.ndim == 1:
                dim, _ = k, v

        bounds = bounds.transpose(..., dim)
        left, right = bounds.data.tolist()
        index = PandasIndex(pd.IntervalIndex.from_arrays(left, right), dim)
        bounds_dim = (set(bounds.dims) - set(dim)).pop()

        return cls(index, bounds_name, str(bounds_dim))

    @classmethod
    def concat(
        cls,
        indexes: Sequence[IntervalIndex],
        dim: Hashable,
        positions: Iterable[Iterable[int]] | None = None,
    ) -> IntervalIndex:
        new_index = PandasIndex.concat(
            [idx._index for idx in indexes], dim, positions=positions
        )

        if indexes:
            bounds_name = indexes[0]._bounds_name
            bounds_dim = indexes[0]._bounds_dim
            if any(
                idx._bounds_name != bounds_name or idx._bounds_dim != bounds_dim
                for idx in indexes
            ):
                raise ValueError(
                    f"Cannot concatenate along dimension {dim!r} indexes with different "
                    "boundary coordinate or dimension names"
                )
        else:
            bounds_name = new_index.index.name + "_bounds"
            bounds_dim = "bnd"

        return cls(new_index, bounds_name, bounds_dim)

    @property
    def _pd_index(self) -> pd.IntervalIndex:
        # For typing purpose only
        # TODO: cleaner to make PandasIndex a generic class, i.e., PandasIndex[pd.IntervalIndex]
        # will be easier once PEP 696 is fully supported (starting from Python 3.13)
        return cast(pd.IntervalIndex, self._index.index)

    def create_variables(
        self, variables: Mapping[Any, Variable] | None = None
    ) -> dict[Any, Variable]:
        if variables is None:
            variables = {}
        empty_var = Variable((), 0)
        bounds_attrs = variables.get(self._bounds_name, empty_var).attrs
        mid_attrs = variables.get(self._index.dim, empty_var).attrs

        bounds_var = Variable(
            dims=(self._bounds_dim, self._index.dim),
            data=np.stack([self._pd_index.left, self._pd_index.right], axis=0),
            attrs=bounds_attrs,
        )
        mid_var = Variable(
            dims=(self._index.dim,),
            data=self._pd_index.mid,
            attrs=mid_attrs,
        )

        return {self._index.dim: mid_var, self._bounds_name: bounds_var}

    def should_add_coord_to_array(
        self,
        name: Hashable,
        var: Variable,
        dims: set[Hashable],
    ) -> bool:
        # add both the mid and boundary coordinates if the index dimension
        # is present in the array dimensions
        if self._index.dim in dims:
            return True
        else:
            return False

    def to_pandas_index(self) -> pd.Index:
        return self._pd_index

    def equals(self, other: Index) -> bool:
        if not isinstance(other, IntervalIndex):
            return False
        return self._index.equals(other._index)

    def sel(self, labels: dict[Any, Any], **kwargs) -> IndexSelResult:
        return self._index.sel(labels, **kwargs)

    def isel(
        self, indexers: Mapping[Any, int | slice | np.ndarray | Variable]
    ) -> Self | None:
        new_index = self._index.isel(indexers)
        if new_index is not None:
            return type(self)(new_index, self._bounds_name, self._bounds_dim)
        else:
            return None

    def roll(self, shifts: Mapping[Any, int]) -> Self | None:
        new_index = self._index.roll(shifts)
        return type(self)(new_index, self._bounds_name, self._bounds_dim)

    def rename(
        self,
        name_dict: Mapping[Any, Hashable],
        dims_dict: Mapping[Any, Hashable],
    ) -> Self:
        new_index = self._index.rename(name_dict, dims_dict)

        bounds_name = name_dict.get(self._bounds_name, self._bounds_name)
        bounds_dim = dims_dict.get(self._bounds_dim, self._bounds_dim)

        return type(self)(new_index, bounds_name, str(bounds_dim))

    def __repr__(self) -> str:
        string = f"{self._index!r}"
        return string
