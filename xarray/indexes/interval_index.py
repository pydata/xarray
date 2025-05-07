from collections.abc import Hashable

import numpy as np
import pandas as pd

from xarray import Variable
from xarray.core.indexes import Index, PandasIndex


class IntervalIndex(Index):
    def __init__(self, index: PandasIndex, bounds_name: Hashable, bounds_dim: str):
        assert isinstance(index.index, pd.IntervalIndex)
        self._index = index
        self._bounds_name = bounds_name
        self._bounds_dim = bounds_dim

    @classmethod
    def from_variables(cls, variables, options):
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

        return cls(index, bounds_name, bounds_dim)

    @classmethod
    def concat(cls, indexes, dim, positions=None):
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

    def create_variables(self, variables=None):
        empty_var = Variable((), 0)
        bounds_attrs = variables.get(self._bounds_name, empty_var).attrs
        mid_attrs = variables.get(self._index.dim, empty_var).attrs

        bounds_var = Variable(
            dims=(self._bounds_dim, self._index.dim),
            data=np.stack([self._index.index.left, self._index.index.right], axis=0),
            attrs=bounds_attrs,
        )
        mid_var = Variable(
            dims=(self._index.dim,),
            data=self._index.index.mid,
            attrs=mid_attrs,
        )

        return {self._index.dim: mid_var, self._bounds_name: bounds_var}

    def should_add_coord_to_array(self, name, var, dims):
        # add both the mid and boundary coordinates if the index dimension
        # is present in the array dimensions
        if self._index.dim in dims:
            return True
        else:
            return False

    def equals(self, other):
        if not isinstance(other, IntervalIndex):
            return False
        return self._index.equals(other._index, exclude_dims=frozenset())

    def sel(self, labels, **kwargs):
        return self._index.sel(labels, **kwargs)

    def isel(self, indexers):
        new_index = self._index.isel(indexers)
        if new_index is not None:
            return type(self)(new_index, self._bounds_name, self._bounds_dim)
        else:
            return None

    def roll(self, shifts):
        new_index = self._index.roll(shifts)
        return type(self)(new_index, self._bounds_name, self._bounds_dim)

    def rename(self, name_dict, dims_dict):
        new_index = self._index.rename(name_dict, dims_dict)

        bounds_name = name_dict.get(self._bounds_name, self._bounds_name)
        bounds_dim = dims_dict.get(self._bounds_dim, self._bounds_dim)

        return type(self)(new_index, bounds_name, bounds_dim)

    def __repr__(self):
        string = f"{self._index!r}"
        return string
