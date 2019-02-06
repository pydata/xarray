import collections.abc
from collections import OrderedDict
from typing import Any, Iterable, Mapping, Optional, Tuple, Union

import pandas as pd

from . import formatting
from .variable import Variable


class Indexes(collections.abc.Mapping):
    """Immutable proxy for Dataset or DataArrary indexes."""
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


def default_indexes(
    coords: Mapping[Any, Variable],
    dims: Iterable,
) -> 'OrderedDict[Any, pd.Index]':
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
    return OrderedDict((key, coords[key].to_index())
                       for key in dims if key in coords)


def isel_variable_and_index(
    variable: Variable,
    index: pd.Index,
    indexers: Mapping[Any, Union[slice, Variable]],
) -> Tuple[Variable, Optional[pd.Index]]:
    """Index a Variable and pandas.Index together."""
    if not indexers:
        # nothing to index
        return variable.copy(deep=False), index

    if len(variable.dims) > 1:
        raise NotImplementedError(
            'indexing multi-dimensional variable with indexes is not '
            'supported yet')

    new_variable = variable.isel(indexers)

    if new_variable.ndim != 1:
        # can't preserve a index if result is not 0D
        return new_variable, None

    # we need to compute the new index
    (dim,) = variable.dims
    indexer = indexers[dim]
    if isinstance(indexer, Variable):
        indexer = indexer.data
    new_index = index[indexer]
    return new_variable, new_index
