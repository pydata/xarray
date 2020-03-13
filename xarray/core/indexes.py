import collections.abc
from typing import Any, Dict, Hashable, Iterable, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from . import formatting
from .utils import is_scalar
from .variable import Variable


def remove_unused_levels_categories(index):
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
            index = pd.MultiIndex.from_arrays(levels, names=index.names)
    elif isinstance(index, pd.CategoricalIndex):
        index = index.remove_unused_categories()
    return index


class Indexes(collections.abc.Mapping):
    """Immutable proxy for Dataset or DataArrary indexes."""

    __slots__ = ("_indexes",)

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
    coords: Mapping[Any, Variable], dims: Iterable
) -> Dict[Hashable, pd.Index]:
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
    return {key: coords[key].to_index() for key in dims if key in coords}


def isel_variable_and_index(
    name: Hashable,
    variable: Variable,
    index: pd.Index,
    indexers: Mapping[Hashable, Union[int, slice, np.ndarray, Variable]],
) -> Tuple[Variable, Optional[pd.Index]]:
    """Index a Variable and pandas.Index together."""
    if not indexers:
        # nothing to index
        return variable.copy(deep=False), index

    if len(variable.dims) > 1:
        raise NotImplementedError(
            "indexing multi-dimensional variable with indexes is not " "supported yet"
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
    new_index = index[indexer]
    return new_variable, new_index


def roll_index(index: pd.Index, count: int, axis: int = 0) -> pd.Index:
    """Roll an pandas.Index."""
    count %= index.shape[0]
    if count != 0:
        return index[-count:].append(index[:-count])
    else:
        return index[:]


def propagate_indexes(
    indexes: Optional[Dict[Hashable, pd.Index]], exclude: Optional[Any] = None
) -> Optional[Dict[Hashable, pd.Index]]:
    """ Creates new indexes dict from existing dict optionally excluding some dimensions.
    """
    if exclude is None:
        exclude = ()

    if is_scalar(exclude):
        exclude = (exclude,)

    if indexes is not None:
        new_indexes = {k: v for k, v in indexes.items() if k not in exclude}
    else:
        new_indexes = None  # type: ignore

    return new_indexes
