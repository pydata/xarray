from collections.abc import Mapping
from collections import OrderedDict

from . import formatting


class Indexes(Mapping):
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


def default_indexes(coords, dims):
    """Default indexes for a Dataset/DataArray.

    Parameters
    ----------
    coords : Mapping[Any, xarray.Variable]
       Coordinate variables from which to draw default indexes.
    dims : iterable
        Iterable of dimension names.

    Returns
    -------
    Mapping[Any, pandas.Index] mapping indexing keys (levels/dimension names)
    to indexes used for indexing along that dimension.
    """
    return OrderedDict((key, coords[key].to_index())
                       for key in dims if key in coords)
