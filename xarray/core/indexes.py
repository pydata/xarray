from __future__ import absolute_import, division, print_function
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping
from collections import OrderedDict

from . import formatting


class Indexes(Mapping, formatting.ReprMixin):
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

    def __unicode__(self):
        return formatting.indexes_repr(self)

# class Indexes(Mapping, formatting.ReprMixin):
#     """Ordered Mapping[str, pandas.Index] for xarray objects.
#     """
#
#     def __init__(self, variables, sizes):
#         """Not for public consumption.
#
#         Parameters
#         ----------
#         variables : OrderedDict[Any, Variable]
#             Reference to OrderedDict holding variable objects. Should be the
#             same dictionary used by the source object.
#         sizes : OrderedDict[Any, int]
#             Map from dimension names to sizes.
#         """
#         self._variables = variables
#         self._sizes = sizes
#
#     def _is_index_variable(self, key):
#         return (key in self._variables and key in self._sizes and
#                 isinstance(self._variables[key], IndexVariable))
#
#     def __iter__(self):
#         for key in self._sizes:
#             if self._is_index_variable(key):
#                 yield key
#
#     def __len__(self):
#         return sum(self._is_index_variable(key) for key in self._sizes)
#
#     def __contains__(self, key):
#         self._is_index_variable(key)
#
#     def __getitem__(self, key):
#         if not self._is_index_variable(key):
#             raise KeyError(key)
#         return self._variables[key].to_index()
#
#     def __unicode__(self):
#         return formatting.indexes_repr(self)


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
