from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

from .. import Variable
from ..core.utils import FrozenOrderedDict, Frozen
from ..core import indexing
from ..core.pycompat import integer_types

from .common import AbstractDataStore, BackendArray, robust_getitem


class PydapArrayWrapper(BackendArray):
    def __init__(self, array):
        self.array = array

    @property
    def dtype(self):
        t = self.array.type
        if t.size is None and t.typecode == 'S':
            # return object dtype because that's the only way in numpy to
            # represent variable length strings; it also prevents automatic
            # string concatenation via conventions.decode_cf_variable
            return np.dtype('O')
        else:
            return np.dtype(t.typecode + str(t.size))

    def __getitem__(self, key):
        key = indexing.unwrap_explicit_indexer(
            key, target=self, allow=indexing.BasicIndexer)

        # pull the data from the array attribute if possible, to avoid
        # downloading coordinate data twice
        array = getattr(self.array, 'array', self.array)
        result = robust_getitem(array, key, catch=ValueError)
        # pydap doesn't squeeze axes automatically like numpy
        axis = tuple(n for n, k in enumerate(key)
                     if isinstance(k, integer_types))
        result = np.squeeze(result, axis)
        return result


def _fix_global_attributes(attributes):
    attributes = dict(attributes)
    for k in list(attributes):
        if k.lower() == 'global' or k.lower().endswith('_global'):
            # move global attributes to the top level, like the netcdf-C
            # DAP client
            attributes.update(attributes.pop(k))
    return attributes


class PydapDataStore(AbstractDataStore):
    """Store for accessing OpenDAP datasets with pydap.

    This store provides an alternative way to access OpenDAP datasets that may
    be useful if the netCDF4 library is not available.
    """
    def __init__(self, ds):
        """
        Parameters
        ----------
        ds : pydap DatasetType
        """
        self.ds = ds

    @classmethod
    def open(cls, url, session=None):
        import pydap.client
        ds = pydap.client.open_url(url, session=session)
        return cls(ds)

    def open_store_variable(self, var):
        data = indexing.LazilyIndexedArray(PydapArrayWrapper(var))
        return Variable(var.dimensions, data, var.attributes)

    def get_variables(self):
        return FrozenOrderedDict((k, self.open_store_variable(self.ds[k]))
                                 for k in self.ds.keys())

    def get_attrs(self):
        return Frozen(_fix_global_attributes(self.ds.attributes))

    def get_dimensions(self):
        return Frozen(self.ds.dimensions)
