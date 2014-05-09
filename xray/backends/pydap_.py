import numpy as np

import xray
from xray.utils import FrozenOrderedDict, Frozen, NDArrayMixin
from xray import indexing

from .common import AbstractDataStore


class PydapArrayWrapper(NDArrayMixin):
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
        if not isinstance(key, tuple):
            key = (key,)
        for k in key:
            if not (isinstance(k, int)
                    or isinstance(k, slice)
                    or k is Ellipsis):
                raise IndexError('pydap only supports indexing with int, '
                                 'slice and Ellipsis objects')
        # pull the data from the array attribute if possible, to avoid
        # downloading coordinate data twice
        return getattr(self.array, 'array', self.array)[key]


class PydapDataStore(AbstractDataStore):
    """Store for accessing OpenDAP datasets with pydap.

    This store provides an alternative way to access OpenDAP datasets that may
    be useful if the netCDF4 library is not available.
    """
    def __init__(self, url):
        import pydap.client
        self.ds = pydap.client.open_url(url)

    def open_store_variable(self, var):
        data = indexing.LazilyIndexedArray(PydapArrayWrapper(var))
        return xray.Variable(var.dimensions, data, var.attributes)

    @property
    def store_variables(self):
        return self.ds

    @property
    def attrs(self):
        return Frozen(self.ds.attributes)
