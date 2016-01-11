import numpy as np

from .. import Variable
from ..core.utils import FrozenOrderedDict, Frozen, NDArrayMixin
from ..core import indexing

from .common import AbstractDataStore


class NioArrayWrapper(NDArrayMixin):
    def __init__(self, array, ds):
        self.array = array
        self._ds = ds  # make an explicit reference because pynio uses weakrefs

    @property
    def dtype(self):
        return np.dtype(self.array.typecode())

    def __getitem__(self, key):
        if key == () and self.ndim == 0:
            return self.array.get_value()
        return self.array[key]


class NioDataStore(AbstractDataStore):
    """Store for accessing datasets via PyNIO
    """
    def __init__(self, filename, mode='r'):
        import Nio
        self.ds = Nio.open_file(filename, mode=mode)

    def open_store_variable(self, var):
        data = indexing.LazilyIndexedArray(NioArrayWrapper(var, self.ds))
        return Variable(var.dimensions, data, var.attributes)

    def get_variables(self):
        return FrozenOrderedDict((k, self.open_store_variable(v))
                                 for k, v in self.ds.variables.iteritems())

    def get_attrs(self):
        return Frozen(self.ds.attributes)

    def get_dimensions(self):
        return Frozen(self.ds.dimensions)

    def close(self):
        self.ds.close()
