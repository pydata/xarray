from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np

from .. import Variable
from ..core.utils import FrozenOrderedDict, Frozen, NDArrayMixin
from ..core import indexing

from .common import AbstractDataStore, DataStorePickleMixin


class NioArrayWrapper(NDArrayMixin):

    def __init__(self, variable_name, datastore):
        self.datastore = datastore
        self.variable_name = variable_name

    @property
    def array(self):
        return self.datastore.ds.variables[self.variable_name]

    @property
    def dtype(self):
        return np.dtype(self.array.typecode())

    def __getitem__(self, key):
        if key == () and self.ndim == 0:
            return self.array.get_value()
        return self.array[key]


class NioDataStore(AbstractDataStore, DataStorePickleMixin):
    """Store for accessing datasets via PyNIO
    """
    def __init__(self, filename, mode='r'):
        import Nio
        opener = functools.partial(Nio.open_file, filename, mode=mode)
        self.ds = opener()
        self._opener = opener
        self._mode = mode

    def open_store_variable(self, name, var):
        data = indexing.LazilyIndexedArray(NioArrayWrapper(name, self))
        return Variable(var.dimensions, data, var.attributes)

    def get_variables(self):
        return FrozenOrderedDict((k, self.open_store_variable(k, v))
                                 for k, v in self.ds.variables.iteritems())

    def get_attrs(self):
        return Frozen(self.ds.attributes)

    def get_dimensions(self):
        return Frozen(self.ds.dimensions)

    def close(self):
        self.ds.close()
