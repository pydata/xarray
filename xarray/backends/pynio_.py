from __future__ import absolute_import, division, print_function

import functools

import numpy as np

from .. import Variable
from ..core import indexing
from ..core.utils import Frozen, FrozenOrderedDict
from .common import AbstractDataStore, BackendArray, DataStorePickleMixin


class NioArrayWrapper(BackendArray):

    def __init__(self, variable_name, datastore):
        self.datastore = datastore
        self.variable_name = variable_name
        array = self.get_array()
        self.shape = array.shape
        self.dtype = np.dtype(array.typecode())

    def get_array(self):
        self.datastore.assert_open()
        return self.datastore.ds.variables[self.variable_name]

    def __getitem__(self, key):
        key, np_inds = indexing.decompose_indexer(
            key, self.shape, indexing.IndexingSupport.BASIC)

        with self.datastore.ensure_open(autoclose=True):
            array = self.get_array()
            if key.tuple == () and self.ndim == 0:
                return array.get_value()

            array = array[key.tuple]
            if len(np_inds.tuple) > 0:
                array = indexing.NumpyIndexingAdapter(array)[np_inds]

            return array


class NioDataStore(AbstractDataStore, DataStorePickleMixin):
    """Store for accessing datasets via PyNIO
    """

    def __init__(self, filename, mode='r', autoclose=False):
        import Nio
        opener = functools.partial(Nio.open_file, filename, mode=mode)
        self._ds = opener()
        self._autoclose = autoclose
        self._isopen = True
        self._opener = opener
        self._mode = mode
        # xarray provides its own support for FillValue,
        # so turn off PyNIO's support for the same.
        self.ds.set_option('MaskedArrayMode', 'MaskedNever')

    def open_store_variable(self, name, var):
        data = indexing.LazilyOuterIndexedArray(NioArrayWrapper(name, self))
        return Variable(var.dimensions, data, var.attributes)

    def get_variables(self):
        with self.ensure_open(autoclose=False):
            return FrozenOrderedDict((k, self.open_store_variable(k, v))
                                     for k, v in self.ds.variables.items())

    def get_attrs(self):
        with self.ensure_open(autoclose=True):
            return Frozen(self.ds.attributes)

    def get_dimensions(self):
        with self.ensure_open(autoclose=True):
            return Frozen(self.ds.dimensions)

    def get_encoding(self):
        encoding = {}
        encoding['unlimited_dims'] = set(
            [k for k in self.ds.dimensions if self.ds.unlimited(k)])
        return encoding

    def close(self):
        if self._isopen:
            self.ds.close()
            self._isopen = False
