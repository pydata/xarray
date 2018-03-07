from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np

from .. import Variable
from ..core.utils import (FrozenOrderedDict, Frozen)
from ..core import indexing

from .common import AbstractDataStore, DataStorePickleMixin, BackendArray


class PncArrayWrapper(BackendArray):

    def __init__(self, variable_name, datastore):
        self.datastore = datastore
        self.variable_name = variable_name
        array = self.get_array()
        self.shape = array.shape
        self.dtype = np.dtype(array.dtype)

    def get_array(self):
        self.datastore.assert_open()
        return self.datastore.ds.variables[self.variable_name]

    def __getitem__(self, key):
        key = indexing.unwrap_explicit_indexer(
            key, target=self, allow=indexing.BasicIndexer)

        with self.datastore.ensure_open(autoclose=True):
            array = self.get_array()
            if key == () and self.ndim == 0:
                return array[...]
            return array[key]


class PncDataStore(AbstractDataStore, DataStorePickleMixin):
    """Store for accessing datasets via PseudoNetCDF
    """

    def __init__(self, filename, mode='r', autoclose=False):
        from PseudoNetCDF import pncopen
        try:
            opener = functools.partial(pncopen, filename, mode=mode)
            self.ds = opener()
        except Exception:
            opener = functools.partial(pncopen, filename)
            self.ds = opener()
        self._autoclose = autoclose
        self._isopen = True
        self._opener = opener
        self._mode = mode

    def open_store_variable(self, name, var):
        data = indexing.LazilyIndexedArray(PncArrayWrapper(name, self))
        return Variable(var.dimensions, data, dict([(k, getattr(var, k))
                                                    for k in var.ncattrs()]))

    def get_variables(self):
        with self.ensure_open(autoclose=False):
            return FrozenOrderedDict((k, self.open_store_variable(k, v))
                                     for k, v in self.ds.variables.items())

    def get_attrs(self):
        with self.ensure_open(autoclose=True):
            return Frozen(dict([(k, getattr(self.ds, k))
                                for k in self.ds.ncattrs()]))

    def get_dimensions(self):
        with self.ensure_open(autoclose=True):
            return Frozen(self.ds.dimensions)

    def get_encoding(self):
        encoding = {}
        encoding['unlimited_dims'] = set(
            [k for k in self.ds.dimensions
             if self.ds.dimensions[k].isunlimited()])
        return encoding

    def close(self):
        if self._isopen:
            self.ds.close()
            self._isopen = False
