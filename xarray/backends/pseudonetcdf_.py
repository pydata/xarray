from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np

from .. import Variable
from ..core.pycompat import OrderedDict
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
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.OUTER_1VECTOR,
            self._getitem)

    def _getitem(self, key):
        with self.datastore.ensure_open(autoclose=True):
            return self.get_array()[key]


class PseudoNetCDFDataStore(AbstractDataStore, DataStorePickleMixin):
    """Store for accessing datasets via PseudoNetCDF
    """
    @classmethod
    def open(cls, filename, format=None, writer=None,
             autoclose=False, **format_kwds):
        from PseudoNetCDF import pncopen
        opener = functools.partial(pncopen, filename, **format_kwds)
        ds = opener()
        mode = format_kwds.get('mode', 'r')
        return cls(ds, mode=mode, writer=writer, opener=opener,
                   autoclose=autoclose)

    def __init__(self, pnc_dataset, mode='r', writer=None, opener=None,
                 autoclose=False):

        if autoclose and opener is None:
            raise ValueError('autoclose requires an opener')

        self._ds = pnc_dataset
        self._autoclose = autoclose
        self._isopen = True
        self._opener = opener
        self._mode = mode
        super(PseudoNetCDFDataStore, self).__init__()

    def open_store_variable(self, name, var):
        with self.ensure_open(autoclose=False):
            data = indexing.LazilyOuterIndexedArray(
                PncArrayWrapper(name, self)
            )
        attrs = OrderedDict((k, getattr(var, k)) for k in var.ncattrs())
        return Variable(var.dimensions, data, attrs)

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
