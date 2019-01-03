from collections import OrderedDict

import numpy as np

from .. import Variable
from ..core import indexing
from ..core.utils import Frozen, FrozenOrderedDict
from .common import AbstractDataStore, BackendArray
from .file_manager import CachingFileManager
from .locks import HDF5_LOCK, NETCDFC_LOCK, combine_locks, ensure_lock

# psuedonetcdf can invoke netCDF libraries internally
PNETCDF_LOCK = combine_locks([HDF5_LOCK, NETCDFC_LOCK])


class PncArrayWrapper(BackendArray):

    def __init__(self, variable_name, datastore):
        self.datastore = datastore
        self.variable_name = variable_name
        array = self.get_array()
        self.shape = array.shape
        self.dtype = np.dtype(array.dtype)

    def get_array(self, needs_lock=True):
        ds = self.datastore._manager.acquire(needs_lock)
        return ds.variables[self.variable_name]

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.OUTER_1VECTOR,
            self._getitem)

    def _getitem(self, key):
        with self.datastore.lock:
            array = self.get_array(needs_lock=False)
            return array[key]


class PseudoNetCDFDataStore(AbstractDataStore):
    """Store for accessing datasets via PseudoNetCDF
    """
    @classmethod
    def open(cls, filename, lock=None, **format_kwds):
        from PseudoNetCDF import pncopen

        keywords = dict(kwargs=format_kwds)
        # only include mode if explicitly passed
        mode = format_kwds.pop('mode', None)
        if mode is not None:
            keywords['mode'] = mode

        if lock is None:
            lock = PNETCDF_LOCK

        manager = CachingFileManager(pncopen, filename, lock=lock, **keywords)
        return cls(manager, lock)

    def __init__(self, manager, lock=None):
        self._manager = manager
        self.lock = ensure_lock(lock)

    @property
    def ds(self):
        return self._manager.acquire()

    def open_store_variable(self, name, var):
        data = indexing.LazilyOuterIndexedArray(
            PncArrayWrapper(name, self)
        )
        attrs = OrderedDict((k, getattr(var, k)) for k in var.ncattrs())
        return Variable(var.dimensions, data, attrs)

    def get_variables(self):
        return FrozenOrderedDict((k, self.open_store_variable(k, v))
                                 for k, v in self.ds.variables.items())

    def get_attrs(self):
        return Frozen(dict([(k, getattr(self.ds, k))
                            for k in self.ds.ncattrs()]))

    def get_dimensions(self):
        return Frozen(self.ds.dimensions)

    def get_encoding(self):
        encoding = {}
        encoding['unlimited_dims'] = set(
            [k for k in self.ds.dimensions
             if self.ds.dimensions[k].isunlimited()])
        return encoding

    def close(self):
        self._manager.close()
