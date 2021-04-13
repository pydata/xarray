import numpy as np

from ..core import indexing
from ..core.utils import Frozen, FrozenDict, close_on_error
from ..core.variable import Variable
from .common import (
    BACKEND_ENTRYPOINTS,
    AbstractDataStore,
    BackendArray,
    BackendEntrypoint,
    _normalize_path,
)
from .file_manager import CachingFileManager
from .locks import HDF5_LOCK, NETCDFC_LOCK, combine_locks, ensure_lock
from .store import StoreBackendEntrypoint

try:
    from PseudoNetCDF import pncopen

    has_pseudonetcdf = True
except ModuleNotFoundError:
    has_pseudonetcdf = False


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
            key, self.shape, indexing.IndexingSupport.OUTER_1VECTOR, self._getitem
        )

    def _getitem(self, key):
        with self.datastore.lock:
            array = self.get_array(needs_lock=False)
            return array[key]


class PseudoNetCDFDataStore(AbstractDataStore):
    """Store for accessing datasets via PseudoNetCDF"""

    @classmethod
    def open(cls, filename, lock=None, mode=None, **format_kwargs):

        keywords = {"kwargs": format_kwargs}
        # only include mode if explicitly passed
        if mode is not None:
            keywords["mode"] = mode

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
        data = indexing.LazilyIndexedArray(PncArrayWrapper(name, self))
        attrs = {k: getattr(var, k) for k in var.ncattrs()}
        return Variable(var.dimensions, data, attrs)

    def get_variables(self):
        return FrozenDict(
            (k, self.open_store_variable(k, v)) for k, v in self.ds.variables.items()
        )

    def get_attrs(self):
        return Frozen({k: getattr(self.ds, k) for k in self.ds.ncattrs()})

    def get_dimensions(self):
        return Frozen(self.ds.dimensions)

    def get_encoding(self):
        return {
            "unlimited_dims": {
                k for k in self.ds.dimensions if self.ds.dimensions[k].isunlimited()
            }
        }

    def close(self):
        self._manager.close()


class PseudoNetCDFBackendEntrypoint(BackendEntrypoint):

    # *args and **kwargs are not allowed in open_backend_dataset_ kwargs,
    # unless the open_dataset_parameters are explicity defined like this:
    open_dataset_parameters = (
        "filename_or_obj",
        "mask_and_scale",
        "decode_times",
        "concat_characters",
        "decode_coords",
        "drop_variables",
        "use_cftime",
        "decode_timedelta",
        "mode",
        "lock",
    )

    def open_dataset(
        self,
        filename_or_obj,
        mask_and_scale=False,
        decode_times=True,
        concat_characters=True,
        decode_coords=True,
        drop_variables=None,
        use_cftime=None,
        decode_timedelta=None,
        mode=None,
        lock=None,
        **format_kwargs,
    ):

        filename_or_obj = _normalize_path(filename_or_obj)
        store = PseudoNetCDFDataStore.open(
            filename_or_obj, lock=lock, mode=mode, **format_kwargs
        )

        store_entrypoint = StoreBackendEntrypoint()
        with close_on_error(store):
            ds = store_entrypoint.open_dataset(
                store,
                mask_and_scale=mask_and_scale,
                decode_times=decode_times,
                concat_characters=concat_characters,
                decode_coords=decode_coords,
                drop_variables=drop_variables,
                use_cftime=use_cftime,
                decode_timedelta=decode_timedelta,
            )
        return ds


if has_pseudonetcdf:
    BACKEND_ENTRYPOINTS["pseudonetcdf"] = PseudoNetCDFBackendEntrypoint
