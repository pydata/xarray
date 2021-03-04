import functools
import operator
import os
import pathlib
from contextlib import suppress

import numpy as np

from ..coding.variables import pop_to
from ..core.utils import close_on_error, is_remote_uri
from .lazy_data_store import LazyDataStore
from .common import (
    BACKEND_ENTRYPOINTS,
    BackendEntrypoint,
    find_root_and_group,
    robust_getitem,
)
from .file_manager import CachingFileManager, DummyFileManager
from .locks import HDF5_LOCK, NETCDFC_LOCK, combine_locks, ensure_lock, get_write_lock
from .store import StoreBackendEntrypoint

try:
    import netCDF4

    has_netcdf4 = True
except ModuleNotFoundError:
    has_netcdf4 = False


# This lookup table maps from dtype.byteorder to a readable endian
# string used by netCDF4.
_endian_lookup = {"=": "native", ">": "big", "<": "little", "|": "native"}


NETCDF4_PYTHON_LOCK = combine_locks([NETCDFC_LOCK, HDF5_LOCK])


class NetCDF4BackendArray:
    def __init__(self, variable_name, datastore):
        self.datastore = datastore
        self.variable_name = variable_name

        array = self.get_array()
        self.shape = array.shape

        dtype = array.dtype
        if dtype is str:
            # use object dtype because that's the only way in numpy to
            # represent variable length strings; it also prevents automatic
            # string concatenation via conventions.decode_cf_variable
            dtype = np.dtype("O")
        self.dtype = dtype
        self.indexing_support = 1
        
    def get_array(self, needs_lock=True):
        ds = self.datastore._acquire(needs_lock)
        variable = ds.variables[self.variable_name]
        variable.set_auto_maskandscale(False)
        # only added in netCDF4-python v1.2.8
        with suppress(AttributeError):
            variable.set_auto_chartostring(False)
        return variable

    def __getitem__(self, key):
        if self.datastore.is_remote:  # pragma: no cover
            getitem = functools.partial(robust_getitem, catch=RuntimeError)
        else:
            getitem = operator.getitem

        try:
            with self.datastore.lock:
                original_array = self.get_array(needs_lock=False)
                array = getitem(original_array, key)
        except IndexError:
            # Catch IndexError in netCDF4 and return a more informative
            # error message.  This is most often called when an unsorted
            # indexer is used before the data is loaded from disk.
            msg = (
                "The indexing operation you are attempting to perform "
                "is not valid on netCDF4.Variable object. Try loading "
                "your data into memory first by calling .load()."
            )
            raise IndexError(msg)
        return array


class NetCDF4StoreVariable:
    def __init__(self, name, var, datastore):
        self.dimensions = var.dimensions
        self.data = NetCDF4BackendArray(name, datastore)
        self.attributes = {k: var.getncattr(k) for k in var.ncattrs()}
        if self.data.dtype.kind == "S" and "_FillValue" in self.attributes:
            self.attributes["_FillValue"] = np.string_(self.attributes["_FillValue"])
        # netCDF4 specific encoding; save _FillValue for later
        self.encoding = {}
        filters = var.filters()
        if filters is not None:
            self.encoding.update(filters)
        chunking = var.chunking()
        if chunking is not None:
            if chunking == "contiguous":
                self.encoding["contiguous"] = True
                self.encoding["chunksizes"] = None
            else:
                self.encoding["contiguous"] = False
                self.encoding["chunksizes"] = tuple(chunking)
        # TODO: figure out how to round-trip "endian-ness" without raising
        # warnings from netCDF4
        # self.encoding['endian'] = var.endian()
        pop_to(self.attributes, self.encoding, "least_significant_digit")
        # save source so __repr__ can detect if it's local or not
        self.encoding["source"] = datastore._filename
        self.encoding["original_shape"] = var.shape
        self.encoding["dtype"] = var.dtype


def _netcdf4_create_group(dataset, name):
    return dataset.createGroup(name)


def _nc4_require_group(ds, group, mode, create_group=_netcdf4_create_group):
    if group in {None, "", "/"}:
        # use the root group
        return ds
    else:
        # make sure it's a string
        if not isinstance(group, str):
            raise ValueError("group must be a string or None")
        # support path-like syntax
        path = group.strip("/").split("/")
        for key in path:
            try:
                ds = ds.groups[key]
            except KeyError as e:
                if mode != "r":
                    ds = create_group(ds, key)
                else:
                    # wrap error to provide slightly more helpful message
                    raise OSError("group not found: %s" % key, e)
        return ds


class ExampleDataStore:
    """Store for reading and writing data via the Python-NetCDF4 library.

    This store supports NetCDF3, NetCDF4 and OpenDAP datasets.
    """

    __slots__ = (
        "autoclose",
        "format",
        "is_remote",
        "lock",
        "_filename",
        "_group",
        "_manager",
        "_mode",
    )

    def __init__(
        self, manager, group=None, mode=None, lock=NETCDF4_PYTHON_LOCK, autoclose=False
    ):

        if isinstance(manager, netCDF4.Dataset):
            if group is None:
                root, group = find_root_and_group(manager)
            else:
                if not type(manager) is netCDF4.Dataset:
                    raise ValueError(
                        "must supply a root netCDF4.Dataset if the group "
                        "argument is provided"
                    )
                root = manager
            manager = DummyFileManager(root)

        self._manager = manager
        self._group = group
        self._mode = mode
        self.format = self.ds.data_model
        self._filename = self.ds.filepath()
        self.is_remote = is_remote_uri(self._filename)
        self.lock = ensure_lock(lock)
        self.autoclose = autoclose

    @classmethod
    def open(
        cls,
        filename,
        mode="r",
        format="NETCDF4",
        group=None,
        clobber=True,
        diskless=False,
        persist=False,
        lock=None,
        lock_maker=None,
        autoclose=False,
    ):

        if isinstance(filename, pathlib.Path):
            filename = os.fspath(filename)

        if not isinstance(filename, str):
            raise ValueError(
                "can only read bytes or file-like objects "
                "with engine='scipy' or 'h5netcdf'"
            )

        if format is None:
            format = "NETCDF4"

        if lock is None:
            if mode == "r":
                if is_remote_uri(filename):
                    lock = NETCDFC_LOCK
                else:
                    lock = NETCDF4_PYTHON_LOCK
            else:
                if format is None or format.startswith("NETCDF4"):
                    base_lock = NETCDF4_PYTHON_LOCK
                else:
                    base_lock = NETCDFC_LOCK
                lock = combine_locks([base_lock, get_write_lock(filename)])

        kwargs = dict(
            clobber=clobber, diskless=diskless, persist=persist, format=format
        )
        manager = CachingFileManager(
            netCDF4.Dataset, filename, mode=mode, kwargs=kwargs
        )
        return cls(manager, group=group, mode=mode, lock=lock, autoclose=autoclose)

    def _acquire(self, needs_lock=True):
        with self._manager.acquire_context(needs_lock) as root:
            ds = _nc4_require_group(root, self._group, self._mode)
        return ds

    @property
    def ds(self):
        return self._acquire()

    def get_variables(self):
        return {
            name: NetCDF4StoreVariable(name, var, self)
            for name, var in self.ds.variables.items()
        }

    def get_attrs(self):
        return {key: self.ds.getncattr(key) for key in self.ds.ncattrs()}

    def get_dimensions(self):
        return {name: len(dim) for name, dim in self.ds.dimensions.items()}

    def get_encoding(self):
        encoding = {}
        encoding["unlimited_dims"] = {
            k for k, v in self.ds.dimensions.items() if v.isunlimited()
        }
        return encoding

    def close(self, **kwargs):
        self._manager.close(**kwargs)


class LazyNetCDF4DataStore(LazyDataStore):

    def __init__(
        self,
        filename,
        mode="r",
        format="NETCDF4",
        group=None,
        clobber=True,
        diskless=False,
        persist=False,
        lock=None,
        lock_maker=None,
        autoclose=False,
    ):
        LazyDataStore.__init__(
            self,
            ExampleDataStore.open(
                filename,
                mode=mode,
                format=format,
                group=group,
                clobber=clobber,
                diskless=diskless,
                persist=persist,
                lock=lock,
                autoclose=autoclose,
            )
        )

        
class ExampleProtocolEntrypoint(BackendEntrypoint):
    def open_dataset(
        self,
        filename_or_obj,
        mask_and_scale=True,
        decode_times=True,
        concat_characters=True,
        decode_coords=True,
        drop_variables=None,
        use_cftime=None,
        decode_timedelta=None,
        group=None,
        mode="r",
        format="NETCDF4",
        clobber=True,
        diskless=False,
        persist=False,
        lock=None,
        autoclose=False,
    ):

        store = LazyNetCDF4DataStore(
            filename_or_obj,
            mode=mode,
            format=format,
            group=group,
            clobber=clobber,
            diskless=diskless,
            persist=persist,
            lock=lock,
            autoclose=autoclose,
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


if has_netcdf4:
    BACKEND_ENTRYPOINTS["protocol"] = ExampleProtocolEntrypoint
