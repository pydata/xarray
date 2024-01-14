from __future__ import annotations

import functools
import operator
import os
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np

from xarray import coding
from xarray.backends.common import (
    BACKEND_ENTRYPOINTS,
    BackendArray,
    BackendEntrypoint,
    WritableCFDataStore,
    _normalize_path,
    find_root_and_group,
    robust_getitem,
)
from xarray.backends.file_manager import CachingFileManager, DummyFileManager
from xarray.backends.locks import (
    HDF5_LOCK,
    NETCDFC_LOCK,
    combine_locks,
    ensure_lock,
    get_write_lock,
)
from xarray.backends.netcdf3 import encode_nc3_attr_value, encode_nc3_variable
from xarray.backends.store import StoreBackendEntrypoint
from xarray.coding.variables import pop_to
from xarray.core import indexing
from xarray.core.types import Self
from xarray.core.utils import (
    FrozenDict,
    close_on_error,
    is_remote_uri,
    try_read_magic_number_from_path,
)
from xarray.core.variable import Variable

if TYPE_CHECKING:
    import netCDF4

    from xarray.backends.file_manager import FileManager
    from xarray.core.dataset import Dataset
    from xarray.core.types import (
        LockLike,
        NetcdfFormats,
        NetCDFOpenModes,
        T_XarrayCanOpen,
    )

# This lookup table maps from dtype.byteorder to a readable endian
# string used by netCDF4.
_endian_lookup = {"=": "native", ">": "big", "<": "little", "|": "native"}


NETCDF4_PYTHON_LOCK = combine_locks([NETCDFC_LOCK, HDF5_LOCK])


class BaseNetCDF4Array(BackendArray):
    __slots__ = ("datastore", "dtype", "shape", "variable_name")

    def __init__(self, variable_name, datastore):
        self.datastore = datastore
        self.variable_name = variable_name

        array = self.get_array()
        self.shape = array.shape

        dtype = array.dtype
        if dtype is str:
            # use object dtype (with additional vlen string metadata) because that's
            # the only way in numpy to represent variable length strings and to
            # check vlen string dtype in further steps
            # it also prevents automatic string concatenation via
            # conventions.decode_cf_variable
            dtype = coding.strings.create_vlen_dtype(str)
        self.dtype = dtype

    def __setitem__(self, key, value):
        with self.datastore.lock:
            data = self.get_array(needs_lock=False)
            data[key] = value
            if self.datastore.autoclose:
                self.datastore.close(needs_lock=False)

    def get_array(self, needs_lock=True):
        raise NotImplementedError("Virtual Method")


class NetCDF4ArrayWrapper(BaseNetCDF4Array):
    __slots__ = ()

    def get_array(self, needs_lock=True):
        ds = self.datastore._acquire(needs_lock)
        variable = ds.variables[self.variable_name]
        variable.set_auto_maskandscale(False)
        # only added in netCDF4-python v1.2.8
        with suppress(AttributeError):
            variable.set_auto_chartostring(False)
        return variable

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.OUTER, self._getitem
        )

    def _getitem(self, key):
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


def _encode_nc4_variable(var):
    for coder in [
        coding.strings.EncodedStringCoder(allows_unicode=True),
        coding.strings.CharacterArrayCoder(),
    ]:
        var = coder.encode(var)
    return var


def _check_encoding_dtype_is_vlen_string(dtype):
    if dtype is not str:
        raise AssertionError(  # pragma: no cover
            f"unexpected dtype encoding {dtype!r}. This shouldn't happen: please "
            "file a bug report at github.com/pydata/xarray"
        )


def _get_datatype(var, nc_format="NETCDF4", raise_on_invalid_encoding=False):
    if nc_format == "NETCDF4":
        return _nc4_dtype(var)
    if "dtype" in var.encoding:
        encoded_dtype = var.encoding["dtype"]
        _check_encoding_dtype_is_vlen_string(encoded_dtype)
        if raise_on_invalid_encoding:
            raise ValueError(
                "encoding dtype=str for vlen strings is only supported "
                "with format='NETCDF4'."
            )
    return var.dtype


def _nc4_dtype(var):
    if "dtype" in var.encoding:
        dtype = var.encoding.pop("dtype")
        _check_encoding_dtype_is_vlen_string(dtype)
    elif coding.strings.is_unicode_dtype(var.dtype):
        dtype = str
    elif var.dtype.kind in ["i", "u", "f", "c", "S"]:
        dtype = var.dtype
    else:
        raise ValueError(f"unsupported dtype for netCDF4 variable: {var.dtype}")
    return dtype


def _netcdf4_create_group(
    dataset: netCDF4.Dataset | netCDF4.Group, name: str
) -> netCDF4.Group:
    return dataset.createGroup(name)


def _nc4_require_group(
    ds: netCDF4.Dataset,
    group: str | None,
    mode: str | None,
    create_group: Callable[
        [netCDF4.Dataset | netCDF4.Group, str], netCDF4.Group
    ] = _netcdf4_create_group,
) -> netCDF4.Dataset | netCDF4.Group:
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
                    raise OSError(f"group not found: {key}", e)
        return ds


def _ensure_no_forward_slash_in_name(name):
    if "/" in name:
        raise ValueError(
            f"Forward slashes '/' are not allowed in variable and dimension names (got {name!r}). "
            "Forward slashes are used as hierarchy-separators for "
            "HDF5-based files ('netcdf4'/'h5netcdf')."
        )


def _ensure_fill_value_valid(data, attributes):
    # work around for netCDF4/scipy issue where _FillValue has the wrong type:
    # https://github.com/Unidata/netcdf4-python/issues/271
    if data.dtype.kind == "S" and "_FillValue" in attributes:
        attributes["_FillValue"] = np.bytes_(attributes["_FillValue"])


def _force_native_endianness(var):
    # possible values for byteorder are:
    #     =    native
    #     <    little-endian
    #     >    big-endian
    #     |    not applicable
    # Below we check if the data type is not native or NA
    if var.dtype.byteorder not in ["=", "|"]:
        # if endianness is specified explicitly, convert to the native type
        data = var.data.astype(var.dtype.newbyteorder("="))
        var = Variable(var.dims, data, var.attrs, var.encoding)
        # if endian exists, remove it from the encoding.
        var.encoding.pop("endian", None)
    # check to see if encoding has a value for endian its 'native'
    if var.encoding.get("endian", "native") != "native":
        raise NotImplementedError(
            "Attempt to write non-native endian type, "
            "this is not supported by the netCDF4 "
            "python library."
        )
    return var


def _extract_nc4_variable_encoding(
    variable,
    raise_on_invalid=False,
    lsd_okay=True,
    h5py_okay=False,
    backend="netCDF4",
    unlimited_dims=None,
):
    if unlimited_dims is None:
        unlimited_dims = ()

    encoding = variable.encoding.copy()

    safe_to_drop = {"source", "original_shape"}
    valid_encodings = {
        "zlib",
        "complevel",
        "fletcher32",
        "contiguous",
        "chunksizes",
        "shuffle",
        "_FillValue",
        "dtype",
        "compression",
        "significant_digits",
        "quantize_mode",
        "blosc_shuffle",
        "szip_coding",
        "szip_pixels_per_block",
        "endian",
    }
    if lsd_okay:
        valid_encodings.add("least_significant_digit")
    if h5py_okay:
        valid_encodings.add("compression_opts")

    if not raise_on_invalid and encoding.get("chunksizes") is not None:
        # It's possible to get encoded chunksizes larger than a dimension size
        # if the original file had an unlimited dimension. This is problematic
        # if the new file no longer has an unlimited dimension.
        chunksizes = encoding["chunksizes"]
        chunks_too_big = any(
            c > d and dim not in unlimited_dims
            for c, d, dim in zip(chunksizes, variable.shape, variable.dims)
        )
        has_original_shape = "original_shape" in encoding
        changed_shape = (
            has_original_shape and encoding.get("original_shape") != variable.shape
        )
        if chunks_too_big or changed_shape:
            del encoding["chunksizes"]

    var_has_unlim_dim = any(dim in unlimited_dims for dim in variable.dims)
    if not raise_on_invalid and var_has_unlim_dim and "contiguous" in encoding.keys():
        del encoding["contiguous"]

    for k in safe_to_drop:
        if k in encoding:
            del encoding[k]

    if raise_on_invalid:
        invalid = [k for k in encoding if k not in valid_encodings]
        if invalid:
            raise ValueError(
                f"unexpected encoding parameters for {backend!r} backend: {invalid!r}. Valid "
                f"encodings are: {valid_encodings!r}"
            )
    else:
        for k in list(encoding):
            if k not in valid_encodings:
                del encoding[k]

    return encoding


def _is_list_of_strings(value):
    arr = np.asarray(value)
    return arr.dtype.kind in ["U", "S"] and arr.size > 1


class NetCDF4DataStore(WritableCFDataStore):
    """Store for reading and writing data via the Python-NetCDF4 library.

    This store supports NetCDF3, NetCDF4 and OpenDAP datasets.
    """

    __slots__ = (
        "_manager",
        "_group",
        "_mode",
        "_filename",
        "format",
        "is_remote",
        "lock",
        "autoclose",
    )

    _manager: FileManager[netCDF4.Dataset]
    _group: str | None
    _mode: NetCDFOpenModes
    _filename: str
    format: NetcdfFormats
    is_remote: bool
    lock: Literal[False] | LockLike | None
    autoclose: bool

    def __init__(
        self,
        manager: netCDF4.Dataset | FileManager[netCDF4.Dataset],
        group: str | None = None,
        mode: NetCDFOpenModes = "r",
        lock: Literal[False] | LockLike | None = NETCDF4_PYTHON_LOCK,
        autoclose: bool = False,
    ):
        import netCDF4

        if isinstance(manager, netCDF4.Dataset):
            if group is None:
                root, group = find_root_and_group(manager)
            else:
                if type(manager) is not netCDF4.Dataset:
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
        filename: T_XarrayCanOpen,
        mode: NetCDFOpenModes = "r",
        format: NetcdfFormats | None = "NETCDF4",
        group: str | None = None,
        clobber: bool = True,
        diskless: bool = False,
        persist: bool = False,
        lock: Literal[False] | LockLike | None = None,
        lock_maker=None,
        autoclose: bool = False,
    ) -> Self:
        import netCDF4

        if isinstance(filename, os.PathLike):
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

    def _acquire(self, needs_lock: bool = True) -> netCDF4.Dataset | netCDF4.Group:
        with self._manager.acquire_context(needs_lock) as root:
            ds = _nc4_require_group(root, self._group, self._mode)
        return ds

    @property
    def ds(self) -> netCDF4.Dataset | netCDF4.Group:
        return self._acquire()

    def open_store_variable(self, name, var):
        dimensions = var.dimensions
        data = indexing.LazilyIndexedArray(NetCDF4ArrayWrapper(name, self))
        attributes = {k: var.getncattr(k) for k in var.ncattrs()}
        _ensure_fill_value_valid(data, attributes)
        # netCDF4 specific encoding; save _FillValue for later
        encoding = {}
        filters = var.filters()
        if filters is not None:
            encoding.update(filters)
        chunking = var.chunking()
        if chunking is not None:
            if chunking == "contiguous":
                encoding["contiguous"] = True
                encoding["chunksizes"] = None
            else:
                encoding["contiguous"] = False
                encoding["chunksizes"] = tuple(chunking)
                encoding["preferred_chunks"] = dict(zip(var.dimensions, chunking))
        # TODO: figure out how to round-trip "endian-ness" without raising
        # warnings from netCDF4
        # encoding['endian'] = var.endian()
        pop_to(attributes, encoding, "least_significant_digit")
        # save source so __repr__ can detect if it's local or not
        encoding["source"] = self._filename
        encoding["original_shape"] = var.shape
        encoding["dtype"] = var.dtype

        return Variable(dimensions, data, attributes, encoding)

    def get_variables(self):
        return FrozenDict(
            (k, self.open_store_variable(k, v)) for k, v in self.ds.variables.items()
        )

    def get_attrs(self):
        return FrozenDict((k, self.ds.getncattr(k)) for k in self.ds.ncattrs())

    def get_dimensions(self):
        return FrozenDict((k, len(v)) for k, v in self.ds.dimensions.items())

    def get_encoding(self):
        return {
            "unlimited_dims": {
                k for k, v in self.ds.dimensions.items() if v.isunlimited()
            }
        }

    def set_dimension(self, name, length, is_unlimited=False):
        _ensure_no_forward_slash_in_name(name)
        dim_length = length if not is_unlimited else None
        self.ds.createDimension(name, size=dim_length)

    def set_attribute(self, key, value):
        if self.format != "NETCDF4":
            value = encode_nc3_attr_value(value)
        if _is_list_of_strings(value):
            # encode as NC_STRING if attr is list of strings
            self.ds.setncattr_string(key, value)
        else:
            self.ds.setncattr(key, value)

    def encode_variable(self, variable):
        variable = _force_native_endianness(variable)
        if self.format == "NETCDF4":
            variable = _encode_nc4_variable(variable)
        else:
            variable = encode_nc3_variable(variable)
        return variable

    def prepare_variable(
        self, name, variable, check_encoding=False, unlimited_dims=None
    ):
        _ensure_no_forward_slash_in_name(name)

        datatype = _get_datatype(
            variable, self.format, raise_on_invalid_encoding=check_encoding
        )
        attrs = variable.attrs.copy()

        fill_value = attrs.pop("_FillValue", None)

        encoding = _extract_nc4_variable_encoding(
            variable, raise_on_invalid=check_encoding, unlimited_dims=unlimited_dims
        )

        if name in self.ds.variables:
            nc4_var = self.ds.variables[name]
        else:
            default_args = dict(
                varname=name,
                datatype=datatype,
                dimensions=variable.dims,
                zlib=False,
                complevel=4,
                shuffle=True,
                fletcher32=False,
                contiguous=False,
                chunksizes=None,
                endian="native",
                least_significant_digit=None,
                fill_value=fill_value,
            )
            default_args.update(encoding)
            default_args.pop("_FillValue", None)
            nc4_var = self.ds.createVariable(**default_args)

        nc4_var.setncatts(attrs)

        target = NetCDF4ArrayWrapper(name, self)

        return target, variable.data

    def sync(self):
        self.ds.sync()

    def close(self, **kwargs):
        self._manager.close(**kwargs)


@dataclass(repr=False)
class NetCDF4BackendEntrypoint(BackendEntrypoint):
    """
    Backend for netCDF files based on the netCDF4 package.

    It can open ".nc", ".nc4", ".cdf" files and will be chosen as default for
    these files.

    Additionally it can open valid HDF5 files, see
    https://h5netcdf.org/#invalid-netcdf-files for more info. It will not be
    detected as valid backend for such files, so make sure to specify
    ``engine="netcdf4"`` in ``open_dataset``.

    For more information about the underlying library, visit:
    https://unidata.github.io/netcdf4-python

    Parameters
    ----------
    group: str or None, optional
        Path to the netCDF4 group in the given file to open. None (default) uses
        the root group.
    mode: {"w", "x", "a", "r+", "r"}, default: "r"
        Access mode of the NetCDF file. "r" means read-only; no data can be
        modified. "w" means write; a new file is created, an existing file with
        the same name is deleted. "x" means write, but fail if an existing file
        with the same name already exists. "a" and "r+" mean append; an existing
        file is opened for reading and writing, if file does not exist already,
        one is created.
    format: {"NETCDF4", "NETCDF4_CLASSIC", "NETCDF3_64BIT", \
            "NETCDF3_64BIT_OFFSET", "NETCDF3_64BIT_DATA", "NETCDF3_CLASSIC"} \
            or None, optional
        Format of the NetCDF file, defaults to "NETCDF4".
    lock: False, None or Lock-like, optional
        Resource lock to use when reading data from disk. Only relevant when
        using dask or another form of parallelism. If None (default) appropriate
        locks are chosen to safely read and write files with the currently
        active dask scheduler.
    autoclose: bool, default: False
        If True, automatically close files to avoid OS Error of too many files
        being open. However, this option doesn't work with streams, e.g.,
        BytesIO.
    clobber: bool, default: False
        If True, opening a file with mode="w" will clobber an existing file with
        the same name. If False, an exception will be raised if a file with the
        same name already exists. mode="x" is identical to mode="w" with
        clobber=False.
    diskless: bool, default: False
        If True, create diskless (in-core) file.
    persist: bool, default: False
        If True, persist file to disk when closed.

    See Also
    --------
    backends.NetCDF4DataStore
    backends.H5netcdfBackendEntrypoint
    backends.ScipyBackendEntrypoint
    """

    description = (
        "Open netCDF (.nc, .nc4 and .cdf) and most HDF5 files using netCDF4 in Xarray"
    )
    url = "https://docs.xarray.dev/en/stable/generated/xarray.backends.NetCDF4BackendEntrypoint.html"
    open_dataset_parameters = (
        "drop_variables",
        "mask_and_scale",
        "decode_times",
        "concat_characters",
        "use_cftime",
        "decode_timedelta",
        "decode_coords",
    )

    group: str | None = None
    mode: NetCDFOpenModes = "r"
    format: NetcdfFormats | None = "NETCDF4"
    lock: Literal[False] | LockLike | None = None
    autoclose: bool = False
    clobber: bool = True
    diskless: bool = False
    persist: bool = False

    def guess_can_open(self, filename_or_obj: T_XarrayCanOpen) -> bool:
        if isinstance(filename_or_obj, str) and is_remote_uri(filename_or_obj):
            return True
        magic_number = try_read_magic_number_from_path(filename_or_obj)
        if magic_number is not None:
            # netcdf 3 or HDF5
            return magic_number.startswith((b"CDF", b"\211HDF\r\n\032\n"))

        if isinstance(filename_or_obj, (str, os.PathLike)):
            _, ext = os.path.splitext(filename_or_obj)
            return ext in {".nc", ".nc4", ".cdf"}

        return False

    def open_dataset(
        self,
        filename_or_obj: T_XarrayCanOpen,
        *,
        drop_variables: str | Iterable[str] | None = None,
        mask_and_scale: bool = True,
        decode_times: bool = True,
        concat_characters: bool = True,
        use_cftime: bool | None = None,
        decode_timedelta: bool | None = None,
        decode_coords: bool | Literal["coordinates", "all"] = True,
        **kwargs: Any,
    ) -> Dataset:
        filename_or_obj = _normalize_path(filename_or_obj)
        store = NetCDF4DataStore.open(
            filename_or_obj,
            mode=kwargs.pop("mode", self.mode),
            format=kwargs.pop("format", self.format),
            group=kwargs.pop("group", self.group),
            clobber=kwargs.pop("clobber", self.clobber),
            diskless=kwargs.pop("diskless", self.diskless),
            persist=kwargs.pop("persist", self.persist),
            lock=kwargs.pop("lock", self.lock),
            autoclose=kwargs.pop("autoclose", self.autoclose),
        )
        if kwargs:
            raise ValueError(f"Unsupported kwargs: {kwargs.values()}")

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


BACKEND_ENTRYPOINTS["netcdf4"] = ("netCDF4", NetCDF4BackendEntrypoint)
