import functools
import operator
from contextlib import suppress

import numpy as np

from .. import coding
from ..coding.variables import pop_to
from ..core import indexing
from ..core.utils import FrozenDict, is_remote_uri
from ..core.variable import Variable
from .common import (
    BackendArray,
    WritableCFDataStore,
    find_root_and_group,
    robust_getitem,
)
from .file_manager import CachingFileManager, DummyFileManager
from .locks import HDF5_LOCK, NETCDFC_LOCK, combine_locks, ensure_lock, get_write_lock
from .netcdf3 import encode_nc3_attr_value, encode_nc3_variable

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
            # use object dtype because that's the only way in numpy to
            # represent variable length strings; it also prevents automatic
            # string concatenation via conventions.decode_cf_variable
            dtype = np.dtype("O")
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
            "unexpected dtype encoding %r. This shouldn't happen: please "
            "file a bug report at github.com/pydata/xarray" % dtype
        )


def _get_datatype(var, nc_format="NETCDF4", raise_on_invalid_encoding=False):
    if nc_format == "NETCDF4":
        datatype = _nc4_dtype(var)
    else:
        if "dtype" in var.encoding:
            encoded_dtype = var.encoding["dtype"]
            _check_encoding_dtype_is_vlen_string(encoded_dtype)
            if raise_on_invalid_encoding:
                raise ValueError(
                    "encoding dtype=str for vlen strings is only supported "
                    "with format='NETCDF4'."
                )
        datatype = var.dtype
    return datatype


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


def _ensure_fill_value_valid(data, attributes):
    # work around for netCDF4/scipy issue where _FillValue has the wrong type:
    # https://github.com/Unidata/netcdf4-python/issues/271
    if data.dtype.kind == "S" and "_FillValue" in attributes:
        attributes["_FillValue"] = np.string_(attributes["_FillValue"])


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
    if not var.encoding.get("endian", "native") == "native":
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
    }
    if lsd_okay:
        valid_encodings.add("least_significant_digit")
    if h5py_okay:
        valid_encodings.add("compression")
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
                "unexpected encoding parameters for %r backend: %r. Valid "
                "encodings are: %r" % (backend, invalid, valid_encodings)
            )
    else:
        for k in list(encoding):
            if k not in valid_encodings:
                del encoding[k]

    return encoding


def _is_list_of_strings(value):
    if np.asarray(value).dtype.kind in ["U", "S"] and np.asarray(value).size > 1:
        return True
    else:
        return False


class NetCDF4DataStore(WritableCFDataStore):
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
        import netCDF4

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
        import netCDF4

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

    def open_store_variable(self, name, var):
        dimensions = var.dimensions
        data = indexing.LazilyOuterIndexedArray(NetCDF4ArrayWrapper(name, self))
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
        dsvars = FrozenDict(
            (k, self.open_store_variable(k, v)) for k, v in self.ds.variables.items()
        )
        return dsvars

    def get_attrs(self):
        attrs = FrozenDict((k, self.ds.getncattr(k)) for k in self.ds.ncattrs())
        return attrs

    def get_dimensions(self):
        dims = FrozenDict((k, len(v)) for k, v in self.ds.dimensions.items())
        return dims

    def get_encoding(self):
        encoding = {}
        encoding["unlimited_dims"] = {
            k for k, v in self.ds.dimensions.items() if v.isunlimited()
        }
        return encoding

    def set_dimension(self, name, length, is_unlimited=False):
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
        datatype = _get_datatype(
            variable, self.format, raise_on_invalid_encoding=check_encoding
        )
        attrs = variable.attrs.copy()

        fill_value = attrs.pop("_FillValue", None)

        if datatype is str and fill_value is not None:
            raise NotImplementedError(
                "netCDF4 does not yet support setting a fill value for "
                "variable-length strings "
                "(https://github.com/Unidata/netcdf4-python/issues/730). "
                "Either remove '_FillValue' from encoding on variable %r "
                "or set {'dtype': 'S1'} in encoding to use the fixed width "
                "NC_CHAR type." % name
            )

        encoding = _extract_nc4_variable_encoding(
            variable, raise_on_invalid=check_encoding, unlimited_dims=unlimited_dims
        )

        if name in self.ds.variables:
            nc4_var = self.ds.variables[name]
        else:
            nc4_var = self.ds.createVariable(
                varname=name,
                datatype=datatype,
                dimensions=variable.dims,
                zlib=encoding.get("zlib", False),
                complevel=encoding.get("complevel", 4),
                shuffle=encoding.get("shuffle", True),
                fletcher32=encoding.get("fletcher32", False),
                contiguous=encoding.get("contiguous", False),
                chunksizes=encoding.get("chunksizes"),
                endian="native",
                least_significant_digit=encoding.get("least_significant_digit"),
                fill_value=fill_value,
            )

        nc4_var.setncatts(attrs)

        target = NetCDF4ArrayWrapper(name, self)

        return target, variable.data

    def sync(self):
        self.ds.sync()

    def close(self, **kwargs):
        self._manager.close(**kwargs)
