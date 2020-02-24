import functools
from distutils.version import LooseVersion

import numpy as np

from ..core import indexing
from ..core.utils import FrozenDict, is_remote_uri
from ..core.variable import Variable
from .common import WritableCFDataStore, find_root_and_group
from .file_manager import CachingFileManager, DummyFileManager
from .locks import HDF5_LOCK, combine_locks, ensure_lock, get_write_lock
from .netCDF4_ import (
    BaseNetCDF4Array,
    _encode_nc4_variable,
    _extract_nc4_variable_encoding,
    _get_datatype,
    _nc4_require_group,
)


class H5NetCDFArrayWrapper(BaseNetCDF4Array):
    def get_array(self, needs_lock=True):
        ds = self.datastore._acquire(needs_lock)
        variable = ds.variables[self.variable_name]
        return variable

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.OUTER_1VECTOR, self._getitem
        )

    def _getitem(self, key):
        # h5py requires using lists for fancy indexing:
        # https://github.com/h5py/h5py/issues/992
        key = tuple(list(k) if isinstance(k, np.ndarray) else k for k in key)
        with self.datastore.lock:
            array = self.get_array(needs_lock=False)
            return array[key]


def maybe_decode_bytes(txt):
    if isinstance(txt, bytes):
        return txt.decode("utf-8")
    else:
        return txt


def _read_attributes(h5netcdf_var):
    # GH451
    # to ensure conventions decoding works properly on Python 3, decode all
    # bytes attributes to strings
    attrs = {}
    for k, v in h5netcdf_var.attrs.items():
        if k not in ["_FillValue", "missing_value"]:
            v = maybe_decode_bytes(v)
        attrs[k] = v
    return attrs


_extract_h5nc_encoding = functools.partial(
    _extract_nc4_variable_encoding, lsd_okay=False, h5py_okay=True, backend="h5netcdf"
)


def _h5netcdf_create_group(dataset, name):
    return dataset.create_group(name)


class H5NetCDFStore(WritableCFDataStore):
    """Store for reading and writing data via h5netcdf
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

    def __init__(self, manager, group=None, mode=None, lock=HDF5_LOCK, autoclose=False):

        import h5netcdf

        if isinstance(manager, (h5netcdf.File, h5netcdf.Group)):
            if group is None:
                root, group = find_root_and_group(manager)
            else:
                if not type(manager) is h5netcdf.File:
                    raise ValueError(
                        "must supply a h5netcdf.File if the group "
                        "argument is provided"
                    )
                root = manager
            manager = DummyFileManager(root)

        self._manager = manager
        self._group = group
        self._mode = mode
        self.format = None
        # todo: utilizing find_root_and_group seems a bit clunky
        #  making filename available on h5netcdf.Group seems better
        self._filename = find_root_and_group(self.ds)[0].filename
        self.is_remote = is_remote_uri(self._filename)
        self.lock = ensure_lock(lock)
        self.autoclose = autoclose

    @classmethod
    def open(
        cls,
        filename,
        mode="r",
        format=None,
        group=None,
        lock=None,
        autoclose=False,
        invalid_netcdf=None,
        phony_dims=None,
    ):
        import h5netcdf

        if format not in [None, "NETCDF4"]:
            raise ValueError("invalid format for h5netcdf backend")

        kwargs = {"invalid_netcdf": invalid_netcdf}
        if phony_dims is not None:
            if LooseVersion(h5netcdf.__version__) >= LooseVersion("0.8.0"):
                kwargs["phony_dims"] = phony_dims
            else:
                raise ValueError(
                    "h5netcdf backend keyword argument 'phony_dims' needs "
                    "h5netcdf >= 0.8.0."
                )

        if lock is None:
            if mode == "r":
                lock = HDF5_LOCK
            else:
                lock = combine_locks([HDF5_LOCK, get_write_lock(filename)])

        manager = CachingFileManager(h5netcdf.File, filename, mode=mode, kwargs=kwargs)
        return cls(manager, group=group, mode=mode, lock=lock, autoclose=autoclose)

    def _acquire(self, needs_lock=True):
        with self._manager.acquire_context(needs_lock) as root:
            ds = _nc4_require_group(
                root, self._group, self._mode, create_group=_h5netcdf_create_group
            )
        return ds

    @property
    def ds(self):
        return self._acquire()

    def open_store_variable(self, name, var):
        import h5py

        dimensions = var.dimensions
        data = indexing.LazilyOuterIndexedArray(H5NetCDFArrayWrapper(name, self))
        attrs = _read_attributes(var)

        # netCDF4 specific encoding
        encoding = {
            "chunksizes": var.chunks,
            "fletcher32": var.fletcher32,
            "shuffle": var.shuffle,
        }
        # Convert h5py-style compression options to NetCDF4-Python
        # style, if possible
        if var.compression == "gzip":
            encoding["zlib"] = True
            encoding["complevel"] = var.compression_opts
        elif var.compression is not None:
            encoding["compression"] = var.compression
            encoding["compression_opts"] = var.compression_opts

        # save source so __repr__ can detect if it's local or not
        encoding["source"] = self._filename
        encoding["original_shape"] = var.shape

        vlen_dtype = h5py.check_dtype(vlen=var.dtype)
        if vlen_dtype is str:
            encoding["dtype"] = str
        elif vlen_dtype is not None:  # pragma: no cover
            # xarray doesn't support writing arbitrary vlen dtypes yet.
            pass
        else:
            encoding["dtype"] = var.dtype

        return Variable(dimensions, data, attrs, encoding)

    def get_variables(self):
        return FrozenDict(
            (k, self.open_store_variable(k, v)) for k, v in self.ds.variables.items()
        )

    def get_attrs(self):
        return FrozenDict(_read_attributes(self.ds))

    def get_dimensions(self):
        return self.ds.dimensions

    def get_encoding(self):
        encoding = {}
        encoding["unlimited_dims"] = {
            k for k, v in self.ds.dimensions.items() if v is None
        }
        return encoding

    def set_dimension(self, name, length, is_unlimited=False):
        if is_unlimited:
            self.ds.dimensions[name] = None
            self.ds.resize_dimension(name, length)
        else:
            self.ds.dimensions[name] = length

    def set_attribute(self, key, value):
        self.ds.attrs[key] = value

    def encode_variable(self, variable):
        return _encode_nc4_variable(variable)

    def prepare_variable(
        self, name, variable, check_encoding=False, unlimited_dims=None
    ):
        import h5py

        attrs = variable.attrs.copy()
        dtype = _get_datatype(variable, raise_on_invalid_encoding=check_encoding)

        fillvalue = attrs.pop("_FillValue", None)
        if dtype is str and fillvalue is not None:
            raise NotImplementedError(
                "h5netcdf does not yet support setting a fill value for "
                "variable-length strings "
                "(https://github.com/shoyer/h5netcdf/issues/37). "
                "Either remove '_FillValue' from encoding on variable %r "
                "or set {'dtype': 'S1'} in encoding to use the fixed width "
                "NC_CHAR type." % name
            )

        if dtype is str:
            dtype = h5py.special_dtype(vlen=str)

        encoding = _extract_h5nc_encoding(variable, raise_on_invalid=check_encoding)
        kwargs = {}

        # Convert from NetCDF4-Python style compression settings to h5py style
        # If both styles are used together, h5py takes precedence
        # If set_encoding=True, raise ValueError in case of mismatch
        if encoding.pop("zlib", False):
            if check_encoding and encoding.get("compression") not in (None, "gzip"):
                raise ValueError("'zlib' and 'compression' encodings mismatch")
            encoding.setdefault("compression", "gzip")

        if (
            check_encoding
            and "complevel" in encoding
            and "compression_opts" in encoding
            and encoding["complevel"] != encoding["compression_opts"]
        ):
            raise ValueError("'complevel' and 'compression_opts' encodings " "mismatch")
        complevel = encoding.pop("complevel", 0)
        if complevel != 0:
            encoding.setdefault("compression_opts", complevel)

        encoding["chunks"] = encoding.pop("chunksizes", None)

        # Do not apply compression, filters or chunking to scalars.
        if variable.shape:
            for key in [
                "compression",
                "compression_opts",
                "shuffle",
                "chunks",
                "fletcher32",
            ]:
                if key in encoding:
                    kwargs[key] = encoding[key]
        if name not in self.ds:
            nc4_var = self.ds.create_variable(
                name,
                dtype=dtype,
                dimensions=variable.dims,
                fillvalue=fillvalue,
                **kwargs,
            )
        else:
            nc4_var = self.ds[name]

        for k, v in attrs.items():
            nc4_var.attrs[k] = v

        target = H5NetCDFArrayWrapper(name, self)

        return target, variable.data

    def sync(self):
        self.ds.sync()

    def close(self, **kwargs):
        self._manager.close(**kwargs)
