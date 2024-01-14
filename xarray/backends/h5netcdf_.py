from __future__ import annotations

import functools
import io
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from xarray.backends.common import (
    BACKEND_ENTRYPOINTS,
    BackendEntrypoint,
    WritableCFDataStore,
    _normalize_path,
    find_root_and_group,
)
from xarray.backends.file_manager import CachingFileManager, DummyFileManager
from xarray.backends.locks import HDF5_LOCK, combine_locks, ensure_lock, get_write_lock
from xarray.backends.netCDF4_ import (
    BaseNetCDF4Array,
    _encode_nc4_variable,
    _ensure_no_forward_slash_in_name,
    _extract_nc4_variable_encoding,
    _get_datatype,
    _nc4_require_group,
)
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.utils import (
    FrozenDict,
    hashable,
    is_remote_uri,
    read_magic_number_from_file,
    try_read_magic_number_from_file_or_path,
)
from xarray.core.variable import Variable

if TYPE_CHECKING:
    import h5netcdf

    from xarray.backends.file_manager import FileManager
    from xarray.core.dataset import Dataset
    from xarray.core.types import H5netcdfOpenModes, LockLike, Self, T_XarrayCanOpen


class H5NetCDFArrayWrapper(BaseNetCDF4Array):
    def get_array(self, needs_lock=True):
        ds = self.datastore._acquire(needs_lock)
        return ds.variables[self.variable_name]

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.OUTER_1VECTOR, self._getitem
        )

    def _getitem(self, key):
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
    _extract_nc4_variable_encoding,
    lsd_okay=False,
    h5py_okay=True,
    backend="h5netcdf",
    unlimited_dims=None,
)


def _h5netcdf_create_group(dataset, name):
    return dataset.create_group(name)


class H5NetCDFStore(WritableCFDataStore):
    """Store for reading and writing data via h5netcdf"""

    __slots__ = (
        "_manager",
        "_group",
        "_mode",
        "_filename",
        "autoclose",
        "format",
        "is_remote",
        "lock",
    )

    _manager: FileManager[h5netcdf.File | h5netcdf.Group]
    _group: str | None
    _mode: H5netcdfOpenModes
    _filename: str
    autoclose: bool
    format: None
    is_remote: bool
    lock: LockLike

    def __init__(
        self,
        manager: h5netcdf.File
        | h5netcdf.Group
        | FileManager[h5netcdf.File | h5netcdf.Group],
        group: str | None = None,
        mode: H5netcdfOpenModes = "r",
        lock: Literal[False] | LockLike | None = HDF5_LOCK,
        autoclose: bool = False,
    ):
        import h5netcdf

        if isinstance(manager, (h5netcdf.File, h5netcdf.Group)):
            if group is None:
                root, group = find_root_and_group(manager)
            else:
                if type(manager) is not h5netcdf.File:
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
        filename: T_XarrayCanOpen,
        mode: H5netcdfOpenModes = "r",
        format: str | None = None,
        group: str | None = None,
        lock: Literal[False] | LockLike | None = None,
        autoclose: bool = False,
        invalid_netcdf: bool | None = None,
        phony_dims: Literal["sort", "access", None] = None,
        decode_vlen_strings: bool = True,
        driver: str | None = None,
        driver_kwds: Mapping[str, Any] | None = None,
    ) -> Self:
        import h5netcdf

        if isinstance(filename, bytes):
            raise ValueError(
                "can't open netCDF4/HDF5 as bytes "
                "try passing a path or file-like object"
            )
        elif isinstance(filename, io.IOBase):
            magic_number = read_magic_number_from_file(filename)
            if not magic_number.startswith(b"\211HDF\r\n\032\n"):
                raise ValueError(
                    f"{magic_number.decode()} is not the signature of a valid netCDF4 file"
                )

        if format not in [None, "NETCDF4"]:
            raise ValueError("invalid format for h5netcdf backend")

        kwargs = {
            "invalid_netcdf": invalid_netcdf,
            "decode_vlen_strings": decode_vlen_strings,
            "driver": driver,
        }
        if driver_kwds is not None:
            kwargs.update(driver_kwds)
        if phony_dims is not None:
            kwargs["phony_dims"] = phony_dims

        if lock is None:
            if mode == "r":
                lock = HDF5_LOCK
            else:
                assert hashable(filename)
                lock = combine_locks([HDF5_LOCK, get_write_lock(filename)])

        manager = CachingFileManager(h5netcdf.File, filename, mode=mode, kwargs=kwargs)
        return cls(manager, group=group, mode=mode, lock=lock, autoclose=autoclose)

    def _acquire(self, needs_lock: bool = True) -> h5netcdf.File | h5netcdf.Group:
        with self._manager.acquire_context(needs_lock) as root:
            ds = _nc4_require_group(
                root, self._group, self._mode, create_group=_h5netcdf_create_group
            )
        return ds

    @property
    def ds(self) -> h5netcdf.File | h5netcdf.Group:
        return self._acquire()

    def open_store_variable(self, name, var):
        import h5py

        dimensions = var.dimensions
        data = indexing.LazilyIndexedArray(H5NetCDFArrayWrapper(name, self))
        attrs = _read_attributes(var)

        # netCDF4 specific encoding
        encoding = {
            "chunksizes": var.chunks,
            "fletcher32": var.fletcher32,
            "shuffle": var.shuffle,
        }
        if var.chunks:
            encoding["preferred_chunks"] = dict(zip(var.dimensions, var.chunks))
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
        return FrozenDict((k, len(v)) for k, v in self.ds.dimensions.items())

    def get_encoding(self):
        return {
            "unlimited_dims": {
                k for k, v in self.ds.dimensions.items() if v.isunlimited()
            }
        }

    def set_dimension(self, name, length, is_unlimited=False):
        _ensure_no_forward_slash_in_name(name)
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

        _ensure_no_forward_slash_in_name(name)
        attrs = variable.attrs.copy()
        dtype = _get_datatype(variable, raise_on_invalid_encoding=check_encoding)

        fillvalue = attrs.pop("_FillValue", None)

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
            raise ValueError("'complevel' and 'compression_opts' encodings mismatch")
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


@dataclass(repr=False)
class H5netcdfBackendEntrypoint(BackendEntrypoint):
    """
    Backend for netCDF files based on the h5netcdf package.

    It can open ".nc", ".nc4", ".cdf" files but will only be
    selected as the default if the "netcdf4" engine is not available.

    Additionally it can open valid HDF5 files, see
    https://h5netcdf.org/#invalid-netcdf-files for more info.
    It will not be detected as valid backend for such files, so make
    sure to specify ``engine="h5netcdf"`` in ``open_dataset``.

    For more information about the underlying library, visit:
    https://h5netcdf.org

    Parameters
    ----------
    group: str or None, optional
        Path to the netCDF4 group in the given file to open. None (default) uses
        the root group.
    mode: {"w", "a", "r+", "r"}, default: "r"
        Access mode of the NetCDF file. "r" means read-only; no data can be
        modified. "w" means write; a new file is created, an existing file with
        the same name is deleted. "a" and "r+" mean append; an existing file is
        opened for reading and writing, if file does not exist already, one is
        created.
    format: "NETCDF4", or None, optional
        Format of the NetCDF file. Only "NETCDF4" is supported by h5netcdf.
    lock: False, None or Lock-like, optional
        Resource lock to use when reading data from disk. Only relevant when
        using dask or another form of parallelism. If None (default) appropriate
        locks are chosen to safely read and write files with the currently
        active dask scheduler.
    autoclose: bool, default: False
        If True, automatically close files to avoid OS Error of too many files
        being open. However, this option doesn't work with streams, e.g.,
        BytesIO.
    invalid_netcdf : bool or None, optional
        Allow writing netCDF4 with data types and attributes that would
        otherwise not generate netCDF4 files that can be read by other
        applications. See https://h5netcdf.org/#invalid-netcdf-files for
        more details.
    phony_dims: {"sort", "access"} or None, optional
        Change how variables with no dimension scales associated with
        one of their axes are accessed.

        - None: raises a ValueError (default)

        - "sort": invent phony dimensions according to netCDF behaviour.
          Note, that this iterates once over the whole group-hierarchy.
          This has affects on performance in case you rely on laziness
          of group access.

        - "access": defer phony dimension creation to group access time.
          The created phony dimension naming will differ from netCDF behaviour.

    decode_vlen_strings: bool, default: True
        Return vlen string data as str instead of bytes.
    driver: str or None, optional
        Name of the driver to use. Legal values are None (default,
        recommended), "core", "sec2", "direct", "stdio", "mpio", "ros3".
    driver_kwds: Mapping or None, optional
        Additional driver options. See h5py.File for more infos.

    See Also
    --------
    backends.H5NetCDFStore
    backends.NetCDF4BackendEntrypoint
    backends.ScipyBackendEntrypoint
    """

    description = (
        "Open netCDF (.nc, .nc4 and .cdf) and most HDF5 files using h5netcdf in Xarray"
    )
    url = "https://docs.xarray.dev/en/stable/generated/xarray.backends.H5netcdfBackendEntrypoint.html"
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
    mode: H5netcdfOpenModes = "r"
    format: str | None = "NETCDF4"
    lock: Literal[False] | LockLike | None = None
    autoclose: bool = False
    invalid_netcdf: bool | None = None
    phony_dims: Literal["sort", "access", None] = None
    decode_vlen_strings: bool = True
    driver: str | None = None
    driver_kwds: Mapping[str, Any] | None = None

    def guess_can_open(self, filename_or_obj: T_XarrayCanOpen) -> bool:
        magic_number = try_read_magic_number_from_file_or_path(filename_or_obj)
        if magic_number is not None:
            return magic_number.startswith(b"\211HDF\r\n\032\n")

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
        store = H5NetCDFStore.open(
            filename_or_obj,
            mode=kwargs.get("mode", self.mode),
            format=kwargs.get("format", self.format),
            group=kwargs.get("group", self.group),
            lock=kwargs.get("lock", self.lock),
            autoclose=kwargs.get("autoclose", self.autoclose),
            invalid_netcdf=kwargs.get("invalid_netcdf", self.invalid_netcdf),
            phony_dims=kwargs.get("phony_dims", self.phony_dims),
            decode_vlen_strings=kwargs.get(
                "decode_vlen_strings", self.decode_vlen_strings
            ),
            driver=kwargs.get("driver", self.driver),
            driver_kwds=kwargs.get("driver_kwds", self.driver_kwds),
        )

        store_entrypoint = StoreBackendEntrypoint()
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


BACKEND_ENTRYPOINTS["h5netcdf"] = ("h5netcdf", H5netcdfBackendEntrypoint)
