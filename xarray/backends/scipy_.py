from __future__ import annotations

import gzip
import io
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from xarray.backends.common import (
    BACKEND_ENTRYPOINTS,
    BackendArray,
    BackendEntrypoint,
    WritableCFDataStore,
    _normalize_path,
)
from xarray.backends.file_manager import CachingFileManager, DummyFileManager
from xarray.backends.locks import ensure_lock, get_write_lock
from xarray.backends.netcdf3 import (
    encode_nc3_attr_value,
    encode_nc3_variable,
    is_valid_nc3_name,
)
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core.indexing import NumpyIndexingAdapter
from xarray.core.utils import (
    Frozen,
    FrozenDict,
    close_on_error,
    try_read_magic_number_from_file_or_path,
)
from xarray.core.variable import Variable

if TYPE_CHECKING:
    import scipy.io

    from xarray.backends import FileManager
    from xarray.core.dataset import Dataset
    from xarray.core.types import (
        LockLike,
        ScipyFormats,
        ScipyOpenModes,
        T_XarrayCanOpen,
    )


def _decode_string(s):
    if isinstance(s, bytes):
        return s.decode("utf-8", "replace")
    return s


def _decode_attrs(d):
    # don't decode _FillValue from bytes -> unicode, because we want to ensure
    # that its type matches the data exactly
    return {k: v if k == "_FillValue" else _decode_string(v) for (k, v) in d.items()}


class ScipyArrayWrapper(BackendArray):
    def __init__(self, variable_name, datastore):
        self.datastore = datastore
        self.variable_name = variable_name
        array = self.get_variable().data
        self.shape = array.shape
        self.dtype = np.dtype(array.dtype.kind + str(array.dtype.itemsize))

    def get_variable(self, needs_lock=True):
        ds = self.datastore._manager.acquire(needs_lock)
        return ds.variables[self.variable_name]

    def __getitem__(self, key):
        data = NumpyIndexingAdapter(self.get_variable().data)[key]
        # Copy data if the source file is mmapped. This makes things consistent
        # with the netCDF4 library by ensuring we can safely read arrays even
        # after closing associated files.
        copy = self.datastore.ds.use_mmap
        return np.array(data, dtype=self.dtype, copy=copy)

    def __setitem__(self, key, value):
        with self.datastore.lock:
            data = self.get_variable(needs_lock=False)
            try:
                data[key] = value
            except TypeError:
                if key is Ellipsis:
                    # workaround for GH: scipy/scipy#6880
                    data[:] = value
                else:
                    raise


def _open_scipy_netcdf(filename, mode, mmap, version) -> scipy.io.netcdf_file:
    import scipy.io

    # if the string ends with .gz, then gunzip and open as netcdf file
    if isinstance(filename, str) and filename.endswith(".gz"):
        try:
            return scipy.io.netcdf_file(
                gzip.open(filename), mode=mode, mmap=mmap, version=version
            )
        except TypeError as e:
            # TODO: gzipped loading only works with NetCDF3 files.
            errmsg = e.args[0]
            if "is not a valid NetCDF 3 file" in errmsg:
                raise ValueError("gzipped file loading only supports NetCDF 3 files.")
            else:
                raise

    if isinstance(filename, bytes) and filename.startswith(b"CDF"):
        # it's a NetCDF3 bytestring
        filename = io.BytesIO(filename)

    try:
        return scipy.io.netcdf_file(filename, mode=mode, mmap=mmap, version=version)
    except TypeError as e:  # netcdf3 message is obscure in this case
        errmsg = e.args[0]
        if "is not a valid NetCDF 3 file" in errmsg:
            msg = """
            If this is a NetCDF4 file, you may need to install the
            netcdf4 library, e.g.,

            $ pip install netcdf4
            """
            errmsg += msg
            raise TypeError(errmsg)
        else:
            raise


class ScipyDataStore(WritableCFDataStore):
    """Store for reading and writing data via scipy.io.netcdf.

    This store has the advantage of being able to be initialized with a
    StringIO object, allow for serialization without writing to disk.

    It only supports the NetCDF3 file-format.
    """

    _manager: FileManager[scipy.io.netcdf_file]
    lock: LockLike

    def __init__(
        self,
        filename_or_obj: T_XarrayCanOpen,
        mode: ScipyOpenModes = "r",
        format: ScipyFormats = None,
        group: str | None = None,
        mmap: bool | None = None,
        lock: Literal[False] | LockLike | None = None,
    ) -> None:
        if group is not None:
            raise ValueError("cannot save to a group with the scipy.io.netcdf backend")

        if format in (None, "NETCDF3_64BIT", "NETCDF3_64BIT_OFFSET"):
            version = 2
        elif format == "NETCDF3_CLASSIC":
            version = 1
        else:
            raise ValueError(f"invalid format for scipy.io.netcdf backend: {format!r}")

        if lock is None and mode != "r" and isinstance(filename_or_obj, str):
            lock = get_write_lock(filename_or_obj)

        self.lock = ensure_lock(lock)

        if isinstance(filename_or_obj, str):
            self._manager = CachingFileManager(
                _open_scipy_netcdf,
                filename_or_obj,
                mode=mode,
                lock=lock,
                kwargs=dict(mmap=mmap, version=version),
            )
        else:
            scipy_dataset = _open_scipy_netcdf(
                filename_or_obj, mode=mode, mmap=mmap, version=version
            )
            self._manager = DummyFileManager(scipy_dataset)

    @property
    def ds(self) -> scipy.io.netcdf_file:
        return self._manager.acquire()

    def open_store_variable(self, name, var):
        return Variable(
            var.dimensions,
            ScipyArrayWrapper(name, self),
            _decode_attrs(var._attributes),
        )

    def get_variables(self):
        return FrozenDict(
            (k, self.open_store_variable(k, v)) for k, v in self.ds.variables.items()
        )

    def get_attrs(self):
        return Frozen(_decode_attrs(self.ds._attributes))

    def get_dimensions(self):
        return Frozen(self.ds.dimensions)

    def get_encoding(self):
        return {
            "unlimited_dims": {k for k, v in self.ds.dimensions.items() if v is None}
        }

    def set_dimension(self, name, length, is_unlimited=False):
        if name in self.ds.dimensions:
            raise ValueError(
                f"{type(self).__name__} does not support modifying dimensions"
            )
        dim_length = length if not is_unlimited else None
        self.ds.createDimension(name, dim_length)

    def _validate_attr_key(self, key):
        if not is_valid_nc3_name(key):
            raise ValueError("Not a valid attribute name")

    def set_attribute(self, key, value):
        self._validate_attr_key(key)
        value = encode_nc3_attr_value(value)
        setattr(self.ds, key, value)

    def encode_variable(self, variable):
        variable = encode_nc3_variable(variable)
        return variable

    def prepare_variable(
        self, name, variable, check_encoding=False, unlimited_dims=None
    ):
        if (
            check_encoding
            and variable.encoding
            and variable.encoding != {"_FillValue": None}
        ):
            raise ValueError(
                f"unexpected encoding for scipy backend: {list(variable.encoding)}"
            )

        data = variable.data
        # nb. this still creates a numpy array in all memory, even though we
        # don't write the data yet; scipy.io.netcdf does not not support
        # incremental writes.
        if name not in self.ds.variables:
            self.ds.createVariable(name, data.dtype, variable.dims)
        scipy_var = self.ds.variables[name]
        for k, v in variable.attrs.items():
            self._validate_attr_key(k)
            setattr(scipy_var, k, v)

        target = ScipyArrayWrapper(name, self)

        return target, data

    def sync(self):
        self.ds.sync()

    def close(self):
        self._manager.close()


@dataclass(repr=False)
class ScipyBackendEntrypoint(BackendEntrypoint):
    """
    Backend for netCDF files based on the scipy package.

    It can open ".nc", ".nc4", ".cdf" and ".gz" files but will only be selected
    as the default if the "netcdf4" and "h5netcdf" engines are not available. It
    has the advantage that is is a lightweight engine that has no system
    requirements (unlike netcdf4 and h5netcdf).

    Additionally it can open gizp compressed (".gz") files.

    For more information about the underlying library, visit:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.netcdf_file.html

    Parameters
    ----------
    group: str or None, optional
        Path to the netCDF4 group in the given file to open. None (default) uses
        the root group.
    mode: {"w", "a", "r"}, default: "r"
        Access mode of the NetCDF file. "r" means read-only; no data can be
        modified. "w" means write; a new file is created, an existing file with
        the same name is deleted. "a" means append; an existing file is opened
        for reading and writing, if file does not exist already, one is created.
    format: {"NETCDF3_64BIT", "NETCDF3_64BIT_OFFSET", "NETCDF3_CLASSIC"} or \
            None, optional
        Format of the NetCDF file. Only classic NetCDF files supported. For newer
        NetCDF version use a different backend.
    lock: False, None or Lock-like, optional
        Resource lock to use when reading data from disk. Only relevant when
        using dask or another form of parallelism. If None (default) appropriate
        locks are chosen to safely read and write files with the currently
        active dask scheduler.
    mmap: bool or None, optional
        Whether to mmap filename when reading. Default is True when filename is
        a file name, False when filename is a file-like object. Note that when
        mmap is in use, data arrays returned refer directly to the mmapped data
        on disk, and the file cannot be closed as long as references to it
        exist.

    See Also
    --------
    backends.ScipyDataStore backends.NetCDF4BackendEntrypoint
    backends.H5netcdfBackendEntrypoint
    """

    description = "Open netCDF files (.nc, .nc4, .cdf and .gz) using scipy in Xarray"
    url = "https://docs.xarray.dev/en/stable/generated/xarray.backends.ScipyBackendEntrypoint.html"
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
    mode: ScipyOpenModes = "r"
    format: ScipyFormats = None
    lock: Literal[False] | LockLike | None = None
    mmap: bool | None = None

    def guess_can_open(self, filename_or_obj: T_XarrayCanOpen) -> bool:
        magic_number = try_read_magic_number_from_file_or_path(filename_or_obj)
        if magic_number is not None and magic_number.startswith(b"\x1f\x8b"):
            with gzip.open(filename_or_obj) as f:  # type: ignore[arg-type]
                magic_number = try_read_magic_number_from_file_or_path(f)
        if magic_number is not None:
            return magic_number.startswith(b"CDF")

        if isinstance(filename_or_obj, (str, os.PathLike)):
            _, ext = os.path.splitext(filename_or_obj)
            return ext in {".nc", ".nc4", ".cdf", ".gz"}

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
        store = ScipyDataStore(
            filename_or_obj,
            mode=kwargs.pop("mode", self.mode),
            format=kwargs.pop("format", self.format),
            group=kwargs.pop("group", self.group),
            mmap=kwargs.pop("mmap", self.mmap),
            lock=kwargs.pop("lock", self.lock),
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


BACKEND_ENTRYPOINTS["scipy"] = ("scipy", ScipyBackendEntrypoint)
