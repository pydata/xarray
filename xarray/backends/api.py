import os.path
import warnings
from glob import glob
from io import BytesIO
from numbers import Number
from pathlib import Path
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Hashable,
    Iterable,
    Mapping,
    Tuple,
    Union,
)

import numpy as np

from .. import DataArray, Dataset, auto_combine, backends, coding, conventions
from ..core import indexing
from ..core.combine import (
    _infer_concat_order_from_positions,
    _nested_combine,
    combine_by_coords,
)
from ..core.utils import close_on_error, is_grib_path, is_remote_uri
from .common import AbstractDataStore, ArrayWriter
from .locks import _get_scheduler

if TYPE_CHECKING:
    try:
        from dask.delayed import Delayed
    except ImportError:
        Delayed = None


DATAARRAY_NAME = "__xarray_dataarray_name__"
DATAARRAY_VARIABLE = "__xarray_dataarray_variable__"


def _get_default_engine_remote_uri():
    try:
        import netCDF4  # noqa: F401

        engine = "netcdf4"
    except ImportError:  # pragma: no cover
        try:
            import pydap  # noqa: F401

            engine = "pydap"
        except ImportError:
            raise ValueError(
                "netCDF4 or pydap is required for accessing "
                "remote datasets via OPeNDAP"
            )
    return engine


def _get_default_engine_grib():
    msgs = []
    try:
        import Nio  # noqa: F401

        msgs += ["set engine='pynio' to access GRIB files with PyNIO"]
    except ImportError:  # pragma: no cover
        pass
    try:
        import cfgrib  # noqa: F401

        msgs += ["set engine='cfgrib' to access GRIB files with cfgrib"]
    except ImportError:  # pragma: no cover
        pass
    if msgs:
        raise ValueError(" or\n".join(msgs))
    else:
        raise ValueError("PyNIO or cfgrib is required for accessing " "GRIB files")


def _get_default_engine_gz():
    try:
        import scipy  # noqa: F401

        engine = "scipy"
    except ImportError:  # pragma: no cover
        raise ValueError("scipy is required for accessing .gz files")
    return engine


def _get_default_engine_netcdf():
    try:
        import netCDF4  # noqa: F401

        engine = "netcdf4"
    except ImportError:  # pragma: no cover
        try:
            import scipy.io.netcdf  # noqa: F401

            engine = "scipy"
        except ImportError:
            raise ValueError(
                "cannot read or write netCDF files without "
                "netCDF4-python or scipy installed"
            )
    return engine


def _get_engine_from_magic_number(filename_or_obj):
    # check byte header to determine file type
    if isinstance(filename_or_obj, bytes):
        magic_number = filename_or_obj[:8]
    else:
        if filename_or_obj.tell() != 0:
            raise ValueError(
                "file-like object read/write pointer not at zero "
                "please close and reopen, or use a context "
                "manager"
            )
        magic_number = filename_or_obj.read(8)
        filename_or_obj.seek(0)

    if magic_number.startswith(b"CDF"):
        engine = "scipy"
    elif magic_number.startswith(b"\211HDF\r\n\032\n"):
        engine = "h5netcdf"
        if isinstance(filename_or_obj, bytes):
            raise ValueError(
                "can't open netCDF4/HDF5 as bytes "
                "try passing a path or file-like object"
            )
    else:
        if isinstance(filename_or_obj, bytes) and len(filename_or_obj) > 80:
            filename_or_obj = filename_or_obj[:80] + b"..."
        raise ValueError(
            "{} is not a valid netCDF file "
            "did you mean to pass a string for a path instead?".format(filename_or_obj)
        )
    return engine


def _get_default_engine(path, allow_remote=False):
    if allow_remote and is_remote_uri(path):
        engine = _get_default_engine_remote_uri()
    elif is_grib_path(path):
        engine = _get_default_engine_grib()
    elif path.endswith(".gz"):
        engine = _get_default_engine_gz()
    else:
        engine = _get_default_engine_netcdf()
    return engine


def _normalize_path(path):
    if is_remote_uri(path):
        return path
    else:
        return os.path.abspath(os.path.expanduser(path))


def _validate_dataset_names(dataset):
    """DataArray.name and Dataset keys must be a string or None"""

    def check_name(name):
        if isinstance(name, str):
            if not name:
                raise ValueError(
                    "Invalid name for DataArray or Dataset key: "
                    "string must be length 1 or greater for "
                    "serialization to netCDF files"
                )
        elif name is not None:
            raise TypeError(
                "DataArray.name or Dataset key must be either a "
                "string or None for serialization to netCDF files"
            )

    for k in dataset.variables:
        check_name(k)


def _validate_attrs(dataset):
    """`attrs` must have a string key and a value which is either: a number,
    a string, an ndarray or a list/tuple of numbers/strings.
    """

    def check_attr(name, value):
        if isinstance(name, str):
            if not name:
                raise ValueError(
                    "Invalid name for attr: string must be "
                    "length 1 or greater for serialization to "
                    "netCDF files"
                )
        else:
            raise TypeError(
                "Invalid name for attr: {} must be a string for "
                "serialization to netCDF files".format(name)
            )

        if not isinstance(value, (str, Number, np.ndarray, np.number, list, tuple)):
            raise TypeError(
                "Invalid value for attr: {} must be a number, "
                "a string, an ndarray or a list/tuple of "
                "numbers/strings for serialization to netCDF "
                "files".format(value)
            )

    # Check attrs on the dataset itself
    for k, v in dataset.attrs.items():
        check_attr(k, v)

    # Check attrs on each variable within the dataset
    for variable in dataset.variables.values():
        for k, v in variable.attrs.items():
            check_attr(k, v)


def _protect_dataset_variables_inplace(dataset, cache):
    for name, variable in dataset.variables.items():
        if name not in variable.dims:
            # no need to protect IndexVariable objects
            data = indexing.CopyOnWriteArray(variable._data)
            if cache:
                data = indexing.MemoryCachedArray(data)
            variable.data = data


def _finalize_store(write, store):
    """ Finalize this store by explicitly syncing and closing"""
    del write  # ensure writing is done first
    store.close()


def load_dataset(filename_or_obj, **kwargs):
    """Open, load into memory, and close a Dataset from a file or file-like
    object.

    This is a thin wrapper around :py:meth:`~xarray.open_dataset`. It differs
    from `open_dataset` in that it loads the Dataset into memory, closes the
    file, and returns the Dataset. In contrast, `open_dataset` keeps the file
    handle open and lazy loads its contents. All parameters are passed directly
    to `open_dataset`. See that documentation for further details.

    Returns
    -------
    dataset : Dataset
        The newly created Dataset.

    See Also
    --------
    open_dataset
    """
    if "cache" in kwargs:
        raise TypeError("cache has no effect in this context")

    with open_dataset(filename_or_obj, **kwargs) as ds:
        return ds.load()


def load_dataarray(filename_or_obj, **kwargs):
    """Open, load into memory, and close a DataArray from a file or file-like
    object containing a single data variable.

    This is a thin wrapper around :py:meth:`~xarray.open_dataarray`. It differs
    from `open_dataarray` in that it loads the Dataset into memory, closes the
    file, and returns the Dataset. In contrast, `open_dataarray` keeps the file
    handle open and lazy loads its contents. All parameters are passed directly
    to `open_dataarray`. See that documentation for further details.

    Returns
    -------
    datarray : DataArray
        The newly created DataArray.

    See Also
    --------
    open_dataarray
    """
    if "cache" in kwargs:
        raise TypeError("cache has no effect in this context")

    with open_dataarray(filename_or_obj, **kwargs) as da:
        return da.load()


def open_dataset(
    filename_or_obj,
    group=None,
    decode_cf=True,
    mask_and_scale=None,
    decode_times=True,
    autoclose=None,
    concat_characters=True,
    decode_coords=True,
    engine=None,
    chunks=None,
    lock=None,
    cache=None,
    drop_variables=None,
    backend_kwargs=None,
    use_cftime=None,
):
    """Open and decode a dataset from a file or file-like object.

    Parameters
    ----------
    filename_or_obj : str, Path, file or xarray.backends.*DataStore
        Strings and Path objects are interpreted as a path to a netCDF file
        or an OpenDAP URL and opened with python-netCDF4, unless the filename
        ends with .gz, in which case the file is gunzipped and opened with
        scipy.io.netcdf (only netCDF3 supported). Byte-strings or file-like
        objects are opened by scipy.io.netcdf (netCDF3) or h5py (netCDF4/HDF).
    group : str, optional
        Path to the netCDF4 group in the given file to open (only works for
        netCDF4 files).
    decode_cf : bool, optional
        Whether to decode these variables, assuming they were saved according
        to CF conventions.
    mask_and_scale : bool, optional
        If True, replace array values equal to `_FillValue` with NA and scale
        values according to the formula `original_values * scale_factor +
        add_offset`, where `_FillValue`, `scale_factor` and `add_offset` are
        taken from variable attributes (if they exist).  If the `_FillValue` or
        `missing_value` attribute contains multiple values a warning will be
        issued and all array values matching one of the multiple values will
        be replaced by NA. mask_and_scale defaults to True except for the
        pseudonetcdf backend.
    decode_times : bool, optional
        If True, decode times encoded in the standard NetCDF datetime format
        into datetime objects. Otherwise, leave them encoded as numbers.
    autoclose : bool, optional
        If True, automatically close files to avoid OS Error of too many files
        being open.  However, this option doesn't work with streams, e.g.,
        BytesIO.
    concat_characters : bool, optional
        If True, concatenate along the last dimension of character arrays to
        form string arrays. Dimensions will only be concatenated over (and
        removed) if they have no corresponding variable and if they are only
        used as the last dimension of character arrays.
    decode_coords : bool, optional
        If True, decode the 'coordinates' attribute to identify coordinates in
        the resulting dataset.
    engine : {'netcdf4', 'scipy', 'pydap', 'h5netcdf', 'pynio', 'cfgrib', \
        'pseudonetcdf'}, optional
        Engine to use when reading files. If not provided, the default engine
        is chosen based on available dependencies, with a preference for
        'netcdf4'.
    chunks : int or dict, optional
        If chunks is provided, it used to load the new dataset into dask
        arrays. ``chunks={}`` loads the dataset with dask using a single
        chunk for all arrays.
    lock : False or duck threading.Lock, optional
        Resource lock to use when reading data from disk. Only relevant when
        using dask or another form of parallelism. By default, appropriate
        locks are chosen to safely read and write files with the currently
        active dask scheduler.
    cache : bool, optional
        If True, cache data loaded from the underlying datastore in memory as
        NumPy arrays when accessed to avoid reading from the underlying data-
        store multiple times. Defaults to True unless you specify the `chunks`
        argument to use dask, in which case it defaults to False. Does not
        change the behavior of coordinates corresponding to dimensions, which
        always load their data from disk into a ``pandas.Index``.
    drop_variables: string or iterable, optional
        A variable or list of variables to exclude from being parsed from the
        dataset. This may be useful to drop variables with problems or
        inconsistent values.
    backend_kwargs: dictionary, optional
        A dictionary of keyword arguments to pass on to the backend. This
        may be useful when backend options would improve performance or
        allow user control of dataset processing.
    use_cftime: bool, optional
        Only relevant if encoded dates come from a standard calendar
        (e.g. 'gregorian', 'proleptic_gregorian', 'standard', or not
        specified).  If None (default), attempt to decode times to
        ``np.datetime64[ns]`` objects; if this is not possible, decode times to
        ``cftime.datetime`` objects. If True, always decode times to
        ``cftime.datetime`` objects, regardless of whether or not they can be
        represented using ``np.datetime64[ns]`` objects.  If False, always
        decode times to ``np.datetime64[ns]`` objects; if this is not possible
        raise an error.

    Returns
    -------
    dataset : Dataset
        The newly created dataset.

    Notes
    -----
    ``open_dataset`` opens the file with read-only access. When you modify
    values of a Dataset, even one linked to files on disk, only the in-memory
    copy you are manipulating in xarray is modified: the original file on disk
    is never touched.

    See Also
    --------
    open_mfdataset
    """
    engines = [
        None,
        "netcdf4",
        "scipy",
        "pydap",
        "h5netcdf",
        "pynio",
        "cfgrib",
        "pseudonetcdf",
    ]
    if engine not in engines:
        raise ValueError(
            "unrecognized engine for open_dataset: {}\n"
            "must be one of: {}".format(engine, engines)
        )

    if autoclose is not None:
        warnings.warn(
            "The autoclose argument is no longer used by "
            "xarray.open_dataset() and is now ignored; it will be removed in "
            "a future version of xarray. If necessary, you can control the "
            "maximum number of simultaneous open files with "
            "xarray.set_options(file_cache_maxsize=...).",
            FutureWarning,
            stacklevel=2,
        )

    if mask_and_scale is None:
        mask_and_scale = not engine == "pseudonetcdf"

    if not decode_cf:
        mask_and_scale = False
        decode_times = False
        concat_characters = False
        decode_coords = False

    if cache is None:
        cache = chunks is None

    if backend_kwargs is None:
        backend_kwargs = {}

    def maybe_decode_store(store, lock=False):
        ds = conventions.decode_cf(
            store,
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            concat_characters=concat_characters,
            decode_coords=decode_coords,
            drop_variables=drop_variables,
            use_cftime=use_cftime,
        )

        _protect_dataset_variables_inplace(ds, cache)

        if chunks is not None:
            from dask.base import tokenize

            # if passed an actual file path, augment the token with
            # the file modification time
            if isinstance(filename_or_obj, str) and not is_remote_uri(filename_or_obj):
                mtime = os.path.getmtime(filename_or_obj)
            else:
                mtime = None
            token = tokenize(
                filename_or_obj,
                mtime,
                group,
                decode_cf,
                mask_and_scale,
                decode_times,
                concat_characters,
                decode_coords,
                engine,
                chunks,
                drop_variables,
                use_cftime,
            )
            name_prefix = "open_dataset-%s" % token
            ds2 = ds.chunk(chunks, name_prefix=name_prefix, token=token)
            ds2._file_obj = ds._file_obj
        else:
            ds2 = ds

        return ds2

    if isinstance(filename_or_obj, Path):
        filename_or_obj = str(filename_or_obj)

    if isinstance(filename_or_obj, AbstractDataStore):
        store = filename_or_obj

    elif isinstance(filename_or_obj, str):
        filename_or_obj = _normalize_path(filename_or_obj)

        if engine is None:
            engine = _get_default_engine(filename_or_obj, allow_remote=True)
        if engine == "netcdf4":
            store = backends.NetCDF4DataStore.open(
                filename_or_obj, group=group, lock=lock, **backend_kwargs
            )
        elif engine == "scipy":
            store = backends.ScipyDataStore(filename_or_obj, **backend_kwargs)
        elif engine == "pydap":
            store = backends.PydapDataStore.open(filename_or_obj, **backend_kwargs)
        elif engine == "h5netcdf":
            store = backends.H5NetCDFStore(
                filename_or_obj, group=group, lock=lock, **backend_kwargs
            )
        elif engine == "pynio":
            store = backends.NioDataStore(filename_or_obj, lock=lock, **backend_kwargs)
        elif engine == "pseudonetcdf":
            store = backends.PseudoNetCDFDataStore.open(
                filename_or_obj, lock=lock, **backend_kwargs
            )
        elif engine == "cfgrib":
            store = backends.CfGribDataStore(
                filename_or_obj, lock=lock, **backend_kwargs
            )

    else:
        if engine not in [None, "scipy", "h5netcdf"]:
            raise ValueError(
                "can only read bytes or file-like objects "
                "with engine='scipy' or 'h5netcdf'"
            )
        engine = _get_engine_from_magic_number(filename_or_obj)
        if engine == "scipy":
            store = backends.ScipyDataStore(filename_or_obj, **backend_kwargs)
        elif engine == "h5netcdf":
            store = backends.H5NetCDFStore(
                filename_or_obj, group=group, lock=lock, **backend_kwargs
            )

    with close_on_error(store):
        ds = maybe_decode_store(store)

    # Ensure source filename always stored in dataset object (GH issue #2550)
    if "source" not in ds.encoding:
        if isinstance(filename_or_obj, str):
            ds.encoding["source"] = filename_or_obj

    return ds


def open_dataarray(
    filename_or_obj,
    group=None,
    decode_cf=True,
    mask_and_scale=None,
    decode_times=True,
    autoclose=None,
    concat_characters=True,
    decode_coords=True,
    engine=None,
    chunks=None,
    lock=None,
    cache=None,
    drop_variables=None,
    backend_kwargs=None,
    use_cftime=None,
):
    """Open an DataArray from a file or file-like object containing a single
    data variable.

    This is designed to read netCDF files with only one data variable. If
    multiple variables are present then a ValueError is raised.

    Parameters
    ----------
    filename_or_obj : str, Path, file or xarray.backends.*DataStore
        Strings and Paths are interpreted as a path to a netCDF file or an
        OpenDAP URL and opened with python-netCDF4, unless the filename ends
        with .gz, in which case the file is gunzipped and opened with
        scipy.io.netcdf (only netCDF3 supported). Byte-strings or file-like
        objects are opened by scipy.io.netcdf (netCDF3) or h5py (netCDF4/HDF).
    group : str, optional
        Path to the netCDF4 group in the given file to open (only works for
        netCDF4 files).
    decode_cf : bool, optional
        Whether to decode these variables, assuming they were saved according
        to CF conventions.
    mask_and_scale : bool, optional
        If True, replace array values equal to `_FillValue` with NA and scale
        values according to the formula `original_values * scale_factor +
        add_offset`, where `_FillValue`, `scale_factor` and `add_offset` are
        taken from variable attributes (if they exist).  If the `_FillValue` or
        `missing_value` attribute contains multiple values a warning will be
        issued and all array values matching one of the multiple values will
        be replaced by NA. mask_and_scale defaults to True except for the
        pseudonetcdf backend.
    decode_times : bool, optional
        If True, decode times encoded in the standard NetCDF datetime format
        into datetime objects. Otherwise, leave them encoded as numbers.
    concat_characters : bool, optional
        If True, concatenate along the last dimension of character arrays to
        form string arrays. Dimensions will only be concatenated over (and
        removed) if they have no corresponding variable and if they are only
        used as the last dimension of character arrays.
    decode_coords : bool, optional
        If True, decode the 'coordinates' attribute to identify coordinates in
        the resulting dataset.
    engine : {'netcdf4', 'scipy', 'pydap', 'h5netcdf', 'pynio', 'cfgrib'}, \
        optional
        Engine to use when reading files. If not provided, the default engine
        is chosen based on available dependencies, with a preference for
        'netcdf4'.
    chunks : int or dict, optional
        If chunks is provided, it used to load the new dataset into dask
        arrays.
    lock : False or duck threading.Lock, optional
        Resource lock to use when reading data from disk. Only relevant when
        using dask or another form of parallelism. By default, appropriate
        locks are chosen to safely read and write files with the currently
        active dask scheduler.
    cache : bool, optional
        If True, cache data loaded from the underlying datastore in memory as
        NumPy arrays when accessed to avoid reading from the underlying data-
        store multiple times. Defaults to True unless you specify the `chunks`
        argument to use dask, in which case it defaults to False. Does not
        change the behavior of coordinates corresponding to dimensions, which
        always load their data from disk into a ``pandas.Index``.
    drop_variables: string or iterable, optional
        A variable or list of variables to exclude from being parsed from the
        dataset. This may be useful to drop variables with problems or
        inconsistent values.
    backend_kwargs: dictionary, optional
        A dictionary of keyword arguments to pass on to the backend. This
        may be useful when backend options would improve performance or
        allow user control of dataset processing.
    use_cftime: bool, optional
        Only relevant if encoded dates come from a standard calendar
        (e.g. 'gregorian', 'proleptic_gregorian', 'standard', or not
        specified).  If None (default), attempt to decode times to
        ``np.datetime64[ns]`` objects; if this is not possible, decode times to
        ``cftime.datetime`` objects. If True, always decode times to
        ``cftime.datetime`` objects, regardless of whether or not they can be
        represented using ``np.datetime64[ns]`` objects.  If False, always
        decode times to ``np.datetime64[ns]`` objects; if this is not possible
        raise an error.

    Notes
    -----
    This is designed to be fully compatible with `DataArray.to_netcdf`. Saving
    using `DataArray.to_netcdf` and then loading with this function will
    produce an identical result.

    All parameters are passed directly to `xarray.open_dataset`. See that
    documentation for further details.

    See also
    --------
    open_dataset
    """

    dataset = open_dataset(
        filename_or_obj,
        group=group,
        decode_cf=decode_cf,
        mask_and_scale=mask_and_scale,
        decode_times=decode_times,
        autoclose=autoclose,
        concat_characters=concat_characters,
        decode_coords=decode_coords,
        engine=engine,
        chunks=chunks,
        lock=lock,
        cache=cache,
        drop_variables=drop_variables,
        backend_kwargs=backend_kwargs,
        use_cftime=use_cftime,
    )

    if len(dataset.data_vars) != 1:
        raise ValueError(
            "Given file dataset contains more than one data "
            "variable. Please read with xarray.open_dataset and "
            "then select the variable you want."
        )
    else:
        data_array, = dataset.data_vars.values()

    data_array._file_obj = dataset._file_obj

    # Reset names if they were changed during saving
    # to ensure that we can 'roundtrip' perfectly
    if DATAARRAY_NAME in dataset.attrs:
        data_array.name = dataset.attrs[DATAARRAY_NAME]
        del dataset.attrs[DATAARRAY_NAME]

    if data_array.name == DATAARRAY_VARIABLE:
        data_array.name = None

    return data_array


class _MultiFileCloser:
    __slots__ = ("file_objs",)

    def __init__(self, file_objs):
        self.file_objs = file_objs

    def close(self):
        for f in self.file_objs:
            f.close()


def open_mfdataset(
    paths,
    chunks=None,
    concat_dim="_not_supplied",
    compat="no_conflicts",
    preprocess=None,
    engine=None,
    lock=None,
    data_vars="all",
    coords="different",
    combine="_old_auto",
    autoclose=None,
    parallel=False,
    join="outer",
    **kwargs,
):
    """Open multiple files as a single dataset.

    If combine='by_coords' then the function ``combine_by_coords`` is used to combine
    the datasets into one before returning the result, and if combine='nested' then
    ``combine_nested`` is used. The filepaths must be structured according to which
    combining function is used, the details of which are given in the documentation for
    ``combine_by_coords`` and ``combine_nested``. By default the old (now deprecated)
    ``auto_combine`` will be used, please specify either ``combine='by_coords'`` or
    ``combine='nested'`` in future. Requires dask to be installed. See documentation for
    details on dask [1]. Attributes from the first dataset file are used for the
    combined dataset.

    Parameters
    ----------
    paths : str or sequence
        Either a string glob in the form "path/to/my/files/*.nc" or an explicit list of
        files to open. Paths can be given as strings or as pathlib Paths. If
        concatenation along more than one dimension is desired, then ``paths`` must be a
        nested list-of-lists (see ``manual_combine`` for details). (A string glob will
        be expanded to a 1-dimensional list.)
    chunks : int or dict, optional
        Dictionary with keys given by dimension names and values given by chunk sizes.
        In general, these should divide the dimensions of each dataset. If int, chunk
        each dimension by ``chunks``. By default, chunks will be chosen to load entire
        input files into memory at once. This has a major impact on performance: please
        see the full documentation for more details [2].
    concat_dim : str, or list of str, DataArray, Index or None, optional
        Dimensions to concatenate files along.  You only need to provide this argument
        if any of the dimensions along which you want to concatenate is not a dimension
        in the original datasets, e.g., if you want to stack a collection of 2D arrays
        along a third dimension. Set ``concat_dim=[..., None, ...]`` explicitly to
        disable concatenation along a particular dimension.
    combine : {'by_coords', 'nested'}, optional
        Whether ``xarray.combine_by_coords`` or ``xarray.combine_nested`` is used to
        combine all the data. If this argument is not provided, `xarray.auto_combine` is
        used, but in the future this behavior will switch to use
        `xarray.combine_by_coords` by default.
    compat : {'identical', 'equals', 'broadcast_equals',
              'no_conflicts', 'override'}, optional
        String indicating how to compare variables of the same name for
        potential conflicts when merging:
         * 'broadcast_equals': all values must be equal when variables are
           broadcast against each other to ensure common dimensions.
         * 'equals': all values and dimensions must be the same.
         * 'identical': all values, dimensions and attributes must be the
           same.
         * 'no_conflicts': only values which are not null in both datasets
           must be equal. The returned dataset then contains the combination
           of all non-null values.
         * 'override': skip comparing and pick variable from first dataset
    preprocess : callable, optional
        If provided, call this function on each dataset prior to concatenation.
        You can find the file-name from which each dataset was loaded in
        ``ds.encoding['source']``.
    engine : {'netcdf4', 'scipy', 'pydap', 'h5netcdf', 'pynio', 'cfgrib'}, \
        optional
        Engine to use when reading files. If not provided, the default engine
        is chosen based on available dependencies, with a preference for
        'netcdf4'.
    lock : False or duck threading.Lock, optional
        Resource lock to use when reading data from disk. Only relevant when
        using dask or another form of parallelism. By default, appropriate
        locks are chosen to safely read and write files with the currently
        active dask scheduler.
    data_vars : {'minimal', 'different', 'all' or list of str}, optional
        These data variables will be concatenated together:
          * 'minimal': Only data variables in which the dimension already
            appears are included.
          * 'different': Data variables which are not equal (ignoring
            attributes) across all datasets are also concatenated (as well as
            all for which dimension already appears). Beware: this option may
            load the data payload of data variables into memory if they are not
            already loaded.
          * 'all': All data variables will be concatenated.
          * list of str: The listed data variables will be concatenated, in
            addition to the 'minimal' data variables.
    coords : {'minimal', 'different', 'all' or list of str}, optional
        These coordinate variables will be concatenated together:
         * 'minimal': Only coordinates in which the dimension already appears
           are included.
         * 'different': Coordinates which are not equal (ignoring attributes)
           across all datasets are also concatenated (as well as all for which
           dimension already appears). Beware: this option may load the data
           payload of coordinate variables into memory if they are not already
           loaded.
         * 'all': All coordinate variables will be concatenated, except
           those corresponding to other dimensions.
         * list of str: The listed coordinate variables will be concatenated,
           in addition the 'minimal' coordinates.
    parallel : bool, optional
        If True, the open and preprocess steps of this function will be
        performed in parallel using ``dask.delayed``. Default is False.
    join : {'outer', 'inner', 'left', 'right', 'exact, 'override'}, optional
        String indicating how to combine differing indexes
        (excluding concat_dim) in objects

        - 'outer': use the union of object indexes
        - 'inner': use the intersection of object indexes
        - 'left': use indexes from the first object with each dimension
        - 'right': use indexes from the last object with each dimension
        - 'exact': instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - 'override': if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.
    **kwargs : optional
        Additional arguments passed on to :py:func:`xarray.open_dataset`.

    Returns
    -------
    xarray.Dataset

    Notes
    -----
    ``open_mfdataset`` opens files with read-only access. When you modify values
    of a Dataset, even one linked to files on disk, only the in-memory copy you
    are manipulating in xarray is modified: the original file on disk is never
    touched.

    See Also
    --------
    combine_by_coords
    combine_nested
    auto_combine
    open_dataset

    References
    ----------

    .. [1] http://xarray.pydata.org/en/stable/dask.html
    .. [2] http://xarray.pydata.org/en/stable/dask.html#chunking-and-performance
    """
    if isinstance(paths, str):
        if is_remote_uri(paths):
            raise ValueError(
                "cannot do wild-card matching for paths that are remote URLs: "
                "{!r}. Instead, supply paths as an explicit list of strings.".format(
                    paths
                )
            )
        paths = sorted(glob(paths))
    else:
        paths = [str(p) if isinstance(p, Path) else p for p in paths]

    if not paths:
        raise OSError("no files to open")

    # If combine='by_coords' then this is unnecessary, but quick.
    # If combine='nested' then this creates a flat list which is easier to
    # iterate over, while saving the originally-supplied structure as "ids"
    if combine == "nested":
        if str(concat_dim) == "_not_supplied":
            raise ValueError("Must supply concat_dim when using " "combine='nested'")
        else:
            if isinstance(concat_dim, (str, DataArray)) or concat_dim is None:
                concat_dim = [concat_dim]
    combined_ids_paths = _infer_concat_order_from_positions(paths)
    ids, paths = (list(combined_ids_paths.keys()), list(combined_ids_paths.values()))

    open_kwargs = dict(
        engine=engine, chunks=chunks or {}, lock=lock, autoclose=autoclose, **kwargs
    )

    if parallel:
        import dask

        # wrap the open_dataset, getattr, and preprocess with delayed
        open_ = dask.delayed(open_dataset)
        getattr_ = dask.delayed(getattr)
        if preprocess is not None:
            preprocess = dask.delayed(preprocess)
    else:
        open_ = open_dataset
        getattr_ = getattr

    datasets = [open_(p, **open_kwargs) for p in paths]
    file_objs = [getattr_(ds, "_file_obj") for ds in datasets]
    if preprocess is not None:
        datasets = [preprocess(ds) for ds in datasets]

    if parallel:
        # calling compute here will return the datasets/file_objs lists,
        # the underlying datasets will still be stored as dask arrays
        datasets, file_objs = dask.compute(datasets, file_objs)

    # Combine all datasets, closing them in case of a ValueError
    try:
        if combine == "_old_auto":
            # Use the old auto_combine for now
            # Remove this after deprecation cycle from #2616 is complete
            basic_msg = dedent(
                """\
            In xarray version 0.15 the default behaviour of `open_mfdataset`
            will change. To retain the existing behavior, pass
            combine='nested'. To use future default behavior, pass
            combine='by_coords'. See
            http://xarray.pydata.org/en/stable/combining.html#combining-multi
            """
            )
            warnings.warn(basic_msg, FutureWarning, stacklevel=2)

            combined = auto_combine(
                datasets,
                concat_dim=concat_dim,
                compat=compat,
                data_vars=data_vars,
                coords=coords,
                join=join,
                from_openmfds=True,
            )
        elif combine == "nested":
            # Combined nested list by successive concat and merge operations
            # along each dimension, using structure given by "ids"
            combined = _nested_combine(
                datasets,
                concat_dims=concat_dim,
                compat=compat,
                data_vars=data_vars,
                coords=coords,
                ids=ids,
                join=join,
            )
        elif combine == "by_coords":
            # Redo ordering from coordinates, ignoring how they were ordered
            # previously
            combined = combine_by_coords(
                datasets, compat=compat, data_vars=data_vars, coords=coords, join=join
            )
        else:
            raise ValueError(
                "{} is an invalid option for the keyword argument"
                " ``combine``".format(combine)
            )
    except ValueError:
        for ds in datasets:
            ds.close()
        raise

    combined._file_obj = _MultiFileCloser(file_objs)
    combined.attrs = datasets[0].attrs
    return combined


WRITEABLE_STORES: Dict[str, Callable] = {
    "netcdf4": backends.NetCDF4DataStore.open,
    "scipy": backends.ScipyDataStore,
    "h5netcdf": backends.H5NetCDFStore,
}


def to_netcdf(
    dataset: Dataset,
    path_or_file=None,
    mode: str = "w",
    format: str = None,
    group: str = None,
    engine: str = None,
    encoding: Mapping = None,
    unlimited_dims: Iterable[Hashable] = None,
    compute: bool = True,
    multifile: bool = False,
    invalid_netcdf: bool = False,
) -> Union[Tuple[ArrayWriter, AbstractDataStore], bytes, "Delayed", None]:
    """This function creates an appropriate datastore for writing a dataset to
    disk as a netCDF file

    See `Dataset.to_netcdf` for full API docs.

    The ``multifile`` argument is only for the private use of save_mfdataset.
    """
    if isinstance(path_or_file, Path):
        path_or_file = str(path_or_file)

    if encoding is None:
        encoding = {}

    if path_or_file is None:
        if engine is None:
            engine = "scipy"
        elif engine != "scipy":
            raise ValueError(
                "invalid engine for creating bytes with "
                "to_netcdf: %r. Only the default engine "
                "or engine='scipy' is supported" % engine
            )
        if not compute:
            raise NotImplementedError(
                "to_netcdf() with compute=False is not yet implemented when "
                "returning bytes"
            )
    elif isinstance(path_or_file, str):
        if engine is None:
            engine = _get_default_engine(path_or_file)
        path_or_file = _normalize_path(path_or_file)
    else:  # file-like object
        engine = "scipy"

    # validate Dataset keys, DataArray names, and attr keys/values
    _validate_dataset_names(dataset)
    _validate_attrs(dataset)

    try:
        store_open = WRITEABLE_STORES[engine]
    except KeyError:
        raise ValueError("unrecognized engine for to_netcdf: %r" % engine)

    if format is not None:
        format = format.upper()

    # handle scheduler specific logic
    scheduler = _get_scheduler()
    have_chunks = any(v.chunks for v in dataset.variables.values())

    autoclose = have_chunks and scheduler in ["distributed", "multiprocessing"]
    if autoclose and engine == "scipy":
        raise NotImplementedError(
            "Writing netCDF files with the %s backend "
            "is not currently supported with dask's %s "
            "scheduler" % (engine, scheduler)
        )

    target = path_or_file if path_or_file is not None else BytesIO()
    kwargs = dict(autoclose=True) if autoclose else {}
    if invalid_netcdf:
        if engine == "h5netcdf":
            kwargs["invalid_netcdf"] = invalid_netcdf
        else:
            raise ValueError(
                "unrecognized option 'invalid_netcdf' for engine %s" % engine
            )
    store = store_open(target, mode, format, group, **kwargs)

    if unlimited_dims is None:
        unlimited_dims = dataset.encoding.get("unlimited_dims", None)
    if unlimited_dims is not None:
        if isinstance(unlimited_dims, str) or not isinstance(unlimited_dims, Iterable):
            unlimited_dims = [unlimited_dims]
        else:
            unlimited_dims = list(unlimited_dims)

    writer = ArrayWriter()

    # TODO: figure out how to refactor this logic (here and in save_mfdataset)
    # to avoid this mess of conditionals
    try:
        # TODO: allow this work (setting up the file for writing array data)
        # to be parallelized with dask
        dump_to_store(
            dataset, store, writer, encoding=encoding, unlimited_dims=unlimited_dims
        )
        if autoclose:
            store.close()

        if multifile:
            return writer, store

        writes = writer.sync(compute=compute)

        if path_or_file is None:
            store.sync()
            return target.getvalue()
    finally:
        if not multifile and compute:
            store.close()

    if not compute:
        import dask

        return dask.delayed(_finalize_store)(writes, store)
    return None


def dump_to_store(
    dataset, store, writer=None, encoder=None, encoding=None, unlimited_dims=None
):
    """Store dataset contents to a backends.*DataStore object."""
    if writer is None:
        writer = ArrayWriter()

    if encoding is None:
        encoding = {}

    variables, attrs = conventions.encode_dataset_coordinates(dataset)

    check_encoding = set()
    for k, enc in encoding.items():
        # no need to shallow copy the variable again; that already happened
        # in encode_dataset_coordinates
        variables[k].encoding = enc
        check_encoding.add(k)

    if encoder:
        variables, attrs = encoder(variables, attrs)

    store.store(variables, attrs, check_encoding, writer, unlimited_dims=unlimited_dims)


def save_mfdataset(
    datasets, paths, mode="w", format=None, groups=None, engine=None, compute=True
):
    """Write multiple datasets to disk as netCDF files simultaneously.

    This function is intended for use with datasets consisting of dask.array
    objects, in which case it can write the multiple datasets to disk
    simultaneously using a shared thread pool.

    When not using dask, it is no different than calling ``to_netcdf``
    repeatedly.

    Parameters
    ----------
    datasets : list of xarray.Dataset
        List of datasets to save.
    paths : list of str or list of Paths
        List of paths to which to save each corresponding dataset.
    mode : {'w', 'a'}, optional
        Write ('w') or append ('a') mode. If mode='w', any existing file at
        these locations will be overwritten.
    format : {'NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_64BIT',
              'NETCDF3_CLASSIC'}, optional

        File format for the resulting netCDF file:

        * NETCDF4: Data is stored in an HDF5 file, using netCDF4 API
          features.
        * NETCDF4_CLASSIC: Data is stored in an HDF5 file, using only
          netCDF 3 compatible API features.
        * NETCDF3_64BIT: 64-bit offset version of the netCDF 3 file format,
          which fully supports 2+ GB files, but is only compatible with
          clients linked against netCDF version 3.6.0 or later.
        * NETCDF3_CLASSIC: The classic netCDF 3 file format. It does not
          handle 2+ GB files very well.

        All formats are supported by the netCDF4-python library.
        scipy.io.netcdf only supports the last two formats.

        The default format is NETCDF4 if you are saving a file to disk and
        have the netCDF4-python library available. Otherwise, xarray falls
        back to using scipy to write netCDF files and defaults to the
        NETCDF3_64BIT format (scipy does not support netCDF4).
    groups : list of str, optional
        Paths to the netCDF4 group in each corresponding file to which to save
        datasets (only works for format='NETCDF4'). The groups will be created
        if necessary.
    engine : {'netcdf4', 'scipy', 'h5netcdf'}, optional
        Engine to use when writing netCDF files. If not provided, the
        default engine is chosen based on available dependencies, with a
        preference for 'netcdf4' if writing to a file on disk.
        See `Dataset.to_netcdf` for additional information.
    compute: boolean
        If true compute immediately, otherwise return a
        ``dask.delayed.Delayed`` object that can be computed later.

    Examples
    --------

    Save a dataset into one netCDF per year of data:

    >>> years, datasets = zip(*ds.groupby('time.year'))
    >>> paths = ['%s.nc' % y for y in years]
    >>> xr.save_mfdataset(datasets, paths)
    """
    if mode == "w" and len(set(paths)) < len(paths):
        raise ValueError(
            "cannot use mode='w' when writing multiple " "datasets to the same path"
        )

    for obj in datasets:
        if not isinstance(obj, Dataset):
            raise TypeError(
                "save_mfdataset only supports writing Dataset "
                "objects, received type %s" % type(obj)
            )

    if groups is None:
        groups = [None] * len(datasets)

    if len({len(datasets), len(paths), len(groups)}) > 1:
        raise ValueError(
            "must supply lists of the same length for the "
            "datasets, paths and groups arguments to "
            "save_mfdataset"
        )

    writers, stores = zip(
        *[
            to_netcdf(
                ds, path, mode, format, group, engine, compute=compute, multifile=True
            )
            for ds, path, group in zip(datasets, paths, groups)
        ]
    )

    try:
        writes = [w.sync(compute=compute) for w in writers]
    finally:
        if compute:
            for store in stores:
                store.close()

    if not compute:
        import dask

        return dask.delayed(
            [dask.delayed(_finalize_store)(w, s) for w, s in zip(writes, stores)]
        )


def _validate_datatypes_for_zarr_append(dataset):
    """DataArray.name and Dataset keys must be a string or None"""

    def check_dtype(var):
        if (
            not np.issubdtype(var.dtype, np.number)
            and not coding.strings.is_unicode_dtype(var.dtype)
            and not var.dtype == object
        ):
            # and not re.match('^bytes[1-9]+$', var.dtype.name)):
            raise ValueError(
                "Invalid dtype for data variable: {} "
                "dtype must be a subtype of number, "
                "a fixed sized string, a fixed size "
                "unicode string or an object".format(var)
            )

    for k in dataset.data_vars.values():
        check_dtype(k)


def _validate_append_dim_and_encoding(
    ds_to_append, store, append_dim, encoding, **open_kwargs
):
    try:
        ds = backends.zarr.open_zarr(store, **open_kwargs)
    except ValueError:  # store empty
        return
    if append_dim:
        if append_dim not in ds.dims:
            raise ValueError(f"{append_dim} not a valid dimension in the Dataset")
    for data_var in ds_to_append:
        if data_var in ds:
            if append_dim is None:
                raise ValueError(
                    "variable '{}' already exists, but append_dim "
                    "was not set".format(data_var)
                )
            if data_var in encoding.keys():
                raise ValueError(
                    "variable '{}' already exists, but encoding was"
                    "provided".format(data_var)
                )


def to_zarr(
    dataset,
    store=None,
    mode=None,
    synchronizer=None,
    group=None,
    encoding=None,
    compute=True,
    consolidated=False,
    append_dim=None,
):
    """This function creates an appropriate datastore for writing a dataset to
    a zarr ztore

    See `Dataset.to_zarr` for full API docs.
    """
    if isinstance(store, Path):
        store = str(store)
    if encoding is None:
        encoding = {}

    # validate Dataset keys, DataArray names, and attr keys/values
    _validate_dataset_names(dataset)
    _validate_attrs(dataset)

    if mode == "a":
        _validate_datatypes_for_zarr_append(dataset)
        _validate_append_dim_and_encoding(
            dataset,
            store,
            append_dim,
            group=group,
            consolidated=consolidated,
            encoding=encoding,
        )

    zstore = backends.ZarrStore.open_group(
        store=store,
        mode=mode,
        synchronizer=synchronizer,
        group=group,
        consolidate_on_close=consolidated,
    )
    zstore.append_dim = append_dim
    writer = ArrayWriter()
    # TODO: figure out how to properly handle unlimited_dims
    dump_to_store(dataset, zstore, writer, encoding=encoding)
    writes = writer.sync(compute=compute)

    if compute:
        _finalize_store(writes, zstore)
    else:
        import dask

        return dask.delayed(_finalize_store)(writes, zstore)

    return zstore
