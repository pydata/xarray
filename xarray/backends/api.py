import os
import warnings
from glob import glob
from io import BytesIO
from numbers import Number
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Hashable,
    Iterable,
    Mapping,
    MutableMapping,
    Tuple,
    Union,
)

import numpy as np

from .. import backends, coding, conventions
from ..core import indexing
from ..core.combine import (
    _infer_concat_order_from_positions,
    _nested_combine,
    combine_by_coords,
)
from ..core.dataarray import DataArray
from ..core.dataset import Dataset, _get_chunk, _maybe_chunk
from ..core.utils import is_remote_uri
from . import plugins
from .common import AbstractDataStore, ArrayWriter, _normalize_path
from .locks import _get_scheduler

if TYPE_CHECKING:
    try:
        from dask.delayed import Delayed
    except ImportError:
        Delayed = None


DATAARRAY_NAME = "__xarray_dataarray_name__"
DATAARRAY_VARIABLE = "__xarray_dataarray_variable__"

ENGINES = {
    "netcdf4": backends.NetCDF4DataStore.open,
    "scipy": backends.ScipyDataStore,
    "pydap": backends.PydapDataStore.open,
    "h5netcdf": backends.H5NetCDFStore.open,
    "pynio": backends.NioDataStore,
    "pseudonetcdf": backends.PseudoNetCDFDataStore.open,
    "cfgrib": backends.CfGribDataStore,
    "zarr": backends.ZarrStore.open_group,
}


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


def _get_default_engine(path: str, allow_remote: bool = False):
    if allow_remote and is_remote_uri(path):
        return _get_default_engine_remote_uri()
    elif path.endswith(".gz"):
        return _get_default_engine_gz()
    else:
        return _get_default_engine_netcdf()


def _validate_dataset_names(dataset):
    """DataArray.name and Dataset keys must be a string or None"""

    def check_name(name):
        if isinstance(name, str):
            if not name:
                raise ValueError(
                    f"Invalid name {name!r} for DataArray or Dataset key: "
                    "string must be length 1 or greater for "
                    "serialization to netCDF files"
                )
        elif name is not None:
            raise TypeError(
                f"Invalid name {name!r} for DataArray or Dataset key: "
                "must be either a string or None for serialization to netCDF "
                "files"
            )

    for k in dataset.variables:
        check_name(k)


def _validate_attrs(dataset, invalid_netcdf=False):
    """`attrs` must have a string key and a value which is either: a number,
    a string, an ndarray, a list/tuple of numbers/strings, or a numpy.bool_.

    Notes
    -----
    A numpy.bool_ is only allowed when using the h5netcdf engine with
    `invalid_netcdf=True`.
    """

    valid_types = (str, Number, np.ndarray, np.number, list, tuple)
    if invalid_netcdf:
        valid_types += (np.bool_,)

    def check_attr(name, value, valid_types):
        if isinstance(name, str):
            if not name:
                raise ValueError(
                    f"Invalid name for attr {name!r}: string must be "
                    "length 1 or greater for serialization to "
                    "netCDF files"
                )
        else:
            raise TypeError(
                f"Invalid name for attr: {name!r} must be a string for "
                "serialization to netCDF files"
            )

        if not isinstance(value, valid_types):
            raise TypeError(
                f"Invalid value for attr {name!r}: {value!r}. For serialization to "
                "netCDF files, its value must be of one of the following types: "
                f"{', '.join([vtype.__name__ for vtype in valid_types])}"
            )

    # Check attrs on the dataset itself
    for k, v in dataset.attrs.items():
        check_attr(k, v, valid_types)

    # Check attrs on each variable within the dataset
    for variable in dataset.variables.values():
        for k, v in variable.attrs.items():
            check_attr(k, v, valid_types)


def _resolve_decoders_kwargs(decode_cf, open_backend_dataset_parameters, **decoders):
    for d in list(decoders):
        if decode_cf is False and d in open_backend_dataset_parameters:
            decoders[d] = False
        if decoders[d] is None:
            decoders.pop(d)
    return decoders


def _get_mtime(filename_or_obj):
    # if passed an actual file path, augment the token with
    # the file modification time
    mtime = None

    try:
        path = os.fspath(filename_or_obj)
    except TypeError:
        path = None

    if path and not is_remote_uri(path):
        mtime = os.path.getmtime(filename_or_obj)

    return mtime


def _protect_dataset_variables_inplace(dataset, cache):
    for name, variable in dataset.variables.items():
        if name not in variable.dims:
            # no need to protect IndexVariable objects
            data = indexing.CopyOnWriteArray(variable._data)
            if cache:
                data = indexing.MemoryCachedArray(data)
            variable.data = data


def _finalize_store(write, store):
    """Finalize this store by explicitly syncing and closing"""
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


def _chunk_ds(
    backend_ds,
    filename_or_obj,
    engine,
    chunks,
    overwrite_encoded_chunks,
    **extra_tokens,
):
    from dask.base import tokenize

    mtime = _get_mtime(filename_or_obj)
    token = tokenize(filename_or_obj, mtime, engine, chunks, **extra_tokens)
    name_prefix = f"open_dataset-{token}"

    variables = {}
    for name, var in backend_ds.variables.items():
        var_chunks = _get_chunk(var, chunks)
        variables[name] = _maybe_chunk(
            name,
            var,
            var_chunks,
            overwrite_encoded_chunks=overwrite_encoded_chunks,
            name_prefix=name_prefix,
            token=token,
        )
    return backend_ds._replace(variables)


def _dataset_from_backend_dataset(
    backend_ds,
    filename_or_obj,
    engine,
    chunks,
    cache,
    overwrite_encoded_chunks,
    **extra_tokens,
):
    if not isinstance(chunks, (int, dict)) and chunks not in {None, "auto"}:
        raise ValueError(
            f"chunks must be an int, dict, 'auto', or None. Instead found {chunks}."
        )

    _protect_dataset_variables_inplace(backend_ds, cache)
    if chunks is None:
        ds = backend_ds
    else:
        ds = _chunk_ds(
            backend_ds,
            filename_or_obj,
            engine,
            chunks,
            overwrite_encoded_chunks,
            **extra_tokens,
        )

    ds.set_close(backend_ds._close)

    # Ensure source filename always stored in dataset object (GH issue #2550)
    if "source" not in ds.encoding and isinstance(filename_or_obj, str):
        ds.encoding["source"] = filename_or_obj

    return ds


def open_dataset(
    filename_or_obj,
    *args,
    engine=None,
    chunks=None,
    cache=None,
    decode_cf=None,
    mask_and_scale=None,
    decode_times=None,
    decode_timedelta=None,
    use_cftime=None,
    concat_characters=None,
    decode_coords=None,
    drop_variables=None,
    backend_kwargs=None,
    **kwargs,
):
    """Open and decode a dataset from a file or file-like object.

    Parameters
    ----------
    filename_or_obj : str, Path, file-like or DataStore
        Strings and Path objects are interpreted as a path to a netCDF file
        or an OpenDAP URL and opened with python-netCDF4, unless the filename
        ends with .gz, in which case the file is gunzipped and opened with
        scipy.io.netcdf (only netCDF3 supported). Byte-strings or file-like
        objects are opened by scipy.io.netcdf (netCDF3) or h5py (netCDF4/HDF).
    engine : {"netcdf4", "scipy", "pydap", "h5netcdf", "pynio", "cfgrib", \
        "pseudonetcdf", "zarr"} or subclass of xarray.backends.BackendEntrypoint, optional
        Engine to use when reading files. If not provided, the default engine
        is chosen based on available dependencies, with a preference for
        "netcdf4". A custom backend class (a subclass of ``BackendEntrypoint``)
        can also be used.
    chunks : int or dict, optional
        If chunks is provided, it is used to load the new dataset into dask
        arrays. ``chunks=-1`` loads the dataset with dask using a single
        chunk for all arrays. `chunks={}`` loads the dataset with dask using
        engine preferred chunks if exposed by the backend, otherwise with
        a single chunk for all arrays.
        ``chunks='auto'`` will use dask ``auto`` chunking taking into account the
        engine preferred chunks. See dask chunking for more details.
    cache : bool, optional
        If True, cache data loaded from the underlying datastore in memory as
        NumPy arrays when accessed to avoid reading from the underlying data-
        store multiple times. Defaults to True unless you specify the `chunks`
        argument to use dask, in which case it defaults to False. Does not
        change the behavior of coordinates corresponding to dimensions, which
        always load their data from disk into a ``pandas.Index``.
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
        pseudonetcdf backend. This keyword may not be supported by all the backends.
    decode_times : bool, optional
        If True, decode times encoded in the standard NetCDF datetime format
        into datetime objects. Otherwise, leave them encoded as numbers.
        This keyword may not be supported by all the backends.
    decode_timedelta : bool, optional
        If True, decode variables and coordinates with time units in
        {"days", "hours", "minutes", "seconds", "milliseconds", "microseconds"}
        into timedelta objects. If False, leave them encoded as numbers.
        If None (default), assume the same value of decode_time.
        This keyword may not be supported by all the backends.
    use_cftime: bool, optional
        Only relevant if encoded dates come from a standard calendar
        (e.g. "gregorian", "proleptic_gregorian", "standard", or not
        specified).  If None (default), attempt to decode times to
        ``np.datetime64[ns]`` objects; if this is not possible, decode times to
        ``cftime.datetime`` objects. If True, always decode times to
        ``cftime.datetime`` objects, regardless of whether or not they can be
        represented using ``np.datetime64[ns]`` objects.  If False, always
        decode times to ``np.datetime64[ns]`` objects; if this is not possible
        raise an error. This keyword may not be supported by all the backends.
    concat_characters : bool, optional
        If True, concatenate along the last dimension of character arrays to
        form string arrays. Dimensions will only be concatenated over (and
        removed) if they have no corresponding variable and if they are only
        used as the last dimension of character arrays.
        This keyword may not be supported by all the backends.
    decode_coords : bool or {"coordinates", "all"}, optional
        Controls which variables are set as coordinate variables:

        - "coordinates" or True: Set variables referred to in the
          ``'coordinates'`` attribute of the datasets or individual variables
          as coordinate variables.
        - "all": Set variables referred to in  ``'grid_mapping'``, ``'bounds'`` and
          other attributes as coordinate variables.
    drop_variables: str or iterable, optional
        A variable or list of variables to exclude from being parsed from the
        dataset. This may be useful to drop variables with problems or
        inconsistent values.
    backend_kwargs: dict
        Additional keyword arguments passed on to the engine open function,
        equivalent to `**kwargs`.
    **kwargs: dict
        Additional keyword arguments passed on to the engine open function.
        For example:

        - 'group': path to the netCDF4 group in the given file to open given as
          a str,supported by "netcdf4", "h5netcdf", "zarr".
        - 'lock': resource lock to use when reading data from disk. Only
          relevant when using dask or another form of parallelism. By default,
          appropriate locks are chosen to safely read and write files with the
          currently active dask scheduler. Supported by "netcdf4", "h5netcdf",
          "scipy", "pynio", "pseudonetcdf", "cfgrib".

        See engine open function for kwargs accepted by each specific engine.

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
    if len(args) > 0:
        raise TypeError(
            "open_dataset() takes only 1 positional argument starting from version 0.18.0, "
            "all other options must be passed as keyword arguments"
        )

    if cache is None:
        cache = chunks is None

    if backend_kwargs is not None:
        kwargs.update(backend_kwargs)

    if engine is None:
        engine = plugins.guess_engine(filename_or_obj)

    backend = plugins.get_backend(engine)

    decoders = _resolve_decoders_kwargs(
        decode_cf,
        open_backend_dataset_parameters=backend.open_dataset_parameters,
        mask_and_scale=mask_and_scale,
        decode_times=decode_times,
        decode_timedelta=decode_timedelta,
        concat_characters=concat_characters,
        use_cftime=use_cftime,
        decode_coords=decode_coords,
    )

    overwrite_encoded_chunks = kwargs.pop("overwrite_encoded_chunks", None)
    backend_ds = backend.open_dataset(
        filename_or_obj,
        drop_variables=drop_variables,
        **decoders,
        **kwargs,
    )
    ds = _dataset_from_backend_dataset(
        backend_ds,
        filename_or_obj,
        engine,
        chunks,
        cache,
        overwrite_encoded_chunks,
        drop_variables=drop_variables,
        **decoders,
        **kwargs,
    )
    return ds


def open_dataarray(
    filename_or_obj,
    *args,
    engine=None,
    chunks=None,
    cache=None,
    decode_cf=None,
    mask_and_scale=None,
    decode_times=None,
    decode_timedelta=None,
    use_cftime=None,
    concat_characters=None,
    decode_coords=None,
    drop_variables=None,
    backend_kwargs=None,
    **kwargs,
):
    """Open an DataArray from a file or file-like object containing a single
    data variable.

    This is designed to read netCDF files with only one data variable. If
    multiple variables are present then a ValueError is raised.

    Parameters
    ----------
    filename_or_obj : str, Path, file-like or DataStore
        Strings and Path objects are interpreted as a path to a netCDF file
        or an OpenDAP URL and opened with python-netCDF4, unless the filename
        ends with .gz, in which case the file is gunzipped and opened with
        scipy.io.netcdf (only netCDF3 supported). Byte-strings or file-like
        objects are opened by scipy.io.netcdf (netCDF3) or h5py (netCDF4/HDF).
    engine : {"netcdf4", "scipy", "pydap", "h5netcdf", "pynio", "cfgrib", \
        "pseudonetcdf", "zarr"}, optional
        Engine to use when reading files. If not provided, the default engine
        is chosen based on available dependencies, with a preference for
        "netcdf4".
    chunks : int or dict, optional
        If chunks is provided, it is used to load the new dataset into dask
        arrays. ``chunks=-1`` loads the dataset with dask using a single
        chunk for all arrays. `chunks={}`` loads the dataset with dask using
        engine preferred chunks if exposed by the backend, otherwise with
        a single chunk for all arrays.
        ``chunks='auto'`` will use dask ``auto`` chunking taking into account the
        engine preferred chunks. See dask chunking for more details.
    cache : bool, optional
        If True, cache data loaded from the underlying datastore in memory as
        NumPy arrays when accessed to avoid reading from the underlying data-
        store multiple times. Defaults to True unless you specify the `chunks`
        argument to use dask, in which case it defaults to False. Does not
        change the behavior of coordinates corresponding to dimensions, which
        always load their data from disk into a ``pandas.Index``.
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
        pseudonetcdf backend. This keyword may not be supported by all the backends.
    decode_times : bool, optional
        If True, decode times encoded in the standard NetCDF datetime format
        into datetime objects. Otherwise, leave them encoded as numbers.
        This keyword may not be supported by all the backends.
    decode_timedelta : bool, optional
        If True, decode variables and coordinates with time units in
        {"days", "hours", "minutes", "seconds", "milliseconds", "microseconds"}
        into timedelta objects. If False, leave them encoded as numbers.
        If None (default), assume the same value of decode_time.
        This keyword may not be supported by all the backends.
    use_cftime: bool, optional
        Only relevant if encoded dates come from a standard calendar
        (e.g. "gregorian", "proleptic_gregorian", "standard", or not
        specified).  If None (default), attempt to decode times to
        ``np.datetime64[ns]`` objects; if this is not possible, decode times to
        ``cftime.datetime`` objects. If True, always decode times to
        ``cftime.datetime`` objects, regardless of whether or not they can be
        represented using ``np.datetime64[ns]`` objects.  If False, always
        decode times to ``np.datetime64[ns]`` objects; if this is not possible
        raise an error. This keyword may not be supported by all the backends.
    concat_characters : bool, optional
        If True, concatenate along the last dimension of character arrays to
        form string arrays. Dimensions will only be concatenated over (and
        removed) if they have no corresponding variable and if they are only
        used as the last dimension of character arrays.
        This keyword may not be supported by all the backends.
    decode_coords : bool or {"coordinates", "all"}, optional
        Controls which variables are set as coordinate variables:

        - "coordinates" or True: Set variables referred to in the
          ``'coordinates'`` attribute of the datasets or individual variables
          as coordinate variables.
        - "all": Set variables referred to in  ``'grid_mapping'``, ``'bounds'`` and
          other attributes as coordinate variables.
    drop_variables: str or iterable, optional
        A variable or list of variables to exclude from being parsed from the
        dataset. This may be useful to drop variables with problems or
        inconsistent values.
    backend_kwargs: dict
        Additional keyword arguments passed on to the engine open function,
        equivalent to `**kwargs`.
    **kwargs: dict
        Additional keyword arguments passed on to the engine open function.
        For example:

        - 'group': path to the netCDF4 group in the given file to open given as
          a str,supported by "netcdf4", "h5netcdf", "zarr".
        - 'lock': resource lock to use when reading data from disk. Only
          relevant when using dask or another form of parallelism. By default,
          appropriate locks are chosen to safely read and write files with the
          currently active dask scheduler. Supported by "netcdf4", "h5netcdf",
          "scipy", "pynio", "pseudonetcdf", "cfgrib".

        See engine open function for kwargs accepted by each specific engine.

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
    if len(args) > 0:
        raise TypeError(
            "open_dataarray() takes only 1 positional argument starting from version 0.18.0, "
            "all other options must be passed as keyword arguments"
        )

    dataset = open_dataset(
        filename_or_obj,
        decode_cf=decode_cf,
        mask_and_scale=mask_and_scale,
        decode_times=decode_times,
        concat_characters=concat_characters,
        decode_coords=decode_coords,
        engine=engine,
        chunks=chunks,
        cache=cache,
        drop_variables=drop_variables,
        backend_kwargs=backend_kwargs,
        use_cftime=use_cftime,
        decode_timedelta=decode_timedelta,
        **kwargs,
    )

    if len(dataset.data_vars) != 1:
        raise ValueError(
            "Given file dataset contains more than one data "
            "variable. Please read with xarray.open_dataset and "
            "then select the variable you want."
        )
    else:
        (data_array,) = dataset.data_vars.values()

    data_array.set_close(dataset._close)

    # Reset names if they were changed during saving
    # to ensure that we can 'roundtrip' perfectly
    if DATAARRAY_NAME in dataset.attrs:
        data_array.name = dataset.attrs[DATAARRAY_NAME]
        del dataset.attrs[DATAARRAY_NAME]

    if data_array.name == DATAARRAY_VARIABLE:
        data_array.name = None

    return data_array


def open_mfdataset(
    paths,
    chunks=None,
    concat_dim=None,
    compat="no_conflicts",
    preprocess=None,
    engine=None,
    data_vars="all",
    coords="different",
    combine="by_coords",
    parallel=False,
    join="outer",
    attrs_file=None,
    combine_attrs="override",
    **kwargs,
):
    """Open multiple files as a single dataset.

    If combine='by_coords' then the function ``combine_by_coords`` is used to combine
    the datasets into one before returning the result, and if combine='nested' then
    ``combine_nested`` is used. The filepaths must be structured according to which
    combining function is used, the details of which are given in the documentation for
    ``combine_by_coords`` and ``combine_nested``. By default ``combine='by_coords'``
    will be used. Requires dask to be installed. See documentation for
    details on dask [1]_. Global attributes from the ``attrs_file`` are used
    for the combined dataset.

    Parameters
    ----------
    paths : str or sequence
        Either a string glob in the form ``"path/to/my/files/*.nc"`` or an explicit list of
        files to open. Paths can be given as strings or as pathlib Paths. If
        concatenation along more than one dimension is desired, then ``paths`` must be a
        nested list-of-lists (see ``combine_nested`` for details). (A string glob will
        be expanded to a 1-dimensional list.)
    chunks : int or dict, optional
        Dictionary with keys given by dimension names and values given by chunk sizes.
        In general, these should divide the dimensions of each dataset. If int, chunk
        each dimension by ``chunks``. By default, chunks will be chosen to load entire
        input files into memory at once. This has a major impact on performance: please
        see the full documentation for more details [2]_.
    concat_dim : str, or list of str, DataArray, Index or None, optional
        Dimensions to concatenate files along.  You only need to provide this argument
        if ``combine='nested'``, and if any of the dimensions along which you want to
        concatenate is not a dimension in the original datasets, e.g., if you want to
        stack a collection of 2D arrays along a third dimension. Set
        ``concat_dim=[..., None, ...]`` explicitly to disable concatenation along a
        particular dimension. Default is None, which for a 1D list of filepaths is
        equivalent to opening the files separately and then merging them with
        ``xarray.merge``.
    combine : {"by_coords", "nested"}, optional
        Whether ``xarray.combine_by_coords`` or ``xarray.combine_nested`` is used to
        combine all the data. Default is to use ``xarray.combine_by_coords``.
    compat : {"identical", "equals", "broadcast_equals", \
              "no_conflicts", "override"}, optional
        String indicating how to compare variables of the same name for
        potential conflicts when merging:

         * "broadcast_equals": all values must be equal when variables are
           broadcast against each other to ensure common dimensions.
         * "equals": all values and dimensions must be the same.
         * "identical": all values, dimensions and attributes must be the
           same.
         * "no_conflicts": only values which are not null in both datasets
           must be equal. The returned dataset then contains the combination
           of all non-null values.
         * "override": skip comparing and pick variable from first dataset

    preprocess : callable, optional
        If provided, call this function on each dataset prior to concatenation.
        You can find the file-name from which each dataset was loaded in
        ``ds.encoding["source"]``.
    engine : {"netcdf4", "scipy", "pydap", "h5netcdf", "pynio", "cfgrib", "zarr"}, \
        optional
        Engine to use when reading files. If not provided, the default engine
        is chosen based on available dependencies, with a preference for
        "netcdf4".
    data_vars : {"minimal", "different", "all"} or list of str, optional
        These data variables will be concatenated together:
          * "minimal": Only data variables in which the dimension already
            appears are included.
          * "different": Data variables which are not equal (ignoring
            attributes) across all datasets are also concatenated (as well as
            all for which dimension already appears). Beware: this option may
            load the data payload of data variables into memory if they are not
            already loaded.
          * "all": All data variables will be concatenated.
          * list of str: The listed data variables will be concatenated, in
            addition to the "minimal" data variables.
    coords : {"minimal", "different", "all"} or list of str, optional
        These coordinate variables will be concatenated together:
         * "minimal": Only coordinates in which the dimension already appears
           are included.
         * "different": Coordinates which are not equal (ignoring attributes)
           across all datasets are also concatenated (as well as all for which
           dimension already appears). Beware: this option may load the data
           payload of coordinate variables into memory if they are not already
           loaded.
         * "all": All coordinate variables will be concatenated, except
           those corresponding to other dimensions.
         * list of str: The listed coordinate variables will be concatenated,
           in addition the "minimal" coordinates.
    parallel : bool, optional
        If True, the open and preprocess steps of this function will be
        performed in parallel using ``dask.delayed``. Default is False.
    join : {"outer", "inner", "left", "right", "exact, "override"}, optional
        String indicating how to combine differing indexes
        (excluding concat_dim) in objects

        - "outer": use the union of object indexes
        - "inner": use the intersection of object indexes
        - "left": use indexes from the first object with each dimension
        - "right": use indexes from the last object with each dimension
        - "exact": instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - "override": if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.
    attrs_file : str or pathlib.Path, optional
        Path of the file used to read global attributes from.
        By default global attributes are read from the first file provided,
        with wildcard matches sorted by filename.
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
    open_dataset

    References
    ----------

    .. [1] http://xarray.pydata.org/en/stable/dask.html
    .. [2] http://xarray.pydata.org/en/stable/dask.html#chunking-and-performance
    """
    if isinstance(paths, str):
        if is_remote_uri(paths) and engine == "zarr":
            try:
                from fsspec.core import get_fs_token_paths
            except ImportError as e:
                raise ImportError(
                    "The use of remote URLs for opening zarr requires the package fsspec"
                ) from e

            fs, _, _ = get_fs_token_paths(
                paths,
                mode="rb",
                storage_options=kwargs.get("backend_kwargs", {}).get(
                    "storage_options", {}
                ),
                expand=False,
            )
            paths = fs.glob(fs._strip_protocol(paths))  # finds directories
            paths = [fs.get_mapper(path) for path in paths]
        elif is_remote_uri(paths):
            raise ValueError(
                "cannot do wild-card matching for paths that are remote URLs: "
                "{!r}. Instead, supply paths as an explicit list of strings.".format(
                    paths
                )
            )
        else:
            paths = sorted(glob(_normalize_path(paths)))
    else:
        paths = [str(p) if isinstance(p, Path) else p for p in paths]

    if not paths:
        raise OSError("no files to open")

    if combine == "nested":
        if isinstance(concat_dim, (str, DataArray)) or concat_dim is None:
            concat_dim = [concat_dim]

        # This creates a flat list which is easier to iterate over, whilst
        # encoding the originally-supplied structure as "ids".
        # The "ids" are not used at all if combine='by_coords`.
        combined_ids_paths = _infer_concat_order_from_positions(paths)
        ids, paths = (
            list(combined_ids_paths.keys()),
            list(combined_ids_paths.values()),
        )

    # TODO raise an error instead of a warning after v0.19
    elif combine == "by_coords" and concat_dim is not None:
        warnings.warn(
            "When combine='by_coords', passing a value for `concat_dim` has no "
            "effect. This combination will raise an error in future. To manually "
            "combine along a specific dimension you should instead specify "
            "combine='nested' along with a value for `concat_dim`.",
            DeprecationWarning,
        )

    open_kwargs = dict(engine=engine, chunks=chunks or {}, **kwargs)

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
    closers = [getattr_(ds, "_close") for ds in datasets]
    if preprocess is not None:
        datasets = [preprocess(ds) for ds in datasets]

    if parallel:
        # calling compute here will return the datasets/file_objs lists,
        # the underlying datasets will still be stored as dask arrays
        datasets, closers = dask.compute(datasets, closers)

    # Combine all datasets, closing them in case of a ValueError
    try:
        if combine == "nested":
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
                combine_attrs=combine_attrs,
            )
        elif combine == "by_coords":
            # Redo ordering from coordinates, ignoring how they were ordered
            # previously
            combined = combine_by_coords(
                datasets,
                compat=compat,
                data_vars=data_vars,
                coords=coords,
                join=join,
                combine_attrs=combine_attrs,
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

    def multi_file_closer():
        for closer in closers:
            closer()

    combined.set_close(multi_file_closer)

    # read global attributes from the attrs_file or from the first dataset
    if attrs_file is not None:
        if isinstance(attrs_file, Path):
            attrs_file = str(attrs_file)
        combined.attrs = datasets[paths.index(attrs_file)].attrs

    return combined


WRITEABLE_STORES: Dict[str, Callable] = {
    "netcdf4": backends.NetCDF4DataStore.open,
    "scipy": backends.ScipyDataStore,
    "h5netcdf": backends.H5NetCDFStore.open,
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
                f"to_netcdf: {engine!r}. Only the default engine "
                "or engine='scipy' is supported"
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
    _validate_attrs(dataset, invalid_netcdf=invalid_netcdf and engine == "h5netcdf")

    try:
        store_open = WRITEABLE_STORES[engine]
    except KeyError:
        raise ValueError(f"unrecognized engine for to_netcdf: {engine!r}")

    if format is not None:
        format = format.upper()

    # handle scheduler specific logic
    scheduler = _get_scheduler()
    have_chunks = any(v.chunks for v in dataset.variables.values())

    autoclose = have_chunks and scheduler in ["distributed", "multiprocessing"]
    if autoclose and engine == "scipy":
        raise NotImplementedError(
            f"Writing netCDF files with the {engine} backend "
            f"is not currently supported with dask's {scheduler} scheduler"
        )

    target = path_or_file if path_or_file is not None else BytesIO()
    kwargs = dict(autoclose=True) if autoclose else {}
    if invalid_netcdf:
        if engine == "h5netcdf":
            kwargs["invalid_netcdf"] = invalid_netcdf
        else:
            raise ValueError(
                f"unrecognized option 'invalid_netcdf' for engine {engine}"
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
    datasets : list of Dataset
        List of datasets to save.
    paths : list of str or list of Path
        List of paths to which to save each corresponding dataset.
    mode : {"w", "a"}, optional
        Write ("w") or append ("a") mode. If mode="w", any existing file at
        these locations will be overwritten.
    format : {"NETCDF4", "NETCDF4_CLASSIC", "NETCDF3_64BIT", \
              "NETCDF3_CLASSIC"}, optional

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
        datasets (only works for format="NETCDF4"). The groups will be created
        if necessary.
    engine : {"netcdf4", "scipy", "h5netcdf"}, optional
        Engine to use when writing netCDF files. If not provided, the
        default engine is chosen based on available dependencies, with a
        preference for "netcdf4" if writing to a file on disk.
        See `Dataset.to_netcdf` for additional information.
    compute : bool
        If true compute immediately, otherwise return a
        ``dask.delayed.Delayed`` object that can be computed later.

    Examples
    --------

    Save a dataset into one netCDF per year of data:

    >>> ds = xr.Dataset(
    ...     {"a": ("time", np.linspace(0, 1, 48))},
    ...     coords={"time": pd.date_range("2010-01-01", freq="M", periods=48)},
    ... )
    >>> ds
    <xarray.Dataset>
    Dimensions:  (time: 48)
    Coordinates:
      * time     (time) datetime64[ns] 2010-01-31 2010-02-28 ... 2013-12-31
    Data variables:
        a        (time) float64 0.0 0.02128 0.04255 0.06383 ... 0.9574 0.9787 1.0
    >>> years, datasets = zip(*ds.groupby("time.year"))
    >>> paths = [f"{y}.nc" for y in years]
    >>> xr.save_mfdataset(datasets, paths)
    """
    if mode == "w" and len(set(paths)) < len(paths):
        raise ValueError(
            "cannot use mode='w' when writing multiple datasets to the same path"
        )

    for obj in datasets:
        if not isinstance(obj, Dataset):
            raise TypeError(
                "save_mfdataset only supports writing Dataset "
                f"objects, received type {type(obj)}"
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
            and not np.issubdtype(var.dtype, np.datetime64)
            and not np.issubdtype(var.dtype, np.bool_)
            and not coding.strings.is_unicode_dtype(var.dtype)
            and not var.dtype == object
        ):
            # and not re.match('^bytes[1-9]+$', var.dtype.name)):
            raise ValueError(
                "Invalid dtype for data variable: {} "
                "dtype must be a subtype of number, "
                "datetime, bool, a fixed sized string, "
                "a fixed size unicode string or an "
                "object".format(var)
            )

    for k in dataset.data_vars.values():
        check_dtype(k)


def _validate_append_dim_and_encoding(
    ds_to_append, store, append_dim, region, encoding, **open_kwargs
):
    try:
        ds = backends.zarr.open_zarr(store, **open_kwargs)
    except ValueError:  # store empty
        return

    if append_dim:
        if append_dim not in ds.dims:
            raise ValueError(
                f"append_dim={append_dim!r} does not match any existing "
                f"dataset dimensions {ds.dims}"
            )
        if region is not None and append_dim in region:
            raise ValueError(
                f"cannot list the same dimension in both ``append_dim`` and "
                f"``region`` with to_zarr(), got {append_dim} in both"
            )

    if region is not None:
        if not isinstance(region, dict):
            raise TypeError(f"``region`` must be a dict, got {type(region)}")
        for k, v in region.items():
            if k not in ds_to_append.dims:
                raise ValueError(
                    f"all keys in ``region`` are not in Dataset dimensions, got "
                    f"{list(region)} and {list(ds_to_append.dims)}"
                )
            if not isinstance(v, slice):
                raise TypeError(
                    "all values in ``region`` must be slice objects, got "
                    f"region={region}"
                )
            if v.step not in {1, None}:
                raise ValueError(
                    "step on all slices in ``region`` must be 1 or None, got "
                    f"region={region}"
                )

        non_matching_vars = [
            k
            for k, v in ds_to_append.variables.items()
            if not set(region).intersection(v.dims)
        ]
        if non_matching_vars:
            raise ValueError(
                f"when setting `region` explicitly in to_zarr(), all "
                f"variables in the dataset to write must have at least "
                f"one dimension in common with the region's dimensions "
                f"{list(region.keys())}, but that is not "
                f"the case for some variables here. To drop these variables "
                f"from this dataset before exporting to zarr, write: "
                f".drop({non_matching_vars!r})"
            )

    for var_name, new_var in ds_to_append.variables.items():
        if var_name in ds.variables:
            existing_var = ds.variables[var_name]
            if new_var.dims != existing_var.dims:
                raise ValueError(
                    f"variable {var_name!r} already exists with different "
                    f"dimension names {existing_var.dims} != "
                    f"{new_var.dims}, but changing variable "
                    f"dimensions is not supported by to_zarr()."
                )

            existing_sizes = {}
            for dim, size in existing_var.sizes.items():
                if region is not None and dim in region:
                    start, stop, stride = region[dim].indices(size)
                    assert stride == 1  # region was already validated above
                    size = stop - start
                if dim != append_dim:
                    existing_sizes[dim] = size

            new_sizes = {
                dim: size for dim, size in new_var.sizes.items() if dim != append_dim
            }
            if existing_sizes != new_sizes:
                raise ValueError(
                    f"variable {var_name!r} already exists with different "
                    f"dimension sizes: {existing_sizes} != {new_sizes}. "
                    f"to_zarr() only supports changing dimension sizes when "
                    f"explicitly appending, but append_dim={append_dim!r}."
                )
            if var_name in encoding.keys():
                raise ValueError(
                    f"variable {var_name!r} already exists, but encoding was provided"
                )


def to_zarr(
    dataset: Dataset,
    store: Union[MutableMapping, str, Path] = None,
    chunk_store=None,
    mode: str = None,
    synchronizer=None,
    group: str = None,
    encoding: Mapping = None,
    compute: bool = True,
    consolidated: bool = False,
    append_dim: Hashable = None,
    region: Mapping[str, slice] = None,
    safe_chunks: bool = True,
):
    """This function creates an appropriate datastore for writing a dataset to
    a zarr ztore

    See `Dataset.to_zarr` for full API docs.
    """

    # expand str and Path arguments
    store = _normalize_path(store)
    chunk_store = _normalize_path(chunk_store)

    if encoding is None:
        encoding = {}

    if mode is None:
        if append_dim is not None or region is not None:
            mode = "a"
        else:
            mode = "w-"

    if mode != "a" and append_dim is not None:
        raise ValueError("cannot set append_dim unless mode='a' or mode=None")

    if mode != "a" and region is not None:
        raise ValueError("cannot set region unless mode='a' or mode=None")

    if mode not in ["w", "w-", "a"]:
        # TODO: figure out how to handle 'r+'
        raise ValueError(
            "The only supported options for mode are 'w', "
            f"'w-' and 'a', but mode={mode!r}"
        )

    if consolidated and region is not None:
        raise ValueError(
            "cannot use consolidated=True when the region argument is set. "
            "Instead, set consolidated=True when writing to zarr with "
            "compute=False before writing data."
        )

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
            region=region,
            encoding=encoding,
            # do we need to pass safe_chunks through here?
        )

    zstore = backends.ZarrStore.open_group(
        store=store,
        mode=mode,
        synchronizer=synchronizer,
        group=group,
        consolidate_on_close=consolidated,
        chunk_store=chunk_store,
        append_dim=append_dim,
        write_region=region,
        safe_chunks=safe_chunks,
    )
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
