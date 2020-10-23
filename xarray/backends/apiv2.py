import os

from ..core.utils import is_remote_uri
from . import cfgrib_, h5netcdf_, zarr
from .api import (
    _autodetect_engine,
    _get_backend_cls,
    _normalize_path,
    _protect_dataset_variables_inplace,
)

ENGINES = {
    "h5netcdf": h5netcdf_.open_backend_dataset_h5necdf,
    "zarr": zarr.open_backend_dataset_zarr,
    "cfgrib": cfgrib_.open_backend_dataset_cfgrib,
}


def dataset_from_backend_dataset(
    ds,
    filename_or_obj,
    engine,
    chunks,
    cache,
    overwrite_encoded_chunks,
    extra_tokens,
):
    if not (isinstance(chunks, (int, dict)) or chunks is None):
        if chunks != "auto":
            raise ValueError(
                "chunks must be an int, dict, 'auto', or None. "
                "Instead found %s. " % chunks
            )

    _protect_dataset_variables_inplace(ds, cache)
    if chunks is not None and engine != "zarr":
        from dask.base import tokenize

        # if passed an actual file path, augment the token with
        # the file modification time
        if isinstance(filename_or_obj, str) and not is_remote_uri(filename_or_obj):
            mtime = os.path.getmtime(filename_or_obj)
        else:
            mtime = None
        token = tokenize(filename_or_obj, mtime, engine, chunks, **extra_tokens)
        name_prefix = "open_dataset-%s" % token
        ds2 = ds.chunk(chunks, name_prefix=name_prefix, token=token)

    elif engine == "zarr":

        if chunks == "auto":
            try:
                import dask.array  # noqa
            except ImportError:
                chunks = None

        if chunks is None:
            return ds

        if isinstance(chunks, int):
            chunks = dict.fromkeys(ds.dims, chunks)

        variables = {
            k: zarr.ZarrStore.maybe_chunk(k, v, chunks, overwrite_encoded_chunks)
            for k, v in ds.variables.items()
        }
        ds2 = ds._replace(variables)

    else:
        ds2 = ds
    ds2._file_obj = ds._file_obj

    # Ensure source filename always stored in dataset object (GH issue #2550)
    if "source" not in ds.encoding:
        if isinstance(filename_or_obj, str):
            ds.encoding["source"] = filename_or_obj

    return ds2


def open_dataset(
    filename_or_obj,
    *,
    engine=None,
    chunks=None,
    cache=None,
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
    engine : {"netcdf4", "scipy", "pydap", "h5netcdf", "pynio", "cfgrib", \
        "pseudonetcdf", "zarr"}, optional
        Engine to use when reading files. If not provided, the default engine
        is chosen based on available dependencies, with a preference for
        "netcdf4".
    chunks : int or dict, optional
        If chunks is provided, it is used to load the new dataset into dask
        arrays. ``chunks={}`` loads the dataset with dask using a single
        chunk for all arrays. When using ``engine="zarr"``, setting
        ``chunks='auto'`` will create dask chunks based on the variable's zarr
        chunks.
    lock : False or lock-like, optional
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
    drop_variables: str or iterable, optional
        A variable or list of variables to exclude from being parsed from the
        dataset. This may be useful to drop variables with problems or
        inconsistent values.
    backend_kwargs: dict, optional
        A dictionary of keyword arguments to pass on to the backend. This
        may be useful when backend options would improve performance or
        allow user control of dataset processing.
    use_cftime: bool, optional
        Only relevant if encoded dates come from a standard calendar
        (e.g. "gregorian", "proleptic_gregorian", "standard", or not
        specified).  If None (default), attempt to decode times to
        ``np.datetime64[ns]`` objects; if this is not possible, decode times to
        ``cftime.datetime`` objects. If True, always decode times to
        ``cftime.datetime`` objects, regardless of whether or not they can be
        represented using ``np.datetime64[ns]`` objects.  If False, always
        decode times to ``np.datetime64[ns]`` objects; if this is not possible
        raise an error.
    decode_timedelta : bool, optional
        If True, decode variables and coordinates with time units in
        {"days", "hours", "minutes", "seconds", "milliseconds", "microseconds"}
        into timedelta objects. If False, leave them encoded as numbers.
        If None (default), assume the same value of decode_time.

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

    if cache is None:
        cache = chunks is None

    if backend_kwargs is None:
        backend_kwargs = {}

    filename_or_obj = _normalize_path(filename_or_obj)

    if engine is None:
        engine = _autodetect_engine(filename_or_obj)

    backend_kwargs = backend_kwargs.copy()
    overwrite_encoded_chunks = backend_kwargs.pop("overwrite_encoded_chunks", None)

    open_backend_dataset = _get_backend_cls(engine, engines=ENGINES)
    backend_ds = open_backend_dataset(
        filename_or_obj,
        **backend_kwargs,
        **{k: v for k, v in kwargs.items() if v is not None},
    )
    ds = dataset_from_backend_dataset(
        backend_ds,
        filename_or_obj,
        engine,
        chunks,
        cache,
        overwrite_encoded_chunks,
        {**backend_kwargs, **kwargs},
    )

    return ds
