import os
import warnings
from pathlib import Path


from ..core.dataset import Dataset
from ..core.utils import close_on_error, is_grib_path, is_remote_uri
from .api import (
    _get_backend_cls,
    _get_default_engine,
    _get_engine_from_magic_number,
    _normalize_path,
    _protect_dataset_variables_inplace,
)
from . import h5netcdf_, cfgrib_

ENGINES = {
    "h5netcdf": h5netcdf_.open_backend_dataset_h5necdf,
    "cfgrib": cfgrib_.open_backend_dataset_cfgrib,
}


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
    decode_timedelta=None,
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
        decode_timedelta = False

    if cache is None:
        cache = chunks is None

    if backend_kwargs is None:
        backend_kwargs = {}
    extra_kwargs = {}

    def chunk_backend_ds(ds, chunks, lock=False):
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
                decode_timedelta,
            )
            name_prefix = "open_dataset-%s" % token
            ds2 = ds.chunk(chunks, name_prefix=name_prefix, token=token)

        else:
            ds2 = ds
        ds2._file_obj = ds._file_obj
        return ds2

    if isinstance(filename_or_obj, Path):
        filename_or_obj = str(filename_or_obj)

    if isinstance(filename_or_obj, str):
        filename_or_obj = _normalize_path(filename_or_obj)

        if engine is None:
            engine = _get_default_engine(filename_or_obj, allow_remote=True)
    elif engine != "zarr":
        if engine not in [None, "scipy", "h5netcdf"]:
            raise ValueError(
                "can only read bytes or file-like objects "
                "with engine='scipy' or 'h5netcdf'"
            )
        engine = _get_engine_from_magic_number(filename_or_obj)

    if engine in ["netcdf4", "h5netcdf"]:
        extra_kwargs["group"] = group
        extra_kwargs["lock"] = lock
    elif engine in ["pynio", "pseudonetcdf", "cfgrib"]:
        extra_kwargs["lock"] = lock
    elif engine == "zarr":
        backend_kwargs = backend_kwargs.copy()
        overwrite_encoded_chunks = backend_kwargs.pop("overwrite_encoded_chunks", None)
        extra_kwargs["mode"] = "r"
        extra_kwargs["group"] = group

    open_backend_dataset = _get_backend_cls(engine, engines=ENGINES)

    backend_ds = open_backend_dataset(
        filename_or_obj,
        mask_and_scale=mask_and_scale,
        decode_times=decode_times,
        concat_characters=concat_characters,
        decode_coords=decode_coords,
        drop_variables=drop_variables,
        use_cftime=use_cftime,
        decode_timedelta=decode_timedelta,
        **backend_kwargs,
        **extra_kwargs,
    )

    ds = chunk_backend_ds(backend_ds, chunks, lock=False)

    # Ensure source filename always stored in dataset object (GH issue #2550)
    if "source" not in ds.encoding:
        if isinstance(filename_or_obj, str):
            ds.encoding["source"] = filename_or_obj

    return ds
