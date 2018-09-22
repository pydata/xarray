from __future__ import absolute_import, division, print_function

import os.path
from glob import glob
from io import BytesIO
from numbers import Number

import numpy as np

from .. import Dataset, backends, conventions
from ..core import indexing
from ..core.combine import auto_combine
from ..core.pycompat import basestring, path_type
from ..core.utils import close_on_error, is_remote_uri
from .common import (
    HDF5_LOCK, ArrayWriter, CombinedLock, _get_scheduler, _get_scheduler_lock)

DATAARRAY_NAME = '__xarray_dataarray_name__'
DATAARRAY_VARIABLE = '__xarray_dataarray_variable__'


def _get_default_engine(path, allow_remote=False):
    if allow_remote and is_remote_uri(path):  # pragma: no cover
        try:
            import netCDF4
            engine = 'netcdf4'
        except ImportError:
            try:
                import pydap  # flake8: noqa
                engine = 'pydap'
            except ImportError:
                raise ValueError('netCDF4 or pydap is required for accessing '
                                 'remote datasets via OPeNDAP')
    else:
        try:
            import netCDF4  # flake8: noqa
            engine = 'netcdf4'
        except ImportError:  # pragma: no cover
            try:
                import scipy.io.netcdf  # flake8: noqa
                engine = 'scipy'
            except ImportError:
                raise ValueError('cannot read or write netCDF files without '
                                 'netCDF4-python or scipy installed')
    return engine


def _normalize_path(path):
    if is_remote_uri(path):
        return path
    else:
        return os.path.abspath(os.path.expanduser(path))


def _default_lock(filename, engine):
    if filename.endswith('.gz'):
        lock = False
    else:
        if engine is None:
            engine = _get_default_engine(filename, allow_remote=True)

        if engine == 'netcdf4':
            if is_remote_uri(filename):
                lock = False
            else:
                # TODO: identify netcdf3 files and don't use the global lock
                # for them
                lock = HDF5_LOCK
        elif engine in {'h5netcdf', 'pynio'}:
            lock = HDF5_LOCK
        else:
            lock = False
    return lock


def _validate_dataset_names(dataset):
    """DataArray.name and Dataset keys must be a string or None"""
    def check_name(name):
        if isinstance(name, basestring):
            if not name:
                raise ValueError('Invalid name for DataArray or Dataset key: '
                                 'string must be length 1 or greater for '
                                 'serialization to netCDF files')
        elif name is not None:
            raise TypeError('DataArray.name or Dataset key must be either a '
                            'string or None for serialization to netCDF files')

    for k in dataset.variables:
        check_name(k)


def _validate_attrs(dataset):
    """`attrs` must have a string key and a value which is either: a number
    a string, an ndarray or a list/tuple of numbers/strings.
    """
    def check_attr(name, value):
        if isinstance(name, basestring):
            if not name:
                raise ValueError('Invalid name for attr: string must be '
                                 'length 1 or greater for serialization to '
                                 'netCDF files')
        else:
            raise TypeError("Invalid name for attr: {} must be a string for "
                            "serialization to netCDF files".format(name))

        if not isinstance(value, (basestring, Number, np.ndarray, np.number,
                                  list, tuple)):
            raise TypeError('Invalid value for attr: {} must be a number '
                            'string, ndarray or a list/tuple of '
                            'numbers/strings for serialization to netCDF '
                            'files'.format(value))

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


def _get_lock(engine, scheduler, format, path_or_file):
    """ Get the lock(s) that apply to a particular scheduler/engine/format"""

    locks = []
    if format in ['NETCDF4', None] and engine in ['h5netcdf', 'netcdf4']:
        locks.append(HDF5_LOCK)
    locks.append(_get_scheduler_lock(scheduler, path_or_file))

    # When we have more than one lock, use the CombinedLock wrapper class
    lock = CombinedLock(locks) if len(locks) > 1 else locks[0]

    return lock


def _finalize_store(write, store):
    """ Finalize this store by explicitly syncing and closing"""
    del write  # ensure writing is done first
    store.sync()
    store.close()


def open_dataset(filename_or_obj, group=None, decode_cf=True,
                 mask_and_scale=None, decode_times=True, autoclose=False,
                 concat_characters=True, decode_coords=True, engine=None,
                 chunks=None, lock=None, cache=None, drop_variables=None,
                 backend_kwargs=None):
    """Load and decode a dataset from a file or file-like object.

    Parameters
    ----------
    filename_or_obj : str, Path, file or xarray.backends.*DataStore
        Strings and Path objects are interpreted as a path to a netCDF file
        or an OpenDAP URL and opened with python-netCDF4, unless the filename
        ends with .gz, in which case the file is gunzipped and opened with
        scipy.io.netcdf (only netCDF3 supported). File-like objects are opened
        with scipy.io.netcdf (only netCDF3 supported).
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
    engine : {'netcdf4', 'scipy', 'pydap', 'h5netcdf', 'pynio', 'pseudonetcdf'}, optional
        Engine to use when reading files. If not provided, the default engine
        is chosen based on available dependencies, with a preference for
        'netcdf4'.
    chunks : int or dict, optional
        If chunks is provided, it used to load the new dataset into dask
        arrays. ``chunks={}`` loads the dataset with dask using a single
        chunk for all arrays.
    lock : False, True or threading.Lock, optional
        If chunks is provided, this argument is passed on to
        :py:func:`dask.array.from_array`. By default, a global lock is
        used when reading data from netCDF files with the netcdf4 and h5netcdf
        engines to avoid issues with concurrent access when using dask's
        multithreaded backend.
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

    Returns
    -------
    dataset : Dataset
        The newly created dataset.

    See Also
    --------
    open_mfdataset
    """

    if mask_and_scale is None:
        mask_and_scale = not engine == 'pseudonetcdf'

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
            store, mask_and_scale=mask_and_scale, decode_times=decode_times,
            concat_characters=concat_characters, decode_coords=decode_coords,
            drop_variables=drop_variables)

        _protect_dataset_variables_inplace(ds, cache)

        if chunks is not None:
            from dask.base import tokenize
            # if passed an actual file path, augment the token with
            # the file modification time
            if (isinstance(filename_or_obj, basestring) and
                    not is_remote_uri(filename_or_obj)):
                mtime = os.path.getmtime(filename_or_obj)
            else:
                mtime = None
            token = tokenize(filename_or_obj, mtime, group, decode_cf,
                             mask_and_scale, decode_times, concat_characters,
                             decode_coords, engine, chunks, drop_variables)
            name_prefix = 'open_dataset-%s' % token
            ds2 = ds.chunk(chunks, name_prefix=name_prefix, token=token,
                           lock=lock)
            ds2._file_obj = ds._file_obj
        else:
            ds2 = ds

        # protect so that dataset store isn't necessarily closed, e.g.,
        # streams like BytesIO  can't be reopened
        # datastore backend is responsible for determining this capability
        if store._autoclose:
            store.close()

        return ds2

    if isinstance(filename_or_obj, path_type):
        filename_or_obj = str(filename_or_obj)

    if isinstance(filename_or_obj, backends.AbstractDataStore):
        store = filename_or_obj
    elif isinstance(filename_or_obj, basestring):

        if (isinstance(filename_or_obj, bytes) and
                filename_or_obj.startswith(b'\x89HDF')):
            raise ValueError('cannot read netCDF4/HDF5 file images')
        elif (isinstance(filename_or_obj, bytes) and
                filename_or_obj.startswith(b'CDF')):
            # netCDF3 file images are handled by scipy
            pass
        elif isinstance(filename_or_obj, basestring):
            filename_or_obj = _normalize_path(filename_or_obj)

        if filename_or_obj.endswith('.gz'):
            if engine is not None and engine != 'scipy':
                raise ValueError('can only read gzipped netCDF files with '
                                 "default engine or engine='scipy'")
            else:
                engine = 'scipy'

        if engine is None:
            engine = _get_default_engine(filename_or_obj,
                                         allow_remote=True)
        if engine == 'netcdf4':
            store = backends.NetCDF4DataStore.open(filename_or_obj,
                                                   group=group,
                                                   autoclose=autoclose,
                                                   **backend_kwargs)
        elif engine == 'scipy':
            store = backends.ScipyDataStore(filename_or_obj,
                                            autoclose=autoclose,
                                            **backend_kwargs)
        elif engine == 'pydap':
            store = backends.PydapDataStore.open(filename_or_obj,
                                                 **backend_kwargs)
        elif engine == 'h5netcdf':
            store = backends.H5NetCDFStore(filename_or_obj, group=group,
                                           autoclose=autoclose,
                                           **backend_kwargs)
        elif engine == 'pynio':
            store = backends.NioDataStore(filename_or_obj,
                                          autoclose=autoclose,
                                           **backend_kwargs)
        elif engine == 'pseudonetcdf':
            store = backends.PseudoNetCDFDataStore.open(
                filename_or_obj, autoclose=autoclose, **backend_kwargs)
        else:
            raise ValueError('unrecognized engine for open_dataset: %r'
                             % engine)

        if lock is None:
            lock = _default_lock(filename_or_obj, engine)
        with close_on_error(store):
            return maybe_decode_store(store, lock)
    else:
        if engine is not None and engine != 'scipy':
            raise ValueError('can only read file-like objects with '
                             "default engine or engine='scipy'")
        # assume filename_or_obj is a file-like object
        store = backends.ScipyDataStore(filename_or_obj)

    return maybe_decode_store(store)


def open_dataarray(filename_or_obj, group=None, decode_cf=True,
                   mask_and_scale=None, decode_times=True, autoclose=False,
                   concat_characters=True, decode_coords=True, engine=None,
                   chunks=None, lock=None, cache=None, drop_variables=None,
                   backend_kwargs=None):
    """Open an DataArray from a netCDF file containing a single data variable.

    This is designed to read netCDF files with only one data variable. If
    multiple variables are present then a ValueError is raised.

    Parameters
    ----------
    filename_or_obj : str, Path, file or xarray.backends.*DataStore
        Strings and Paths are interpreted as a path to a netCDF file or an
        OpenDAP URL and opened with python-netCDF4, unless the filename ends
        with .gz, in which case the file is gunzipped and opened with
        scipy.io.netcdf (only netCDF3 supported). File-like objects are opened
        with scipy.io.netcdf (only netCDF3 supported).
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
    engine : {'netcdf4', 'scipy', 'pydap', 'h5netcdf', 'pynio'}, optional
        Engine to use when reading files. If not provided, the default engine
        is chosen based on available dependencies, with a preference for
        'netcdf4'.
    chunks : int or dict, optional
        If chunks is provided, it used to load the new dataset into dask
        arrays.
    lock : False, True or threading.Lock, optional
        If chunks is provided, this argument is passed on to
        :py:func:`dask.array.from_array`. By default, a global lock is
        used when reading data from netCDF files with the netcdf4 and h5netcdf
        engines to avoid issues with concurrent access when using dask's
        multithreaded backend.
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

    dataset = open_dataset(filename_or_obj, group=group, decode_cf=decode_cf,
                           mask_and_scale=mask_and_scale,
                           decode_times=decode_times, autoclose=autoclose,
                           concat_characters=concat_characters,
                           decode_coords=decode_coords, engine=engine,
                           chunks=chunks, lock=lock, cache=cache,
                           drop_variables=drop_variables,
                           backend_kwargs=backend_kwargs)

    if len(dataset.data_vars) != 1:
        raise ValueError('Given file dataset contains more than one data '
                         'variable. Please read with xarray.open_dataset and '
                         'then select the variable you want.')
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


class _MultiFileCloser(object):
    def __init__(self, file_objs):
        self.file_objs = file_objs

    def close(self):
        for f in self.file_objs:
            f.close()


_CONCAT_DIM_DEFAULT = '__infer_concat_dim__'


def open_mfdataset(paths, chunks=None, concat_dim=_CONCAT_DIM_DEFAULT,
                   compat='no_conflicts', preprocess=None, engine=None,
                   lock=None, data_vars='all', coords='different',
                   autoclose=False, parallel=False, **kwargs):
    """Open multiple files as a single dataset.

    Requires dask to be installed. See documentation for details on dask [1].
    Attributes from the first dataset file are used for the combined dataset.

    Parameters
    ----------
    paths : str or sequence
        Either a string glob in the form "path/to/my/files/*.nc" or an explicit
        list of files to open.  Paths can be given as strings or as pathlib
        Paths.
    chunks : int or dict, optional
        Dictionary with keys given by dimension names and values given by chunk
        sizes. In general, these should divide the dimensions of each dataset.
        If int, chunk each dimension by ``chunks``.
        By default, chunks will be chosen to load entire input files into
        memory at once. This has a major impact on performance: please see the
        full documentation for more details [2].
    concat_dim : None, str, DataArray or Index, optional
        Dimension to concatenate files along. This argument is passed on to
        :py:func:`xarray.auto_combine` along with the dataset objects. You only
        need to provide this argument if the dimension along which you want to
        concatenate is not a dimension in the original datasets, e.g., if you
        want to stack a collection of 2D arrays along a third dimension.
        By default, xarray attempts to infer this argument by examining
        component files. Set ``concat_dim=None`` explicitly to disable
        concatenation.
    compat : {'identical', 'equals', 'broadcast_equals',
              'no_conflicts'}, optional
        String indicating how to compare variables of the same name for
        potential conflicts when merging:

        - 'broadcast_equals': all values must be equal when variables are
          broadcast against each other to ensure common dimensions.
        - 'equals': all values and dimensions must be the same.
        - 'identical': all values, dimensions and attributes must be the
          same.
        - 'no_conflicts': only values which are not null in both datasets
          must be equal. The returned dataset then contains the combination
          of all non-null values.
    preprocess : callable, optional
        If provided, call this function on each dataset prior to concatenation.
    engine : {'netcdf4', 'scipy', 'pydap', 'h5netcdf', 'pynio'}, optional
        Engine to use when reading files. If not provided, the default engine
        is chosen based on available dependencies, with a preference for
        'netcdf4'.
    autoclose : bool, optional
        If True, automatically close files to avoid OS Error of too many files
        being open.  However, this option doesn't work with streams, e.g.,
        BytesIO.
    lock : False, True or threading.Lock, optional
        This argument is passed on to :py:func:`dask.array.from_array`. By
        default, a per-variable lock is used when reading data from netCDF
        files with the netcdf4 and h5netcdf engines to avoid issues with
        concurrent access when using dask's multithreaded backend.
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
    coords : {'minimal', 'different', 'all' o list of str}, optional
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
    **kwargs : optional
        Additional arguments passed on to :py:func:`xarray.open_dataset`.

    Returns
    -------
    xarray.Dataset

    See Also
    --------
    auto_combine
    open_dataset

    References
    ----------
    .. [1] http://xarray.pydata.org/en/stable/dask.html
    .. [2] http://xarray.pydata.org/en/stable/dask.html#chunking-and-performance
    """
    if isinstance(paths, basestring):
        if is_remote_uri(paths):
            raise ValueError(
                'cannot do wild-card matching for paths that are remote URLs: '
                '{!r}. Instead, supply paths as an explicit list of strings.'
                .format(paths))
        paths = sorted(glob(paths))
    else:
        paths = [str(p) if isinstance(p, path_type) else p for p in paths]

    if not paths:
        raise IOError('no files to open')

    if lock is None:
        lock = _default_lock(paths[0], engine)

    open_kwargs = dict(engine=engine, chunks=chunks or {}, lock=lock,
                       autoclose=autoclose, **kwargs)

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
    file_objs = [getattr_(ds, '_file_obj') for ds in datasets]
    if preprocess is not None:
        datasets = [preprocess(ds) for ds in datasets]

    if parallel:
        # calling compute here will return the datasets/file_objs lists,
        # the underlying datasets will still be stored as dask arrays
        datasets, file_objs = dask.compute(datasets, file_objs)

    # close datasets in case of a ValueError
    try:
        if concat_dim is _CONCAT_DIM_DEFAULT:
            combined = auto_combine(datasets, compat=compat,
                                    data_vars=data_vars, coords=coords)
        else:
            combined = auto_combine(datasets, concat_dim=concat_dim,
                                    compat=compat,
                                    data_vars=data_vars, coords=coords)
    except ValueError:
        for ds in datasets:
            ds.close()
        raise

    combined._file_obj = _MultiFileCloser(file_objs)
    combined.attrs = datasets[0].attrs
    return combined


WRITEABLE_STORES = {'netcdf4': backends.NetCDF4DataStore.open,
                    'scipy': backends.ScipyDataStore,
                    'h5netcdf': backends.H5NetCDFStore}


def to_netcdf(dataset, path_or_file=None, mode='w', format=None, group=None,
              engine=None, writer=None, encoding=None, unlimited_dims=None,
              compute=True):
    """This function creates an appropriate datastore for writing a dataset to
    disk as a netCDF file

    See `Dataset.to_netcdf` for full API docs.

    The ``writer`` argument is only for the private use of save_mfdataset.
    """
    if isinstance(path_or_file, path_type):
        path_or_file = str(path_or_file)
    if encoding is None:
        encoding = {}
    if path_or_file is None:
        if engine is None:
            engine = 'scipy'
        elif engine != 'scipy':
            raise ValueError('invalid engine for creating bytes with '
                             'to_netcdf: %r. Only the default engine '
                             "or engine='scipy' is supported" % engine)
    elif isinstance(path_or_file, basestring):
        if engine is None:
            engine = _get_default_engine(path_or_file)
        path_or_file = _normalize_path(path_or_file)
    else:  # file-like object
        engine = 'scipy'

    # validate Dataset keys, DataArray names, and attr keys/values
    _validate_dataset_names(dataset)
    _validate_attrs(dataset)

    try:
        store_open = WRITEABLE_STORES[engine]
    except KeyError:
        raise ValueError('unrecognized engine for to_netcdf: %r' % engine)

    if format is not None:
        format = format.upper()

    # if a writer is provided, store asynchronously
    sync = writer is None

    # handle scheduler specific logic
    scheduler = _get_scheduler()
    have_chunks = any(v.chunks for v in dataset.variables.values())
    if (have_chunks and scheduler in ['distributed', 'multiprocessing'] and
            engine != 'netcdf4'):
        raise NotImplementedError("Writing netCDF files with the %s backend "
                                  "is not currently supported with dask's %s "
                                  "scheduler" % (engine, scheduler))
    lock = _get_lock(engine, scheduler, format, path_or_file)
    autoclose = (have_chunks and
                 scheduler in ['distributed', 'multiprocessing'])

    target = path_or_file if path_or_file is not None else BytesIO()
    store = store_open(target, mode, format, group, writer,
                       autoclose=autoclose, lock=lock)

    if unlimited_dims is None:
        unlimited_dims = dataset.encoding.get('unlimited_dims', None)
    if isinstance(unlimited_dims, basestring):
        unlimited_dims = [unlimited_dims]

    try:
        dataset.dump_to_store(store, sync=sync, encoding=encoding,
                              unlimited_dims=unlimited_dims, compute=compute)
        if path_or_file is None:
            return target.getvalue()
    finally:
        if sync and isinstance(path_or_file, basestring):
            store.close()

    if not compute:
        import dask
        return dask.delayed(_finalize_store)(store.delayed_store, store)

    if not sync:
        return store

def save_mfdataset(datasets, paths, mode='w', format=None, groups=None,
                   engine=None, compute=True):
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
    if mode == 'w' and len(set(paths)) < len(paths):
        raise ValueError("cannot use mode='w' when writing multiple "
                         'datasets to the same path')

    for obj in datasets:
        if not isinstance(obj, Dataset):
            raise TypeError('save_mfdataset only supports writing Dataset '
                            'objects, received type %s' % type(obj))

    if groups is None:
        groups = [None] * len(datasets)

    if len(set([len(datasets), len(paths), len(groups)])) > 1:
        raise ValueError('must supply lists of the same length for the '
                         'datasets, paths and groups arguments to '
                         'save_mfdataset')

    writer = ArrayWriter() if compute else None
    stores = [to_netcdf(ds, path, mode, format, group, engine, writer,
                        compute=compute)
              for ds, path, group in zip(datasets, paths, groups)]

    if not compute:
        import dask
        return dask.delayed(stores)

    try:
        delayed = writer.sync(compute=compute)
        for store in stores:
            store.sync()
    finally:
        for store in stores:
            store.close()


def to_zarr(dataset, store=None, mode='w-', synchronizer=None, group=None,
            encoding=None, compute=True):
    """This function creates an appropriate datastore for writing a dataset to
    a zarr ztore

    See `Dataset.to_zarr` for full API docs.
    """
    if isinstance(store, path_type):
        store = str(store)
    if encoding is None:
        encoding = {}

    # validate Dataset keys, DataArray names, and attr keys/values
    _validate_dataset_names(dataset)
    _validate_attrs(dataset)

    store = backends.ZarrStore.open_group(store=store, mode=mode,
                                          synchronizer=synchronizer,
                                          group=group, writer=None)

    # I think zarr stores should always be sync'd immediately
    # TODO: figure out how to properly handle unlimited_dims
    dataset.dump_to_store(store, sync=True, encoding=encoding, compute=compute)

    if not compute:
        import dask
        return dask.delayed(_finalize_store)(store.delayed_store, store)
    return store
