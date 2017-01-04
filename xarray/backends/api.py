from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gzip
import os.path
from distutils.version import StrictVersion
from glob import glob
from io import BytesIO
from numbers import Number

import numpy as np

from .. import backends, conventions
from .common import ArrayWriter, GLOBAL_LOCK
from ..core import indexing
from ..core.combine import auto_combine
from ..core.utils import close_on_error, is_remote_uri
from ..core.pycompat import basestring

DATAARRAY_NAME = '__xarray_dataarray_name__'
DATAARRAY_VARIABLE = '__xarray_dataarray_variable__'


def _get_default_engine(path, allow_remote=False):
    if allow_remote and is_remote_uri(path):  # pragma: no cover
        try:
            import netCDF4
            engine = 'netcdf4'
        except ImportError:
            try:
                import pydap
                engine = 'pydap'
            except ImportError:
                raise ValueError('netCDF4 or pydap is required for accessing '
                                 'remote datasets via OPeNDAP')
    else:
        try:
            import netCDF4
            engine = 'netcdf4'
        except ImportError:  # pragma: no cover
            try:
                import scipy.io.netcdf
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
                lock = GLOBAL_LOCK
        elif engine in {'h5netcdf', 'pynio'}:
            lock = GLOBAL_LOCK
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

    for k in dataset:
        check_name(k)


def _validate_attrs(dataset):
    """`attrs` must have a string key and a value which is either: a number
    a string, an ndarray or a list/tuple of numbers/strings.
    """
    def check_attr(name, value):
        if isinstance(name, basestring):
            if not name:
                raise ValueError('Invalid name for attr: string must be length '
                                 '1 or greater for serialization to netCDF '
                                 'files')
        else:
            raise TypeError("Invalid name for attr: {} must be a string for "
                            "serialization to netCDF files".format(name))

        if not isinstance(value, (basestring, Number, np.ndarray, np.number,
                                  list, tuple)):
            raise TypeError('Invalid value for attr: {} must be a number '
                            'string, ndarray or a list/tuple of numbers/strings '
                            'for serialization to netCDF '
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


def open_dataset(filename_or_obj, group=None, decode_cf=True,
                 mask_and_scale=True, decode_times=True,
                 concat_characters=True, decode_coords=True, engine=None,
                 chunks=None, lock=None, cache=None, drop_variables=None):
    """Load and decode a dataset from a file or file-like object.

    Parameters
    ----------
    filename_or_obj : str, file or xarray.backends.*DataStore
        Strings are interpreted as a path to a netCDF file or an OpenDAP URL
        and opened with python-netCDF4, unless the filename ends with .gz, in
        which case the file is gunzipped and opened with scipy.io.netcdf (only
        netCDF3 supported). File-like objects are opened with scipy.io.netcdf
        (only netCDF3 supported).
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
        be replaced by NA.
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
    engine : {'netcdf4', 'scipy', 'pydap', 'h5netcdf', 'pynio'}, optional
        Engine to use when reading files. If not provided, the default engine
        is chosen based on available dependencies, with a preference for
        'netcdf4'.
    chunks : int or dict, optional
        If chunks is provided, it used to load the new dataset into dask
        arrays. ``chunks={}`` loads the dataset with dask using a single
        chunk for all arrays. This is an experimental feature; see the
        documentation for more details.
    lock : False, True or threading.Lock, optional
        If chunks is provided, this argument is passed on to
        :py:func:`dask.array.from_array`. By default, a per-variable lock is
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

    Returns
    -------
    dataset : Dataset
        The newly created dataset.

    See Also
    --------
    open_mfdataset
    """
    if not decode_cf:
        mask_and_scale = False
        decode_times = False
        concat_characters = False
        decode_coords = False

    if cache is None:
        cache = chunks is None

    def maybe_decode_store(store, lock=False):
        ds = conventions.decode_cf(
            store, mask_and_scale=mask_and_scale, decode_times=decode_times,
            concat_characters=concat_characters, decode_coords=decode_coords,
            drop_variables=drop_variables)

        _protect_dataset_variables_inplace(ds, cache)

        if chunks is not None:
            try:
                from dask.base import tokenize
            except ImportError:
                import dask  # raise the usual error if dask is entirely missing
                if StrictVersion(dask.__version__) < StrictVersion('0.6'):
                    raise ImportError('xarray requires dask version 0.6 or newer')
                else:
                    raise

            if (isinstance(filename_or_obj, basestring) and
                    not is_remote_uri(filename_or_obj)):
                file_arg = os.path.getmtime(filename_or_obj)
            else:
                file_arg = filename_or_obj
            token = tokenize(file_arg, group, decode_cf, mask_and_scale,
                             decode_times, concat_characters, decode_coords,
                             engine, chunks, drop_variables)
            name_prefix = '%s:%s/' % (filename_or_obj, group or '')
            ds2 = ds.chunk(chunks, name_prefix=name_prefix, token=token,
                           lock=lock)
            ds2._file_obj = ds._file_obj
        else:
            ds2 = ds

        return ds2

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
            # if the string ends with .gz, then gunzip and open as netcdf file
            try:
                store = backends.ScipyDataStore(gzip.open(filename_or_obj))
            except TypeError as e:
                # TODO: gzipped loading only works with NetCDF3 files.
                if 'is not a valid NetCDF 3 file' in e.message:
                    raise ValueError('gzipped file loading only supports '
                                     'NetCDF 3 files.')
                else:
                    raise
        else:
            if engine is None:
                engine = _get_default_engine(filename_or_obj,
                                             allow_remote=True)
            if engine == 'netcdf4':
                store = backends.NetCDF4DataStore(filename_or_obj, group=group)
            elif engine == 'scipy':
                store = backends.ScipyDataStore(filename_or_obj)
            elif engine == 'pydap':
                store = backends.PydapDataStore(filename_or_obj)
            elif engine == 'h5netcdf':
                store = backends.H5NetCDFStore(filename_or_obj, group=group)
            elif engine == 'pynio':
                store = backends.NioDataStore(filename_or_obj)
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
                   mask_and_scale=True, decode_times=True,
                   concat_characters=True, decode_coords=True, engine=None,
                   chunks=None, lock=None, cache=None, drop_variables=None):
    """
    Opens an DataArray from a netCDF file containing a single data variable.

    This is designed to read netCDF files with only one data variable. If
    multiple variables are present then a ValueError is raised.

    Parameters
    ----------
    filename_or_obj : str, file or xarray.backends.*DataStore
        Strings are interpreted as a path to a netCDF file or an OpenDAP URL
        and opened with python-netCDF4, unless the filename ends with .gz, in
        which case the file is gunzipped and opened with scipy.io.netcdf (only
        netCDF3 supported). File-like objects are opened with scipy.io.netcdf
        (only netCDF3 supported).
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
        be replaced by NA.
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
    engine : {'netcdf4', 'scipy', 'pydap', 'h5netcdf', 'pynio'}, optional
        Engine to use when reading files. If not provided, the default engine
        is chosen based on available dependencies, with a preference for
        'netcdf4'.
    chunks : int or dict, optional
        If chunks is provided, it used to load the new dataset into dask
        arrays. This is an experimental feature; see the documentation for more
        details.
    lock : False, True or threading.Lock, optional
        If chunks is provided, this argument is passed on to
        :py:func:`dask.array.from_array`. By default, a per-variable lock is
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
    dataset = open_dataset(filename_or_obj, group, decode_cf,
                           mask_and_scale, decode_times,
                           concat_characters, decode_coords, engine,
                           chunks, lock, cache, drop_variables)

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
                   lock=None, **kwargs):
    """Open multiple files as a single dataset.

    Experimental. Requires dask to be installed.

    Parameters
    ----------
    paths : str or sequence
        Either a string glob in the form "path/to/my/files/*.nc" or an explicit
        list of files to open.
    chunks : int or dict, optional
        Dictionary with keys given by dimension names and values given by chunk
        sizes. In general, these should divide the dimensions of each dataset.
        If int, chunk each dimension by ``chunks``.
        By default, chunks will be chosen to load entire input files into
        memory at once. This has a major impact on performance: please see the
        full documentation for more details.
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
    lock : False, True or threading.Lock, optional
        This argument is passed on to :py:func:`dask.array.from_array`. By
        default, a per-variable lock is used when reading data from netCDF
        files with the netcdf4 and h5netcdf engines to avoid issues with
        concurrent access when using dask's multithreaded backend.
    **kwargs : optional
        Additional arguments passed on to :py:func:`xarray.open_dataset`.

    Returns
    -------
    xarray.Dataset

    See Also
    --------
    auto_combine
    open_dataset
    """
    if isinstance(paths, basestring):
        paths = sorted(glob(paths))
    if not paths:
        raise IOError('no files to open')

    if lock is None:
        lock = _default_lock(paths[0], engine)
    datasets = [open_dataset(p, engine=engine, chunks=chunks or {}, lock=lock,
                             **kwargs) for p in paths]
    file_objs = [ds._file_obj for ds in datasets]

    if preprocess is not None:
        datasets = [preprocess(ds) for ds in datasets]

    if concat_dim is _CONCAT_DIM_DEFAULT:
        combined = auto_combine(datasets, compat=compat)
    else:
        combined = auto_combine(datasets, concat_dim=concat_dim, compat=compat)
    combined._file_obj = _MultiFileCloser(file_objs)
    return combined


WRITEABLE_STORES = {'netcdf4': backends.NetCDF4DataStore,
                    'scipy': backends.ScipyDataStore,
                    'h5netcdf': backends.H5NetCDFStore}


def to_netcdf(dataset, path=None, mode='w', format=None, group=None,
              engine=None, writer=None, encoding=None):
    """This function creates an appropriate datastore for writing a dataset to
    disk as a netCDF file

    See `Dataset.to_netcdf` for full API docs.

    The ``writer`` argument is only for the private use of save_mfdataset.
    """
    if encoding is None:
        encoding = {}
    if path is None:
        path = BytesIO()
        if engine is None:
            engine = 'scipy'
        elif engine is not None:
            raise ValueError('invalid engine for creating bytes with '
                             'to_netcdf: %r. Only the default engine '
                             "or engine='scipy' is supported" % engine)
    else:
        if engine is None:
            engine = _get_default_engine(path)
        path = _normalize_path(path)

    # validate Dataset keys, DataArray names, and attr keys/values
    _validate_dataset_names(dataset)
    _validate_attrs(dataset)

    try:
        store_cls = WRITEABLE_STORES[engine]
    except KeyError:
        raise ValueError('unrecognized engine for to_netcdf: %r' % engine)

    if format is not None:
        format = format.upper()

    # if a writer is provided, store asynchronously
    sync = writer is None

    store = store_cls(path, mode, format, group, writer)
    try:
        dataset.dump_to_store(store, sync=sync, encoding=encoding)
        if isinstance(path, BytesIO):
            return path.getvalue()
    finally:
        if sync:
            store.close()

    if not sync:
        return store


def save_mfdataset(datasets, paths, mode='w', format=None, groups=None,
                   engine=None):
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
    paths : list of str
        List of paths to which to save each corresponding dataset.
    mode : {'w', 'a'}, optional
        Write ('w') or append ('a') mode. If mode='w', any existing file at
        these locations will be overwritten.
    format : {'NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_64BIT', 'NETCDF3_CLASSIC'}, optional
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

    if groups is None:
        groups = [None] * len(datasets)

    if len(set([len(datasets), len(paths), len(groups)])) > 1:
        raise ValueError('must supply lists of the same length for the '
                         'datasets, paths and groups arguments to '
                         'save_mfdataset')

    writer = ArrayWriter()
    stores = [to_netcdf(ds, path, mode, format, group, engine, writer)
              for ds, path, group in zip(datasets, paths, groups)]
    try:
        writer.sync()
        for store in stores:
            store.sync()
    finally:
        for store in stores:
            store.close()
