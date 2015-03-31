import sys
import gzip
import itertools
from glob import glob
from io import BytesIO

import numpy as np

from .. import backends, conventions
from ..core.dataset import Dataset
from ..core.alignment import auto_combine
from ..core.utils import close_on_error
from ..core.variable import Variable
from ..core.pycompat import basestring, OrderedDict, range


def _get_default_netcdf_engine(engine):
    try:
        import netCDF4
        engine = 'netcdf4'
    except ImportError: # pragma: no cover
        try:
            import scipy.io.netcdf
            engine = 'scipy'
        except ImportError:
            raise ValueError('cannot read or write netCDF files without '
                             'netCDF4-python or scipy installed')
    return engine


counter = itertools.count()


def _lazify_dataset(dataset, blockshapes):
    """Make a dataset lazy by converting all its arrays into dask arrays

    Currently only tested for converting datasets opened from disk.
    """
    import dask.array as da

    if set(blockshapes) != set(dataset.dims):
        raise ValueError('one blockshape for each dimension is required. '
                         'In general, these should divide the dimensions of '
                         'each dataset: %s' % dict(dataset.dims))

    variables = OrderedDict()
    for k, v in dataset.variables.items():
        if v.ndim > 0:
            array = v._data.array  # undo the LazilyIndexedArray
            if isinstance(array, range):
                # dask can't handle range objects, currently
                array = np.asarray(array)
            blockshape = tuple(blockshapes[d] for d in v.dims)
            name = 'xray_%s_%s' % (k, next(counter))
            data = da.from_array(array, blockshape=blockshape, name=name)
            variables[k] = Variable(v.dims, data, v.attrs, v.encoding)
        else:
            variables[k] = v
    return Dataset(variables, attrs=dataset.attrs).set_coords(dataset.coords)


def open_dataset(filename_or_obj, group=None, decode_cf=True,
                 mask_and_scale=True, decode_times=True,
                 concat_characters=True, decode_coords=True, engine=None,
                 blockshapes=None):
    """Load and decode a dataset from a file or file-like object.

    Parameters
    ----------
    filename_or_obj : str or file
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
        taken from variable attributes (if they exist).
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
    engine : 'netcdf4' or 'scipy', optional
        Engine to use when reading netCDF files. If not provided, the default
        engine is chosen based on available dependencies, with a preference for
        'netcdf4' if reading a file on disk.
    blockshapes : dict, optional
        If blockshapes are provided, they are used to load the new dataset
        into dask arrays. See the documentation for more details.

    Returns
    -------
    dataset : Dataset
        The newly created dataset.
    """
    if not decode_cf:
        mask_and_scale = False
        decode_times = False
        concat_characters = False
        decode_coords = False

    def maybe_decode_store(store):
        ds = conventions.decode_cf(
            store, mask_and_scale=mask_and_scale, decode_times=decode_times,
            concat_characters=concat_characters, decode_coords=decode_coords)
        if blockshapes is not None:
            ds = _lazify_dataset(ds, blockshapes)
        return ds

    if isinstance(filename_or_obj, basestring):
        if filename_or_obj.endswith('.gz'):
            if engine is not None and engine != 'scipy':
                raise ValueError('can only read gzipped netCDF files with '
                                 "default engine or engine='scipy'")
            # if the string ends with .gz, then gunzip and open as netcdf file
            if sys.version_info[:2] < (2, 7):
                raise ValueError('reading a gzipped netCDF not '
                                 'supported on Python 2.6')
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
                engine = _get_default_netcdf_engine(engine)
            if engine == 'netcdf4':
                store = backends.NetCDF4DataStore(filename_or_obj, group=group)
            elif engine == 'scipy':
                store = backends.ScipyDataStore(filename_or_obj)
            else:
                raise ValueError('unrecognized engine for open_dataset: %r'
                                 % engine)

        with close_on_error(store):
            return maybe_decode_store(store)
    else:
        if engine is not None and engine != 'scipy':
            raise ValueError('can only read file-like objects with '
                             "default engine or engine='scipy'")
        # assume filename_or_obj is a file-like object
        store = backends.ScipyDataStore(filename_or_obj)
        return maybe_decode_store(store)


class _FileCloser(object):
    def __init__(self, file_objs):
        self.file_objs = file_objs

    def close(self):
        for f in self.file_objs:
            f.close()


def open_mfdataset(paths, blockshapes={}, concat_dim=None, **kwargs):
    """Open multiple files as a single dataset.

    Experimental. Requires dask to be installed.

    Parameters
    ----------
    paths : str or sequence
        Either a str glob in the form "path/to/my/files/*.nc" or an explicit
        list of files to open.
    blockshapes : dict, optional
        Dictionary with keys given by dimension names and values given by
        blockshapes. In general, these should divide the dimensions of each
        dataset. By default, blockshapes will be chosen to load an entire
        input file into memory at once. This has a major impact on performance:
        please see the full documentation for more details.
    concat_dim : str or DataArray or Index, optional
        Dimension to concatenate files along. This argument is passed on to
        ``auto_combine`` along with the dataset objects.
    **kwargs : optional
        Additional arguments passed on to ``open_dataset``.

    Returns
    -------
    xray.Dataset
    """
    if isinstance(paths, basestring):
        paths = sorted(glob(paths))
    if not paths:
        raise IOError('no files to open')
    datasets = [open_dataset(p, **kwargs) for p in paths]

    blockshapes = dict(blockshapes)
    for k, v in datasets[0].dims.items():
        blockshapes.setdefault(k, v)

    file_objs = [ds._file_obj for ds in datasets]
    datasets = [_lazify_dataset(ds, blockshapes) for ds in datasets]
    combined = auto_combine(datasets, concat_dim=concat_dim)
    combined._file_obj = _FileCloser(file_objs)
    return combined


def to_netcdf(dataset, path=None, mode='w', format=None, group=None,
              engine=None):
    if path is None:
        path = BytesIO()
        if engine is None:
            engine = 'scipy'
        elif engine is not None:
            raise ValueError('invalid engine for creating bytes with '
                             'to_netcdf: %r. Only the default engine '
                             "or engine='scipy' is supported" % engine)
    elif engine is None:
        engine = _get_default_netcdf_engine(engine)

    if engine == 'netcdf4':
        to_netcdf_func = _to_netcdf4
    elif engine == 'scipy':
        to_netcdf_func = _to_scipy_netcdf
    else:
        raise ValueError('unrecognized engine for to_netcdf: %r' % engine)

    return to_netcdf_func(dataset, path, mode, format, group)


def _to_netcdf4(dataset, path, mode, format, group):
    if format is None:
        format = 'NETCDF4'
    with backends.NetCDF4DataStore(path, mode=mode, format=format,
                                   group=group) as store:
        dataset.dump_to_store(store)


def _to_scipy_netcdf(dataset, path, mode, format, group):
    if group is not None:
        raise ValueError('cannot save to a group with the '
                         'scipy.io.netcdf backend')

    if format is None or format == 'NETCDF3_64BIT':
        version = 2
    elif format == 'NETCDF3_CLASSIC':
        version = 1
    else:
        raise ValueError('invalid format for scipy.io.netcdf backend: %r'
                         % format)

    with backends.ScipyDataStore(path, mode='w', version=version) as store:
        dataset.dump_to_store(store)

        if isinstance(path, BytesIO):
            return path.getvalue()
