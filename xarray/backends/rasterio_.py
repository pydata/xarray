import os
from collections import OrderedDict
from distutils.version import LooseVersion
import warnings

import numpy as np

from .. import DataArray
from ..core import indexing
from ..core.utils import is_scalar
from .common import BackendArray, PickleByReconstructionWrapper

try:
    from dask.utils import SerializableLock as Lock
except ImportError:
    from threading import Lock

RASTERIO_LOCK = Lock()

_ERROR_MSG = ('The kind of indexing operation you are trying to do is not '
              'valid on rasterio files. Try to load your data with ds.load()'
              'first.')


class RasterioArrayWrapper(BackendArray):
    """A wrapper around rasterio dataset objects"""

    def __init__(self, riods):
        self.riods = riods
        self._shape = (riods.value.count, riods.value.height,
                       riods.value.width)
        self._ndims = len(self.shape)

    @property
    def dtype(self):
        dtypes = self.riods.value.dtypes
        if not np.all(np.asarray(dtypes) == dtypes[0]):
            raise ValueError('All bands should have the same dtype')
        return np.dtype(dtypes[0])

    @property
    def shape(self):
        return self._shape

    def _get_indexer(self, key):
        """ Get indexer for rasterio array.

        Parameter
        ---------
        key: tuple of int

        Returns
        -------
        band_key: an indexer for the 1st dimension
        window: two tuples. Each consists of (start, stop).
        squeeze_axis: axes to be squeezed
        np_ind: indexer for loaded numpy array

        See also
        --------
        indexing.decompose_indexer
        """
        assert len(key) == 3, 'rasterio datasets should always be 3D'

        # bands cannot be windowed but they can be listed
        band_key = key[0]
        np_inds = []
        # bands (axis=0) cannot be windowed but they can be listed
        if isinstance(band_key, slice):
            start, stop, step = band_key.indices(self.shape[0])
            band_key = np.arange(start, stop, step)
        # be sure we give out a list
        band_key = (np.asarray(band_key) + 1).tolist()
        if isinstance(band_key, list):  # if band_key is not a scalar
            np_inds.append(slice(None))

        # but other dims can only be windowed
        window = []
        squeeze_axis = []
        for i, (k, n) in enumerate(zip(key[1:], self.shape[1:])):
            if isinstance(k, slice):
                # step is always positive. see indexing.decompose_indexer
                start, stop, step = k.indices(n)
                np_inds.append(slice(None, None, step))
            elif is_scalar(k):
                # windowed operations will always return an array
                # we will have to squeeze it later
                squeeze_axis.append(- (2 - i))
                start = k
                stop = k + 1
            else:
                start, stop = np.min(k), np.max(k) + 1
                np_inds.append(k - start)
            window.append((start, stop))

        if isinstance(key[1], np.ndarray) and isinstance(key[2], np.ndarray):
            # do outer-style indexing
            np_inds[1:] = np.ix_(*np_inds[1:])

        return band_key, tuple(window), tuple(squeeze_axis), tuple(np_inds)

    def _getitem(self, key):
        band_key, window, squeeze_axis, np_inds = self._get_indexer(key)

        if not band_key or any(start == stop for (start, stop) in window):
            # no need to do IO
            shape = (len(band_key),) + tuple(
                stop - start for (start, stop) in window)
            out = np.zeros(shape, dtype=self.dtype)
        else:
            out = self.riods.value.read(band_key, window=window)

        if squeeze_axis:
            out = np.squeeze(out, axis=squeeze_axis)
        return out[np_inds]

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.OUTER, self._getitem)


def _parse_envi(meta):
    """Parse ENVI metadata into Python data structures.

    See the link for information on the ENVI header file format:
    http://www.harrisgeospatial.com/docs/enviheaderfiles.html

    Parameters
    ----------
    meta : dict
        Dictionary of keys and str values to parse, as returned by the rasterio
        tags(ns='ENVI') call.

    Returns
    -------
    parsed_meta : dict
        Dictionary containing the original keys and the parsed values

    """

    def parsevec(s):
        return np.fromstring(s.strip('{}'), dtype='float', sep=',')

    def default(s):
        return s.strip('{}')

    parse = {'wavelength': parsevec,
             'fwhm': parsevec}
    parsed_meta = {k: parse.get(k, default)(v) for k, v in meta.items()}
    return parsed_meta


def open_rasterio(filename, parse_coordinates=None, chunks=None, cache=None,
                  lock=None):
    """Open a file with rasterio (experimental).

    This should work with any file that rasterio can open (most often:
    geoTIFF). The x and y coordinates are generated automatically from the
    file's geoinformation, shifted to the center of each pixel (see
    `"PixelIsArea" Raster Space
    <http://web.archive.org/web/20160326194152/http://remotesensing.org/geotiff/spec/geotiff2.5.html#2.5.2>`_
    for more information).

    You can generate 2D coordinates from the file's attributes with::

        from affine import Affine
        da = xr.open_rasterio('path_to_file.tif')
        transform = Affine(*da.attrs['transform'])
        nx, ny = da.sizes['x'], da.sizes['y']
        x, y = np.meshgrid(np.arange(nx)+0.5, np.arange(ny)+0.5) * transform


    Parameters
    ----------
    filename : str
        Path to the file to open.
    parse_coordinates : bool, optional
        Whether to parse the x and y coordinates out of the file's
        ``transform`` attribute or not. The default is to automatically
        parse the coordinates only if they are rectilinear (1D).
        It can be useful to set ``parse_coordinates=False``
        if your files are very large or if you don't need the coordinates.
    chunks : int, tuple or dict, optional
        Chunk sizes along each dimension, e.g., ``5``, ``(5, 5)`` or
        ``{'x': 5, 'y': 5}``. If chunks is provided, it used to load the new
        DataArray into a dask array.
    cache : bool, optional
        If True, cache data loaded from the underlying datastore in memory as
        NumPy arrays when accessed to avoid reading from the underlying data-
        store multiple times. Defaults to True unless you specify the `chunks`
        argument to use dask, in which case it defaults to False.
    lock : False, True or threading.Lock, optional
        If chunks is provided, this argument is passed on to
        :py:func:`dask.array.from_array`. By default, a global lock is
        used to avoid issues with concurrent access to the same file when using
        dask's multithreaded backend.

    Returns
    -------
    data : DataArray
        The newly created DataArray.
    """

    import rasterio

    riods = PickleByReconstructionWrapper(rasterio.open, filename, mode='r')

    if cache is None:
        cache = chunks is None

    coords = OrderedDict()

    # Get bands
    if riods.value.count < 1:
        raise ValueError('Unknown dims')
    coords['band'] = np.asarray(riods.value.indexes)

    # Get coordinates
    if LooseVersion(rasterio.__version__) < '1.0':
        transform = riods.value.affine
    else:
        transform = riods.value.transform
    if transform.is_rectilinear:
        # 1d coordinates
        parse = True if parse_coordinates is None else parse_coordinates
        if parse:
            nx, ny = riods.value.width, riods.value.height
            # xarray coordinates are pixel centered
            x, _ = (np.arange(nx) + 0.5, np.zeros(nx) + 0.5) * transform
            _, y = (np.zeros(ny) + 0.5, np.arange(ny) + 0.5) * transform
            coords['y'] = y
            coords['x'] = x
    else:
        # 2d coordinates
        parse = False if (parse_coordinates is None) else parse_coordinates
        if parse:
            warnings.warn("The file coordinates' transformation isn't "
                          "rectilinear: xarray won't parse the coordinates "
                          "in this case. Set `parse_coordinates=False` to "
                          "suppress this warning.",
                          RuntimeWarning, stacklevel=3)

    # Attributes
    attrs = dict()
    # Affine transformation matrix (always available)
    # This describes coefficients mapping pixel coordinates to CRS
    # For serialization store as tuple of 6 floats, the last row being
    # always (0, 0, 1) per definition (see https://github.com/sgillies/affine)
    attrs['transform'] = tuple(transform)[:6]
    if hasattr(riods.value, 'crs') and riods.value.crs:
        # CRS is a dict-like object specific to rasterio
        # If CRS is not None, we convert it back to a PROJ4 string using
        # rasterio itself
        attrs['crs'] = riods.value.crs.to_string()
    if hasattr(riods.value, 'res'):
        # (width, height) tuple of pixels in units of CRS
        attrs['res'] = riods.value.res
    if hasattr(riods.value, 'is_tiled'):
        # Is the TIF tiled? (bool)
        # We cast it to an int for netCDF compatibility
        attrs['is_tiled'] = np.uint8(riods.value.is_tiled)
    if hasattr(riods.value, 'nodatavals'):
        # The nodata values for the raster bands
        attrs['nodatavals'] = tuple([np.nan if nodataval is None else nodataval
                                     for nodataval in riods.value.nodatavals])

    # Parse extra metadata from tags, if supported
    parsers = {'ENVI': _parse_envi}

    driver = riods.value.driver
    if driver in parsers:
        meta = parsers[driver](riods.value.tags(ns=driver))

        for k, v in meta.items():
            # Add values as coordinates if they match the band count,
            # as attributes otherwise
            if (isinstance(v, (list, np.ndarray)) and
               len(v) == riods.value.count):
                coords[k] = ('band', np.asarray(v))
            else:
                attrs[k] = v

    data = indexing.LazilyOuterIndexedArray(RasterioArrayWrapper(riods))

    # this lets you write arrays loaded with rasterio
    data = indexing.CopyOnWriteArray(data)
    if cache and (chunks is None):
        data = indexing.MemoryCachedArray(data)

    result = DataArray(data=data, dims=('band', 'y', 'x'),
                       coords=coords, attrs=attrs)

    if chunks is not None:
        from dask.base import tokenize
        # augment the token with the file modification time
        try:
            mtime = os.path.getmtime(filename)
        except OSError:
            # the filename is probably an s3 bucket rather than a regular file
            mtime = None
        token = tokenize(filename, mtime, chunks)
        name_prefix = 'open_rasterio-%s' % token
        if lock is None:
            lock = RASTERIO_LOCK
        result = result.chunk(chunks, name_prefix=name_prefix, token=token,
                              lock=lock)

    # Make the file closeable
    result._file_obj = riods

    return result
