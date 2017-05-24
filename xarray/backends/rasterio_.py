import numpy as np

try:
    import rasterio
except ImportError:
    rasterio = False

from .. import DataArray
from ..core.utils import NDArrayMixin, is_scalar
from ..core import indexing

_ERROR_MSG = ('The kind of indexing operation you are trying to do is not '
              'valid on rasterio files. Try to load your data with ds.load()'
              'first.')


class RasterioArrayWrapper(NDArrayMixin):
    """A wrapper around rasterio dataset objects"""
    def __init__(self, riods):
        self.riods = riods
        self._shape = self.riods.count, self.riods.height, self.riods.width
        self._ndims = len(self.shape)

    @property
    def dtype(self):
        return np.dtype(self.riods.dtypes[0])

    @property
    def shape(self):
        return self._shape

    def __exit__(self, exception_type, exception_value, traceback):
        self.riods.close()

    def __getitem__(self, key):

        # make our job a bit easier
        key = indexing.canonicalize_indexer(key, self._ndims)

        # bands cannot be windowed but they can be listed
        band_key = key[0]
        n_bands = self.shape[0]
        if isinstance(band_key, slice):
            start, stop, step = band_key.indices(n_bands)
            if step is not None and step != 1:
                raise IndexError(_ERROR_MSG)
            band_key = np.arange(start, stop)
        # be sure we give out a list
        band_key = (np.asarray(band_key) + 1).tolist()

        # but other dims can only be windowed
        window = []
        squeeze_axis = []
        for i, (k, n) in enumerate(zip(key[1:], self.shape[1:])):
            if isinstance(k, slice):
                start, stop, step = k.indices(n)
                if step is not None and step != 1:
                    raise IndexError(_ERROR_MSG)
            else:
                if is_scalar(k):
                    # windowed operations will always return an array
                    # we will have to squeeze it later
                    squeeze_axis.append(i+1)
                    start = k
                    stop = k+1
                else:
                    start = k[0]
                    stop = k[-1] + 1
                    if not np.all(k == np.arange(start, stop)):
                        raise IndexError(_ERROR_MSG)
            window.append((start, stop))

        out = self.riods.read(band_key, window=window)
        if squeeze_axis:
            out = np.squeeze(out, axis=squeeze_axis)
        return out


def rasterio_to_dataarray(filename):
    """Open a file with rasterio.

    This should work with any file that rasterio can open (most often: 
    geoTIFF). The x and y coordinates are generated automatically from the 
    file's geoinformation.
    """

    riods = rasterio.open(filename, mode='r')

    # Get geo coords
    nx, ny = riods.width, riods.height
    dx, dy = riods.res[0], -riods.res[1]
    x0 = riods.bounds.right if dx < 0 else riods.bounds.left
    y0 = riods.bounds.top if dy < 0 else riods.bounds.bottom
    x = np.linspace(start=x0, num=nx, stop=(x0 + (nx - 1) * dx))
    y = np.linspace(start=y0, num=ny, stop=(y0 + (ny - 1) * dy))

    # Get bands
    if riods.count < 1:
        raise ValueError('Unknown dims')
    bands = np.asarray(riods.indexes)

    # Attributes
    attrs = {}
    if hasattr(riods, 'crs'):
        # CRS is a dict-like object specific to rasterio
        # We convert it back to a PROJ4 string using rasterio itself
        attrs['crs'] = riods.crs.to_string()
    # Maybe we'd like to parse other attributes here (for later)

    data = indexing.LazilyIndexedArray(RasterioArrayWrapper(riods))
    return DataArray(data=data, dims=('band', 'y', 'x'),
                     coords={'band': bands, 'y': y, 'x': x},
                     attrs=attrs)
