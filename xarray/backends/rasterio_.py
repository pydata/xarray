import numpy as np

try:
    import rasterio
except ImportError:
    rasterio = False

from .. import Variable
from ..core.utils import FrozenOrderedDict, Frozen, NDArrayMixin
from ..core import indexing
from ..core.pycompat import OrderedDict, suppress

from .common import AbstractDataStore

_rio_varname = 'raster'

_error_mess = 'The kind of indexing operation you are trying to do is not '
'valid on RasterIO files. Try to load your data with ds.load()'
'first.'

class RasterioArrayWrapper(NDArrayMixin):
    def __init__(self, ds):
        self.ds = ds
        self._shape = self.ds.count, self.ds.height, self.ds.width
        self._ndims = len(self.shape)

    @property
    def dtype(self):
        return np.dtype(self.ds.dtypes[0])

    @property
    def shape(self):
        return self._shape

    def __getitem__(self, key):

        # make our job a bit easier
        key = indexing.canonicalize_indexer(key, self._ndims)

        # bands cannot be windowed but they can be listed
        bands, n = key[0], self.shape[0]
        if isinstance(bands, slice):
            start = bands.start if bands.start is not None else 0
            stop = bands.stop if bands.stop is not None else n
            if bands.step is not None and bands.step != 1:
                raise IndexError(_error_mess)
            bands = np.arange(start, stop)
        # be sure we give out a list
        bands = (np.asarray(bands) + 1).tolist()

        # but other dims can
        window = []
        for k, n in zip(key[1:], self.shape[1:]):
            if isinstance(k, slice):
                start = k.start if k.start is not None else 0
                stop = k.stop if k.stop is not None else n
                if k.step is not None and k.step != 1:
                    raise IndexError(_error_mess)
            else:
                k = np.asarray(k).flatten()
                start = k[0]
                stop = k[-1] + 1
                if (stop - start) != len(k):
                    raise IndexError(_error_mess)
            window.append((start, stop))

        return self.ds.read(bands, window=window)


class RasterioDataStore(AbstractDataStore):
    """Store for accessing datasets via Rasterio
    """
    def __init__(self, filename, mode='r'):

        # TODO: is the rasterio.Env() really necessary, and if yes: when?
        with rasterio.Env():
            self.ds = rasterio.open(filename, mode=mode)

        # Get coords
        nx, ny = self.ds.width, self.ds.height
        dx, dy = self.ds.res[0], -self.ds.res[1]
        x0 = self.ds.bounds.right if dx < 0 else self.ds.bounds.left
        y0 = self.ds.bounds.top if dy < 0 else self.ds.bounds.bottom
        x = np.linspace(start=x0, num=nx, stop=(x0 + (nx - 1) * dx))
        y = np.linspace(start=y0, num=ny, stop=(y0 + (ny - 1) * dy))

        self._vars = OrderedDict()
        self._vars['y'] = Variable(('y',), y)
        self._vars['x'] = Variable(('x',), x)

        # Get dims
        if self.ds.count >= 1:
            self.dims = ('band', 'y', 'x')
            self._vars['band'] = Variable(('band',),
                                          np.atleast_1d(self.ds.indexes))
        else:
            raise ValueError('Unknown dims')

        self._attrs = OrderedDict()
        with suppress(AttributeError):
            for attr_name in ['crs']:
                self._attrs[attr_name] = getattr(self.ds, attr_name)

        # Get data
        self._vars[_rio_varname] = self.open_store_variable(_rio_varname)

    def open_store_variable(self, var):
        if var != _rio_varname:
            raise ValueError('Rasterio variables are named %s' % _rio_varname)
        data = indexing.LazilyIndexedArray(RasterioArrayWrapper(self.ds))
        return Variable(self.dims, data, self._attrs)

    def get_variables(self):
        return FrozenOrderedDict(self._vars)

    def get_attrs(self):
        return Frozen(self._attrs)

    def get_dimensions(self):
        return Frozen(self.dims)

    def close(self):
        self.ds.close()
