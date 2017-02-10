import copy
import numpy as np

try:
    import rasterio
except ImportError:
    rasterio = False

from .. import Variable, DataArray
from ..core.utils import FrozenOrderedDict, Frozen, NDArrayMixin
from ..core import indexing
from ..core.pycompat import OrderedDict, suppress

from .common import AbstractDataStore

__rio_varname__ = 'raster'


class RasterioArrayWrapper(NDArrayMixin):
    def __init__(self, ds):
        self._ds = ds
        self.array = ds.read()

    @property
    def dtype(self):
        return np.dtype(self._ds.dtypes[0])

    def __getitem__(self, key):
        if key == () and self.ndim == 0:
            return self.array.get_value()
        return self.array[key]


class RasterioDataStore(AbstractDataStore):
    """Store for accessing datasets via Rasterio
    """
    def __init__(self, filename, mode='r'):

        # TODO: is the rasterio.Env() really necessary, and if yes where?
        with rasterio.Env():
            self.ds = rasterio.open(filename, mode=mode)

        # Get coords
        nx, ny = self.ds.width, self.ds.height
        dx, dy = self.ds.res[0], -self.ds.res[1]
        x0 = self.ds.bounds.right if dx < 0 else self.ds.bounds.left
        y0 = self.ds.bounds.top if dy < 0 else self.ds.bounds.bottom
        y = np.linspace(start=y0, num=ny, stop=(y0 + (ny-1) * dy))
        x = np.linspace(start=x0, num=nx, stop=(x0 + (nx-1) * dx))

        self.coords = OrderedDict()
        self.coords['y'] = Variable(('y', ), y)
        self.coords['x'] = Variable(('x', ), x)

        # Get dims
        if self.ds.count >= 1:
            self.dims = ('band', 'y', 'x')
            self.coords['band'] = Variable(('band', ),
                                           np.atleast_1d(self.ds.indexes))
        else:
            raise ValueError('unknown dims')

        self._attrs = OrderedDict()
        with suppress(AttributeError):
            for attr_name in ['crs', 'transform', 'proj']:
                self._attrs[attr_name] = getattr(self.ds, attr_name)

    def open_store_variable(self, var):
        if var != __rio_varname__:
            raise ValueError(
                'Rasterio variables are all named %s' % __rio_varname__)
        data = indexing.LazilyIndexedArray(
            RasterioArrayWrapper(self.ds))
        return Variable(self.dims, data, self._attrs)

    def get_variables(self):
        # Get lat lon coordinates
        coords = _try_to_get_latlon_coords(self.coords, self._attrs)
        rio_vars = {__rio_varname__: self.open_store_variable(__rio_varname__)}
        rio_vars.update(coords)
        return FrozenOrderedDict(rio_vars)

    def get_attrs(self):
        return Frozen(self._attrs)

    def get_dimensions(self):
        return Frozen(self.ds.dims)

    def close(self):
        self.ds.close()


def _try_to_get_latlon_coords(coords, attrs):

    from rasterio.warp import transform

    coords_out = coords
    if 'crs' in attrs:
        proj = attrs['crs']
        # TODO: if the proj is already PlateCarree, making 2D coordinates
        # is not the best thing to do here.
        ny, nx = len(coords['y']), len(coords['x'])
        x, y = np.meshgrid(coords['x'], coords['y'])
        # Rasterio works with 1D arrays
        xc, yc = transform(proj, {'init': 'EPSG:4326'},
                           x.flatten(), y.flatten())
        xc = np.asarray(xc).reshape((ny, nx))
        yc = np.asarray(yc).reshape((ny, nx))
        dims = ('y', 'x')

        coords_out['lat'] = Variable(dims, yc,
                                     attrs={'units': 'degrees_north',
                                            'long_name': 'latitude',
                                            'standard_name': 'latitude'})
        coords_out['lon'] = Variable(dims, xc,
                                     attrs={'units': 'degrees_east',
                                            'long_name': 'longitude',
                                            'standard_name': 'longitude'})
    return coords_out
