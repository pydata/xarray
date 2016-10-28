import numpy as np

try:
    import rasterio
except ImportError:
    rasterio = False

from .. import Variable, DataArray
from ..core.utils import FrozenOrderedDict, Frozen, NDArrayMixin
from ..core import indexing
from ..core.pycompat import OrderedDict

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

        with rasterio.Env():
            self.ds = rasterio.open(filename, mode=mode, )

            # Get coords
            nx, ny = self.ds.width, self.ds.height
            x0, y0 = self.ds.bounds.left, self.ds.bounds.top
            dx, dy = self.ds.res[0], -self.ds.res[1]

        self.coords = {'y': np.arange(start=y0, stop=(y0 + ny * dy), step=dy),
                       'x': np.arange(start=x0, stop=(x0 + nx * dx), step=dx)}

        # Get dims
        if self.ds.count >= 2:
            self.dims = ('band', 'y', 'x')
            self.coords['band'] = self.ds.indexes
        elif self.ds.count == 1:
            self.dims = ('y', 'x')
        else:
            raise ValueError('unknown dims')

        self._attrs = OrderedDict()
        for attr_name in ['crs', 'transform', 'proj']:
            try:
                self._attrs[attr_name] = getattr(self.ds, attr_name)
            except AttributeError:
                pass

        self.coords = _try_to_get_latlon_coords(self.coords, self._attrs)



    # def get_vardata(self, var_id=1):
    #     """Read the geotiff band.
    #     Parameters
    #     ----------
    #     var_id: the variable name (here the band number)
    #     """
    #     # wx = (self.sub_x[0], self.sub_x[1] + 1)
    #     # wy = (self.sub_y[0], self.sub_y[1] + 1)
    #     with rasterio.Env():
    #         band = self.ds.read()  # var_id, window=(wy, wx))
    #     return band

    def open_store_variable(self, var):
        if var != __rio_varname__:
            raise ValueError(
                'Rasterio variables are all named %s' % __rio_varname__)
        data = indexing.LazilyIndexedArray(
            RasterioArrayWrapper(self.ds))
        return Variable(self.dims, data, self._attrs)

    def get_variables(self):
        return FrozenOrderedDict(
            {__rio_varname__: self.open_store_variable(__rio_varname__)})

    def get_attrs(self):
        return Frozen(self._attrs)

    def get_dimensions(self):
        return Frozen(self.ds.dims)

    def close(self):
        self.ds.close()


def _transform_proj(p1, p2, x, y, nocopy=False):
    """Wrapper around the pyproj transform.
    When two projections are equal, this function avoids quite a bunch of
    useless calculations. See https://github.com/jswhit/pyproj/issues/15
    """
    import pyproj
    import copy

    if p1.srs == p2.srs:
        if nocopy:
            return x, y
        else:
            return copy.deepcopy(x), copy.deepcopy(y)

    return pyproj.transform(p1, p2, x, y)


def _try_to_get_latlon_coords(coords, attrs):
    try:
        import pyproj
    except ImportError:
        pyproj = False
    if 'crs' in attrs and pyproj:
        proj = pyproj.Proj(attrs['crs'])
        x, y = np.meshgrid(coords['x'], coords['y'])
        proj_out = pyproj.Proj("+init=EPSG:4326", preserve_units=True)
        xc, yc = _transform_proj(proj, proj_out, x, y)
        coords = dict(y=coords['y'], x=coords['x'])
        dims = ('y', 'x')

        coords['latitude'] = DataArray(
            data=yc, coords=coords, dims=dims, name='latitude',
            attrs={'units': 'degrees_north', 'long_name': 'latitude',
                   'standard_name': 'latitude'})
        coords['longitude'] = DataArray(
            data=xc, coords=coords, dims=dims, name='latitude',
            attrs={'units': 'degrees_east', 'long_name': 'longitude',
                   'standard_name': 'longitude'})

    return coords
