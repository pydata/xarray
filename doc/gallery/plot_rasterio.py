# -*- coding: utf-8 -*-
"""
.. _recipes.rasterio:

=================================
Parsing rasterio's geocoordinates
=================================


The example illustrates how to use an accessor (see :ref:`internals.accessors`)
to  convert a projection's cartesian coordinates into 2D longitudes and
latitudes.

These new coordinates might be handy for plotting and indexing, but it should
be kept in mind that a grid which is regular in projection coordinates will
likely be irregular in lon/lat. It is often recommended to work in the data's
original map projection.
"""

import os
import urllib.request
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from rasterio.warp import transform


# Define the accessor
@xr.register_dataarray_accessor('rasterio')
class RasterioAccessor(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def add_lonlat_coords(self):
        """Compute the lon/lat coordinates out of the dataset's crs.

        This adds two non-dimension coordinates ('lon' and 'lat') to the
        original dataarray.
        """

        ny, nx = len(self._obj['y']), len(self._obj['x'])
        x, y = np.meshgrid(self._obj['x'], self._obj['y'])

        # Rasterio works with 1D arrays
        lon, lat = transform(self._obj.crs, {'init': 'EPSG:4326'},
                             x.flatten(), y.flatten())
        lon = np.asarray(lon).reshape((ny, nx))
        lat = np.asarray(lat).reshape((ny, nx))
        self._obj.coords['lon'] = (('y', 'x'), lon)
        self._obj.coords['lat'] = (('y', 'x'), lat)


# Download the file from rasterio's repository
url = 'https://github.com/mapbox/rasterio/raw/master/tests/data/RGB.byte.tif'
urllib.request.urlretrieve(url, 'RGB.byte.tif')

# Read the data
rioda = xr.open_rasterio('RGB.byte.tif')

# Compute the coordinates
rioda.rasterio.add_lonlat_coords()

# Compute a greyscale out of the rgb image
greyscale = rioda.mean(dim='band')

# Plot on a map
ax = plt.subplot(projection=ccrs.PlateCarree())
greyscale.plot(ax=ax, x='lon', y='lat', transform=ccrs.PlateCarree(),
               cmap='Greys_r', add_colorbar=False)
ax.coastlines('10m', color='r')
plt.show()

# Delete the file
os.remove('RGB.byte.tif')
