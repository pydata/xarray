# -*- coding: utf-8 -*-
"""
.. _recipes.rasterio:

=================================
Parsing rasterio's geocoordinates
=================================

Convert cartesian coordinates into 2D longitudes and latitudes

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

# Download the file from rasterio's repository
url = 'https://github.com/mapbox/rasterio/raw/master/tests/data/RGB.byte.tif'
urllib.request.urlretrieve(url, 'RGB.byte.tif')

# Read the data
rioda = xr.open_rasterio('RGB.byte.tif')

# Compute the lons and lats using rasterio
ny, nx = len(rioda['y']), len(rioda['x'])
x, y = np.meshgrid(rioda['x'], rioda['y'])

# Rasterio works with 1D arrays
lon, lat = transform(rioda.crs, {'init': 'EPSG:4326'},
                     x.flatten(), y.flatten())
lon = np.asarray(lon).reshape((ny, nx))
lat = np.asarray(lat).reshape((ny, nx))

# Convert the DataArray to a dataset and set them as non-dimension coordinates
riods = rioda.to_dataset(name='img')
riods.coords['lon'] = (('y', 'x'), lon)
riods.coords['lat'] = (('y', 'x'), lat)

# Compute a greyscale out of the rgb image
riods['greyscale'] = riods.img.mean(dim='band')

# Plot on a map
ax = plt.subplot(projection=ccrs.PlateCarree())
riods.greyscale.plot(ax=ax, x='lon', y='lat', transform=ccrs.PlateCarree(),
                     cmap='Greys_r', add_colorbar=False)
ax.coastlines('10m', color='r')
plt.show()

# Delete the file
os.remove('RGB.byte.tif')
