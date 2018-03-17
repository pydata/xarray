# -*- coding: utf-8 -*-
"""
.. _recipes.rasterio_rgb:

============================
imshow() and map projections
============================

Using rasterio's projection information for more accurate plots.

This example extends :ref:`recipes.rasterio` and plots the image in the
original map projection instead of relying on pcolormesh and a map
transformation.
"""

import os
import urllib.request

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import xarray as xr

# Download the file from rasterio's repository
url = 'https://github.com/mapbox/rasterio/raw/master/tests/data/RGB.byte.tif'
urllib.request.urlretrieve(url, 'RGB.byte.tif')

# Read the data
da = xr.open_rasterio('RGB.byte.tif')

# The data is in UTM projection. We have to set it manually until
# https://github.com/SciTools/cartopy/issues/813 is implemented
crs = ccrs.UTM('18N')

# Plot on a map
ax = plt.subplot(projection=crs)
da.plot.imshow(ax=ax, rgb='band', transform=crs)
ax.coastlines('10m', color='r')
plt.show()

# Delete the file
os.remove('RGB.byte.tif')
