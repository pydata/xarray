"""
.. _recipes.rasterio:

=================================
Parsing rasterio's geocoordinates
=================================


Converting a projection's cartesian coordinates into 2D longitudes and
latitudes.

These new coordinates might be handy for plotting and indexing, but it should
be kept in mind that a grid which is regular in projection coordinates will
likely be irregular in lon/lat. It is often recommended to work in the data's
original map projection (see :ref:`recipes.rasterio_rgb`).
"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer

import xarray as xr

# Read the data
url = "https://github.com/rasterio/rasterio/raw/master/tests/data/RGB.byte.tif"
da = xr.open_rasterio(url)

# Compute the lon/lat coordinates with pyproj
transformer = Transformer.from_crs(da.crs, "EPSG:4326", always_xy=True)
lon, lat = transformer.transform(*np.meshgrid(da["x"], da["y"]))
da.coords["lon"] = (("y", "x"), lon)
da.coords["lat"] = (("y", "x"), lat)

# Compute a greyscale out of the rgb image
greyscale = da.mean(dim="band")

# Plot on a map
ax = plt.subplot(projection=ccrs.PlateCarree())
greyscale.plot(
    ax=ax,
    x="lon",
    y="lat",
    transform=ccrs.PlateCarree(),
    cmap="Greys_r",
    add_colorbar=False,
)
ax.coastlines("10m", color="r")
plt.show()
