import xray
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

nlat = 15
nlon = 5
arr = np.random.randn(nlat, nlon)
arr[0, 0] = np.nan
atlantic = xray.DataArray(arr,
        coords = (np.linspace(50, 20, nlat), np.linspace(-60, -20, nlon)),
        dims = ('latitude', 'longitude'))

ax = plt.axes(projection=ccrs.Orthographic(-50, 30))

atlantic.plot(ax=ax, origin='upper', aspect='equal',
              transform=ccrs.PlateCarree())

ax.set_global()
ax.coastlines()

plt.savefig('atlantic_noise.png')
