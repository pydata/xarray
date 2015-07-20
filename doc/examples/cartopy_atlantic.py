import xray
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

nlat = 15
nlon = 5
atlantic = xray.DataArray(np.random.randn(nlat, nlon),
        coords = (np.linspace(50, 20, nlat), np.linspace(-60, -20, nlon)),
        dims = ('latitude', 'longitude'))

ax = plt.axes(projection=ccrs.PlateCarree())

atlantic.plot(ax=ax)

ax.set_ylim(0, 90)
ax.set_xlim(-180, 30)
ax.coastlines()

plt.savefig('atlantic_noise.png')
