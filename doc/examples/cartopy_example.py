import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


air = (xr.tutorial
       .load_dataset('air_temperature')
       .air
       .isel(time=0))

ax = plt.axes(projection=ccrs.Orthographic(-80, 35))
ax.set_global()
air.plot.contourf(ax=ax, transform=ccrs.PlateCarree())
ax.coastlines()

plt.savefig('cartopy_example.png')
