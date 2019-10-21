"""
==================================
Multiple lines from a 2d DataArray
==================================


Use :py:func:`xarray.plot.line` on a 2d DataArray to plot selections as
multiple lines.

See :ref:`plotting.multiplelines` for more details.

"""

import matplotlib.pyplot as plt

import xarray as xr

# Load the data
ds = xr.tutorial.load_dataset("air_temperature")
air = ds.air - 273.15  # to celsius

# Prepare the figure
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

# Selected latitude indices
isel_lats = [10, 15, 20]

# Temperature vs longitude plot - illustrates the "hue" kwarg
air.isel(time=0, lat=isel_lats).plot.line(ax=ax1, hue="lat")
ax1.set_ylabel("°C")

# Temperature vs time plot - illustrates the "x" and "add_legend" kwargs
air.isel(lon=30, lat=isel_lats).plot.line(ax=ax2, x="time", add_legend=False)
ax2.set_ylabel("")

# Show
plt.tight_layout()
plt.show()
