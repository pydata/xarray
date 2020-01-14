"""
================================================
Compare weighted and unweighted mean temperature
================================================


Use ``air.weighted(weights).mean()`` to calculate the area-weighted temperature
for the air_temperature example dataset. This dataset has a regular latitude/ longitude
grid, thus the gridcell area decreases towards the pole. For this grid we can use the
cosine of the latitude as proxy for the grid cell area. Note how the weighted mean
temperature is higher than the unweighted.


"""

import matplotlib.pyplot as plt
import numpy as np

import xarray as xr

# Load the data
ds = xr.tutorial.load_dataset("air_temperature")
air = ds.air - 273.15  # to celsius

# resample from 6-hourly to daily values
air = air.resample(time="D").mean()

# the cosine of the latitude is proportional to the grid cell area (for a rectangular grid)
weights = np.cos(np.deg2rad(air.lat))

mean_air = air.weighted(weights).mean(("lat", "lon"))

# Prepare the figure
f, ax = plt.subplots(1, 1)

mean_air.plot(label="Area weighted mean")
air.mean(("lat", "lon")).plot(label="Unweighted mean")

ax.legend()

# Show
plt.tight_layout()
plt.show()
