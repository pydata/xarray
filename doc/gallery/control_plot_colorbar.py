# -*- coding: utf-8 -*-
"""
==========================
Control a plot's colorbar
==========================

Use ``cbar_kwargs`` keyword to specify the number of ticks.
``spacing`` can be used to draw proportional ticks..
"""
import xarray as xr
import matplotlib.pyplot as plt

# Load the data
air_temp = xr.tutorial.load_dataset('air_temperature')
air2d = air_temp.air.isel(time=500)

# Prepare the figure
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

# Levels increment by 10 except last value to show a proportional colorbar
levels = [200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 320]

# Plot data
air2d.plot(ax=ax1, levels=levels)
air2d.plot(ax=ax2, levels=levels, cbar_kwargs={'ticks': levels})
air2d.plot(ax=ax3, levels=levels, cbar_kwargs={
           'ticks': levels, 'spacing': 'proportional'})

# Show plots
plt.tight_layout()
plt.show()
