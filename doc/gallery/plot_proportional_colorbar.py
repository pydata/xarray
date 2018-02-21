# -*- coding: utf-8 -*-
"""
=================================
Proportional colorbar for discrete colormaps
=================================

While plotting a discrete colormap you may wish to label every colorbar ticks. By default, the tick labels are limited to a number of 10.
However, you can use the ``cbar_kwargs`` keyword to specify the number of ticks. It is also possible to arrange the spacing between ticks if ``levels`` are not regular.

"""

import xarray as xr
import matplotlib.pyplot as plt

# Load the data
air_temp = xr.tutorial.load_dataset('air_temperature')
air = air_temp.air
air2d = air.isel(time=500)

# Prepare the figure
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

# Different levels
levels = [200,210,220,230,240,250,260,270,280,290,320]

# Plot data
air2d.plot(ax=ax1, levels = levels)
air2d.plot(ax=ax2, levels = levels, cbar_kwargs={'ticks':levels})
air2d.plot(ax=ax3, levels = levels, cbar_kwargs={'ticks':levels, 'spacing':'proportional'})

# Show plots
plt.tight_layout()
plt.show()