.. _examples.multidim_lookup:

Multidimensional Lookup with Vectorized Indexing
=================================================

Author: `Keisuke Fujii <http://github.org/fujiisoup>`__

:ref:`vectorized_indexing` can be used to project object to another coordinate by nearest neighbor lookup.

.. ipython:: python

    import numpy as np
    import pandas as pd
    import xarray as xr
    import netCDF4
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt


ds.sel(latitude=latitude_grid, longitude=longitude_grid, method='nearest', tolerance=0.1).
