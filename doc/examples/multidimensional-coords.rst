.. _examples.multidim:

Working with Multidimensional Coordinates
=========================================

Author: `Ryan Abernathey <https://github.com/rabernat>`__

Many datasets have *physical coordinates* which differ from their
*logical coordinates*. Xarray provides several ways to plot and analyze
such datasets.


.. ipython:: python

    import numpy as np
    import pandas as pd
    import xarray as xr
    import netCDF4
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

As an example, consider this dataset from the
`xarray-data <https://github.com/pydata/xarray-data>`__ repository.


.. ipython:: python

    ds = xr.tutorial.open_dataset('rasm').load()
    ds

In this example, the *logical coordinates* are ``x`` and ``y``, while
the *physical coordinates* are ``xc`` and ``yc``, which represent the
latitudes and longitude of the data.


.. ipython:: python

    ds.xc.attrs
    ds.yc.attrs


Plotting
--------

Let's examine these coordinate variables by plotting them.

.. ipython:: python

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9,3))
    ds.xc.plot(ax=ax1);
    @savefig xarray_multidimensional_coords_8_2.png width=100%
    ds.yc.plot(ax=ax2);

Note that the variables ``xc`` (longitude) and ``yc`` (latitude) are
two-dimensional scalar fields.

If we try to plot the data variable ``Tair``, by default we get the
logical coordinates.

.. ipython:: python
   :suppress:

    f = plt.figure(figsize=(6, 4))

.. ipython:: python

    @savefig xarray_multidimensional_coords_10_1.png width=5in
    ds.Tair[0].plot();


In order to visualize the data on a conventional latitude-longitude
grid, we can take advantage of xarray's ability to apply
`cartopy <http://scitools.org.uk/cartopy/index.html>`__ map projections.

.. ipython:: python

    plt.figure(figsize=(7,2));
    ax = plt.axes(projection=ccrs.PlateCarree());
    ds.Tair[0].plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(),
                               x='xc', y='yc', add_colorbar=False);
    @savefig xarray_multidimensional_coords_12_0.png width=100%
    ax.coastlines();

Multidimensional Groupby
------------------------

The above example allowed us to visualize the data on a regular
latitude-longitude grid. But what if we want to do a calculation that
involves grouping over one of these physical coordinates (rather than
the logical coordinates), for example, calculating the mean temperature
at each latitude. This can be achieved using xarray's ``groupby``
function, which accepts multidimensional variables. By default,
``groupby`` will use every unique value in the variable, which is
probably not what we want. Instead, we can use the ``groupby_bins``
function to specify the output coordinates of the group.

.. ipython:: python
   :suppress:

    f = plt.figure(figsize=(6, 4.5))

.. ipython:: python

    # define two-degree wide latitude bins
    lat_bins = np.arange(0, 91, 2)
    # define a label for each bin corresponding to the central latitude
    lat_center = np.arange(1, 90, 2)
    # group according to those bins and take the mean
    Tair_lat_mean = (ds.Tair.groupby_bins('xc', lat_bins, labels=lat_center)
	             .mean(xr.ALL_DIMS))
    # plot the result
    @savefig xarray_multidimensional_coords_14_1.png width=5in
    Tair_lat_mean.plot();


Note that the resulting coordinate for the ``groupby_bins`` operation
got the ``_bins`` suffix appended: ``xc_bins``. This help us distinguish
it from the original multidimensional variable ``xc``.
