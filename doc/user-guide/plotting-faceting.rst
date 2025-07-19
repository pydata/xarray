.. currentmodule:: xarray
.. _plotting.faceting:

Faceting
========

Faceting here refers to splitting an array along one or two dimensions and
plotting each group.
Xarray's basic plotting is useful for plotting two dimensional arrays. What
about three or four dimensional arrays? That's where facets become helpful.
The general approach to plotting here is called "small multiples", where the
same kind of plot is repeated multiple times, and the specific use of small
multiples to display the same relationship conditioned on one or more other
variables is often called a "trellis plot".

.. jupyter-execute::

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import xarray as xr

    # Load example data
    airtemps = xr.tutorial.open_dataset("air_temperature")
    air = airtemps.air - 273.15
    air.attrs = airtemps.air.attrs
    air.attrs["units"] = "deg C"

Consider the temperature data set. There are 4 observations per day for two
years which makes for 2920 values along the time dimension.
One way to visualize this data is to make a
separate plot for each time period.

The faceted dimension should not have too many values;
faceting on the time dimension will produce 2920 plots. That's
too much to be helpful. To handle this situation try performing
an operation that reduces the size of the data in some way. For example, we
could compute the average air temperature for each month and reduce the
size of this dimension from 2920 -> 12. A simpler way is
to just take a slice on that dimension.
So let's use a slice to pick 6 times throughout the first year.

.. jupyter-execute::

    t = air.isel(time=slice(0, 365 * 4, 250))
    t.coords

================
 Simple Example
================

The easiest way to create faceted plots is to pass in ``row`` or ``col``
arguments to the xarray plotting methods/functions. This returns a
:py:class:`xarray.plot.FacetGrid` object.

.. jupyter-execute::

    g_simple = t.plot(x="lon", y="lat", col="time", col_wrap=3);

Faceting also works for line plots.

.. jupyter-execute::

    g_simple_line = t.isel(lat=slice(0, None, 4)).plot(
        x="lon", hue="lat", col="time", col_wrap=3
    );

===============
 4 dimensional
===============

For 4 dimensional arrays we can use the rows and columns of the grids.
Here we create a 4 dimensional array by taking the original data and adding
a fixed amount. Now we can see how the temperature maps would compare if
one were much hotter.

.. jupyter-execute::

    t2 = t.isel(time=slice(0, 2))
    t4d = xr.concat([t2, t2 + 40], pd.Index(["normal", "hot"], name="fourth_dim"))
    # This is a 4d array
    t4d.coords

    t4d.plot(x="lon", y="lat", col="time", row="fourth_dim");

================
 Other features
================

Faceted plotting supports other arguments common to xarray 2d plots.

.. jupyter-execute::

    hasoutliers = t.isel(time=slice(0, 5)).copy()
    hasoutliers[0, 0, 0] = -100
    hasoutliers[-1, -1, -1] = 400

    g = hasoutliers.plot.pcolormesh(
        x="lon",
        y="lat",
        col="time",
        col_wrap=3,
        robust=True,
        cmap="viridis",
        cbar_kwargs={"label": "this has outliers"},
    )

===================
 FacetGrid Objects
===================

The object returned, ``g`` in the above examples, is a :py:class:`~xarray.plot.FacetGrid` object
that links a :py:class:`DataArray` to a matplotlib figure with a particular structure.
This object can be used to control the behavior of the multiple plots.
It borrows an API and code from `Seaborn's FacetGrid
<https://seaborn.pydata.org/tutorial/axis_grids.html>`_.
The structure is contained within the ``axs`` and ``name_dicts``
attributes, both 2d NumPy object arrays.

.. jupyter-execute::

    g.axs

.. jupyter-execute::

    g.name_dicts

It's possible to select the :py:class:`xarray.DataArray` or
:py:class:`xarray.Dataset` corresponding to the FacetGrid through the
``name_dicts``.

.. jupyter-execute::

    g.data.loc[g.name_dicts[0, 0]]

Here is an example of using the lower level API and then modifying the axes after
they have been plotted.

.. jupyter-execute::


    g = t.plot.imshow(x="lon", y="lat", col="time", col_wrap=3, robust=True)

    for i, ax in enumerate(g.axs.flat):
        ax.set_title("Air Temperature %d" % i)

    bottomright = g.axs[-1, -1]
    bottomright.annotate("bottom right", (240, 40));


:py:class:`~xarray.plot.FacetGrid` objects have methods that let you customize the automatically generated
axis labels, axis ticks and plot titles. See :py:meth:`~xarray.plot.FacetGrid.set_titles`,
:py:meth:`~xarray.plot.FacetGrid.set_xlabels`, :py:meth:`~xarray.plot.FacetGrid.set_ylabels` and
:py:meth:`~xarray.plot.FacetGrid.set_ticks` for more information.
Plotting functions can be applied to each subset of the data by calling
:py:meth:`~xarray.plot.FacetGrid.map_dataarray` or to each subplot by calling :py:meth:`~xarray.plot.FacetGrid.map`.
