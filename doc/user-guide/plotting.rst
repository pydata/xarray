.. currentmodule:: xarray
.. _plotting:

Plotting
========

Introduction
------------

Labeled data enables expressive computations. These same
labels can also be used to easily create informative plots.

Xarray's plotting capabilities are centered around
:py:class:`DataArray` objects.
To plot :py:class:`Dataset` objects
simply access the relevant DataArrays, i.e. ``dset['var1']``.
Dataset specific plotting routines are also available (see :ref:`plot-dataset`).
Here we focus mostly on arrays 2d or larger. If your data fits
nicely into a pandas DataFrame then you're better off using one of the more
developed tools there.

Xarray plotting functionality is a thin wrapper around the popular
`matplotlib <https://matplotlib.org/>`_ library.
Matplotlib syntax and function names were copied as much as possible, which
makes for an easy transition between the two.
Matplotlib must be installed before xarray can plot.

To use xarray's plotting capabilities with time coordinates containing
``cftime.datetime`` objects
`nc-time-axis <https://github.com/SciTools/nc-time-axis>`_ v1.3.0 or later
needs to be installed.

For more extensive plotting applications consider the following projects:

- `Seaborn <https://seaborn.pydata.org/>`_: "provides
  a high-level interface for drawing attractive statistical graphics."
  Integrates well with pandas.

- `HoloViews <https://holoviews.org/>`_
  and `GeoViews <https://geoviews.org/>`_: "Composable, declarative
  data structures for building even complex visualizations easily." Includes
  native support for xarray objects.

- `hvplot <https://hvplot.pyviz.org/>`_: ``hvplot`` makes it very easy to produce
  dynamic plots (backed by ``Holoviews`` or ``Geoviews``) by adding a ``hvplot``
  accessor to DataArrays.

- `Cartopy <https://scitools.org.uk/cartopy/docs/latest/>`_: Provides cartographic
  tools.

Details
-------

Ways to Use
~~~~~~~~~~~

There are three ways to use the xarray plotting functionality:

1. Use ``plot`` as a convenience method for a DataArray.

2. Access a specific plotting method from the ``plot`` attribute of a
   DataArray.

3. Directly from the xarray plot submodule.

These are provided for user convenience; they all call the same code.

.. jupyter-execute::

    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr

    da = xr.DataArray(range(5))
    fig, axs = plt.subplots(ncols=2, nrows=2)
    da.plot(ax=axs[0, 0])
    da.plot.line(ax=axs[0, 1])
    xr.plot.plot(da, ax=axs[1, 0])
    xr.plot.line(da, ax=axs[1, 1]);

Here the output is the same. Since the data is 1 dimensional the line plot
was used.

The convenience method :py:meth:`xarray.DataArray.plot` dispatches to an appropriate
plotting function based on the dimensions of the ``DataArray`` and whether
the coordinates are sorted and uniformly spaced. This table
describes what gets plotted:

=============== ===========================
Dimensions      Plotting function
--------------- ---------------------------
1               :py:func:`xarray.plot.line`
2               :py:func:`xarray.plot.pcolormesh`
Anything else   :py:func:`xarray.plot.hist`
=============== ===========================

Coordinates
~~~~~~~~~~~

If you'd like to find out what's really going on in the coordinate system,
read on.

.. jupyter-execute::

    import cartopy.crs as ccrs

    a0 = xr.DataArray(np.zeros((4, 3, 2)), dims=("y", "x", "z"), name="temperature")
    a0[0, 0, 0] = 1
    a = a0.isel(z=0)
    a

The plot will produce an image corresponding to the values of the array.
Hence the top left pixel will be a different color than the others.
Before reading on, you may want to look at the coordinates and
think carefully about what the limits, labels, and orientation for
each of the axes should be.

.. jupyter-execute::

    a.plot();

It may seem strange that
the values on the y axis are decreasing with -0.5 on the top. This is because
the pixels are centered over their coordinates, and the
axis labels and ranges correspond to the values of the
coordinates.

Multidimensional coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See also: :ref:`/examples/multidimensional-coords.ipynb`.

You can plot irregular grids defined by multidimensional coordinates with
xarray, but you'll have to tell the plot function to use these coordinates
instead of the default ones:

.. jupyter-execute::

    lon, lat = np.meshgrid(np.linspace(-20, 20, 5), np.linspace(0, 30, 4))
    lon += lat / 10
    lat += lon / 10
    da = xr.DataArray(
        np.arange(20).reshape(4, 5),
        dims=["y", "x"],
        coords={"lat": (("y", "x"), lat), "lon": (("y", "x"), lon)},
    )

    da.plot.pcolormesh(x="lon", y="lat");

Note that in this case, xarray still follows the pixel centered convention.
This might be undesirable in some cases, for example when your data is defined
on a polar projection (:issue:`781`). This is why the default is to not follow
this convention when plotting on a map:

.. jupyter-execute::
    :stderr:

    ax = plt.subplot(projection=ccrs.PlateCarree())
    da.plot.pcolormesh(x="lon", y="lat", ax=ax)
    ax.scatter(lon, lat, transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(draw_labels=True);

You can however decide to infer the cell boundaries and use the
``infer_intervals`` keyword:

.. jupyter-execute::

    ax = plt.subplot(projection=ccrs.PlateCarree())
    da.plot.pcolormesh(x="lon", y="lat", ax=ax, infer_intervals=True)
    ax.scatter(lon, lat, transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(draw_labels=True);

.. note::
    The data model of xarray does not support datasets with `cell boundaries`_
    yet. If you want to use these coordinates, you'll have to make the plots
    outside the xarray framework.

.. _cell boundaries: https://cfconventions.org/cf-conventions/v1.6.0/cf-conventions.html#cell-boundaries

One can also make line plots with multidimensional coordinates. In this case, ``hue`` must be a dimension name, not a coordinate name.

.. jupyter-execute::

    f, ax = plt.subplots(2, 1)
    da.plot.line(x="lon", hue="y", ax=ax[0])
    da.plot.line(x="lon", hue="x", ax=ax[1]);

.. toctree:::
   :maxdepth: 2

   plotting-lines
   plotting-2d
   plotting-faceting
   plotting-scatter-quiver

.. note::
   This guide covers the core plotting functionality. For additional features like maps, see the individual plotting sections.
