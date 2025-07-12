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

.. toctree::
   :maxdepth: 2

   plotting-lines
   plotting-2d
   plotting-faceting
   plotting-scatter-quiver

.. note::
   This guide covers the core plotting functionality. For additional features like maps, see the individual plotting sections.
