Plotting
========

Introduction
------------

The goal of xray's plotting is to make exploratory plotting quick
and easy by using metadata from :py:class:`xray.DataArray` objects to add
informative labels. 

Xray plotting functionality is a thin wrapper around the popular
`matplotlib <http://matplotlib.org/>`__ library. 
Matplotlib syntax and function names were copied as much as possible, which
makes for an easy transition between the two.

For more specialized plotting applications consider the following packages:

- `Seaborn <http://stanford.edu/~mwaskom/software/seaborn/>`__: "provides
  a high-level interface for drawing attractive statistical graphics."
  Integrates well with pandas.

- `Holoviews <http://ioam.github.io/holoviews/>`__: "Composable, declarative
  data structures for building even complex visualizations easily."
  Works for 2d datasets.

- `Cartopy <http://scitools.org.uk/cartopy/>`__: provides cartographic
  tools

Imports
~~~~~~~

Begin by importing the necessary modules:

.. ipython:: python

    import numpy as np
    import xray
    import matplotlib.pyplot as plt

One Dimension
-------------

Simple Example
~~~~~~~~~~~~~~

Here is a simple example of plotting.
Xray uses the coordinate name to label the x axis:

.. ipython:: python

    t = np.linspace(0, 2*np.pi)
    sinpts = xray.DataArray(np.sin(t), {'t': t}, name='sin(t)')

    @savefig plotting_example_sin.png width=4in
    sinpts.plot()

Additional Arguments 
~~~~~~~~~~~~~~~~~~~~~

Additional arguments are passed directly to the matplotlib function which
does the work. 
For example, :py:meth:`xray.DataArray.plot_line` calls ``plt.plot``,
passing in the index and the array values as x and y, respectively.
So to make a line plot with blue triangles a `matplotlib format string
<http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot>`__ 
can be used:

.. ipython:: python

    @savefig plotting_example_sin2.png width=4in
    sinpts.plot_line('b-^')

.. warning:: 
    Not all xray plotting methods support passing positional arguments 
    to the underlying matplotlib functions, but they do all
    support keyword arguments. Check the documentation for each
    function to make sure.

Keyword arguments work the same way:

.. ipython:: python

    @savefig plotting_example_sin3.png width=4in
    sinpts.plot_line(color='purple', marker='o')

Adding to Existing Axis
~~~~~~~~~~~~~~~~~~~~~~~

To add the plot to an existing axis pass in the axis as a keyword argument
``ax``. This works for all xray plotting methods.
In this example ``axes`` is a tuple consisting of the left and right
axes created by ``plt.subplots``.

.. ipython:: python

    fig, axes = plt.subplots(ncols=2)

    axes

    sinpts.plot(ax=axes[0])
    sinpts.plot_hist(ax=axes[1])

    @savefig plotting_example_existing_axes.png width=6in
    plt.show()

Instead of using the default :py:meth:`xray.DataArray.plot` we see a
histogram created by :py:meth:`xray.DataArray.plot_hist`.

Time Series
~~~~~~~~~~~

The index may be a date.

.. ipython:: python

    import pandas as pd
    npts = 20
    time = pd.date_range('2015-01-01', periods=npts)
    noise = xray.DataArray(np.random.randn(npts), {'time': time})

    @savefig plotting_example_time.png width=6in
    noise.plot_line()

TODO- rotate dates printed on x axis.

Two Dimensions
--------------

Simple Example
~~~~~~~~~~~~~~

The default method :py:meth:`xray.DataArray.plot` sees that the data is 
2 dimensional. If the coordinates are uniformly spaced then it
calls :py:meth:`xray.DataArray.plot_imshow`. 

.. ipython:: python

    a = xray.DataArray(np.zeros((4, 3)), dims=('y', 'x'))
    a[0, 0] = 1
    a

The plot will produce an image corresponding to the values of the array. 
Hence the top left pixel will be a different color than the others. 
Before reading on, you may want to look at the coordinates and 
think carefully about what the limits, labels, and orientation for 
each of the axes should be.

.. ipython:: python

    @savefig plotting_example_2d_simple.png width=4in
    a.plot()

It may seem strange that
the the values on the y axis are decreasing with -0.5 on the top. This is because 
the pixels are centered over their coordinates, and the
axis labels and ranges correspond to the values of the
coordinates. 

An `extended slice <http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html>`__
can be used to reverse the order of the rows, producing a
more conventional plot where the coordinates increase in the y axis.

.. ipython:: python

    a[::-1, :]

    @savefig plotting_example_2d_simple_reversed.png width=4in
    a[::-1, :].plot()

Simulated Data
~~~~~~~~~~~~~~

For further examples we generate two dimensional data by computing the distance
from a 2d grid point to the origin.

.. ipython:: python

    x = np.arange(start=0, stop=10, step=2)
    y = np.arange(start=9, stop=-7, step=-3)
    xy = np.dstack(np.meshgrid(x, y))

    distance = np.linalg.norm(xy, axis=2)

    distance = xray.DataArray(distance, {'x': x, 'y': y})
    distance

Note the coordinate ``y`` here is decreasing. 
This makes the y axes appear in the conventional way.

.. ipython:: python

    @savefig plotting_2d_simulated.png width=4in
    distance.plot()
 
Changing Axes
~~~~~~~~~~~~~

To swap the variables plotted on vertical and horizontal axes one can 
transpose the array.

.. ipython:: python

    @savefig plotting_changing_axes.png width=4in
    distance.T.plot()

TODO: Feedback here please. This requires the user to put the array into
the order they want for plotting. To plot with sorted coordinates they
would have to write something
like this: ``distance.T[::-1, ::-1].plot()``. 
This requires the user to be aware of how the array is organized.

Alternatively, this could be implemented in
xray plotting as: ``distance.plot(xvar='y', sortx=True,
sorty=True)``. 
This allows the use of the dimension
name to describe which coordinate should appear as the x variable on the
plot, and is probably more convenient.

Nonuniform Coordinates
~~~~~~~~~~~~~~~~~~~~~~

It's not necessary for the coordinates to be evenly spaced. If not, then
:py:meth:`xray.DataArray.plot` produces a filled contour plot by calling
:py:meth:`xray.DataArray.plot_contourf`. This example demonstrates that by
using one coordinate with logarithmic spacing.

.. ipython:: python

    x = np.linspace(0, 500)
    y = np.logspace(0, 3)
    xy = np.dstack(np.meshgrid(x, y))
    d_ylog = np.linalg.norm(xy, axis=2)
    d_ylog = xray.DataArray(d_ylog, {'x': x, 'y': y})

    @savefig plotting_nonuniform_coords.png width=4in
    d_ylog.plot()

Calling Matplotlib
~~~~~~~~~~~~~~~~~~

Since this is a thin wrapper around matplotlib, all the functionality of 
matplotlib is available. For example, use a different color map and add a title.

.. ipython:: python

    d_ylog.plot(cmap=plt.cm.Blues)
    plt.title('Euclidean distance from point to origin')

    @savefig plotting_2d_call_matplotlib.png width=4in
    plt.show()

Colormaps
~~~~~~~~~

Suppose we want two plots to share the same color scale. This can be
achieved by passing in a color map.

TODO- Don't actually know how to do this yet. Will probably want it for the
Faceting

.. ipython:: python

    colors = plt.cm.Blues

    fig, axes = plt.subplots(ncols=2)

    distance.plot(ax=axes[0], cmap=colors, )

    halfd = distance / 2
    halfd.plot(ax=axes[1], cmap=colors)

    @savefig plotting_same_color_scale.png width=6in
    plt.show()

Maps
----

To follow this section you'll need to have Cartopy installed and working.

Plot an image over the Atlantic ocean.

.. ipython:: python

    import cartopy.crs as ccrs

    nlat = 15
    nlon = 5
    atlantic = xray.DataArray(np.random.randn(nlat, nlon),
            coords = (np.linspace(50, 20, nlat), np.linspace(-60, -20, nlon)),
            dims = ('latitude', 'longitude'))

    ax = plt.axes(projection=ccrs.PlateCarree())

    atlantic.plot(ax=ax)

    ax.set_ylim(0, 90)
    ax.set_xlim(-180, 30)

    ax.coastlines()

    @savefig simple_map.png width=6in
    plt.show()

Details
-------

There are two ways to use the xray plotting functionality:

1. Use the ``plot`` convenience methods of :py:class:`xray.DataArray` 
2. Directly from the xray plotting submodule::

    import xray.plotting as xplt

The convenience method :py:meth:`xray.DataArray.plot` dispatches to an appropriate
plotting function based on the dimensions of the ``DataArray`` and whether
the coordinates are sorted and uniformly spaced. This table
describes what gets plotted:

=============== =========== ===========================
Dimensions      Coordinates Plotting function
--------------- ----------- ---------------------------
1                           :py:meth:`xray.DataArray.plot_line` 
2               Uniform     :py:meth:`xray.DataArray.plot_imshow` 
2               Irregular   :py:meth:`xray.DataArray.plot_contourf` 
Anything else               :py:meth:`xray.DataArray.plot_hist` 
=============== =========== ===========================
