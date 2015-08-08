Plotting
========

Introduction
------------

The goal of xray's plotting is to make exploratory plotting quick
and easy by using metadata from :py:class:`xray.DataArray` objects to add
informative labels. To plot :py:class:`xray.Dataset` objects 
simply access the relevant DataArrays, ie ``dset['var1']``.

Xray plotting functionality is a thin wrapper around the popular
`matplotlib <http://matplotlib.org/>`_ library.
Matplotlib syntax and function names were copied as much as possible, which
makes for an easy transition between the two.
Matplotlib must be installed before xray can plot.

For more extensive plotting applications consider the following projects:

- `Seaborn <http://stanford.edu/~mwaskom/software/seaborn/>`_: "provides
  a high-level interface for drawing attractive statistical graphics."
  Integrates well with pandas.

- `Holoviews <http://ioam.github.io/holoviews/>`_: "Composable, declarative
  data structures for building even complex visualizations easily." Works
  for 2d datasets.

- `Cartopy <http://scitools.org.uk/cartopy/>`_: Provides cartographic
  tools.

Imports
~~~~~~~

.. ipython:: python

    # Use defaults so we don't get gridlines in generated docs
    import matplotlib as mpl
    mpl.rcdefaults()

The following imports are necessary for all of the examples.

.. ipython:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import xray

One Dimension
-------------

Simple Example
~~~~~~~~~~~~~~

Xray uses the coordinate name to label the x axis:

.. ipython:: python

    t = np.linspace(0, np.pi, num=20)
    sinpts = xray.DataArray(np.sin(t), {'t': t}, name='sin(t)')
    sinpts

    @savefig plotting_example_sin.png width=4in
    sinpts.plot()

Additional Arguments
~~~~~~~~~~~~~~~~~~~~~

Additional arguments are passed directly to the matplotlib function which
does the work.
For example, :py:func:`xray.plot.line` calls 
matplotlib.pyplot.plot_ passing in the index and the array values as x and y, respectively.
So to make a line plot with blue triangles a matplotlib format string
can be used:

.. _matplotlib.pyplot.plot: http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot

.. ipython:: python

    @savefig plotting_example_sin2.png width=4in
    sinpts.plot.line('b-^')

.. warning::
    Not all xray plotting methods support passing positional arguments
    to the wrapped matplotlib functions, but they do all
    support keyword arguments.

Keyword arguments work the same way, and are more explicit.

.. ipython:: python

    @savefig plotting_example_sin3.png width=4in
    sinpts.plot.line(color='purple', marker='o')

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
    sinpts.plot.hist(ax=axes[1])

    @savefig plotting_example_existing_axes.png width=6in
    plt.show()

On the right is a histogram created by :py:func:`xray.plot.hist`.

Time Series
~~~~~~~~~~~

The index may be a date.

.. ipython:: python

    import pandas as pd
    npts = 20
    time = pd.date_range('2015-01-01', periods=npts)
    noise = xray.DataArray(np.random.randn(npts), {'time': time})

    @savefig plotting_example_time.png width=6in
    noise.plot.line()

TODO- rotate dates printed on x axis.


Two Dimensions
--------------

Simple Example
~~~~~~~~~~~~~~

The default method :py:meth:`xray.DataArray.plot` sees that the data is
2 dimensional. If the coordinates are uniformly spaced then it
calls :py:func:`xray.plot.imshow`.

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
the values on the y axis are decreasing with -0.5 on the top. This is because
the pixels are centered over their coordinates, and the
axis labels and ranges correspond to the values of the
coordinates. 

All 2d plots in xray allow the use of the keyword arguments ``yincrease=True``
to produce a
more conventional plot where the coordinates increase in the y axis.
``xincrease`` works similarly.

.. ipython:: python

    @savefig 2d_simple_yincrease.png width=4in
    a.plot(yincrease=True)

Missing Values
~~~~~~~~~~~~~~

Xray plots data with :ref:`missing_values`.

.. ipython:: python

    # This data has holes in it!
    a[1, 1] = np.nan

    @savefig plotting_missing_values.png width=4in
    a.plot()

Simulated Data
~~~~~~~~~~~~~~

For further examples we generate two dimensional data by computing the Euclidean
distance from a 2d grid point to the origin.

.. ipython:: python

    x = np.arange(start=0, stop=10, step=2)
    y = np.arange(start=9, stop=-7, step=-3)
    xy = np.dstack(np.meshgrid(x, y))

    distance = np.linalg.norm(xy, axis=2)

    distance = xray.DataArray(distance, zip(('y', 'x'), (y, x)))
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

To make x and y increase:

.. ipython:: python

    @savefig plotting_changing_axes2.png width=4in
    distance.T.plot(xincrease=True, yincrease=True)

Nonuniform Coordinates
~~~~~~~~~~~~~~~~~~~~~~

It's not necessary for the coordinates to be evenly spaced. If not, then
:py:meth:`xray.DataArray.plot` produces a filled contour plot by calling
:py:func:`xray.plot.contourf`. This example demonstrates that by
using one coordinate with logarithmic spacing.

.. ipython:: python

    x = np.linspace(0, 500)
    y = np.logspace(0, 3)
    xy = np.dstack(np.meshgrid(x, y))
    d_ylog = np.linalg.norm(xy, axis=2)
    d_ylog = xray.DataArray(d_ylog, zip(('y', 'x'), (y, x)))

    @savefig plotting_nonuniform_coords.png width=4in
    d_ylog.plot()

Calling Matplotlib
~~~~~~~~~~~~~~~~~~

Since this is a thin wrapper around matplotlib, all the functionality of
matplotlib is available. 

.. ipython:: python

    d_ylog.plot(cmap=plt.cm.Blues)
    plt.title('Euclidean distance from point to origin')
    plt.xlabel('temperature (C)')

    @savefig plotting_2d_call_matplotlib.png width=4in
    plt.show()

.. warning::

    Xray methods update label information and generally play around with the
    axes. So any kind of updates to the plot 
    should be done *after* the call to the xray's plot.
    In the example below, ``plt.xlabel`` effectively does nothing, since 
    ``d_ylog.plot()`` updates the xlabel.

.. ipython:: python

    plt.xlabel('temperature (C)')
    d_ylog.plot()

    @savefig plotting_2d_call_matplotlib2.png width=4in
    plt.show()

Contour plots can have missing values also.

.. ipython:: python

    d_ylog[30:48, 10:30] = np.nan

    d_ylog.plot()

    plt.text(100, 600, 'So common...')

    @savefig plotting_nonuniform_coords_missing.png width=4in
    plt.show()

Colormaps
~~~~~~~~~

Suppose we want two plots to share the same color scale. This can be
achieved by passing in axes and adding the color bar
later.

.. ipython:: python

    fig, axes = plt.subplots(ncols=2)

    kwargs = {'cmap': plt.cm.Blues, 'vmin': distance.min(), 'vmax': distance.max(), 'add_colorbar': False}

    im = distance.plot(ax=axes[0], **kwargs)

    halfd = distance / 2
    halfd.plot(ax=axes[1], **kwargs)

    plt.colorbar(im, ax=axes.tolist())

    @savefig plotting_same_color_scale.png width=6in
    plt.show()

Here we've used the object returned by :py:meth:`xray.DataArray.plot` to
pass in as an argument to
`plt.colorbar <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.colorbar>`_.
Take a closer look:

.. ipython:: python
    
    im

In general xray's plotting functions modify the axes and
return the same objects that the wrapped
matplotlib functions return.

Discrete Colormaps
~~~~~~~~~~~~~~~~~~

It is often useful, when visualizing 2d data, to use a discrete colormap,
rather than the default continuous colormaps that matplotlib uses. The
``levels`` keyword argument can be used to generate plots with discrete
colormaps. For example, to make a plot with 8 discrete color intervals:

.. ipython:: python

    @savefig plotting_discrete_levels.png width=4in
    distance.plot(levels=8)

It is also possible to use a list of levels to specify the boundaries of the
discrete colormap:

.. ipython:: python

    @savefig plotting_listed_levels.png width=4in
    distance.plot(levels=[2, 5, 10, 11])

Finally, if you are have `Seaborn <http://stanford.edu/~mwaskom/software/seaborn/>`_ installed, you can also specify a `seaborn` color palete or a list of colors as the ``cmap`` argument:

.. ipython:: python

    flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    @savefig plotting_custom_colors_levels.png width=4in
    distance.plot(levels=[1, 2, 4, 5, 7], cmap=flatui)

Maps
----

To follow this section you'll need to have Cartopy installed and working.

This script will plot an image over the Atlantic ocean.

.. literalinclude:: examples/cartopy_atlantic.py

Here is the resulting image:

.. image:: examples/atlantic_noise.png

Details
-------

Ways to Use
~~~~~~~~~~~

There are three ways to use the xray plotting functionality:

1. Use ``plot`` as a convenience method for a DataArray.

2. Access a specific plotting method from the ``plot`` attribute of a
   DataArray.

3. Directly from the xray plot submodule.

These are provided for user convenience; they all call the same code.

.. ipython:: python

    import xray.plot as xplt
    da = xray.DataArray(range(5))
    fig, axes = plt.subplots(ncols=2, nrows=2)
    da.plot(ax=axes[0, 0])
    da.plot.line(ax=axes[0, 1])
    xplt.plot(da, ax=axes[1, 0])
    xplt.line(da, ax=axes[1, 1])
    @savefig plotting_ways_to_use.png width=6in
    plt.show()

Here the output is the same. Since the data is 1 dimensional the line plot
was used.

The convenience method :py:meth:`xray.DataArray.plot` dispatches to an appropriate
plotting function based on the dimensions of the ``DataArray`` and whether
the coordinates are sorted and uniformly spaced. This table
describes what gets plotted:

=============== =========== ===========================
Dimensions      Coordinates Plotting function
--------------- ----------- ---------------------------
1                           :py:func:`xray.plot.line`
2               Uniform     :py:func:`xray.plot.imshow`
2               Irregular   :py:func:`xray.plot.contourf`
Anything else               :py:func:`xray.plot.hist`
=============== =========== ===========================
