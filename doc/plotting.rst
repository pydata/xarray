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

- `Cartopy <http://scitools.org.uk/cartopy/>`__: provides cartographic
  tools

Imports
~~~~~~~

Begin by importing the necessary modules:

.. ipython:: python

    import numpy as np
    import xray
    import matplotlib.pyplot as plt

The following line is not necessary, but it makes for a nice style.

.. ipython:: python

    plt.style.use('ggplot')

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
For example, for a 1 dimensional DataArray, :py:meth:`xray.DataArray.plot_line` calls ``plt.plot``,
passing in the index and the array values as x and y, respectively.
So to make a line plot with blue triangles a `matplotlib format string
<http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot>`__ 
can be used:

.. ipython:: python

    @savefig plotting_example_sin2.png width=4in
    sinpts.plot_line('b-^')

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

The index may be a time series.

.. ipython:: python

    import pandas as pd
    npts = 50
    time = pd.date_range('2015-01-01', periods=npts)
    noise = xray.DataArray(np.random.randn(npts), {'time': time})

    @savefig plotting_example_time.png width=6in
    noise.plot_line()


Two Dimensions
--------------

Simple Example
~~~~~~~~~~~~~~

The default :py:meth:`xray.DataArray.plot` sees that the data is 2 dimensional
and calls :py:meth:`xray.DataArray.plot_imshow`. 

.. ipython:: python

    a = np.zeros((5, 3))
    a[0, 0] = 1
    xa = xray.DataArray(a)
    xa

    @savefig plotting_example_2d.png width=4in
    xa.plot()

The top left pixel is 1, and the others are 0.

Simulated Data
~~~~~~~~~~~~~~

For further examples we generate two dimensional data by computing the distance
from a 2d grid point to the origin.
It's not necessary for the grid to be evenly spaced.

.. ipython:: python

    x = np.linspace(-5, 10, num=6)
    y = np.logspace(1.2, 0, num=7)
    xy = np.dstack(np.meshgrid(x, y))

    distance = np.linalg.norm(xy, axis=2)

    distance = xray.DataArray(distance, {'x': x, 'y': y})
    distance

Note the coordinate ``y`` here is decreasing. 
This makes the axes of the image plot in the expected way.

# TODO- Edge case- what if the coordinates are not sorted? Is this
possible? What if coordinates increasing?

Calling Matplotlib
~~~~~~~~~~~~~~~~~~

Use matplotlib to adjust plot parameters. For example, the
y grid points were generated from a log scale, so we can use matplotlib
to adjust the scale on y:

.. ipython:: python

    #plt.yscale('log')

    @savefig plotting_example_2d3.png width=4in
    distance.plot()

Changing Axes
~~~~~~~~~~~~~

Two dimensional plotting in xray uses the 
Swap the variables plotted on vertical and horizontal axes by transposing the array.

.. ipython:: python

    @savefig plotting_example_2d2.png width=4in
    distance.T.plot()

Contour Plot
~~~~~~~~~~~~

Visualization is 

.. ipython:: python

    @savefig plotting_example_contour.png width=4in
    distance.plot_contourf()
 
TODO- This  is the same plot as ``imshow``.

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
