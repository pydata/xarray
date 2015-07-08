Plotting
========

Introduction
------------

Xray plotting functionality is a thin wrapper around the popular
`matplotlib <http://matplotlib.org/>`__ library. 
Metadata from :py:class:`xray.DataArray` objects are used to add
informative labels. Matplotlib is required for plotting with xray.
Matplotlib syntax and function names were copied as much as possible, which
makes for an easy transition between the two.

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
does the work. For example,
for a plot with blue triangles marking the data points one can use a
matplotlib format string:

.. ipython:: python

    @savefig plotting_example_sin2.png width=4in
    sinpts.plot('b-^')

Keyword arguments work the same way.

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

Two Dimensions
--------------

For these examples we generate two dimensional data by computing the distance
from a 2d grid point to the origin

.. ipython:: python

    x = np.linspace(-5, 10, num=6)
    y = np.logspace(0, 1.2, num=7)
    xy = np.dstack(np.meshgrid(x, y))

    distance = np.linalg.norm(xy, axis=2)

    distance = xray.DataArray(distance, {'x': x, 'y': y})
    distance

The default :py:meth:`xray.DataArray.plot` sees that the data is 2 dimenstional
and calls :py:meth:`xray.DataArray.plot_imshow`. 

.. ipython:: python

    @savefig plotting_example_2d.png width=4in
    distance.plot()

The y grid points were generated from a log scale, so we can use matplotlib
to adjust the scale on y:

.. ipython:: python

    plt.yscale('log')

    @savefig plotting_example_2d3.png width=4in
    distance.plot()

Swap the variables plotted on vertical and horizontal axes by transposing the array.

TODO: This is easy, but is it better to have an argument for which variable
should appear on x and y axis?

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
plotting function based on the dimensions of the ``DataArray``. This table
describes what gets plotted:

=============== ======================================
Dimensions      Plotting function
--------------- --------------------------------------
1               :py:meth:`xray.DataArray.plot_line` 
2               :py:meth:`xray.DataArray.plot_imshow` 
Anything else   :py:meth:`xray.DataArray.plot_hist` 
=============== ======================================
