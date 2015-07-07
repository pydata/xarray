Plotting
========

Introduction
------------

Xray plotting functionality is a thin wrapper around the popular
`matplotlib <http://matplotlib.org/>`__ library. 
The metadata from :py:class:`xray.DataArray` objects are used to add
informative labels.
We copy matplotlib syntax and function names as much as possible.

Hence matplotlib is a
dependency for plotting. 

xray tries to create reasonable labeled plots based on metadata and the array
dimensions.

But it's not always obvious what to plot. A wise man once said:
'In the face of ambiguity, refuse the temptation to guess.'
So don't be scared if you see some ``ValueError``'s when 
trying to plot, it just means you may need to get the data into a form
where plotting is more natural.

To begin, import numpy, pandas and xray:

.. ipython:: python

    import numpy as np
    import pandas as pd
    import xray
    import matplotlib.pyplot as plt

The following line is not necessary, but it makes for a nice style.

.. ipython:: python

    plt.style.use('ggplot')

1 Dimension
-----------

Here is a simple example of plotting. 
Xray uses the coordinate name to label the x axis.

.. ipython:: python

    x = np.linspace(0, 2*np.pi)
    sinpts = xray.DataArray(np.sin(x), {'t': x}, name='sin(t)')

    @savefig plotting_example_sin.png width=4in
    sinpts.plot()


Histogram
~~~~~~~~~

A histogram of the same data.

.. ipython:: python

    @savefig plotting_example_hist.png width=4in
    sinpts.plot_hist()

Additional arguments are passed directly to ``matplotlib.pyplot.hist``,
which handles the plotting.

.. ipython:: python

    @savefig plotting_example_hist2.png width=4in
    sinpts.plot_hist(bins=3)

2 Dimensions
------------

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
and calls :py:meth:`xray.DataArray.plot_imshow`. This was chosen as a
default
since it does not perform any smoothing or interpolation; it just shows the
raw data.

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

Multivariate Normal Density
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider the density for a two dimensional normal distribution
evaluated on a square grid::
    
    # TODO this requires scipy as a dependency for docs to build

    from scipy.stats import multivariate_normal

    g = np.linspace(-3, 3)
    xy = np.dstack(np.meshgrid(g, g))

    # 2d Normal distribution centered at 1, 0
    rv = multivariate_normal(mean=(1, 0))

    normal = xray.DataArray(rv.pdf(xy), {'x': g, 'y': g})

    # TODO- use xray method
    @savefig plotting_example_2dnormal.png
    plt.contourf(normal.x, normal.y, normal.data)


Rules
-----

The following is a more complete description of how xray determines what
and how to plot.

The method :py:meth:`xray.DataArray.plot` dispatches to an appropriate
plotting function based on the dimensions of the ``DataArray``.

=============== ======================================
Dimensions      Plotting function
--------------- --------------------------------------
1               :py:meth:`xray.DataArray.plot_line` 
2               :py:meth:`xray.DataArray.plot_imshow` 
Anything else   :py:meth:`xray.DataArray.plot_hist` 
=============== ======================================
