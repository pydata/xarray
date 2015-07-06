Plotting
========

xray plotting functionality is a thin wrapper around the popular
`matplotlib <http://matplotlib.org/>`__ library. Hence matplotlib is a
dependency for plotting. The metadata is used to
add informative labels.

xray tries to create reasonable labeled plots based on metadata and the array
dimensions.

But it's not always obvious what to plot. A wise man once said:
'In the face of ambiguity, refuse the temptation to guess.'
So don't be scared if you see some ``ValueError``'s when 
trying to plot, it just means you may need to get the data into a form
where plotting is more natural.

Examples
--------

To begin, import numpy, pandas and xray:

.. ipython:: python

    import numpy as np
    import pandas as pd
    import xray
    import matplotlib.pyplot as plt

The following line is not necessary, but it makes for a nice style.

.. ipython:: python

    plt.style.use('ggplot')


Sin Function
~~~~~~~~~~~~

Here is a simple example of plotting. 
Xray uses the coordinate name to label the x axis.

.. ipython:: python

    x = np.linspace(0, 2*np.pi)
    sinpts = xray.DataArray(np.sin(x), {'time': x}, name='sin(x)')

    @savefig plotting_example_sin.png
    sinpts.plot()


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
