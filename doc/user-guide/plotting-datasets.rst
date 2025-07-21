.. currentmodule:: xarray
.. _plot-dataset:

Dataset Plotting
================

Xarray has limited support for plotting Dataset variables against each other.
Consider this dataset

.. jupyter-execute::

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import pandas as pd
    import xarray as xr
    %matplotlib inline

    # Load air temperature dataset (needed for complete context)
    airtemps = xr.tutorial.open_dataset("air_temperature")

    # Convert to celsius
    air = airtemps.air - 273.15

    # copy attributes to get nice figure labels and change Kelvin to Celsius
    air.attrs = airtemps.air.attrs
    air.attrs["units"] = "deg C"

    ds = xr.tutorial.scatter_example_dataset(seed=42)
    ds


Scatter
-------

Let's plot the ``A`` DataArray as a function of the ``y`` coord

.. jupyter-execute::

    with xr.set_options(display_expand_data=False):
        display(ds.A)

.. jupyter-execute::

    ds.A.plot.scatter(x="y");

Same plot can be displayed using the dataset:

.. jupyter-execute::

    ds.plot.scatter(x="y", y="A");

Now suppose we want to scatter the ``A`` DataArray against the ``B`` DataArray

.. jupyter-execute::

    ds.plot.scatter(x="A", y="B");

The ``hue`` kwarg lets you vary the color by variable value

.. jupyter-execute::

    ds.plot.scatter(x="A", y="B", hue="w");

You can force a legend instead of a colorbar by setting ``add_legend=True, add_colorbar=False``.

.. jupyter-execute::

    ds.plot.scatter(x="A", y="B", hue="w", add_legend=True, add_colorbar=False);

.. jupyter-execute::

    ds.plot.scatter(x="A", y="B", hue="w", add_legend=False, add_colorbar=True);

The ``markersize`` kwarg lets you vary the point's size by variable value.
You can additionally pass ``size_norm`` to control how the variable's values are mapped to point sizes.

.. jupyter-execute::

    ds.plot.scatter(x="A", y="B", hue="y", markersize="z");

The ``z`` kwarg lets you plot the data along the z-axis as well.

.. jupyter-execute::

    ds.plot.scatter(x="A", y="B", z="z", hue="y", markersize="x");

Faceting is also possible

.. jupyter-execute::

    ds.plot.scatter(x="A", y="B", hue="y", markersize="x", row="x", col="w");

And adding the z-axis

.. jupyter-execute::

    ds.plot.scatter(x="A", y="B", z="z", hue="y", markersize="x", row="x", col="w");

For more advanced scatter plots, we recommend converting the relevant data variables
to a pandas DataFrame and using the extensive plotting capabilities of ``seaborn``.

Quiver
------

Visualizing vector fields is supported with quiver plots:

.. jupyter-execute::

    ds.isel(w=1, z=1).plot.quiver(x="x", y="y", u="A", v="B");


where ``u`` and ``v`` denote the x and y direction components of the arrow vectors. Again, faceting is also possible:

.. jupyter-execute::

    ds.plot.quiver(x="x", y="y", u="A", v="B", col="w", row="z", scale=4);

``scale`` is required for faceted quiver plots.
The scale determines the number of data units per arrow length unit, i.e. a smaller scale parameter makes the arrow longer.

Streamplot
----------

Visualizing vector fields is also supported with streamline plots:

.. jupyter-execute::

    ds.isel(w=1, z=1).plot.streamplot(x="x", y="y", u="A", v="B");


where ``u`` and ``v`` denote the x and y direction components of the vectors tangent to the streamlines.
Again, faceting is also possible:

.. jupyter-execute::

    ds.plot.streamplot(x="x", y="y", u="A", v="B", col="w", row="z");
