.. currentmodule:: xarray
.. _plotting-plotly:

Interactive Plotting with Plotly
================================

Introduction
------------

Xarray provides interactive plotting capabilities using
`Plotly Express <https://plotly.com/python/plotly-express/>`_ through the
``.plotly`` accessor on DataArrays. This enables creating interactive,
zoomable, and hoverable plots directly from xarray data structures.

Plotly must be installed before using this functionality:

.. code-block:: bash

    pip install plotly

The ``.plotly`` accessor automatically assigns dimensions to plot "slots"
(x-axis, color, facets, animation) based on their order in the DataArray.
You can override these assignments explicitly or skip slots entirely.

Imports and Data
----------------

.. jupyter-execute::

    import numpy as np
    import pandas as pd
    import xarray as xr

For these examples we'll use the North American air temperature dataset.

.. jupyter-execute::

    airtemps = xr.tutorial.open_dataset("air_temperature")

    # Convert to celsius
    air = airtemps.air - 273.15
    air.attrs = airtemps.air.attrs
    air.attrs["units"] = "deg C"
    air

Line Plots
----------

Simple Line Plot
~~~~~~~~~~~~~~~~

The simplest way to create an interactive line plot:

.. jupyter-execute::

    air1d = air.isel(lat=10, lon=10)
    fig = air1d.plotly.line()
    fig

With 2D data, dimensions are automatically assigned to slots.
The default slot order for line plots is:
``x → color → facet_col → facet_row → animation_frame``

.. jupyter-execute::

    air2d = air.isel(lon=10, lat=[10, 15, 20])
    # time → x, lat → color (automatic assignment)
    fig = air2d.plotly.line()
    fig

Explicit Dimension Assignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can explicitly assign dimensions to slots:

.. jupyter-execute::

    # Swap the default assignment
    fig = air2d.plotly.line(x="lat", color="time")
    fig

Skipping Slots
~~~~~~~~~~~~~~

Use ``None`` to skip a slot and let remaining dimensions fill later slots:

.. jupyter-execute::

    air3d = air.isel(lon=10, lat=[10, 15], time=slice(0, 100, 10))
    # Skip color → time goes to x, lat to facet_col, nothing to color
    fig = air3d.plotly.line(color=None)
    fig

Passing Plotly Arguments
~~~~~~~~~~~~~~~~~~~~~~~~

Additional keyword arguments are passed directly to Plotly Express:

.. jupyter-execute::

    fig = air2d.plotly.line(title="Air Temperature Over Time")
    fig

Bar Charts
----------

Bar charts work similarly to line plots:

.. jupyter-execute::

    # Monthly mean temperature
    monthly = air.isel(lat=10, lon=10).resample(time="ME").mean()
    monthly = monthly.isel(time=slice(0, 12))
    fig = monthly.plotly.bar()
    fig

With multiple dimensions:

.. jupyter-execute::

    seasonal = air.isel(lon=10).groupby("time.season").mean()
    seasonal_subset = seasonal.isel(lat=[10, 15, 20])
    fig = seasonal_subset.plotly.bar()
    fig

Area Charts
-----------

Stacked area charts are useful for showing composition:

.. jupyter-execute::

    fig = seasonal_subset.plotly.area()
    fig

Scatter Plots
-------------

By default, scatter plots show DataArray values on the y-axis:

.. jupyter-execute::

    air_sample = air.isel(time=slice(0, 50), lat=10, lon=10)
    fig = air_sample.plotly.scatter()
    fig

With additional dimensions mapped to color:

.. jupyter-execute::

    air_sample2d = air.isel(time=slice(0, 20), lat=[10, 15], lon=10)
    # time → x, values → y, lat → color
    fig = air_sample2d.plotly.scatter()
    fig

Dimension vs Dimension
~~~~~~~~~~~~~~~~~~~~~~

To plot one dimension against another (instead of values on y-axis),
explicitly set ``y`` to a dimension name:

.. jupyter-execute::

    # Plot lat vs lon coordinates, colored by temperature
    air_snapshot = air.isel(time=0)
    fig = air_snapshot.plotly.scatter(x="lon", y="lat", color="value")
    fig

Box Plots
---------

Box plots show the distribution of values:

.. jupyter-execute::

    seasonal_box = air.isel(lon=10, lat=10).groupby("time.season").map(lambda x: x)
    fig = air.isel(lon=10, lat=10, time=slice(0, 365)).plotly.box(x="time")
    fig

Grouped by a dimension:

.. jupyter-execute::

    air_box = air.isel(lon=10, lat=[10, 15, 20], time=slice(0, 100))
    fig = air_box.plotly.box()
    fig

Heatmaps with imshow
--------------------

For 2D data, ``imshow`` creates heatmap visualizations:

.. jupyter-execute::

    air2d_spatial = air.isel(time=0)
    fig = air2d_spatial.plotly.imshow()
    fig

Controlling Axis Orientation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``x`` and ``y`` parameters control which dimension appears on which axis
by transposing the data:

.. jupyter-execute::

    # Default: first dim → x, second dim → y
    fig = air2d_spatial.plotly.imshow()
    fig

.. jupyter-execute::

    # Swap axes: lon on x, lat on y
    fig = air2d_spatial.plotly.imshow(x="lon", y="lat")
    fig

Faceted Heatmaps
~~~~~~~~~~~~~~~~

Use ``facet_col`` to create small multiples:

.. jupyter-execute::

    air_monthly = air.resample(time="ME").mean().isel(time=slice(0, 4))
    fig = air_monthly.plotly.imshow(x="lon", y="lat", facet_col="time")
    fig

Animated Heatmaps
~~~~~~~~~~~~~~~~~

Use ``animation_frame`` to create animated plots:

.. jupyter-execute::

    air_anim = air.isel(time=slice(0, 20, 2))
    fig = air_anim.plotly.imshow(x="lon", y="lat", animation_frame="time")
    fig

Customizing Plots
-----------------

All methods return a ``plotly.graph_objects.Figure`` that can be further
customized using Plotly's API:

.. jupyter-execute::

    fig = air2d.plotly.line()
    fig.update_layout(
        title="Customized Air Temperature Plot",
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        template="plotly_white"
    )
    fig

Dimension-to-Slot Assignment
----------------------------

The ``.plotly`` accessor uses automatic dimension-to-slot assignment.
Dimensions fill slots in order based on the plot type:

.. list-table:: Slot Orders by Plot Type
   :header-rows: 1

   * - Method
     - Slot Order
   * - ``line``
     - x → color → line_dash → symbol → facet_col → facet_row → animation_frame
   * - ``bar``
     - x → color → pattern_shape → facet_col → facet_row → animation_frame
   * - ``area``
     - x → color → pattern_shape → facet_col → facet_row → animation_frame
   * - ``scatter``
     - x → color → size → symbol → facet_col → facet_row → animation_frame (y="value" default)
   * - ``box``
     - x → color → facet_col → facet_row → animation_frame
   * - ``imshow``
     - x → y → facet_col → animation_frame

Assignment Rules
~~~~~~~~~~~~~~~~

1. **Explicit assignments** lock a dimension to a slot
2. **None** skips a slot entirely
3. **Remaining dimensions** fill remaining slots by position
4. **Error** if dimensions remain after all slots are filled

Examples:

.. code-block:: python

    # 3D DataArray with dims: ["time", "lat", "lon"]

    da.plotly.line()
    # → time→x, lat→color, lon→facet_col

    da.plotly.line(color="lon")
    # → time→x, lon→color, lat→facet_col

    da.plotly.line(color=None)
    # → time→x, lat→facet_col, lon→facet_row

    da.plotly.line(x="lat", color="time")
    # → lat→x, time→color, lon→facet_col

Handling Too Many Dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your DataArray has more dimensions than available slots, you'll get an error:

.. code-block:: python

    # 6D array with only 5 slots available
    da_6d.plotly.line()
    # ValueError: Unassigned dimension(s): ['f'].
    # Reduce with .sel(), .isel(), or .mean() before plotting.

Solution: reduce dimensions before plotting:

.. code-block:: python

    da_6d.mean("f").plotly.line()  # Average over dimension 'f'
    da_6d.isel(f=0).plotly.line()  # Select first index of 'f'
