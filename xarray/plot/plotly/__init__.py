"""
Plotly Express plotting for xarray.

This module provides interactive plotting capabilities using Plotly Express.
Use as an accessor on DataArray:

    >>> da.plotly.line()

The accessor automatically assigns dimensions to plot slots (x, color,
facet_col, facet_row, animation_frame) based on their order. Override
with explicit assignments or use None to skip a slot.

Examples
--------
>>> import xarray as xr
>>> import numpy as np

>>> da = xr.DataArray(
...     np.random.rand(10, 3, 2),
...     dims=["time", "city", "scenario"],
... )

>>> # Auto-assignment: time→x, city→color, scenario→facet_col
>>> fig = da.plotly.line()

>>> # Explicit assignment
>>> fig = da.plotly.line(x="time", color="scenario", facet_col="city")

>>> # Skip a slot
>>> fig = da.plotly.line(color=None)  # time→x, city→facet_col, scenario→facet_row
"""

from xarray.plot.plotly.accessor import DataArrayPlotlyAccessor
from xarray.plot.plotly.common import auto

__all__ = [
    "DataArrayPlotlyAccessor",
    "auto",
]
