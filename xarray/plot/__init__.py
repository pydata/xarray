"""
Use this module directly:
    import xarray.plot as xplt

Or use the methods on a DataArray or Dataset:
    DataArray.plot._____
    Dataset.plot._____
"""
from .dataarray_plot import (
    contour,
    contourf,
    hist,
    imshow,
    line,
    pcolormesh,
    plot,
    step,
    surface,
)
from .dataset_plot import scatter
from .facetgrid import FacetGrid

__all__ = [
    "plot",
    "line",
    "step",
    "contour",
    "contourf",
    "hist",
    "imshow",
    "pcolormesh",
    "FacetGrid",
    "scatter",
    "surface",
]
