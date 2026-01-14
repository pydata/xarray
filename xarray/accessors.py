"""
External accessor support for xarray.

This module provides mixin classes with typed properties for external accessor
packages, enabling full IDE support (autocompletion, parameter hints, docstrings)
for packages like hvplot, cf-xarray, pint-xarray, rioxarray, and xarray-plotly.

Properties are defined statically for IDE support
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from cf_xarray.accessor import CFAccessor
    from hvplot.xarray import hvPlotAccessor
    from pint_xarray import PintDataArrayAccessor, PintDatasetAccessor
    from rioxarray import RasterArray, RasterDataset
    from xarray_plotly import DataArrayPlotlyAccessor, DatasetPlotlyAccessor


class DataArrayExternalAccessorMixin:
    """Mixin providing typed external accessor properties for DataArray."""

    __slots__ = ()

    hvplot: ClassVar[type[hvPlotAccessor]]
    cf: ClassVar[type[CFAccessor]]
    pint: ClassVar[type[PintDataArrayAccessor]]
    rio: ClassVar[type[RasterArray]]
    plotly: ClassVar[type[DataArrayPlotlyAccessor]]


class DatasetExternalAccessorMixin:
    """Mixin providing typed external accessor properties for Dataset."""

    __slots__ = ()

    hvplot: ClassVar[type[hvPlotAccessor]]
    cf: ClassVar[type[CFAccessor]]
    pint: ClassVar[type[PintDatasetAccessor]]
    rio: ClassVar[type[RasterDataset]]
    plotly: ClassVar[type[DatasetPlotlyAccessor]]


class DataTreeExternalAccessorMixin:
    """Mixin providing typed external accessor properties for DataTree."""

    __slots__ = ()

    hvplot: ClassVar[type[hvPlotAccessor]]
    cf: ClassVar[type[CFAccessor]]
