"""Static type hints for popular external accessors.

Accessors registered with :py:func:`register_dataarray_accessor` and
:py:func:`register_dataset_accessor` only exist at runtime, so type checkers
and IDEs can't know about them. The mixins below declare the accessors of some
widely used packages, which gives autocompletion, parameter hints and
docstrings for them.

The accessor classes are only imported for type checking, so none of these
packages is a dependency of xarray. If a package isn't installed, type checkers
treat its accessor as ``Any``, and at runtime the attribute doesn't exist.

Packages are listed here if they are widely used, maintained and the accessor is
their main API. Their modules need an ``ignore_missing_imports`` entry for mypy
in ``pyproject.toml``. ``xarray/tests/test_accessors.py`` checks that the imports
below resolve when the packages are installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cf_xarray.accessor import CFAccessor
    from hvplot.plotting.core import hvPlot
    from pint_xarray.accessors import PintDataArrayAccessor, PintDatasetAccessor
    from rioxarray.raster_array import RasterArray
    from rioxarray.raster_dataset import RasterDataset
    from xarray_plotly import DataArrayPlotlyAccessor, DatasetPlotlyAccessor

    from xarray.core.extensions import _CachedAccessor


class DataArrayExternalAccessorMixin:
    """Mixin declaring the accessors of external packages for DataArray."""

    __slots__ = ()

    cf: _CachedAccessor[CFAccessor]
    hvplot: _CachedAccessor[hvPlot]
    pint: _CachedAccessor[PintDataArrayAccessor]
    plotly: _CachedAccessor[DataArrayPlotlyAccessor]
    rio: _CachedAccessor[RasterArray]


class DatasetExternalAccessorMixin:
    """Mixin declaring the accessors of external packages for Dataset."""

    __slots__ = ()

    cf: _CachedAccessor[CFAccessor]
    hvplot: _CachedAccessor[hvPlot]
    pint: _CachedAccessor[PintDatasetAccessor]
    plotly: _CachedAccessor[DatasetPlotlyAccessor]
    rio: _CachedAccessor[RasterDataset]
