# None of these packages is a dependency of xarray
# pyright: reportMissingImports=false
"""Static type hints for popular external accessors.

Accessors registered with :py:func:`register_dataarray_accessor` and
:py:func:`register_dataset_accessor` only exist at runtime, so type checkers
and IDEs can't know about them. The mixins below declare the accessors of some
widely used packages, which gives autocompletion, parameter hints and
docstrings for them.

The accessor classes are only imported for type checking, so none of these
packages is a dependency of xarray. If a package isn't installed, type checkers
treat its accessor as ``Any``, and at runtime the attribute doesn't exist.

Packages are listed here if they are maintained, the accessor is a main part of
their API and they are widely used (roughly more than 10k downloads per month on
PyPI) or the maintainers decide to include them. Their modules need an ``ignore_missing_imports`` entry for mypy
in ``pyproject.toml``. ``xarray/tests/test_accessors.py`` checks that the imports
below resolve when the packages are installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cf_xarray.accessor import CFAccessor
    from hvplot.plotting.core import hvPlot
    from metpy.xarray import MetPyDataArrayAccessor, MetPyDatasetAccessor
    from odc.geo.xr import ODCExtensionDa, ODCExtensionDs
    from pint_xarray.accessors import PintDataArrayAccessor, PintDatasetAccessor
    from rioxarray.raster_array import RasterArray
    from rioxarray.raster_dataset import RasterDataset
    from virtualizarr.accessor import (
        VirtualiZarrDatasetAccessor,
        VirtualiZarrDataTreeAccessor,
    )
    from xarray_plotly import DataArrayPlotlyAccessor, DatasetPlotlyAccessor
    from xrspatial.accessor import (
        XrsSpatialDataArrayAccessor,
        XrsSpatialDatasetAccessor,
    )
    from xvec.accessor import XvecAccessor

    from xarray.core.extensions import _CachedAccessor


class DataArrayExternalAccessorMixin:
    """Mixin declaring the accessors of external packages for DataArray."""

    __slots__ = ()

    cf: _CachedAccessor[CFAccessor]
    hvplot: _CachedAccessor[hvPlot]
    metpy: _CachedAccessor[MetPyDataArrayAccessor]
    odc: _CachedAccessor[ODCExtensionDa]
    pint: _CachedAccessor[PintDataArrayAccessor]
    plotly: _CachedAccessor[DataArrayPlotlyAccessor]
    rio: _CachedAccessor[RasterArray]
    xrs: _CachedAccessor[XrsSpatialDataArrayAccessor]
    xvec: _CachedAccessor[XvecAccessor]


class DatasetExternalAccessorMixin:
    """Mixin declaring the accessors of external packages for Dataset."""

    __slots__ = ()

    cf: _CachedAccessor[CFAccessor]
    hvplot: _CachedAccessor[hvPlot]
    metpy: _CachedAccessor[MetPyDatasetAccessor]
    odc: _CachedAccessor[ODCExtensionDs]
    pint: _CachedAccessor[PintDatasetAccessor]
    plotly: _CachedAccessor[DatasetPlotlyAccessor]
    rio: _CachedAccessor[RasterDataset]
    vz: _CachedAccessor[VirtualiZarrDatasetAccessor]
    xrs: _CachedAccessor[XrsSpatialDatasetAccessor]
    xvec: _CachedAccessor[XvecAccessor]


class DataTreeExternalAccessorMixin:
    """Mixin declaring the accessors of external packages for DataTree."""

    __slots__ = ()

    vz: _CachedAccessor[VirtualiZarrDataTreeAccessor]
