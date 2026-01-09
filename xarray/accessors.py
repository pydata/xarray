"""
External accessor support for xarray.

This module provides mixin classes with typed properties for external accessor
packages, enabling full IDE support (autocompletion, parameter hints, docstrings)
for packages like hvplot, cf-xarray, pint-xarray, rioxarray, and xarray-plotly.

Properties are defined statically for IDE support, but raise AttributeError
for uninstalled packages (making hasattr() return False).
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # External accessor types (for IDE support)
    from cf_xarray.accessor import CFAccessor
    from hvplot.xarray import hvPlotAccessor
    from pint_xarray import PintDataArrayAccessor, PintDatasetAccessor
    from rioxarray import RasterArray, RasterDataset
    from xarray_plotly import DataArrayPlotlyAccessor, DatasetPlotlyAccessor

    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset
    from xarray.core.datatree import DataTree

# Registry of known external accessors
# Format: name -> (module_path, class_name, install_name, top_level_package)
DATAARRAY_ACCESSORS: dict[str, tuple[str, str, str, str]] = {
    "hvplot": ("hvplot.xarray", "hvPlotAccessor", "hvplot", "hvplot"),
    "cf": ("cf_xarray.accessor", "CFAccessor", "cf-xarray", "cf_xarray"),
    "pint": ("pint_xarray", "PintDataArrayAccessor", "pint-xarray", "pint_xarray"),
    "rio": ("rioxarray", "RasterArray", "rioxarray", "rioxarray"),
    "plotly": (
        "xarray_plotly",
        "DataArrayPlotlyAccessor",
        "xarray-plotly",
        "xarray_plotly",
    ),
}

DATASET_ACCESSORS: dict[str, tuple[str, str, str, str]] = {
    "hvplot": ("hvplot.xarray", "hvPlotAccessor", "hvplot", "hvplot"),
    "cf": ("cf_xarray.accessor", "CFAccessor", "cf-xarray", "cf_xarray"),
    "pint": ("pint_xarray", "PintDatasetAccessor", "pint-xarray", "pint_xarray"),
    "rio": ("rioxarray", "RasterDataset", "rioxarray", "rioxarray"),
    "plotly": (
        "xarray_plotly",
        "DatasetPlotlyAccessor",
        "xarray-plotly",
        "xarray_plotly",
    ),
}

DATATREE_ACCESSORS: dict[str, tuple[str, str, str, str]] = {
    "hvplot": ("hvplot.xarray", "hvPlotAccessor", "hvplot", "hvplot"),
    "cf": ("cf_xarray.accessor", "CFAccessor", "cf-xarray", "cf_xarray"),
}

# Cache for package availability checks
_package_available_cache: dict[str, bool] = {}


def _is_package_available(package_name: str) -> bool:
    """Check if a package is available without importing it."""
    if package_name not in _package_available_cache:
        _package_available_cache[package_name] = (
            importlib.util.find_spec(package_name) is not None
        )
    return _package_available_cache[package_name]


def _get_external_accessor(
    name: str,
    obj: DataArray | Dataset | DataTree,
    accessor_registry: dict[str, tuple[str, str, str, str]],
) -> Any:
    """
    Get an external accessor instance, using the object's cache if available.

    Parameters
    ----------
    name : str
        Name of the accessor
    obj : DataArray | Dataset | DataTree
        The xarray object
    accessor_registry : dict
        The accessor registry to use

    Returns
    -------
    Any
        An instance of the accessor class

    Raises
    ------
    AttributeError
        If the accessor package is not installed (makes hasattr return False)
    """
    package, cls_name, install_name, top_pkg = accessor_registry[name]

    # Check if package is available - raise AttributeError if not
    # This makes hasattr() return False for uninstalled packages
    if not _is_package_available(top_pkg):
        raise AttributeError(
            f"'{type(obj).__name__}' object has no attribute '{name}'. "
            f"Install with: pip install {install_name}"
        )

    # Check cache
    try:
        cache = obj._cache
    except AttributeError:
        cache = obj._cache = {}

    cache_key = f"_external_{name}"
    if cache_key in cache:
        return cache[cache_key]

    # Import and instantiate the accessor
    try:
        module = importlib.import_module(package)
    except ImportError as err:
        raise AttributeError(
            f"'{type(obj).__name__}' object has no attribute '{name}'. "
            f"Install with: pip install {install_name}"
        ) from err

    try:
        accessor_cls = getattr(module, cls_name)
        accessor = accessor_cls(obj)
    except AttributeError as err:
        # __getattr__ on data object will swallow any AttributeErrors
        # raised when initializing the accessor, so we need to raise as
        # something else (same pattern as _CachedAccessor in extensions.py)
        raise RuntimeError(f"Error initializing {name!r} accessor.") from err

    cache[cache_key] = accessor
    return accessor


class DataArrayExternalAccessorMixin:
    """
    Mixin class providing typed external accessor properties for DataArray.

    These properties enable IDE support (autocompletion, parameter hints, docstrings)
    for external accessor packages. For uninstalled packages, hasattr() returns False.
    """

    __slots__ = ()

    @property
    def hvplot(self) -> hvPlotAccessor:
        """
        hvPlot accessor for interactive plotting.

        Requires: ``pip install hvplot``

        See Also
        --------
        hvplot : https://hvplot.holoviz.org/
        """
        return _get_external_accessor("hvplot", self, DATAARRAY_ACCESSORS)  # type: ignore[arg-type]

    @property
    def cf(self) -> CFAccessor:
        """
        CF conventions accessor.

        Requires: ``pip install cf-xarray``

        See Also
        --------
        cf_xarray : https://cf-xarray.readthedocs.io/
        """
        return _get_external_accessor("cf", self, DATAARRAY_ACCESSORS)  # type: ignore[arg-type]

    @property
    def pint(self) -> PintDataArrayAccessor:
        """
        Pint unit accessor for unit-aware arrays.

        Requires: ``pip install pint-xarray``

        See Also
        --------
        pint_xarray : https://pint-xarray.readthedocs.io/
        """
        return _get_external_accessor("pint", self, DATAARRAY_ACCESSORS)  # type: ignore[arg-type]

    @property
    def rio(self) -> RasterArray:
        """
        Rasterio accessor for geospatial raster data.

        Requires: ``pip install rioxarray``

        See Also
        --------
        rioxarray : https://corteva.github.io/rioxarray/
        """
        return _get_external_accessor("rio", self, DATAARRAY_ACCESSORS)  # type: ignore[arg-type]

    @property
    def plotly(self) -> DataArrayPlotlyAccessor:
        """
        Plotly accessor for interactive Plotly visualizations.

        Requires: ``pip install xarray-plotly``

        See Also
        --------
        xarray_plotly : https://github.com/xarray-contrib/xarray-plotly
        """
        return _get_external_accessor("plotly", self, DATAARRAY_ACCESSORS)  # type: ignore[arg-type]


class DatasetExternalAccessorMixin:
    """
    Mixin class providing typed external accessor properties for Dataset.

    These properties enable IDE support (autocompletion, parameter hints, docstrings)
    for external accessor packages. For uninstalled packages, hasattr() returns False.
    """

    __slots__ = ()

    @property
    def hvplot(self) -> hvPlotAccessor:
        """
        hvPlot accessor for interactive plotting.

        Requires: ``pip install hvplot``

        See Also
        --------
        hvplot : https://hvplot.holoviz.org/
        """
        return _get_external_accessor("hvplot", self, DATASET_ACCESSORS)  # type: ignore[arg-type]

    @property
    def cf(self) -> CFAccessor:
        """
        CF conventions accessor.

        Requires: ``pip install cf-xarray``

        See Also
        --------
        cf_xarray : https://cf-xarray.readthedocs.io/
        """
        return _get_external_accessor("cf", self, DATASET_ACCESSORS)  # type: ignore[arg-type]

    @property
    def pint(self) -> PintDatasetAccessor:
        """
        Pint unit accessor for unit-aware arrays.

        Requires: ``pip install pint-xarray``

        See Also
        --------
        pint_xarray : https://pint-xarray.readthedocs.io/
        """
        return _get_external_accessor("pint", self, DATASET_ACCESSORS)  # type: ignore[arg-type]

    @property
    def rio(self) -> RasterDataset:
        """
        Rasterio accessor for geospatial raster data.

        Requires: ``pip install rioxarray``

        See Also
        --------
        rioxarray : https://corteva.github.io/rioxarray/
        """
        return _get_external_accessor("rio", self, DATASET_ACCESSORS)  # type: ignore[arg-type]

    @property
    def plotly(self) -> DatasetPlotlyAccessor:
        """
        Plotly accessor for interactive Plotly visualizations.

        Requires: ``pip install xarray-plotly``

        See Also
        --------
        xarray_plotly : https://github.com/xarray-contrib/xarray-plotly
        """
        return _get_external_accessor("plotly", self, DATASET_ACCESSORS)  # type: ignore[arg-type]


class DataTreeExternalAccessorMixin:
    """
    Mixin class providing typed external accessor properties for DataTree.

    These properties enable IDE support (autocompletion, parameter hints, docstrings)
    for external accessor packages. For uninstalled packages, hasattr() returns False.
    """

    __slots__ = ()

    @property
    def hvplot(self) -> hvPlotAccessor:
        """
        hvPlot accessor for interactive plotting.

        Requires: ``pip install hvplot``

        See Also
        --------
        hvplot : https://hvplot.holoviz.org/
        """
        return _get_external_accessor("hvplot", self, DATATREE_ACCESSORS)  # type: ignore[arg-type]

    @property
    def cf(self) -> CFAccessor:
        """
        CF conventions accessor.

        Requires: ``pip install cf-xarray``

        See Also
        --------
        cf_xarray : https://cf-xarray.readthedocs.io/
        """
        return _get_external_accessor("cf", self, DATATREE_ACCESSORS)  # type: ignore[arg-type]
