""" isort:skip_file """

import pkg_resources

try:
    __version__ = pkg_resources.get_distribution("xarray").version
except Exception:
    # Local copy, not installed with setuptools
    __version__ = "unknown"

from .core.alignment import align, broadcast
from .core.common import full_like, zeros_like, ones_like
from .core.concat import concat
from .core.combine import combine_by_coords, combine_nested, auto_combine
from .core.computation import apply_ufunc, dot, where
from .core.extensions import register_dataarray_accessor, register_dataset_accessor
from .core.variable import as_variable, Variable, IndexVariable, Coordinate
from .core.dataset import Dataset
from .core.dataarray import DataArray
from .core.merge import merge, MergeError
from .core.options import set_options
from .core.parallel import map_blocks

from .backends.api import (
    open_dataset,
    open_dataarray,
    open_mfdataset,
    save_mfdataset,
    load_dataset,
    load_dataarray,
)
from .backends.rasterio_ import open_rasterio
from .backends.zarr import open_zarr

from .conventions import decode_cf, SerializationWarning

from .coding.cftime_offsets import cftime_range
from .coding.cftimeindex import CFTimeIndex

from .util.print_versions import show_versions

from . import tutorial
from . import ufuncs
from . import testing

from .core.common import ALL_DIMS

# A hardcoded __all__ variable is necessary to appease
# `mypy --strict` running in projects that import xarray.
__all__ = (
    # Sub-packages
    "ufuncs",
    "testing",
    "tutorial",
    # Top-level functions
    "align",
    "apply_ufunc",
    "as_variable",
    "auto_combine",
    "broadcast",
    "cftime_range",
    "combine_by_coords",
    "combine_nested",
    "concat",
    "decode_cf",
    "dot",
    "full_like",
    "load_dataarray",
    "load_dataset",
    "map_blocks",
    "merge",
    "ones_like",
    "open_dataarray",
    "open_dataset",
    "open_mfdataset",
    "open_rasterio",
    "open_zarr",
    "register_dataarray_accessor",
    "register_dataset_accessor",
    "save_mfdataset",
    "set_options",
    "show_versions",
    "where",
    "zeros_like",
    # Classes
    "CFTimeIndex",
    "Coordinate",
    "DataArray",
    "Dataset",
    "IndexVariable",
    "Variable",
    # Exceptions
    "MergeError",
    "SerializationWarning",
    # Constants
    "__version__",
    "ALL_DIMS",
)
