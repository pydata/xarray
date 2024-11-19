from importlib.metadata import version as _version

from xarray import groupers, testing, tutorial, ufuncs
from xarray.backends.api import (
    load_dataarray,
    load_dataset,
    open_dataarray,
    open_dataset,
    open_datatree,
    open_groups,
    open_mfdataset,
    save_mfdataset,
)
from xarray.backends.zarr import open_zarr
from xarray.coding.cftime_offsets import cftime_range, date_range, date_range_like
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.coding.frequencies import infer_freq
from xarray.conventions import SerializationWarning, decode_cf
from xarray.core.alignment import align, broadcast
from xarray.core.combine import combine_by_coords, combine_nested
from xarray.core.common import ALL_DIMS, full_like, ones_like, zeros_like
from xarray.core.computation import (
    apply_ufunc,
    corr,
    cov,
    cross,
    dot,
    polyval,
    unify_chunks,
    where,
)
from xarray.core.concat import concat
from xarray.core.coordinates import Coordinates
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.datatree import DataTree
from xarray.core.datatree_mapping import map_over_datasets
from xarray.core.extensions import (
    register_dataarray_accessor,
    register_dataset_accessor,
    register_datatree_accessor,
)
from xarray.core.indexes import Index
from xarray.core.indexing import IndexSelResult
from xarray.core.merge import Context, MergeError, merge
from xarray.core.options import get_options, set_options
from xarray.core.parallel import map_blocks
from xarray.core.treenode import (
    InvalidTreeError,
    NotFoundInTreeError,
    TreeIsomorphismError,
    group_subtrees,
)
from xarray.core.variable import IndexVariable, Variable, as_variable
from xarray.namedarray.core import NamedArray
from xarray.util.print_versions import show_versions

try:
    __version__ = _version("xarray")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "9999"

# A hardcoded __all__ variable is necessary to appease
# `mypy --strict` running in projects that import xarray.
__all__ = (
    # Sub-packages
    "groupers",
    "testing",
    "tutorial",
    "ufuncs",
    # Top-level functions
    "align",
    "apply_ufunc",
    "as_variable",
    "broadcast",
    "cftime_range",
    "combine_by_coords",
    "combine_nested",
    "concat",
    "corr",
    "cov",
    "cross",
    "date_range",
    "date_range_like",
    "decode_cf",
    "dot",
    "full_like",
    "get_options",
    "group_subtrees",
    "infer_freq",
    "load_dataarray",
    "load_dataset",
    "map_blocks",
    "map_over_datasets",
    "merge",
    "ones_like",
    "open_dataarray",
    "open_dataset",
    "open_datatree",
    "open_groups",
    "open_mfdataset",
    "open_zarr",
    "polyval",
    "register_dataarray_accessor",
    "register_dataset_accessor",
    "register_datatree_accessor",
    "save_mfdataset",
    "set_options",
    "show_versions",
    "unify_chunks",
    "where",
    "zeros_like",
    # Classes
    "CFTimeIndex",
    "Context",
    "Coordinates",
    "DataArray",
    "Dataset",
    "DataTree",
    "Index",
    "IndexSelResult",
    "IndexVariable",
    "NamedArray",
    "Variable",
    # Exceptions
    "InvalidTreeError",
    "MergeError",
    "NotFoundInTreeError",
    "SerializationWarning",
    "TreeIsomorphismError",
    # Constants
    "__version__",
    "ALL_DIMS",
)
