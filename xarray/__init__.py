# flake8: noqa

from . import testing, tutorial, ufuncs
from ._version import get_versions
from .backends.api import (
    load_dataarray, load_dataset, open_dataarray, open_dataset, open_mfdataset,
    save_mfdataset)
from .backends.rasterio_ import open_rasterio
from .backends.zarr import open_zarr
from .coding.cftime_offsets import cftime_range
from .coding.cftimeindex import CFTimeIndex
from .conventions import SerializationWarning, decode_cf
from .core.alignment import align, broadcast, broadcast_arrays
from .core.combine import auto_combine, concat
from .core.common import ALL_DIMS, full_like, ones_like, zeros_like
from .core.computation import apply_ufunc, dot, where
from .core.dataarray import DataArray
from .core.dataset import Dataset
from .core.extensions import (
    register_dataarray_accessor, register_dataset_accessor)
from .core.merge import MergeError, merge
from .core.options import set_options
from .core.variable import Coordinate, IndexVariable, Variable, as_variable
from .util.print_versions import show_versions

__version__ = get_versions()['version']
del get_versions
