""" isort:skip_file """
# flake8: noqa

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

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
