# flake8: noqa
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .core.alignment import align, broadcast, broadcast_arrays
from .core.common import full_like, zeros_like, ones_like
from .core.combine import concat, auto_combine
from .core.computation import apply_ufunc, where
from .core.extensions import (register_dataarray_accessor,
                              register_dataset_accessor)
from .core.variable import as_variable, Variable, IndexVariable, Coordinate
from .core.dataset import Dataset
from .core.dataarray import DataArray
from .core.merge import merge, MergeError
from .core.options import set_options

from .backends.api import (open_dataset, open_dataarray, open_mfdataset,
                           save_mfdataset)
from .backends.rasterio_ import open_rasterio

from .conventions import decode_cf, SerializationWarning

try:
    from .version import version as __version__
except ImportError:  # pragma: no cover
    raise ImportError('xarray not properly installed. If you are running from '
                      'the source directory, please instead create a new '
                      'virtual environment (using conda or virtualenv) and '
                      'then install it in-place by running: pip install -e .')
from .util.print_versions import show_versions

from . import tutorial
from . import ufuncs
from . import testing
