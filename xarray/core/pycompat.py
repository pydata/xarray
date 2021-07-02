from distutils.version import LooseVersion

import numpy as np

from .utils import is_duck_array

integer_types = (int, np.integer)

try:
    import dask
    import dask.array
    from dask.base import is_dask_collection

    dask_version = LooseVersion(dask.__version__)

    # solely for isinstance checks
    dask_array_type = (dask.array.Array,)

    def is_duck_dask_array(x):
        return is_duck_array(x) and is_dask_collection(x)


except ImportError:  # pragma: no cover
    dask_version = LooseVersion("0.0.0")
    dask_array_type = ()
    is_duck_dask_array = lambda _: False
    is_dask_collection = lambda _: False

try:
    # solely for isinstance checks
    import sparse

    sparse_version = LooseVersion(sparse.__version__)
    sparse_array_type = (sparse.SparseArray,)
except ImportError:  # pragma: no cover
    sparse_version = LooseVersion("0.0.0")
    sparse_array_type = ()

try:
    # solely for isinstance checks
    import cupy

    cupy_version = LooseVersion(cupy.__version__)
    cupy_array_type = (cupy.ndarray,)
except ImportError:  # pragma: no cover
    cupy_version = LooseVersion("0.0.0")
    cupy_array_type = ()
