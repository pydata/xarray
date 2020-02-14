import numpy as np

integer_types = (int, np.integer)

try:
    # solely for isinstance checks
    import dask.array

    dask_array_type = (dask.array.Array,)
except ImportError:  # pragma: no cover
    dask_array_type = ()

try:
    # solely for isinstance checks
    import sparse

    sparse_array_type = (sparse.SparseArray,)
except ImportError:  # pragma: no cover
    sparse_array_type = ()
