# flake8: noqa
import sys
import typing

import numpy as np

integer_types = (int, np.integer, )

try:
    # solely for isinstance checks
    import dask.array
    dask_array_type = (dask.array.Array,)
except ImportError:  # pragma: no cover
    dask_array_type = ()

# Ensure we have some more recent additions to the typing module.
# Note that TYPE_CHECKING itself is not available on Python 3.5.1.
TYPE_CHECKING = sys.version >= '3.5.3' and typing.TYPE_CHECKING
