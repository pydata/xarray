from __future__ import absolute_import, division, print_function

import numpy as np
import dask.array as da

try:
    from dask.array import isin
except ImportError:  # pragma: no cover
    # Copied from dask v0.17.3.
    # Used under the terms of Dask's license, see licenses/DASK_LICENSE.

    def _isin_kernel(element, test_elements, assume_unique=False):
        values = np.in1d(element.ravel(), test_elements,
                         assume_unique=assume_unique)
        return values.reshape(element.shape + (1,) * test_elements.ndim)

    def isin(element, test_elements, assume_unique=False, invert=False):
        element = da.asarray(element)
        test_elements = da.asarray(test_elements)
        element_axes = tuple(range(element.ndim))
        test_axes = tuple(i + element.ndim for i in range(test_elements.ndim))
        mapped = da.atop(_isin_kernel, element_axes + test_axes,
                         element, element_axes,
                         test_elements, test_axes,
                         adjust_chunks={axis: lambda _: 1
                                        for axis in test_axes},
                         dtype=bool,
                         assume_unique=assume_unique)
        result = mapped.any(axis=test_axes)
        if invert:
            result = ~result
        return result
