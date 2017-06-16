from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools
import random

import numpy as np

np.random.seed(10)
_counter = itertools.count()


def randn(shape, frac_nan=None, chunks=None):
    if chunks is None:
        x = np.random.standard_normal(shape)
    else:
        import dask.array as da
        x = da.random.standard_normal(shape, chunks=chunks)

    if frac_nan is not None:
        inds = random.sample(range(x.size), int(x.size * frac_nan))
        x.flat[inds] = np.nan

    return x
