"""Define core operations for xarray objects.
"""
import numpy as np
from . import nputils

try:
    import dask.array as da
except ImportError:
    pass


def dask_rolling_wrapper(moving_func, a, window, min_count=None, axis=-1):
    '''wrapper to apply bottleneck moving window funcs on dask arrays'''
    # inputs for ghost
    if axis < 0:
        axis = a.ndim + axis
    depth = {d: 0 for d in range(a.ndim)}
    depth[axis] = window - 1
    boundary = {d: np.nan for d in range(a.ndim)}
    # create ghosted arrays
    ag = da.ghost.ghost(a, depth=depth, boundary=boundary)
    # apply rolling func
    out = ag.map_blocks(moving_func, window, min_count=min_count,
                        axis=axis, dtype=a.dtype)
    # trim array
    result = da.ghost.trim_internal(out, depth)
    return result


def rolling_window(a, window, axis=-1):
    """ Dask's equivalence to np.utils.rolling_window """
    # inputs for ghost
    if axis < 0:
        axis = a.ndim + axis
    depth = {d: 0 for d in range(a.ndim)}
    if window % 2 == 0:
        depth[axis] = int((window - 1) / 2 + 1)
        offset = 1
    else:
        depth[axis] = int((window - 1) / 2)
        offset = 0

    if depth[axis] > min(a.chunks[axis]):
        raise ValueError(
            "The window size %d is larger than your\n"
            "smallest chunk size %d + 1. Rechunk your array\n"
            "with a larger chunk size or a chunk size that\n"
            "more evenly divides the shape of your array." %
            (window, min(a.chunks[axis])))

    boundary = {d: np.nan for d in range(a.ndim)}
    # create ghosted arrays
    ag = da.ghost.ghost(a, depth=depth, boundary=boundary)
    # apply rolling func
    def func(x, window, axis=-1):
        x = np.asarray(x)
        rolling = nputils.rolling_window(x, window, axis)
        return rolling[(slice(None), ) * axis + (slice(offset, None), )]

    chunks = list(a.chunks)
    chunks.append(window)
    out = ag.map_blocks(func, dtype=a.dtype, new_axis=a.ndim, chunks=chunks,
                        window=window, axis=axis)
    # crop the edge points
    index = (slice(None),) * axis + (slice(depth[axis] - offset,
                                           - depth[axis]),)
    return out[index]
