from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
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


def rolling_window(a, axis, window, center, fill_value):
    """ Dask's equivalence to np.utils.rolling_window """
    orig_shape = a.shape
    # inputs for ghost
    if axis < 0:
        axis = a.ndim + axis
    depth = {d: 0 for d in range(a.ndim)}
    depth[axis] = int(window / 2)

    offset = 1 if window % 2 == 0 else 0

    # pad the original array before the operation in order to avoid copying
    # the output array (output array is just a view).
    if center:
        start = int(window / 2)  # 10 -> 5,  9 -> 4
        end = window - 1 - start
    else:
        start, end = window - 1, 0

    drop_size = depth[axis] - offset - np.maximum(start, end)
    if drop_size < 0:
        # ghosting requires each chunk should be larger than depth.
        if -drop_size < depth[axis]:
            pad_size = depth[axis]
            drop_size = depth[axis] + drop_size
        else:
            pad_size = -drop_size
            drop_size = 0
        shape = list(a.shape)
        shape[axis] = pad_size
        chunks = list(a.chunks)
        chunks[axis] = (pad_size, )
        fill_array = da.full(shape, fill_value, dtype=a.dtype, chunks=chunks)
        a = da.concatenate([fill_array, a], axis=axis)

    if depth[axis] > min(a.chunks[axis]):
        raise ValueError(
            "For window size %d, every chunk should be larger than %d, "
            "but the smallest chunk size is %d. Rechunk your array\n"
            "with a larger chunk size or a chunk size that\n"
            "more evenly divides the shape of your array." %
            (window, depth[axis], min(a.chunks[axis])))

    # We temporary use `reflect` boundary here, but the edge portion is
    # truncated later.
    boundary = {d: fill_value for d in range(a.ndim)}

    # create ghosted arrays
    ag = da.ghost.ghost(a, depth=depth, boundary=boundary)

    # apply rolling func
    def func(x, window, axis=-1):
        x = np.asarray(x)
        rolling = nputils._rolling_window(x, window, axis)
        return rolling[(slice(None), ) * axis + (slice(offset, None), )]

    chunks = list(a.chunks)
    chunks.append(window)
    out = ag.map_blocks(func, dtype=a.dtype, new_axis=a.ndim, chunks=chunks,
                        window=window, axis=axis)

    # crop the edge points
    index = (slice(None),) * axis + (slice(drop_size,
                                           drop_size + orig_shape[axis]), )
    return out[index]
