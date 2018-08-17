from __future__ import absolute_import, division, print_function
from distutils.version import LooseVersion

import numpy as np

from . import nputils
from . import dtypes

try:
    import dask
    import dask.array as da
    # Note: dask has used `ghost` before 0.18.2
    if LooseVersion(dask.__version__) <= LooseVersion('0.18.2'):
        overlap = da.ghost.ghost
        trim_internal = da.ghost.trim_internal
    else:
        overlap = da.overlap.overlap
        trim_internal = da.overlap.trim_internal
except ImportError:
    pass


def dask_rolling_wrapper(moving_func, a, window, min_count=None, axis=-1):
    '''wrapper to apply bottleneck moving window funcs on dask arrays'''
    dtype, fill_value = dtypes.maybe_promote(a.dtype)
    a = a.astype(dtype)
    # inputs for overlap
    if axis < 0:
        axis = a.ndim + axis
    depth = {d: 0 for d in range(a.ndim)}
    depth[axis] = (window + 1) // 2
    boundary = {d: fill_value for d in range(a.ndim)}
    # Create overlap array.
    ag = overlap(a, depth=depth, boundary=boundary)
    # apply rolling func
    out = ag.map_blocks(moving_func, window, min_count=min_count,
                        axis=axis, dtype=a.dtype)
    # trim array
    result = trim_internal(out, depth)
    return result


def rolling_window(a, axis, window, center, fill_value):
    """ Dask's equivalence to np.utils.rolling_window """
    orig_shape = a.shape
    if axis < 0:
        axis = a.ndim + axis
    depth = {d: 0 for d in range(a.ndim)}
    depth[axis] = int(window / 2)
    # For evenly sized window, we need to crop the first point of each block.
    offset = 1 if window % 2 == 0 else 0

    if depth[axis] > min(a.chunks[axis]):
        raise ValueError(
            "For window size %d, every chunk should be larger than %d, "
            "but the smallest chunk size is %d. Rechunk your array\n"
            "with a larger chunk size or a chunk size that\n"
            "more evenly divides the shape of your array." %
            (window, depth[axis], min(a.chunks[axis])))

    # Although dask.overlap pads values to boundaries of the array,
    # the size of the generated array is smaller than what we want
    # if center == False.
    if center:
        start = int(window / 2)  # 10 -> 5,  9 -> 4
        end = window - 1 - start
    else:
        start, end = window - 1, 0
    pad_size = max(start, end) + offset - depth[axis]
    drop_size = 0
    # pad_size becomes more than 0 when the overlapped array is smaller than
    # needed. In this case, we need to enlarge the original array by padding
    # before overlapping.
    if pad_size > 0:
        if pad_size < depth[axis]:
            # overlapping requires each chunk larger than depth. If pad_size is
            # smaller than the depth, we enlarge this and truncate it later.
            drop_size = depth[axis] - pad_size
            pad_size = depth[axis]
        shape = list(a.shape)
        shape[axis] = pad_size
        chunks = list(a.chunks)
        chunks[axis] = (pad_size, )
        fill_array = da.full(shape, fill_value, dtype=a.dtype, chunks=chunks)
        a = da.concatenate([fill_array, a], axis=axis)

    boundary = {d: fill_value for d in range(a.ndim)}

    # create overlap arrays
    ag = overlap(a, depth=depth, boundary=boundary)

    # apply rolling func
    def func(x, window, axis=-1):
        x = np.asarray(x)
        rolling = nputils._rolling_window(x, window, axis)
        return rolling[(slice(None), ) * axis + (slice(offset, None), )]

    chunks = list(a.chunks)
    chunks.append(window)
    out = ag.map_blocks(func, dtype=a.dtype, new_axis=a.ndim, chunks=chunks,
                        window=window, axis=axis)

    # crop boundary.
    index = (slice(None),) * axis + (slice(drop_size,
                                           drop_size + orig_shape[axis]), )
    return out[index]
