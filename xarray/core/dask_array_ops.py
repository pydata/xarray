import numpy as np

from . import dtypes, nputils


def dask_rolling_wrapper(moving_func, a, window, min_count=None, axis=-1):
    """Wrapper to apply bottleneck moving window funcs on dask arrays"""
    import dask.array as da

    dtype, fill_value = dtypes.maybe_promote(a.dtype)
    a = a.astype(dtype)
    # inputs for overlap
    if axis < 0:
        axis = a.ndim + axis
    depth = {d: 0 for d in range(a.ndim)}
    depth[axis] = (window + 1) // 2
    boundary = {d: fill_value for d in range(a.ndim)}
    # Create overlap array.
    ag = da.overlap.overlap(a, depth=depth, boundary=boundary)
    # apply rolling func
    out = da.map_blocks(
        moving_func, ag, window, min_count=min_count, axis=axis, dtype=a.dtype
    )
    # trim array
    result = da.overlap.trim_internal(out, depth)
    return result


def rolling_window(a, axis, window, center, fill_value):
    """Dask's equivalence to np.utils.rolling_window"""
    import dask.array as da

    if not hasattr(axis, "__len__"):
        axis = [axis]
        window = [window]
        center = [center]

    orig_shape = a.shape
    depth = {d: 0 for d in range(a.ndim)}
    offset = [0] * a.ndim
    drop_size = [0] * a.ndim
    pad_size = [0] * a.ndim
    for ax, win, cent in zip(axis, window, center):
        if ax < 0:
            ax = a.ndim + ax
        depth[ax] = int(win / 2)
        # For evenly sized window, we need to crop the first point of each block.
        offset[ax] = 1 if win % 2 == 0 else 0

        if depth[ax] > min(a.chunks[ax]):
            raise ValueError(
                "For window size %d, every chunk should be larger than %d, "
                "but the smallest chunk size is %d. Rechunk your array\n"
                "with a larger chunk size or a chunk size that\n"
                "more evenly divides the shape of your array."
                % (win, depth[ax], min(a.chunks[ax]))
            )

        # Although da.overlap pads values to boundaries of the array,
        # the size of the generated array is smaller than what we want
        # if center == False.
        if cent:
            start = int(win / 2)  # 10 -> 5,  9 -> 4
            end = win - 1 - start
        else:
            start, end = win - 1, 0
        pad_size[ax] = max(start, end) + offset[ax] - depth[ax]
        drop_size[ax] = 0
        # pad_size becomes more than 0 when the overlapped array is smaller than
        # needed. In this case, we need to enlarge the original array by padding
        # before overlapping.
        if pad_size[ax] > 0:
            if pad_size[ax] < depth[ax]:
                # overlapping requires each chunk larger than depth. If pad_size is
                # smaller than the depth, we enlarge this and truncate it later.
                drop_size[ax] = depth[ax] - pad_size[ax]
                pad_size[ax] = depth[ax]

    # TODO maybe following two lines can be summarized.
    a = da.pad(
        a, [(p, 0) for p in pad_size], mode="constant", constant_values=fill_value
    )
    boundary = {d: fill_value for d in range(a.ndim)}

    # create overlap arrays
    ag = da.overlap.overlap(a, depth=depth, boundary=boundary)

    def func(x, window, axis):
        x = np.asarray(x)
        index = [slice(None)] * x.ndim
        for ax, win in zip(axis, window):
            x = nputils._rolling_window(x, win, ax)
            index[ax] = slice(offset[ax], None)
        return x[tuple(index)]

    chunks = list(a.chunks) + window
    new_axis = [a.ndim + i for i in range(len(axis))]
    out = da.map_blocks(
        func,
        ag,
        dtype=a.dtype,
        new_axis=new_axis,
        chunks=chunks,
        window=window,
        axis=axis,
    )

    # crop boundary.
    index = [slice(None)] * a.ndim
    for ax in axis:
        index[ax] = slice(drop_size[ax], drop_size[ax] + orig_shape[ax])
    return out[tuple(index)]


def least_squares(lhs, rhs, rcond=None, skipna=False):
    import dask.array as da

    lhs_da = da.from_array(lhs, chunks=(rhs.chunks[0], lhs.shape[1]))
    if skipna:
        added_dim = rhs.ndim == 1
        if added_dim:
            rhs = rhs.reshape(rhs.shape[0], 1)
        results = da.apply_along_axis(
            nputils._nanpolyfit_1d,
            0,
            rhs,
            lhs_da,
            dtype=float,
            shape=(lhs.shape[1] + 1,),
            rcond=rcond,
        )
        coeffs = results[:-1, ...]
        residuals = results[-1, ...]
        if added_dim:
            coeffs = coeffs.reshape(coeffs.shape[0])
            residuals = residuals.reshape(residuals.shape[0])
    else:
        # Residuals here are (1, 1) but should be (K,) as rhs is (N, K)
        # See issue dask/dask#6516
        coeffs, residuals, _, _ = da.linalg.lstsq(lhs_da, rhs)
    return coeffs, residuals
