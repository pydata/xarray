from __future__ import annotations

from xarray.core import dtypes, nputils


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


def push(array, n, axis, method="blelloch"):
    """
    Dask-aware bottleneck.push
    """
    import dask.array as da
    import numpy as np

    from xarray.core.duck_array_ops import _push, last

    # TODO: Replace all this function
    #  once https://github.com/pydata/xarray/issues/9229 is implemented

    def _fill_with_last_one(a, b):
        # cumreduction apply the push func over all the blocks first so,
        # the only missing part is filling the missing values using the
        # last data of the previous chunk
        return np.where(np.isnan(b), a, b)

    def _dtype_push(a, axis, dtype=None):
        # Not sure why the blelloch algorithm force to receive a dtype
        return _push(a, axis=axis)

    pushed_array = da.reductions.cumreduction(
        func=_dtype_push,
        binop=_fill_with_last_one,
        ident=np.nan,
        x=array,
        axis=axis,
        dtype=array.dtype,
        method=method,
        preop=last,
    )

    if n is not None and 0 < n < array.shape[axis] - 1:
        def _reset_cumsum(a, axis, dtype=None):
            cumsum = np.cumsum(a, axis=axis)
            reset_points = np.maximum.accumulate(
                np.where(a == 0, cumsum, 0),
                axis=axis
            )
            return cumsum - reset_points

        def _last_reset_cumsum(a, axis, keepdims=None):
            # Take the last cumulative sum taking into account the reset
            # This is useful for blelloch method
            return np.take(_reset_cumsum(a, axis=axis), axis=axis, indices=[-1])

        def _combine_reset_cumsum(a, b):
            # It is going to sum the previous result until the first
            # non nan value
            bitmask = np.cumprod(b != 0, axis=axis)
            return np.where(bitmask, b + a, b)

        valid_positions = da.reductions.cumreduction(
            func=_reset_cumsum,
            binop=_combine_reset_cumsum,
            ident=0,
            x=da.isnan(array, dtype=int),
            axis=axis,
            dtype=int,
            method=method,
            preop=_last_reset_cumsum,
        )
        pushed_array = da.where(
            valid_positions <= n,
            pushed_array,
            np.nan
        )

    return pushed_array
