from . import nputils


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
