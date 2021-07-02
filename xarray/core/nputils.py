import warnings

import numpy as np
import pandas as pd
from numpy.core.multiarray import normalize_axis_index  # type: ignore[attr-defined]

try:
    import bottleneck as bn

    _USE_BOTTLENECK = True
except ImportError:
    # use numpy methods instead
    bn = np
    _USE_BOTTLENECK = False


def _select_along_axis(values, idx, axis):
    other_ind = np.ix_(*[np.arange(s) for s in idx.shape])
    sl = other_ind[:axis] + (idx,) + other_ind[axis:]
    return values[sl]


def nanfirst(values, axis):
    axis = normalize_axis_index(axis, values.ndim)
    idx_first = np.argmax(~pd.isnull(values), axis=axis)
    return _select_along_axis(values, idx_first, axis)


def nanlast(values, axis):
    axis = normalize_axis_index(axis, values.ndim)
    rev = (slice(None),) * axis + (slice(None, None, -1),)
    idx_last = -1 - np.argmax(~pd.isnull(values)[rev], axis=axis)
    return _select_along_axis(values, idx_last, axis)


def inverse_permutation(indices):
    """Return indices for an inverse permutation.

    Parameters
    ----------
    indices : 1D np.ndarray with dtype=int
        Integer positions to assign elements to.

    Returns
    -------
    inverse_permutation : 1D np.ndarray with dtype=int
        Integer indices to take from the original array to create the
        permutation.
    """
    # use intp instead of int64 because of windows :(
    inverse_permutation = np.empty(len(indices), dtype=np.intp)
    inverse_permutation[indices] = np.arange(len(indices), dtype=np.intp)
    return inverse_permutation


def _ensure_bool_is_ndarray(result, *args):
    # numpy will sometimes return a scalar value from binary comparisons if it
    # can't handle the comparison instead of broadcasting, e.g.,
    # In [10]: 1 == np.array(['a', 'b'])
    # Out[10]: False
    # This function ensures that the result is the appropriate shape in these
    # cases
    if isinstance(result, bool):
        shape = np.broadcast(*args).shape
        constructor = np.ones if result else np.zeros
        result = constructor(shape, dtype=bool)
    return result


def array_eq(self, other):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"elementwise comparison failed")
        return _ensure_bool_is_ndarray(self == other, self, other)


def array_ne(self, other):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"elementwise comparison failed")
        return _ensure_bool_is_ndarray(self != other, self, other)


def _is_contiguous(positions):
    """Given a non-empty list, does it consist of contiguous integers?"""
    previous = positions[0]
    for current in positions[1:]:
        if current != previous + 1:
            return False
        previous = current
    return True


def _advanced_indexer_subspaces(key):
    """Indices of the advanced indexes subspaces for mixed indexing and vindex."""
    if not isinstance(key, tuple):
        key = (key,)
    advanced_index_positions = [
        i for i, k in enumerate(key) if not isinstance(k, slice)
    ]

    if not advanced_index_positions or not _is_contiguous(advanced_index_positions):
        # Nothing to reorder: dimensions on the indexing result are already
        # ordered like vindex. See NumPy's rule for "Combining advanced and
        # basic indexing":
        # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#combining-advanced-and-basic-indexing
        return (), ()

    non_slices = [k for k in key if not isinstance(k, slice)]
    ndim = len(np.broadcast(*non_slices).shape)
    mixed_positions = advanced_index_positions[0] + np.arange(ndim)
    vindex_positions = np.arange(ndim)
    return mixed_positions, vindex_positions


class NumpyVIndexAdapter:
    """Object that implements indexing like vindex on a np.ndarray.

    This is a pure Python implementation of (some of) the logic in this NumPy
    proposal: https://github.com/numpy/numpy/pull/6256
    """

    def __init__(self, array):
        self._array = array

    def __getitem__(self, key):
        mixed_positions, vindex_positions = _advanced_indexer_subspaces(key)
        return np.moveaxis(self._array[key], mixed_positions, vindex_positions)

    def __setitem__(self, key, value):
        """Value must have dimensionality matching the key."""
        mixed_positions, vindex_positions = _advanced_indexer_subspaces(key)
        self._array[key] = np.moveaxis(value, vindex_positions, mixed_positions)


def _create_bottleneck_method(name, npmodule=np):
    def f(values, axis=None, **kwargs):
        dtype = kwargs.get("dtype", None)
        bn_func = getattr(bn, name, None)

        if (
            _USE_BOTTLENECK
            and isinstance(values, np.ndarray)
            and bn_func is not None
            and not isinstance(axis, tuple)
            and values.dtype.kind in "uifc"
            and values.dtype.isnative
            and (dtype is None or np.dtype(dtype) == values.dtype)
        ):
            # bottleneck does not take care dtype, min_count
            kwargs.pop("dtype", None)
            result = bn_func(values, axis=axis, **kwargs)
        else:
            result = getattr(npmodule, name)(values, axis=axis, **kwargs)

        return result

    f.__name__ = name
    return f


def _nanpolyfit_1d(arr, x, rcond=None):
    out = np.full((x.shape[1] + 1,), np.nan)
    mask = np.isnan(arr)
    if not np.all(mask):
        out[:-1], resid, rank, _ = np.linalg.lstsq(x[~mask, :], arr[~mask], rcond=rcond)
        out[-1] = resid if resid.size > 0 else np.nan
        warn_on_deficient_rank(rank, x.shape[1])
    return out


def warn_on_deficient_rank(rank, order):
    if rank != order:
        warnings.warn("Polyfit may be poorly conditioned", np.RankWarning, stacklevel=2)


def least_squares(lhs, rhs, rcond=None, skipna=False):
    if skipna:
        added_dim = rhs.ndim == 1
        if added_dim:
            rhs = rhs.reshape(rhs.shape[0], 1)
        nan_cols = np.any(np.isnan(rhs), axis=0)
        out = np.empty((lhs.shape[1] + 1, rhs.shape[1]))
        if np.any(nan_cols):
            out[:, nan_cols] = np.apply_along_axis(
                _nanpolyfit_1d, 0, rhs[:, nan_cols], lhs
            )
        if np.any(~nan_cols):
            out[:-1, ~nan_cols], resids, rank, _ = np.linalg.lstsq(
                lhs, rhs[:, ~nan_cols], rcond=rcond
            )
            out[-1, ~nan_cols] = resids if resids.size > 0 else np.nan
            warn_on_deficient_rank(rank, lhs.shape[1])
        coeffs = out[:-1, :]
        residuals = out[-1, :]
        if added_dim:
            coeffs = coeffs.reshape(coeffs.shape[0])
            residuals = residuals.reshape(residuals.shape[0])
    else:
        coeffs, residuals, rank, _ = np.linalg.lstsq(lhs, rhs, rcond=rcond)
        if residuals.size == 0:
            residuals = coeffs[0] * np.nan
        warn_on_deficient_rank(rank, lhs.shape[1])
    return coeffs, residuals


nanmin = _create_bottleneck_method("nanmin")
nanmax = _create_bottleneck_method("nanmax")
nanmean = _create_bottleneck_method("nanmean")
nanmedian = _create_bottleneck_method("nanmedian")
nanvar = _create_bottleneck_method("nanvar")
nanstd = _create_bottleneck_method("nanstd")
nanprod = _create_bottleneck_method("nanprod")
nancumsum = _create_bottleneck_method("nancumsum")
nancumprod = _create_bottleneck_method("nancumprod")
nanargmin = _create_bottleneck_method("nanargmin")
nanargmax = _create_bottleneck_method("nanargmax")
