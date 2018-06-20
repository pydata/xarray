from __future__ import absolute_import, division, print_function

import functools

import numpy as np

from . import dtypes
from .pycompat import dask_array_type


try:
    import bottleneck as bn
    _USE_BOTTLENECK = True
except ImportError:
    # use numpy methods instead
    bn = np
    _USE_BOTTLENECK = False

try:
    import dask.array as dask_array
    from . import dask_array_compat
except ImportError:
    dask_array = None
    dask_array_compat = None


class bottleneck_switch(object):
    """ xarray-version of pandas.core.nanops.bottleneck_switch """
    def __call__(self, alt):
        bn_name = alt.__name__

        try:
            bn_func = getattr(bn, bn_name)
        except (AttributeError, NameError):  # pragma: no cover
            bn_func = None

        @functools.wraps(alt)
        def f(values, axis=None, **kwds):
            dtype = kwds.get('dtype', None)
            min_count = kwds.get('min_count', None)

            if (not isinstance(values, dask_array_type) and _USE_BOTTLENECK and
                    not isinstance(axis, tuple) and
                    values.dtype.kind in 'uifc' and
                    values.dtype.isnative and
                    (dtype is None or np.dtype(dtype) == values.dtype) and
                    min_count is None):
                # bottleneck does not take care dtype, min_count
                kwds.pop('dtype', None)
                kwds.pop('min_count', 1)
                result = bn_func(values, axis=axis, **kwds)
            else:
                result = alt(values, axis=axis, **kwds)

            return result

        return f


def _replace_nan(a, val):
    """
    If `a` is of inexact type, make a copy of `a`, replace NaNs with
    the `val` value, and return the copy together with a boolean mask
    marking the locations where NaNs were present. If `a` is not of
    inexact type, do nothing and return `a` together with a mask of None.
    Note that scalars will end up as array scalars, which is important
    for using the result as the value of the out argument in some
    operations.
    Parameters
    ----------
    a : array-like
        Input array.
    val : float
        NaN values are set to val before doing the operation.
    Returns
    -------
    y : ndarray
        If `a` is of inexact type, return a copy of `a` with the NaNs
        replaced by the fill value, otherwise return `a`.
    mask: {bool, None}
        If `a` is of inexact type, return a boolean mask marking locations of
        NaNs, otherwise return None.

    This function is taken from
    https://github.com/numpy/numpy/blob/v1.14.0/numpy/lib/nanfunctions.py
    but slightly modified to take care of dask.array
    """
    if a.dtype == np.object_:
        # object arrays do not support `isnan` (gh-9009), so make a guess
        mask = a != a
    elif issubclass(a.dtype.type, np.inexact):
        mask = np.isnan(a)
    else:
        mask = None

    if mask is not None:
        if isinstance(a, dask_array_type):
            return dask_array.where(mask, val, a), mask
        return np.where(mask, val, a), mask

    return a, mask


def _maybe_null_out(result, axis, mask, min_count=1):
    """
    xarray version of pandas.core.nanops._maybe_null_out
    """
    if hasattr(axis, '__len__'):  # if tuple or list
        raise ValueError('min_count is not available for reduction '
                         'with more than one dimensions.')

    if axis is not None and getattr(result, 'ndim', False):
        null_mask = (mask.shape[axis] - mask.sum(axis) - min_count) < 0
        if np.any(null_mask):
            dtype, fill_value = dtypes.maybe_promote(result.dtype)
            result = result.astype(dtype)
            result[null_mask] = fill_value

    elif getattr(result, 'dtype', None) not in dtypes.NAT_TYPES:
        null_mask = mask.size - mask.sum()
        if null_mask < min_count:
            result = np.nan

    return result


@bottleneck_switch()
def nanmin(a, axis=None, out=None):
    if a.dtype.kind == 'O':
        return _nan_minmax_object('min', dtypes.get_pos_infinity, a, axis)

    if isinstance(a, dask_array_type):
        return dask_array.nanmin(a, axis=axis)
    return np.nanmin(a, axis=axis)


@bottleneck_switch()
def nanmax(a, axis=None, out=None):
    if a.dtype.kind == 'O':
        return _nan_minmax_object('max', dtypes.get_neg_infinity, a, axis)

    if isinstance(a, dask_array_type):
        return dask_array.nanmax(a, axis=axis)
    return np.nanmax(a, axis=axis)


def _nan_argminmax_object(func, get_fill_value, value, axis=None, **kwargs):
    """ In house nanargmin, nanargmax for object arrays. Always return integer
    type """
    from .duck_array_ops import count, fillna

    fill_value = get_fill_value(value.dtype)
    valid_count = count(value, axis=axis)
    value = fillna(value, fill_value)
    data = getattr(np, func)(value, axis=axis, **kwargs)
    # dask seems return non-integer type
    if isinstance(value, dask_array_type):
        data = data.astype(int)

    if (valid_count == 0).any():
        raise ValueError('All-NaN slice encountered')

    return np.array(data, dtype=int)


def _nan_minmax_object(func, get_fill_value, value, axis=None, **kwargs):
    """ In house nanmin and nanmax for object array """
    from .duck_array_ops import count, fillna, where_method

    fill_value = get_fill_value(value.dtype)
    valid_count = count(value, axis=axis)
    filled_value = fillna(value, fill_value)
    data = getattr(np, func)(filled_value, axis=axis, **kwargs)
    if not hasattr(data, 'dtype'):  # scalar case
        data = dtypes.fill_value(value.dtype) if valid_count == 0 else data
        return np.array(data, dtype=value.dtype)
    return where_method(data, valid_count != 0)


@bottleneck_switch()
def nanargmin(a, axis=None):
    if a.dtype.kind == 'O':
        return _nan_argminmax_object('argmin', dtypes.get_pos_infinity,
                                     a, axis=axis)
    a, mask = _replace_nan(a, np.inf)
    res = np.argmin(a, axis=axis)
    if mask is not None:
        mask = np.all(mask, axis=axis)
        if np.any(mask):
            raise ValueError("All-NaN slice encountered")
    return res


@bottleneck_switch()
def nanargmax(a, axis=None):
    """
    taken from
    https://github.com/numpy/numpy/blob/v1.14.0/numpy/lib/nanfunctions.py
    """
    if a.dtype.kind == 'O':
        return _nan_argminmax_object('argmax', dtypes.get_neg_infinity,
                                     a, axis=axis)
    a, mask = _replace_nan(a, -np.inf)
    res = np.argmax(a, axis=axis)
    if mask is not None:
        mask = np.all(mask, axis=axis)
        if np.any(mask):
            raise ValueError("All-NaN slice encountered")
    return res


@bottleneck_switch()
def nansum(a, axis=None, dtype=None, out=None, min_count=None):
    a, mask = _replace_nan(a, 0)
    result = np.sum(a, axis=axis, dtype=dtype)
    if min_count is not None:
        return _maybe_null_out(result, axis, mask, min_count)
    else:
        return result


def _nanmean_ddof_object(ddof, value, axis=None, **kwargs):
    """ In house nanmean. ddof argument will be used in _nanvar method """
    from .duck_array_ops import (count, fillna, _dask_or_eager_func,
                                 where_method)

    valid_count = count(value, axis=axis)
    value = fillna(value, 0)
    # As dtype inference is impossible for object dtype, we assume float
    # https://github.com/dask/dask/issues/3162
    dtype = kwargs.pop('dtype', None)
    if dtype is None and value.dtype.kind == 'O':
        dtype = value.dtype if value.dtype.kind in ['cf'] else float

    data = _dask_or_eager_func('sum')(value, axis=axis, dtype=dtype, **kwargs)
    data = data / (valid_count - ddof)
    return where_method(data, valid_count != 0)


@bottleneck_switch()
def nanmean(a, axis=None, dtype=None, out=None):
    if a.dtype.kind == 'O':
        return _nanmean_ddof_object(0, a, axis=axis, dtype=dtype)

    if isinstance(a, dask_array_type):
        return dask_array.nanmean(a, axis=axis, dtype=dtype)

    return np.nanmean(a, axis=axis, dtype=dtype)


def _nanvar_object(value, axis=None, **kwargs):
    ddof = kwargs.pop('ddof', 0)
    kwargs_mean = kwargs.copy()
    kwargs_mean.pop('keepdims', None)
    value_mean = _nanmean_ddof_object(ddof=0, value=value, axis=axis,
                                      keepdims=True, **kwargs_mean)
    squared = (value.astype(value_mean.dtype) - value_mean)**2
    return _nanmean_ddof_object(ddof, squared, axis=axis, **kwargs)


@bottleneck_switch()
def nanvar(a, axis=None, dtype=None, out=None, ddof=0):
    if a.dtype.kind == 'O':
        return _nanvar_object(a, axis=axis, dtype=dtype, ddof=ddof)

    if isinstance(a, dask_array_type):
        return dask_array.nanvar(a, axis=axis, dtype=dtype, ddof=ddof)

    return np.nanvar(a, axis=axis, dtype=dtype, ddof=ddof)


def nanprod(a, axis=None, dtype=None, out=None, min_count=None):
    a, mask = _replace_nan(a, 1)
    result = np.prod(a, axis=axis, dtype=dtype, out=out)
    if min_count is not None:
        return _maybe_null_out(result, axis, mask, min_count)
    else:
        return result
