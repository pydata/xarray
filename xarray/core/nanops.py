from __future__ import absolute_import, division, print_function

import numpy as np

from . import dtypes
from .pycompat import dask_array_type
from . duck_array_ops import (count, isnull, fillna, where_method,
                              _dask_or_eager_func)
from . import nputils

try:
    import dask.array as dask_array
except ImportError:
    dask_array = None


def _replace_nan(a, val):
    """
    replace nan in a by val, and returns the replaced array and the nan
    position
    """
    mask = isnull(a)
    return where_method(val, mask, a), mask


def _maybe_null_out(result, axis, mask, min_count=1):
    """
    xarray version of pandas.core.nanops._maybe_null_out
    """
    if hasattr(axis, '__len__'):  # if tuple or list
        raise ValueError('min_count is not available for reduction '
                         'with more than one dimensions.')

    if axis is not None and getattr(result, 'ndim', False):
        null_mask = (mask.shape[axis] - mask.sum(axis) - min_count) < 0
        if null_mask.any():
            dtype, fill_value = dtypes.maybe_promote(result.dtype)
            result = result.astype(dtype)
            result[null_mask] = fill_value

    elif getattr(result, 'dtype', None) not in dtypes.NAT_TYPES:
        null_mask = mask.size - mask.sum()
        if null_mask < min_count:
            result = np.nan

    return result


def _nan_argminmax_object(func, fill_value, value, axis=None, **kwargs):
    """ In house nanargmin, nanargmax for object arrays. Always return integer
    type
    """
    valid_count = count(value, axis=axis)
    value = fillna(value, fill_value)
    data = _dask_or_eager_func(func)(value, axis=axis, **kwargs)

    # TODO This will evaluate dask arrays and might be costly.
    if (valid_count == 0).any():
        raise ValueError('All-NaN slice encountered')

    return data


def _nan_minmax_object(func, fill_value, value, axis=None, **kwargs):
    """ In house nanmin and nanmax for object array """
    valid_count = count(value, axis=axis)
    filled_value = fillna(value, fill_value)
    data = getattr(np, func)(filled_value, axis=axis, **kwargs)
    if not hasattr(data, 'dtype'):  # scalar case
        data = dtypes.fill_value(value.dtype) if valid_count == 0 else data
        return np.array(data, dtype=value.dtype)
    return where_method(data, valid_count != 0)


def nanmin(a, axis=None, out=None):
    if a.dtype.kind == 'O':
        return _nan_minmax_object(
            'min', dtypes.get_pos_infinity(a.dtype), a, axis)

    module = dask_array if isinstance(a, dask_array_type) else nputils
    return module.nanmin(a, axis=axis)


def nanmax(a, axis=None, out=None):
    if a.dtype.kind == 'O':
        return _nan_minmax_object(
            'max', dtypes.get_neg_infinity(a.dtype), a, axis)

    module = dask_array if isinstance(a, dask_array_type) else nputils
    return module.nanmax(a, axis=axis)


def nanargmin(a, axis=None):
    fill_value = dtypes.get_pos_infinity(a.dtype)
    if a.dtype.kind == 'O':
        return _nan_argminmax_object('argmin', fill_value, a, axis=axis)
    a, mask = _replace_nan(a, fill_value)
    if isinstance(a, dask_array_type):
        res = dask_array.argmin(a, axis=axis)
    else:
        res = np.argmin(a, axis=axis)

    if mask is not None:
        mask = mask.all(axis=axis)
        if mask.any():
            raise ValueError("All-NaN slice encountered")
    return res


def nanargmax(a, axis=None):
    fill_value = dtypes.get_neg_infinity(a.dtype)
    if a.dtype.kind == 'O':
        return _nan_argminmax_object('argmax', fill_value, a, axis=axis)

    a, mask = _replace_nan(a, fill_value)
    if isinstance(a, dask_array_type):
        res = dask_array.argmax(a, axis=axis)
    else:
        res = np.argmax(a, axis=axis)

    if mask is not None:
        mask = mask.all(axis=axis)
        if mask.any():
            raise ValueError("All-NaN slice encountered")
    return res


def nansum(a, axis=None, dtype=None, out=None, min_count=None):
    a, mask = _replace_nan(a, 0)
    result = _dask_or_eager_func('sum')(a, axis=axis, dtype=dtype)
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


def nanmean(a, axis=None, dtype=None, out=None):
    if a.dtype.kind == 'O':
        return _nanmean_ddof_object(0, a, axis=axis, dtype=dtype)

    if isinstance(a, dask_array_type):
        return dask_array.nanmean(a, axis=axis, dtype=dtype)

    return np.nanmean(a, axis=axis, dtype=dtype)


def nanmedian(a, axis=None, out=None):
    return _dask_or_eager_func('nanmedian', eager_module=nputils)(a, axis=axis)


def _nanvar_object(value, axis=None, **kwargs):
    ddof = kwargs.pop('ddof', 0)
    kwargs_mean = kwargs.copy()
    kwargs_mean.pop('keepdims', None)
    value_mean = _nanmean_ddof_object(ddof=0, value=value, axis=axis,
                                      keepdims=True, **kwargs_mean)
    squared = (value.astype(value_mean.dtype) - value_mean)**2
    return _nanmean_ddof_object(ddof, squared, axis=axis, **kwargs)


def nanvar(a, axis=None, dtype=None, out=None, ddof=0):
    if a.dtype.kind == 'O':
        return _nanvar_object(a, axis=axis, dtype=dtype, ddof=ddof)

    return _dask_or_eager_func('nanvar', eager_module=nputils)(
        a, axis=axis, dtype=dtype, ddof=ddof)


def nanstd(a, axis=None, dtype=None, out=None):
    return _dask_or_eager_func('nanstd', eager_module=nputils)(
        a, axis=axis, dtype=dtype)


def nanprod(a, axis=None, dtype=None, out=None, min_count=None):
    a, mask = _replace_nan(a, 1)
    result = _dask_or_eager_func('nanprod')(a, axis=axis, dtype=dtype, out=out)
    if min_count is not None:
        return _maybe_null_out(result, axis, mask, min_count)
    else:
        return result


def nancumsum(a, axis=None, dtype=None, out=None):
    return _dask_or_eager_func('nancumsum', eager_module=nputils)(
        a, axis=axis, dtype=dtype)


def nancumprod(a, axis=None, dtype=None, out=None):
    return _dask_or_eager_func('nancumprod', eager_module=nputils)(
        a, axis=axis, dtype=dtype)
