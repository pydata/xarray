import numpy as np

from . import dtypes, nputils, utils
from .duck_array_ops import _dask_or_eager_func, count, fillna, isnull, where_method
from .pycompat import dask_array_type

try:
    import dask.array as dask_array
    from . import dask_array_compat
except ImportError:
    dask_array = None
    dask_array_compat = None  # type: ignore


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
    if hasattr(axis, "__len__"):  # if tuple or list
        raise ValueError(
            "min_count is not available for reduction with more than one dimensions."
        )

    if axis is not None and getattr(result, "ndim", False):
        null_mask = (mask.shape[axis] - mask.sum(axis) - min_count) < 0
        if null_mask.any():
            dtype, fill_value = dtypes.maybe_promote(result.dtype)
            result = result.astype(dtype)
            result[null_mask] = fill_value

    elif getattr(result, "dtype", None) not in dtypes.NAT_TYPES:
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
        raise ValueError("All-NaN slice encountered")

    return data


def _nan_minmax_object(func, fill_value, value, axis=None, **kwargs):
    """ In house nanmin and nanmax for object array """
    valid_count = count(value, axis=axis)
    filled_value = fillna(value, fill_value)
    data = getattr(np, func)(filled_value, axis=axis, **kwargs)
    if not hasattr(data, "dtype"):  # scalar case
        data = fill_value if valid_count == 0 else data
        # we've computed a single min, max value of type object.
        # don't let np.array turn a tuple back into an array
        return utils.to_0d_object_array(data)
    return where_method(data, valid_count != 0)


def nanmin(a, axis=None, out=None):
    if a.dtype.kind == "O":
        return _nan_minmax_object("min", dtypes.get_pos_infinity(a.dtype), a, axis)

    module = dask_array if isinstance(a, dask_array_type) else nputils
    return module.nanmin(a, axis=axis)


def nanmax(a, axis=None, out=None):
    if a.dtype.kind == "O":
        return _nan_minmax_object("max", dtypes.get_neg_infinity(a.dtype), a, axis)

    module = dask_array if isinstance(a, dask_array_type) else nputils
    return module.nanmax(a, axis=axis)


def nanargmin(a, axis=None):
    if a.dtype.kind == "O":
        fill_value = dtypes.get_pos_infinity(a.dtype)
        return _nan_argminmax_object("argmin", fill_value, a, axis=axis)

    module = dask_array if isinstance(a, dask_array_type) else nputils
    return module.nanargmin(a, axis=axis)


def nanargmax(a, axis=None):
    if a.dtype.kind == "O":
        fill_value = dtypes.get_neg_infinity(a.dtype)
        return _nan_argminmax_object("argmax", fill_value, a, axis=axis)

    module = dask_array if isinstance(a, dask_array_type) else nputils
    return module.nanargmax(a, axis=axis)


def nansum(a, axis=None, dtype=None, out=None, min_count=None):
    a, mask = _replace_nan(a, 0)
    result = _dask_or_eager_func("sum")(a, axis=axis, dtype=dtype)
    if min_count is not None:
        return _maybe_null_out(result, axis, mask, min_count)
    else:
        return result


def _nanmean_ddof_object(ddof, value, axis=None, dtype=None, **kwargs):
    """ In house nanmean. ddof argument will be used in _nanvar method """
    from .duck_array_ops import count, fillna, _dask_or_eager_func, where_method

    valid_count = count(value, axis=axis)
    value = fillna(value, 0)
    # As dtype inference is impossible for object dtype, we assume float
    # https://github.com/dask/dask/issues/3162
    if dtype is None and value.dtype.kind == "O":
        dtype = value.dtype if value.dtype.kind in ["cf"] else float

    data = _dask_or_eager_func("sum")(value, axis=axis, dtype=dtype, **kwargs)
    data = data / (valid_count - ddof)
    return where_method(data, valid_count != 0)


def nanmean(a, axis=None, dtype=None, out=None):
    if a.dtype.kind == "O":
        return _nanmean_ddof_object(0, a, axis=axis, dtype=dtype)

    if isinstance(a, dask_array_type):
        return dask_array.nanmean(a, axis=axis, dtype=dtype)

    return np.nanmean(a, axis=axis, dtype=dtype)


def nanmedian(a, axis=None, out=None):
    # The dask algorithm works by rechunking to one chunk along axis
    # Make sure we trigger the dask error when passing all dimensions
    # so that we don't rechunk the entire array to one chunk and
    # possibly blow memory
    if axis is not None and len(np.atleast_1d(axis)) == a.ndim:
        axis = None
    return _dask_or_eager_func(
        "nanmedian", dask_module=dask_array_compat, eager_module=nputils
    )(a, axis=axis)


def _nanvar_object(value, axis=None, ddof=0, keepdims=False, **kwargs):
    value_mean = _nanmean_ddof_object(
        ddof=0, value=value, axis=axis, keepdims=True, **kwargs
    )
    squared = (value.astype(value_mean.dtype) - value_mean) ** 2
    return _nanmean_ddof_object(ddof, squared, axis=axis, keepdims=keepdims, **kwargs)


def nanvar(a, axis=None, dtype=None, out=None, ddof=0):
    if a.dtype.kind == "O":
        return _nanvar_object(a, axis=axis, dtype=dtype, ddof=ddof)

    return _dask_or_eager_func("nanvar", eager_module=nputils)(
        a, axis=axis, dtype=dtype, ddof=ddof
    )


def nanstd(a, axis=None, dtype=None, out=None, ddof=0):
    return _dask_or_eager_func("nanstd", eager_module=nputils)(
        a, axis=axis, dtype=dtype, ddof=ddof
    )


def nanprod(a, axis=None, dtype=None, out=None, min_count=None):
    a, mask = _replace_nan(a, 1)
    result = _dask_or_eager_func("nanprod")(a, axis=axis, dtype=dtype, out=out)
    if min_count is not None:
        return _maybe_null_out(result, axis, mask, min_count)
    else:
        return result


def nancumsum(a, axis=None, dtype=None, out=None):
    return _dask_or_eager_func("nancumsum", eager_module=nputils)(
        a, axis=axis, dtype=dtype
    )


def nancumprod(a, axis=None, dtype=None, out=None):
    return _dask_or_eager_func("nancumprod", eager_module=nputils)(
        a, axis=axis, dtype=dtype
    )
