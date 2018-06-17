from __future__ import absolute_import, division, print_function

import contextlib
import inspect
import warnings
import functools

import numpy as np
import pandas as pd
from pandas.core.nanops import disallow

from . import dask_array_ops, dtypes, npcompat, nputils
from .nputils import nanfirst, nanlast
from .pycompat import dask_array_type


try:
    import bottleneck as bn
    _USE_BOTTLENECK = True
except ImportError:
    # use numpy methods instead
    bn = np
    _USE_BOTTLENECK = False


def _bn_ok_dtype(dt, name):
    # This function is taken from pandas.core.nanops
    # Bottleneck chokes on datetime64
    if (not is_object_dtype(dt) and not is_datetime_or_timedelta_dtype(dt)):

        # GH 15507
        # bottleneck does not properly upcast during the sum
        # so can overflow

        # GH 9422
        # further we also want to preserve NaN when all elements
        # are NaN, unlinke bottleneck/numpy which consider this
        # to be 0
        if name in ['nansum', 'nanprod']:
            return False

        return True
    return False


class bottleneck_switch(object):
    # This function is taken from pandas.core.nanops

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, alt):
        bn_name = alt.__name__

        try:
            bn_func = getattr(bn, bn_name)
        except (AttributeError, NameError):  # pragma: no cover
            bn_func = None

        @functools.wraps(alt)
        def f(values, axis=None, **kwds):
            if len(self.kwargs) > 0:
                for k, v in compat.iteritems(self.kwargs):
                    if k not in kwds:
                        kwds[k] = v
            try:
                if values.size == 0 and kwds.get('min_count') is None:
                    # We are empty, returning NA for our type
                    # Only applies for the default `min_count` of None
                    # since that affects how empty arrays are handled.
                    # TODO(GH-18976) update all the nanops methods to
                    # correctly handle empty inputs and remove this check.
                    # It *may* just be `var`
                    return _na_for_min_count(values, axis)

                if (_USE_BOTTLENECK and not isinstance(value, dask_array_type)
                        and _bn_ok_dtype(values.dtype, bn_name)):
                    result = bn_func(values, axis=axis, **kwds)

                    # prefer to treat inf/-inf as NA, but must compute the func
                    # twice :(
                    if _has_infs(result):
                        result = alt(values, axis=axis, **kwds)
                else:
                    result = alt(values, axis=axis, **kwds)
            except Exception:
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
    """
    a = np.array(a, subok=True, copy=True)

    if a.dtype == np.object_:
        # object arrays do not support `isnan` (gh-9009), so make a guess
        mask = a != a
    elif issubclass(a.dtype.type, np.inexact):
        mask = np.isnan(a)
    else:
        mask = None

    if mask is not None:
        np.copyto(a, val, where=mask)

    return a, mask


def _copyto(a, val, mask):
    """
    Replace values in `a` with NaN where `mask` is True.  This differs from
    copyto in that it will deal with the case where `a` is a numpy scalar.
    Parameters
    ----------
    a : ndarray or numpy scalar
        Array or numpy scalar some of whose values are to be replaced
        by val.
    val : numpy scalar
        Value used a replacement.
    mask : ndarray, scalar
        Boolean array. Where True the corresponding element of `a` is
        replaced by `val`. Broadcasts.
    Returns
    -------
    res : ndarray, scalar
        Array with elements replaced or scalar `val`.

    This function is taken from
    https://github.com/numpy/numpy/blob/v1.14.0/numpy/lib/nanfunctions.py
    """
    if isinstance(a, np.ndarray):
        np.copyto(a, val, where=mask, casting='unsafe')
    else:
        a = a.dtype.type(val)
    return a


def _divide_by_count(a, b, out=None):
    """
    Compute a/b ignoring invalid results. If `a` is an array the division
    is done in place. If `a` is a scalar, then its type is preserved in the
    output. If out is None, then then a is used instead so that the
    division is in place. Note that this is only called with `a` an inexact
    type.
    Parameters
    ----------
    a : {ndarray, numpy scalar}
        Numerator. Expected to be of inexact type but not checked.
    b : {ndarray, numpy scalar}
        Denominator.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.
    Returns
    -------
    ret : {ndarray, numpy scalar}
        The return value is a/b. If `a` was an ndarray the division is done
        in place. If `a` is a numpy scalar, the division preserves its type.

    This function is taken from
    https://github.com/numpy/numpy/blob/v1.14.0/numpy/lib/nanfunctions.py
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        if isinstance(a, np.ndarray):
            if out is None:
                return np.divide(a, b, out=a, casting='unsafe')
            else:
                return np.divide(a, b, out=out, casting='unsafe')
        else:
            if out is None:
                return a.dtype.type(a / b)
            else:
                # This is questionable, but currently a numpy scalar can
                # be output to a zero dimensional array.
                return np.divide(a, b, out=out, casting='unsafe')


@bottleneck_switch()
def nanmin(a, axis=None, out=None, keepdims=np._NoValue):
    """
    taken from
    https://github.com/numpy/numpy/blob/v1.14.0/numpy/lib/nanfunctions.py
    """
    if a.dtype.kind == 'O':
        return _nan_minmax_object('min', dtypes.get_pos_infinity, a, axis)
    kwargs = {}
    if keepdims is not np._NoValue:
        kwargs['keepdims'] = keepdims
    if type(a) is np.ndarray and a.dtype != np.object_:
        # Fast, but not safe for subclasses of ndarray, or object arrays,
        # which do not implement isnan (gh-9009), or fmin correctly (gh-8975)
        res = np.fmin.reduce(a, axis=axis, out=out, **kwargs)
        if np.isnan(res).any():
            warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=2)
    else:
        # Slow, but safe for subclasses of ndarray
        a, mask = _replace_nan(a, +np.inf)
        res = np.amin(a, axis=axis, out=out, **kwargs)
        if mask is None:
            return res

        # Check for all-NaN axis
        mask = np.all(mask, axis=axis, **kwargs)
        if np.any(mask):
            res = _copyto(res, np.nan, mask)
            warnings.warn("All-NaN axis encountered", RuntimeWarning, stacklevel=2)
    return res


@bottleneck_switch()
def nanmax(a, axis=None, out=None, keepdims=np._NoValue):
    """
    taken from
    https://github.com/numpy/numpy/blob/v1.14.0/numpy/lib/nanfunctions.py
    """
    if a.dtype.kind == 'O':
        return _nan_minmax_object('max', dtypes.get_neg_infinity, a, axis)
    kwargs = {}
    if keepdims is not np._NoValue:
        kwargs['keepdims'] = keepdims
    if type(a) is np.ndarray and a.dtype != np.object_:
        # Fast, but not safe for subclasses of ndarray, or object arrays,
        # which do not implement isnan (gh-9009), or fmax correctly (gh-8975)
        res = np.fmax.reduce(a, axis=axis, out=out, **kwargs)
        if np.isnan(res).any():
            warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=2)
    else:
        # Slow, but safe for subclasses of ndarray
        a, mask = _replace_nan(a, -np.inf)
        res = np.amax(a, axis=axis, out=out, **kwargs)
        if mask is None:
            return res

        # Check for all-NaN axis
        mask = np.all(mask, axis=axis, **kwargs)
        if np.any(mask):
            res = _copyto(res, np.nan, mask)
            warnings.warn("All-NaN axis encountered", RuntimeWarning, stacklevel=2)
    return res


def _nan_argminmax_object(func, get_fill_value, value, axis=None, **kwargs):
    """ In house nanargmin, nanargmax for object arrays. Always return integer
    type """
    from .duck_array_ops import isnull, count, fillna

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
    from .duck_array_ops import isnull, count, fillna, where_method

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
def nansum(a, axis=None, dtype=None, out=None, keepdims=np._NoValue):
    a, mask = _replace_nan(a, 0)
    return np.sum(a, axis=axis, dtype=dtype, keepdims=keepdims)


@bottleneck_switch()
def nanprod(a, axis=None, dtype=None, out=None, keepdims=np._NoValue):
    a, mask = _replace_nan(a, 1)
    return np.prod(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)


@bottleneck_switch()
def nancumsum(a, axis=None, dtype=None, out=None):
    a, mask = _replace_nan(a, 0)
    return np.cumsum(a, axis=axis, dtype=dtype, out=out)


@bottleneck_switch()
def nancumprod(a, axis=None, dtype=None, out=None):
    a, mask = _replace_nan(a, 1)
    return np.cumprod(a, axis=axis, dtype=dtype, out=out)


@bottleneck_switch()
def nanmean(a, axis=None, dtype=None, out=None, keepdims=np._NoValue):
    arr, mask = _replace_nan(a, 0)
    if mask is None:
        return np.mean(arr, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    if dtype is not None:
        dtype = np.dtype(dtype)
    if dtype is not None and not issubclass(dtype.type, np.inexact):
        raise TypeError("If a is inexact, then dtype must be inexact")
    if out is not None and not issubclass(out.dtype.type, np.inexact):
        raise TypeError("If a is inexact, then out must be inexact")

    cnt = np.sum(~mask, axis=axis, dtype=np.intp, keepdims=keepdims)
    tot = np.sum(arr, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    avg = _divide_by_count(tot, cnt, out=out)

    isbad = (cnt == 0)
    if isbad.any():
        warnings.warn("Mean of empty slice", RuntimeWarning, stacklevel=2)
        # NaN is the only possible bad value, so no further
        # action is needed to handle bad results.
    return avg


@bottleneck_switch()
def _nanmedian1d(arr1d, overwrite_input=False):
    """
    Private function for rank 1 arrays. Compute the median ignoring NaNs.
    See nanmedian for parameter usage
    """
    arr1d, overwrite_input = _remove_nan_1d(arr1d,
        overwrite_input=overwrite_input)
    if arr1d.size == 0:
        return np.nan

    return np.median(arr1d, overwrite_input=overwrite_input)


@bottleneck_switch()
def _nanmedian(a, axis=None, out=None, overwrite_input=False):
    """
    Private function that doesn't support extended axis or keepdims.
    These methods are extended to this function using _ureduce
    See nanmedian for parameter usage
    """
    if axis is None or a.ndim == 1:
        part = a.ravel()
        if out is None:
            return _nanmedian1d(part, overwrite_input)
        else:
            out[...] = _nanmedian1d(part, overwrite_input)
            return out
    else:
        # for small medians use sort + indexing which is still faster than
        # apply_along_axis
        # benchmarked with shuffled (50, 50, x) containing a few NaN
        if a.shape[axis] < 600:
            return _nanmedian_small(a, axis, out, overwrite_input)
        result = np.apply_along_axis(_nanmedian1d, axis, a, overwrite_input)
        if out is not None:
            out[...] = result
        return result


def _nanmedian_small(a, axis=None, out=None, overwrite_input=False):
    """
    sort + indexing median, faster for small medians along multiple
    dimensions due to the high overhead of apply_along_axis
    see nanmedian for parameter usage
    """
    a = np.ma.masked_array(a, np.isnan(a))
    m = np.ma.median(a, axis=axis, overwrite_input=overwrite_input)
    for i in range(np.count_nonzero(m.mask.ravel())):
        warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=3)
    if out is not None:
        out[...] = m.filled(np.nan)
        return out
    return m.filled(np.nan)


@bottleneck_switch()
def nanmedian(a, axis=None, out=None, overwrite_input=False, keepdims=np._NoValue):
    a = np.asanyarray(a)
    # apply_along_axis in _nanmedian doesn't handle empty arrays well,
    # so deal them upfront
    if a.size == 0:
        return np.nanmean(a, axis, out=out, keepdims=keepdims)

    r, k = _ureduce(a, func=_nanmedian, axis=axis, out=out,
                    overwrite_input=overwrite_input)
    if keepdims and keepdims is not np._NoValue:
        return r.reshape(k)
    else:
        return r


@bottleneck_switch()
def nanpercentile(a, q, axis=None, out=None, overwrite_input=False,
                  interpolation='linear', keepdims=np._NoValue):
    a = np.asanyarray(a)
    q = np.asanyarray(q)
    # apply_along_axis in _nanpercentile doesn't handle empty arrays well,
    # so deal them upfront
    if a.size == 0:
        return np.nanmean(a, axis, out=out, keepdims=keepdims)

    r, k = _ureduce(a, func=_nanpercentile, q=q, axis=axis, out=out,
                    overwrite_input=overwrite_input,
                    interpolation=interpolation)
    if keepdims and keepdims is not np._NoValue:
        return r.reshape(q.shape + k)
    else:
        return r


def _nanpercentile(a, q, axis=None, out=None, overwrite_input=False,
                   interpolation='linear'):
    """
    Private function that doesn't support extended axis or keepdims.
    These methods are extended to this function using _ureduce
    See nanpercentile for parameter usage
    """
    if axis is None or a.ndim == 1:
        part = a.ravel()
        result = _nanpercentile1d(part, q, overwrite_input, interpolation)
    else:
        result = np.apply_along_axis(_nanpercentile1d, axis, a, q,
                                     overwrite_input, interpolation)
        # apply_along_axis fills in collapsed axis with results.
        # Move that axis to the beginning to match percentile's
        # convention.
        if q.ndim != 0:
            result = np.moveaxis(result, axis, 0)

    if out is not None:
        out[...] = result
    return result


def _nanpercentile1d(arr1d, q, overwrite_input=False, interpolation='linear'):
    """
    Private function for rank 1 arrays. Compute percentile ignoring NaNs.
    See nanpercentile for parameter usage
    """
    arr1d, overwrite_input = _remove_nan_1d(arr1d,
        overwrite_input=overwrite_input)
    if arr1d.size == 0:
        return np.full(q.shape, np.nan)[()]  # convert to scalar

    return np.percentile(arr1d, q, overwrite_input=overwrite_input,
                         interpolation=interpolation)


@bottleneck_switch()
def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue):
    arr, mask = _replace_nan(a, 0)
    if mask is None:
        return np.var(arr, axis=axis, dtype=dtype, out=out, ddof=ddof,
                      keepdims=keepdims)

    if dtype is not None:
        dtype = np.dtype(dtype)
    if dtype is not None and not issubclass(dtype.type, np.inexact):
        raise TypeError("If a is inexact, then dtype must be inexact")
    if out is not None and not issubclass(out.dtype.type, np.inexact):
        raise TypeError("If a is inexact, then out must be inexact")

    # Compute mean
    if type(arr) is np.matrix:
        _keepdims = np._NoValue
    else:
        _keepdims = True
    # we need to special case matrix for reverse compatibility
    # in order for this to work, these sums need to be called with
    # keepdims=True, however matrix now raises an error in this case, but
    # the reason that it drops the keepdims kwarg is to force keepdims=True
    # so this used to work by serendipity.
    cnt = np.sum(~mask, axis=axis, dtype=np.intp, keepdims=_keepdims)
    avg = np.sum(arr, axis=axis, dtype=dtype, keepdims=_keepdims)
    avg = _divide_by_count(avg, cnt)

    # Compute squared deviation from mean.
    np.subtract(arr, avg, out=arr, casting='unsafe')
    arr = _copyto(arr, 0, mask)
    if issubclass(arr.dtype.type, np.complexfloating):
        sqr = np.multiply(arr, arr.conj(), out=arr).real
    else:
        sqr = np.multiply(arr, arr, out=arr)

    # Compute variance.
    var = np.sum(sqr, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    if var.ndim < cnt.ndim:
        # Subclasses of ndarray may ignore keepdims, so check here.
        cnt = cnt.squeeze(axis)
    dof = cnt - ddof
    var = _divide_by_count(var, dof)

    isbad = (dof <= 0)
    if np.any(isbad):
        warnings.warn("Degrees of freedom <= 0 for slice.", RuntimeWarning, stacklevel=2)
        # NaN, inf, or negative numbers are all possible bad
        # values, so explicitly replace them with NaN.
        var = _copyto(var, np.nan, isbad)
    return var


@bottleneck_switch()
def nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue):
    var = nanvar(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
                 keepdims=keepdims)
    if isinstance(var, np.ndarray):
        std = np.sqrt(var, out=var)
    else:
        std = var.dtype.type(np.sqrt(var))
    return std
