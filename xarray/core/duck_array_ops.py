"""Compatibility module defining operations on duck numpy-arrays.

Currently, this means Dask or NumPy arrays. None of these functions should
accept or return xarray objects.
"""
from __future__ import absolute_import, division, print_function

import contextlib
import inspect
import warnings
from functools import partial

import numpy as np
import pandas as pd

from . import dask_array_ops, dtypes, npcompat, nputils
from .nputils import nanfirst, nanlast
from .pycompat import dask_array_type

try:
    import dask.array as dask_array
    from . import dask_array_compat
except ImportError:
    dask_array = None
    dask_array_compat = None


def _dask_or_eager_func(name, eager_module=np, dask_module=dask_array,
                        list_of_args=False, array_args=slice(1),
                        requires_dask=None):
    """Create a function that dispatches to dask for dask array inputs."""
    if dask_module is not None:
        def f(*args, **kwargs):
            if list_of_args:
                dispatch_args = args[0]
            else:
                dispatch_args = args[array_args]
            if any(isinstance(a, dask_array.Array) for a in dispatch_args):
                try:
                    wrapped = getattr(dask_module, name)
                except AttributeError as e:
                    raise AttributeError("%s: requires dask >=%s" %
                                         (e, requires_dask))
            else:
                wrapped = getattr(eager_module, name)
            return wrapped(*args, ** kwargs)
    else:
        def f(data, *args, **kwargs):
            return getattr(eager_module, name)(data, *args, **kwargs)
    return f


def fail_on_dask_array_input(values, msg=None, func_name=None):
    if isinstance(values, dask_array_type):
        if msg is None:
            msg = '%r is not yet a valid method on dask arrays'
        if func_name is None:
            func_name = inspect.stack()[1][3]
        raise NotImplementedError(msg % func_name)


around = _dask_or_eager_func('around')
isclose = _dask_or_eager_func('isclose')
notnull = _dask_or_eager_func('notnull', eager_module=pd)
_isnull = _dask_or_eager_func('isnull', eager_module=pd)


def isnull(data):
    # GH837, GH861
    # isnull fcn from pandas will throw TypeError when run on numpy structured
    # array therefore for dims that are np structured arrays we assume all
    # data is present
    try:
        return _isnull(data)
    except TypeError:
        return np.zeros(data.shape, dtype=bool)


transpose = _dask_or_eager_func('transpose')
_where = _dask_or_eager_func('where', array_args=slice(3))
isin = _dask_or_eager_func('isin', eager_module=npcompat,
                           dask_module=dask_array_compat, array_args=slice(2))
take = _dask_or_eager_func('take')
broadcast_to = _dask_or_eager_func('broadcast_to')

_concatenate = _dask_or_eager_func('concatenate', list_of_args=True)
_stack = _dask_or_eager_func('stack', list_of_args=True)

array_all = _dask_or_eager_func('all')
array_any = _dask_or_eager_func('any')

tensordot = _dask_or_eager_func('tensordot', array_args=slice(2))
einsum = _dask_or_eager_func('einsum', array_args=slice(1, None),
                             requires_dask='0.17.3')


def gradient(x, coord, axis, edge_order):
    if isinstance(x, dask_array_type):
        return dask_array_compat.gradient(
            x, coord, axis=axis, edge_order=edge_order)
    return npcompat.gradient(x, coord, axis=axis, edge_order=edge_order)


masked_invalid = _dask_or_eager_func(
    'masked_invalid', eager_module=np.ma,
    dask_module=getattr(dask_array, 'ma', None))


def asarray(data):
    return data if isinstance(data, dask_array_type) else np.asarray(data)


def as_shared_dtype(scalars_or_arrays):
    """Cast a arrays to a shared dtype using xarray's type promotion rules."""
    arrays = [asarray(x) for x in scalars_or_arrays]
    # Pass arrays directly instead of dtypes to result_type so scalars
    # get handled properly.
    # Note that result_type() safely gets the dtype from dask arrays without
    # evaluating them.
    out_type = dtypes.result_type(*arrays)
    return [x.astype(out_type, copy=False) for x in arrays]


def as_like_arrays(*data):
    if all(isinstance(d, dask_array_type) for d in data):
        return data
    else:
        return tuple(np.asarray(d) for d in data)


def allclose_or_equiv(arr1, arr2, rtol=1e-5, atol=1e-8):
    """Like np.allclose, but also allows values to be NaN in both arrays
    """
    arr1, arr2 = as_like_arrays(arr1, arr2)
    if arr1.shape != arr2.shape:
        return False
    return bool(
        isclose(arr1, arr2, rtol=rtol, atol=atol, equal_nan=True).all())


def array_equiv(arr1, arr2):
    """Like np.array_equal, but also allows values to be NaN in both arrays
    """
    arr1, arr2 = as_like_arrays(arr1, arr2)
    if arr1.shape != arr2.shape:
        return False

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', "In the future, 'NAT == x'")

        flag_array = (arr1 == arr2)
        flag_array |= (isnull(arr1) & isnull(arr2))

        return bool(flag_array.all())


def array_notnull_equiv(arr1, arr2):
    """Like np.array_equal, but also allows values to be NaN in either or both
    arrays
    """
    arr1, arr2 = as_like_arrays(arr1, arr2)
    if arr1.shape != arr2.shape:
        return False

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', "In the future, 'NAT == x'")

        flag_array = (arr1 == arr2)
        flag_array |= isnull(arr1)
        flag_array |= isnull(arr2)

        return bool(flag_array.all())


def count(data, axis=None):
    """Count the number of non-NA in this array along the given axis or axes
    """
    return np.sum(~isnull(data), axis=axis)


def where(condition, x, y):
    """Three argument where() with better dtype promotion rules."""
    return _where(condition, *as_shared_dtype([x, y]))


def where_method(data, cond, other=dtypes.NA):
    if other is dtypes.NA:
        other = dtypes.get_fill_value(data.dtype)
    return where(cond, data, other)


def fillna(data, other):
    return where(isnull(data), other, data)


def concatenate(arrays, axis=0):
    """concatenate() with better dtype promotion rules."""
    return _concatenate(as_shared_dtype(arrays), axis=axis)


def stack(arrays, axis=0):
    """stack() with better dtype promotion rules."""
    return _stack(as_shared_dtype(arrays), axis=axis)


@contextlib.contextmanager
def _ignore_warnings_if(condition):
    if condition:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            yield
    else:
        yield


def _create_nan_agg_method(name, coerce_strings=False):
    from . import nanops

    def f(values, axis=None, skipna=None, **kwargs):
        if kwargs.pop('out', None) is not None:
            raise TypeError('`out` is not valid for {}'.format(name))

        values = asarray(values)

        if coerce_strings and values.dtype.kind in 'SU':
            values = values.astype(object)

        func = None
        if skipna or (skipna is None and values.dtype.kind in 'cfO'):
            nanname = 'nan' + name
            func = getattr(nanops, nanname)
        else:
            func = _dask_or_eager_func(name)

        try:
            return func(values, axis=axis, **kwargs)
        except AttributeError:
            if isinstance(values, dask_array_type):
                try:  # dask/dask#3133 dask sometimes needs dtype argument
                    # if func does not accept dtype, then raises TypeError
                    return func(values, axis=axis, dtype=values.dtype,
                                **kwargs)
                except (AttributeError, TypeError):
                    msg = '%s is not yet implemented on dask arrays' % name
            else:
                msg = ('%s is not available with skipna=False with the '
                       'installed version of numpy; upgrade to numpy 1.12 '
                       'or newer to use skipna=True or skipna=None' % name)
            raise NotImplementedError(msg)

    f.__name__ = name
    return f


# Attributes `numeric_only`, `available_min_count` is used for docs.
# See ops.inject_reduce_methods
argmax = _create_nan_agg_method('argmax', coerce_strings=True)
argmin = _create_nan_agg_method('argmin', coerce_strings=True)
max = _create_nan_agg_method('max', coerce_strings=True)
min = _create_nan_agg_method('min', coerce_strings=True)
sum = _create_nan_agg_method('sum')
sum.numeric_only = True
sum.available_min_count = True
mean = _create_nan_agg_method('mean')
mean.numeric_only = True
std = _create_nan_agg_method('std')
std.numeric_only = True
var = _create_nan_agg_method('var')
var.numeric_only = True
median = _create_nan_agg_method('median')
median.numeric_only = True
prod = _create_nan_agg_method('prod')
prod.numeric_only = True
sum.available_min_count = True
cumprod_1d = _create_nan_agg_method('cumprod')
cumprod_1d.numeric_only = True
cumsum_1d = _create_nan_agg_method('cumsum')
cumsum_1d.numeric_only = True


def _nd_cum_func(cum_func, array, axis, **kwargs):
    array = asarray(array)
    if axis is None:
        axis = tuple(range(array.ndim))
    if isinstance(axis, int):
        axis = (axis,)

    out = array
    for ax in axis:
        out = cum_func(out, axis=ax, **kwargs)
    return out


def cumprod(array, axis=None, **kwargs):
    """N-dimensional version of cumprod."""
    return _nd_cum_func(cumprod_1d, array, axis, **kwargs)


def cumsum(array, axis=None, **kwargs):
    """N-dimensional version of cumsum."""
    return _nd_cum_func(cumsum_1d, array, axis, **kwargs)


_fail_on_dask_array_input_skipna = partial(
    fail_on_dask_array_input,
    msg='%r with skipna=True is not yet implemented on dask arrays')


def first(values, axis, skipna=None):
    """Return the first non-NA elements in this array along the given axis
    """
    if (skipna or skipna is None) and values.dtype.kind not in 'iSU':
        # only bother for dtypes that can hold NaN
        _fail_on_dask_array_input_skipna(values)
        return nanfirst(values, axis)
    return take(values, 0, axis=axis)


def last(values, axis, skipna=None):
    """Return the last non-NA elements in this array along the given axis
    """
    if (skipna or skipna is None) and values.dtype.kind not in 'iSU':
        # only bother for dtypes that can hold NaN
        _fail_on_dask_array_input_skipna(values)
        return nanlast(values, axis)
    return take(values, -1, axis=axis)


def rolling_window(array, axis, window, center, fill_value):
    """
    Make an ndarray with a rolling window of axis-th dimension.
    The rolling dimension will be placed at the last dimension.
    """
    if isinstance(array, dask_array_type):
        return dask_array_ops.rolling_window(
            array, axis, window, center, fill_value)
    else:  # np.ndarray
        return nputils.rolling_window(
            array, axis, window, center, fill_value)
