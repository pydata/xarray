"""Compatibility module defining operations on duck numpy-arrays.

Currently, this means Dask or NumPy arrays. None of these functions should
accept or return xarray objects.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import contextlib
import inspect
import warnings

import numpy as np
import pandas as pd

from . import npcompat
from . import dtypes
from .pycompat import dask_array_type
from .nputils import nanfirst, nanlast

try:
    import bottleneck as bn
    has_bottleneck = True
except ImportError:
    # use numpy methods instead
    bn = np
    has_bottleneck = False

try:
    import dask.array as da
    has_dask = True
except ImportError:
    has_dask = False


def _dask_or_eager_func(name, eager_module=np, list_of_args=False,
                        n_array_args=1):
    """Create a function that dispatches to dask for dask array inputs."""
    if has_dask:
        def f(*args, **kwargs):
            if list_of_args:
                dispatch_args = args[0]
            else:
                dispatch_args = args[:n_array_args]
            if any(isinstance(a, da.Array) for a in dispatch_args):
                module = da
            else:
                module = eager_module
            return getattr(module, name)(*args, **kwargs)
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
notnull = _dask_or_eager_func('notnull', pd)
_isnull = _dask_or_eager_func('isnull', pd)


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
where = _dask_or_eager_func('where', n_array_args=3)
insert = _dask_or_eager_func('insert')
take = _dask_or_eager_func('take')
broadcast_to = _dask_or_eager_func('broadcast_to')

concatenate = _dask_or_eager_func('concatenate', list_of_args=True)
stack = _dask_or_eager_func('stack', list_of_args=True)

array_all = _dask_or_eager_func('all')
array_any = _dask_or_eager_func('any')

tensordot = _dask_or_eager_func('tensordot', n_array_args=2)


def asarray(data):
    return data if isinstance(data, dask_array_type) else np.asarray(data)


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

    flag_array = (arr1 == arr2)
    flag_array |= isnull(arr1)
    flag_array |= isnull(arr2)

    return bool(flag_array.all())


def count(data, axis=None):
    """Count the number of non-NA in this array along the given axis or axes
    """
    return sum(~isnull(data), axis=axis)


def where_method(data, cond, other=dtypes.NA):
    if other is dtypes.NA:
        other = dtypes.get_fill_value(data.dtype)
    return where(cond, data, other)


def fillna(data, other):
    return where(isnull(data), other, data)


@contextlib.contextmanager
def _ignore_warnings_if(condition):
    if condition:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            yield
    else:
        yield


def _create_nan_agg_method(name, numeric_only=False, np_compat=False,
                           no_bottleneck=False, coerce_strings=False,
                           keep_dims=False):
    def f(values, axis=None, skipna=None, **kwargs):
        # ignore keyword args inserted by np.mean and other numpy aggregators
        # automatically:
        kwargs.pop('dtype', None)
        kwargs.pop('out', None)

        values = asarray(values)

        if coerce_strings and values.dtype.kind in 'SU':
            values = values.astype(object)

        if skipna or (skipna is None and values.dtype.kind in 'cf'):
            if values.dtype.kind not in ['u', 'i', 'f', 'c']:
                raise NotImplementedError(
                    'skipna=True not yet implemented for %s with dtype %s'
                    % (name, values.dtype))
            nanname = 'nan' + name
            if (isinstance(axis, tuple) or not values.dtype.isnative or
                    no_bottleneck):
                # bottleneck can't handle multiple axis arguments or non-native
                # endianness
                if np_compat:
                    eager_module = npcompat
                else:
                    eager_module = np
            else:
                eager_module = bn
            func = _dask_or_eager_func(nanname, eager_module)
            using_numpy_nan_func = (eager_module is np or
                                    eager_module is npcompat)
        else:
            func = _dask_or_eager_func(name)
            using_numpy_nan_func = False
        with _ignore_warnings_if(using_numpy_nan_func):
            try:
                return func(values, axis=axis, **kwargs)
            except AttributeError:
                if isinstance(values, dask_array_type):
                    msg = '%s is not yet implemented on dask arrays' % name
                else:
                    assert using_numpy_nan_func
                    msg = ('%s is not available with skipna=False with the '
                           'installed version of numpy; upgrade to numpy 1.12 '
                           'or newer to use skipna=True or skipna=None' % name)
                raise NotImplementedError(msg)
    f.numeric_only = numeric_only
    f.keep_dims = keep_dims
    f.__name__ = name
    return f


argmax = _create_nan_agg_method('argmax', coerce_strings=True)
argmin = _create_nan_agg_method('argmin', coerce_strings=True)
max = _create_nan_agg_method('max', coerce_strings=True)
min = _create_nan_agg_method('min', coerce_strings=True)
sum = _create_nan_agg_method('sum', numeric_only=True)
mean = _create_nan_agg_method('mean', numeric_only=True)
std = _create_nan_agg_method('std', numeric_only=True)
var = _create_nan_agg_method('var', numeric_only=True)
median = _create_nan_agg_method('median', numeric_only=True)
prod = _create_nan_agg_method('prod', numeric_only=True, no_bottleneck=True)
cumprod = _create_nan_agg_method('cumprod', numeric_only=True, np_compat=True,
                                 no_bottleneck=True, keep_dims=True)
cumsum = _create_nan_agg_method('cumsum', numeric_only=True, np_compat=True,
                                no_bottleneck=True, keep_dims=True)

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
