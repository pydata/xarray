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
    import bottleneck as bn
    has_bottleneck = True
except ImportError:
    # use numpy methods instead
    bn = np
    has_bottleneck = False

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
    return sum(~isnull(data), axis=axis)


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


def _nansum_object(value, axis=None, **kwargs):
    """ In house nansum for object array """
    value = fillna(value, 0)
    return _dask_or_eager_func('sum')(value, axis=axis, **kwargs)


def _nan_minmax_object(func, get_fill_value, value, axis=None, **kwargs):
    """ In house nanmin and nanmax for object array """
    fill_value = get_fill_value(value.dtype)
    valid_count = count(value, axis=axis)
    filled_value = fillna(value, fill_value)
    data = _dask_or_eager_func(func)(filled_value, axis=axis, **kwargs)
    if not hasattr(data, 'dtype'):  # scalar case
        data = dtypes.fill_value(value.dtype) if valid_count == 0 else data
        return np.array(data, dtype=value.dtype)
    return where_method(data, valid_count != 0)


def _nan_argminmax_object(func, get_fill_value, value, axis=None, **kwargs):
    """ In house nanargmin, nanargmax for object arrays. Always return integer
    type """
    fill_value = get_fill_value(value.dtype)
    valid_count = count(value, axis=axis)
    value = fillna(value, fill_value)
    data = _dask_or_eager_func(func)(value, axis=axis, **kwargs)
    # dask seems return non-integer type
    if isinstance(value, dask_array_type):
        data = data.astype(int)

    if (valid_count == 0).any():
        raise ValueError('All-NaN slice encountered')

    return np.array(data, dtype=int)


def _nanmean_ddof_object(ddof, value, axis=None, **kwargs):
    """ In house nanmean. ddof argument will be used in _nanvar method """
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


def _nanvar_object(value, axis=None, **kwargs):
    ddof = kwargs.pop('ddof', 0)
    kwargs_mean = kwargs.copy()
    kwargs_mean.pop('keepdims', None)
    value_mean = _nanmean_ddof_object(ddof=0, value=value, axis=axis,
                                      keepdims=True, **kwargs_mean)
    squared = (value.astype(value_mean.dtype) - value_mean)**2
    return _nanmean_ddof_object(ddof, squared, axis=axis, **kwargs)


_nan_object_funcs = {
    'sum': _nansum_object,
    'min': partial(_nan_minmax_object, 'min', dtypes.get_pos_infinity),
    'max': partial(_nan_minmax_object, 'max', dtypes.get_neg_infinity),
    'argmin': partial(_nan_argminmax_object, 'argmin',
                      dtypes.get_pos_infinity),
    'argmax': partial(_nan_argminmax_object, 'argmax',
                      dtypes.get_neg_infinity),
    'mean': partial(_nanmean_ddof_object, 0),
    'var': _nanvar_object,
}


def _create_nan_agg_method(name, numeric_only=False, np_compat=False,
                           no_bottleneck=False, coerce_strings=False):
    def f(values, axis=None, skipna=None, **kwargs):
        if kwargs.pop('out', None) is not None:
            raise TypeError('`out` is not valid for {}'.format(name))

        # If dtype is supplied, we use numpy's method.
        dtype = kwargs.get('dtype', None)
        values = asarray(values)

        # dask requires dtype argument for object dtype
        if (values.dtype == 'object' and name in ['sum', ]):
            kwargs['dtype'] = values.dtype if dtype is None else dtype

        if coerce_strings and values.dtype.kind in 'SU':
            values = values.astype(object)

        if skipna or (skipna is None and values.dtype.kind in 'cfO'):
            if values.dtype.kind not in ['u', 'i', 'f', 'c']:
                func = _nan_object_funcs.get(name, None)
                using_numpy_nan_func = True
                if func is None or values.dtype.kind not in 'Ob':
                    raise NotImplementedError(
                        'skipna=True not yet implemented for %s with dtype %s'
                        % (name, values.dtype))
            else:
                nanname = 'nan' + name
                if (isinstance(axis, tuple) or not values.dtype.isnative or
                        no_bottleneck or (dtype is not None and
                                          np.dtype(dtype) != values.dtype)):
                    # bottleneck can't handle multiple axis arguments or
                    # non-native endianness
                    if np_compat:
                        eager_module = npcompat
                    else:
                        eager_module = np
                else:
                    kwargs.pop('dtype', None)
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
                    try:  # dask/dask#3133 dask sometimes needs dtype argument
                        return func(values, axis=axis, dtype=values.dtype,
                                    **kwargs)
                    except AttributeError:
                        msg = '%s is not yet implemented on dask arrays' % name
                else:
                    assert using_numpy_nan_func
                    msg = ('%s is not available with skipna=False with the '
                           'installed version of numpy; upgrade to numpy 1.12 '
                           'or newer to use skipna=True or skipna=None' % name)
                raise NotImplementedError(msg)
    f.numeric_only = numeric_only
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
cumprod_1d = _create_nan_agg_method(
    'cumprod', numeric_only=True, no_bottleneck=True)
cumsum_1d = _create_nan_agg_method(
    'cumsum', numeric_only=True, no_bottleneck=True)


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
