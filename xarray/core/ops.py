from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from functools import partial
import contextlib
import inspect
import operator
import warnings
from distutils.version import StrictVersion

import numpy as np
import pandas as pd

from . import npcompat
from .pycompat import PY3, dask_array_type
from .nputils import nanfirst, nanlast, array_eq, array_ne


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


UNARY_OPS = ['neg', 'pos', 'abs', 'invert']
CMP_BINARY_OPS = ['lt', 'le', 'ge', 'gt']
NUM_BINARY_OPS = ['add', 'sub', 'mul', 'truediv', 'floordiv', 'mod',
                  'pow', 'and', 'xor', 'or']
if not PY3:
    NUM_BINARY_OPS.append('div')

# methods which pass on the numpy return value unchanged
# be careful not to list methods that we would want to wrap later
NUMPY_SAME_METHODS = ['item', 'searchsorted']
# methods which don't modify the data shape, so the result should still be
# wrapped in an Variable/DataArray
NUMPY_UNARY_METHODS = ['astype', 'argsort', 'clip', 'conj', 'conjugate']
PANDAS_UNARY_FUNCTIONS = ['isnull', 'notnull']
# methods which remove an axis
REDUCE_METHODS = ['all', 'any']
NAN_REDUCE_METHODS = ['argmax', 'argmin', 'max', 'min', 'mean', 'prod', 'sum',
                      'std', 'var', 'median']
NAN_CUM_METHODS = ['cumsum', 'cumprod']
BOTTLENECK_ROLLING_METHODS = {'move_sum': 'sum', 'move_mean': 'mean',
                              'move_std': 'std', 'move_min': 'min',
                              'move_max': 'max'}
# TODO: wrap take, dot, sort


def _dask_or_eager_func(name, eager_module=np, list_of_args=False,
                        n_array_args=1):
    if has_dask:
        def f(*args, **kwargs):
            dispatch_args = args[0] if list_of_args else args
            if any(isinstance(a, da.Array)
                   for a in dispatch_args[:n_array_args]):
                module = da
            else:
                module = eager_module
            return getattr(module, name)(*args, **kwargs)
    else:
        def f(data, *args, **kwargs):
            return getattr(eager_module, name)(data, *args, **kwargs)
    return f


def _fail_on_dask_array_input(values, msg=None, func_name=None):
    if isinstance(values, dask_array_type):
        if msg is None:
            msg = '%r is not a valid method on dask arrays'
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
broadcast_to = _dask_or_eager_func('broadcast_to', npcompat)

concatenate = _dask_or_eager_func('concatenate', list_of_args=True)
stack = _dask_or_eager_func('stack', npcompat, list_of_args=True)

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
    return bool(isclose(arr1, arr2, rtol=rtol, atol=atol, equal_nan=True).all())


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


def _call_possibly_missing_method(arg, name, args, kwargs):
    try:
        method = getattr(arg, name)
    except AttributeError:
        _fail_on_dask_array_input(arg, func_name=name)
        if hasattr(arg, 'data'):
            _fail_on_dask_array_input(arg.data, func_name=name)
        raise
    else:
        return method(*args, **kwargs)


def _values_method_wrapper(name):
    def func(self, *args, **kwargs):
        return _call_possibly_missing_method(self.data, name, args, kwargs)
    func.__name__ = name
    func.__doc__ = getattr(np.ndarray, name).__doc__
    return func


def _method_wrapper(name):
    def func(self, *args, **kwargs):
        return _call_possibly_missing_method(self, name, args, kwargs)
    func.__name__ = name
    func.__doc__ = getattr(np.ndarray, name).__doc__
    return func


def _func_slash_method_wrapper(f, name=None):
    # try to wrap a method, but if not found use the function
    # this is useful when patching in a function as both a DataArray and
    # Dataset method
    if name is None:
        name = f.__name__

    def func(self, *args, **kwargs):
        try:
            return getattr(self, name)(*args, **kwargs)
        except AttributeError:
            return f(self, *args, **kwargs)
    func.__name__ = name
    func.__doc__ = f.__doc__
    return func

_CUM_DOCSTRING_TEMPLATE = \
        """Apply `{name}` along some dimension of {cls}.

        Parameters
        ----------
        {extra_args}
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, the attributes (`attrs`) will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to `{name}`.

        Returns
        -------
        cumvalue : {cls}
            New {cls} object with `{name}` applied to its data along the
            indicated dimension.
        """

_REDUCE_DOCSTRING_TEMPLATE = \
        """Reduce this {cls}'s data by applying `{name}` along some
        dimension(s).

        Parameters
        ----------
        {extra_args}
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, the attributes (`attrs`) will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to `{name}`.

        Returns
        -------
        reduced : {cls}
            New {cls} object with `{name}` applied to its data and the
            indicated dimension(s) removed.
        """

_ROLLING_REDUCE_DOCSTRING_TEMPLATE = \
        """Reduce this DataArrayRolling's data windows by applying `{name}`
        along its dimension.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed on to `{name}`.

        Returns
        -------
        reduced : DataArray
            New DataArray object with `{name}` applied along its rolling dimnension.
        """


def count(data, axis=None):
    """Count the number of non-NA in this array along the given axis or axes
    """
    return sum(~isnull(data), axis=axis)


def fillna(data, other):
    """Fill missing values in this object with data from the other object.
    Follows normal broadcasting and alignment rules.
    """
    return where(isnull(data), other, data)


def where_method(data, cond, other=np.nan):
    """Select values from this object that are True in cond. Everything else
    gets masked with other. Follows normal broadcasting and alignment rules.
    """
    return where(cond, data, other)


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
            if values.dtype.kind not in ['i', 'f', 'c']:
                raise NotImplementedError(
                    'skipna=True not yet implemented for %s with dtype %s'
                    % (name, values.dtype))
            nanname = 'nan' + name
            if isinstance(axis, tuple) or not values.dtype.isnative or no_bottleneck:
                # bottleneck can't handle multiple axis arguments or non-native
                # endianness
                if np_compat:
                    eager_module = npcompat
                else:
                    eager_module = np
            else:
                eager_module = bn
            func = _dask_or_eager_func(nanname, eager_module)
            using_numpy_nan_func = eager_module is np or eager_module is npcompat
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
prod = _create_nan_agg_method('prod', numeric_only=True, np_compat=True,
                              no_bottleneck=True)
cumprod = _create_nan_agg_method('cumprod', numeric_only=True, np_compat=True,
                                 no_bottleneck=True, keep_dims=True)
cumsum = _create_nan_agg_method('cumsum', numeric_only=True, np_compat=True,
                                no_bottleneck=True, keep_dims=True)

_fail_on_dask_array_input_skipna = partial(
    _fail_on_dask_array_input,
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


def inject_reduce_methods(cls):
    methods = ([(name, globals()['array_%s' % name], False) for name
               in REDUCE_METHODS] +
               [(name, globals()[name], True) for name in NAN_REDUCE_METHODS] +
               [('count', count, False)])
    for name, f, include_skipna in methods:
        numeric_only = getattr(f, 'numeric_only', False)
        func = cls._reduce_method(f, include_skipna, numeric_only)
        func.__name__ = name
        func.__doc__ = _REDUCE_DOCSTRING_TEMPLATE.format(
            name=name, cls=cls.__name__,
            extra_args=cls._reduce_extra_args_docstring)
        setattr(cls, name, func)

def inject_cum_methods(cls):
    methods = ([(name, globals()[name], True) for name in NAN_CUM_METHODS])
    for name, f, include_skipna in methods:
        numeric_only = getattr(f, 'numeric_only', False)
        func = cls._reduce_method(f, include_skipna, numeric_only)
        func.__name__ = name
        func.__doc__ = _CUM_DOCSTRING_TEMPLATE.format(
            name=name, cls=cls.__name__,
            extra_args=cls._cum_extra_args_docstring)
        setattr(cls, name, func)


def op_str(name):
    return '__%s__' % name


def get_op(name):
    return getattr(operator, op_str(name))


NON_INPLACE_OP = dict((get_op('i' + name), get_op(name))
                      for name in NUM_BINARY_OPS)


def inplace_to_noninplace_op(f):
    return NON_INPLACE_OP[f]


def inject_binary_ops(cls, inplace=False):
    for name in CMP_BINARY_OPS + NUM_BINARY_OPS:
        setattr(cls, op_str(name), cls._binary_op(get_op(name)))

    for name, f in [('eq', array_eq), ('ne', array_ne)]:
        setattr(cls, op_str(name), cls._binary_op(f))

    # patch in fillna
    f = _func_slash_method_wrapper(fillna)
    method = cls._binary_op(f, join='left', fillna=True)
    setattr(cls, '_fillna', method)

    # patch in where
    f = _func_slash_method_wrapper(where_method, 'where')
    setattr(cls, '_where', cls._binary_op(f))

    for name in NUM_BINARY_OPS:
        # only numeric operations have in-place and reflexive variants
        setattr(cls, op_str('r' + name),
                cls._binary_op(get_op(name), reflexive=True))
        if inplace:
            setattr(cls, op_str('i' + name),
                    cls._inplace_binary_op(get_op('i' + name)))


def inject_all_ops_and_reduce_methods(cls, priority=50, array_only=True):
    # prioritize our operations over those of numpy.ndarray (priority=1)
    # and numpy.matrix (priority=10)
    cls.__array_priority__ = priority

    # patch in standard special operations
    for name in UNARY_OPS:
        setattr(cls, op_str(name), cls._unary_op(get_op(name)))
    inject_binary_ops(cls, inplace=True)

    # patch in numpy/pandas methods
    for name in NUMPY_UNARY_METHODS:
        setattr(cls, name, cls._unary_op(_method_wrapper(name)))

    for name in PANDAS_UNARY_FUNCTIONS:
        f = _func_slash_method_wrapper(getattr(pd, name))
        setattr(cls, name, cls._unary_op(f))

    f = _func_slash_method_wrapper(around, name='round')
    setattr(cls, 'round', cls._unary_op(f))

    if array_only:
        # these methods don't return arrays of the same shape as the input, so
        # don't try to patch these in for Dataset objects
        for name in NUMPY_SAME_METHODS:
            setattr(cls, name, _values_method_wrapper(name))

    inject_reduce_methods(cls)
    inject_cum_methods(cls)


def inject_bottleneck_rolling_methods(cls):
    # standard numpy reduce methods
    methods = [(name, globals()[name]) for name in NAN_REDUCE_METHODS]
    for name, f in methods:
        func = cls._reduce_method(f)
        func.__name__ = name
        func.__doc__ = _ROLLING_REDUCE_DOCSTRING_TEMPLATE.format(name=func.__name__)
        setattr(cls, name, func)

    # bottleneck rolling methods
    if has_bottleneck:
        if StrictVersion(bn.__version__) < StrictVersion('1.0'):
            return

        for bn_name, method_name in BOTTLENECK_ROLLING_METHODS.items():
            f = getattr(bn, bn_name)
            func = cls._bottleneck_reduce(f)
            func.__name__ = method_name
            func.__doc__ = _ROLLING_REDUCE_DOCSTRING_TEMPLATE.format(name=func.__name__)
            setattr(cls, method_name, func)

        # bottleneck rolling methods without min_count
        f = getattr(bn, 'move_median')
        func = cls._bottleneck_reduce_without_min_count(f)
        func.__name__ = 'median'
        func.__doc__ = _ROLLING_REDUCE_DOCSTRING_TEMPLATE.format(name=func.__name__)
        setattr(cls, 'median', func)
