import operator

import numpy as np
import pandas as pd

from . import utils
from .pycompat import PY3

try:
    import bottleneck as bn
except ImportError:
    # use numpy methods instead
    bn = np


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
NUMPY_UNARY_METHODS = ['astype', 'argsort', 'clip', 'conj', 'conjugate',
                       'round']
PANDAS_UNARY_FUNCTIONS = ['isnull', 'notnull']
# methods which remove an axis
NUMPY_REDUCE_METHODS = ['all', 'any']
NAN_REDUCE_METHODS = ['argmax', 'argmin', 'max', 'min', 'mean', 'sum',
                      'std', 'var', 'median']
# TODO: wrap cumprod/cumsum, take, dot, sort


def _values_method_wrapper(name):
    def func(self, *args, **kwargs):
        return getattr(self.values, name)(*args, **kwargs)
    func.__name__ = name
    func.__doc__ = getattr(np.ndarray, name).__doc__
    return func


def _method_wrapper(name):
    def func(self, *args, **kwargs):
        return getattr(self, name)(*args, **kwargs)
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


def count(values, axis=None):
    return np.sum(~pd.isnull(values), axis=axis)


def _create_nan_agg_method(name, numeric_only=False):
    def f(values, axis=None, skipna=None, **kwargs):
        # ignore keyword args inserted by np.mean and other numpy aggreagators
        # automatically:
        kwargs.pop('dtype', None)
        kwargs.pop('out', None)

        if skipna or (skipna is None and values.dtype.kind == 'f'):
            if values.dtype.kind not in ['i', 'f']:
                raise NotImplementedError(
                    'skipna=True not yet implemented for %s with dtype %s'
                    % (name, values.dtype))
            nanname = 'nan' + name
            try:
                if isinstance(axis, tuple):
                    func = getattr(np, nanname)
                else:
                    func = getattr(bn, nanname)
            except AttributeError:
                raise NotImplementedError(
                    '%s is not available with skipna=False with the '
                    'installed version of numpy; upgrade to numpy 1.9 or '
                    'newer to use skipna=True or skipna=None' % name)
        else:
            func = getattr(np, name)
        return func(values, axis=axis, **kwargs)
    f.numeric_only = numeric_only
    return f


argmax = _create_nan_agg_method('argmax')
argmin = _create_nan_agg_method('argmin')
max = _create_nan_agg_method('max')
min = _create_nan_agg_method('min')
sum = _create_nan_agg_method('sum', numeric_only=True)
mean = _create_nan_agg_method('mean', numeric_only=True)
std = _create_nan_agg_method('std', numeric_only=True)
var = _create_nan_agg_method('var', numeric_only=True)
median = _create_nan_agg_method('median', numeric_only=True)


def numeric_only(f):
    f.numeric_only = True
    return f


def _replace_nan(a, val):
    # copied from np.lib.nanfunctions
    # remove this if/when https://github.com/numpy/numpy/pull/5418 is merged
    is_new = not isinstance(a, np.ndarray)
    if is_new:
        a = np.array(a)
    if not issubclass(a.dtype.type, np.inexact):
        return a, None
    if not is_new:
        # need copy
        a = np.array(a, subok=True)

    mask = np.isnan(a)
    np.copyto(a, val, where=mask)
    return a, mask


@numeric_only
def prod(values, axis=None, skipna=None, **kwargs):
    if skipna or (skipna is None and values.dtype.kind == 'f'):
        if values.dtype.kind not in ['i', 'f']:
            raise NotImplementedError(
                'skipna=True not yet implemented for prod with dtype %s'
                % values.dtype)
        values, mask = _replace_nan(values, 1)
    return np.prod(values, axis=axis, **kwargs)


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
    return _ensure_bool_is_ndarray(self == other, self, other)


def array_ne(self, other):
    return _ensure_bool_is_ndarray(self != other, self, other)


def inject_reduce_methods(cls):
    methods = ([(name, getattr(np, name), False) for name
               in NUMPY_REDUCE_METHODS]
               + [(name, globals()[name], True) for name
                  in ['prod'] + NAN_REDUCE_METHODS]
               + [('count', count, False)])
    for name, f, include_skipna in methods:
        numeric_only = getattr(f, 'numeric_only', False)
        func = cls._reduce_method(f, include_skipna, numeric_only)
        func.__name__ = name
        func.__doc__ = _REDUCE_DOCSTRING_TEMPLATE.format(
            name=name, cls=cls.__name__,
            extra_args=cls._reduce_extra_args_docstring)
        setattr(cls, name, func)


def op_str(name):
    return '__%s__' % name


def op(name):
    return getattr(operator, op_str(name))


def inject_binary_ops(cls, inplace=False):
    for name in CMP_BINARY_OPS + NUM_BINARY_OPS:
        setattr(cls, op_str(name), cls._binary_op(op(name)))
    for name, f in [('eq', array_eq), ('ne', array_ne)]:
        setattr(cls, op_str(name), cls._binary_op(f))
    for name in NUM_BINARY_OPS:
        # only numeric operations have in-place and reflexive variants
        setattr(cls, op_str('r' + name),
                cls._binary_op(op(name), reflexive=True))
        if inplace:
            setattr(cls, op_str('i' + name),
                    cls._inplace_binary_op(op('i' + name)))


def inject_all_ops_and_reduce_methods(cls, priority=50, array_only=True):
    # priortize our operations over those of numpy.ndarray (priority=1)
    # and numpy.matrix (priority=10)
    cls.__array_priority__ = priority
    # patch in standard special operations
    for name in UNARY_OPS:
        setattr(cls, op_str(name), cls._unary_op(op(name)))
    inject_binary_ops(cls, inplace=True)
    # patch in numpy/pandas methods
    for name in NUMPY_UNARY_METHODS:
        setattr(cls, name, cls._unary_op(_method_wrapper(name)))
    for name in PANDAS_UNARY_FUNCTIONS:
        f = _func_slash_method_wrapper(getattr(pd, name))
        setattr(cls, name, cls._unary_op(f))
    if array_only:
        # these methods don't return arrays of the same shape as the input, so
        # don't try to patch these in for Dataset objects
        for name in NUMPY_SAME_METHODS:
            setattr(cls, name, _values_method_wrapper(name))
    inject_reduce_methods(cls)
