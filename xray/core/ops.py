import operator

import numpy as np
import pandas as pd

from . import utils
from .pycompat import PY3


UNARY_OPS = ['neg', 'pos', 'abs', 'invert']
CMP_BINARY_OPS = ['lt', 'le', 'eq', 'ne', 'ge', 'gt']
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
NUMPY_REDUCE_METHODS = ['all', 'any', 'argmax', 'argmin', 'max', 'mean', 'min',
                        'prod', 'ptp', 'std', 'sum', 'var']
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


def count(self, axis=None):
    nulls = np.asarray(utils.isnull(self))
    not_nulls = np.logical_not(nulls, nulls)
    return np.sum(not_nulls, axis=axis)


def inject_reduce_methods(cls):
    # change these to use methods instead of numpy functions?
    methods = [(name, getattr(np, name), True)
               for name in NUMPY_REDUCE_METHODS]
    methods += [('count', count, False)]
    for name, f, is_numpy_func in methods:
        func = cls._reduce_method(f)
        func.__name__ = name
        func.__doc__ = _REDUCE_DOCSTRING_TEMPLATE.format(
            name=('numpy.' if is_numpy_func else '') + name, cls=cls.__name__,
            extra_args=cls._reduce_extra_args_docstring)
        setattr(cls, name, func)


def op_str(name):
    return '__%s__' % name


def op(name):
    return getattr(operator, op_str(name))


def inject_binary_ops(cls, inplace=False):
    for name in CMP_BINARY_OPS + NUM_BINARY_OPS:
        setattr(cls, op_str(name), cls._binary_op(op(name)))
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
