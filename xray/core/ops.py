import operator

import numpy as np
import pandas as pd

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


def _values_method_wrapper(f):
    def func(self, *args, **kwargs):
        return getattr(self.values, f)(*args, **kwargs)
    func.__name__ = f
    return func


def _method_wrapper(f):
    def func(self, *args, **kwargs):
        return getattr(self, f)(*args, **kwargs)
    func.__name__ = f
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


def inject_reduce_methods(cls):
    # TODO: change these to use methods instead of numpy functions
    for name in NUMPY_REDUCE_METHODS:
        func = cls._reduce_method(getattr(np, name))
        func.__name__ = name
        func.__doc__ = _REDUCE_DOCSTRING_TEMPLATE.format(
            name='numpy.' + name, cls=cls.__name__,
            extra_args=cls._reduce_extra_args_docstring)
        setattr(cls, name, func)


def inject_special_operations(cls, priority=50):
    # priortize our operations over those of numpy.ndarray (priority=1)
    # and numpy.matrix (priority=10)
    cls.__array_priority__ = priority
    op_str = lambda name: '__%s__' % name
    op = lambda name: getattr(operator, op_str(name))
    # patch in standard special operations
    for op_names, op_wrap in [(UNARY_OPS, cls._unary_op),
                              (CMP_BINARY_OPS + NUM_BINARY_OPS,
                               cls._binary_op)]:
        for name in op_names:
            setattr(cls, op_str(name), op_wrap(op(name)))
    # only numeric operations have in-place and reflexive variants
    for name in NUM_BINARY_OPS:
        setattr(cls, op_str('r' + name),
                cls._binary_op(op(name), reflexive=True))
        setattr(cls, op_str('i' + name),
                cls._inplace_binary_op(op('i' + name)))
    # patch in numpy methods
    for name in NUMPY_SAME_METHODS:
        setattr(cls, name, _values_method_wrapper(name))
    for name in NUMPY_UNARY_METHODS:
        setattr(cls, name, cls._unary_op(_method_wrapper(name)))
    for name in PANDAS_UNARY_FUNCTIONS:
        setattr(cls, name, cls._unary_op(getattr(pd, name)))
    inject_reduce_methods(cls)
