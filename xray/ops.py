import operator

import numpy as np

from .pycompat import PY3


UNARY_OPS = ['neg', 'pos', 'abs', 'invert']
CMP_BINARY_OPS = ['lt', 'le', 'eq', 'ne', 'ge', 'gt']
NUM_BINARY_OPS = ['add', 'sub', 'mul', 'truediv', 'floordiv', 'mod',
                  'pow', 'and', 'xor', 'or']
if not PY3:
    NUM_BINARY_OPS.append('div')

# methods which don't modify the data shape, so the result should still be
# wrapped in an Variable/DataArray
NUMPY_UNARY_METHODS = ['astype', 'argsort', 'clip', 'conj', 'conjugate',
                       'round']
# methods which remove an axis
NUMPY_REDUCE_METHODS = ['all', 'any', 'argmax', 'argmin', 'max', 'mean', 'min',
                        'prod', 'ptp', 'std', 'sum', 'var']
# TODO: wrap cumprod, cumsum, take and dot


def _method_wrapper(f):
    def func(self, *args, **kwargs):
        return getattr(self, f)(*args, **kwargs)
    func.__name__ = f
    return func


def inject_reduce_methods(cls):
    # TODO: change these to use methods instead of numpy functions
    for name in NUMPY_REDUCE_METHODS:
        setattr(cls, name, cls._reduce_method(getattr(np, name),
                                              name, 'numpy'))


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
    for name in NUMPY_UNARY_METHODS:
        setattr(cls, name, cls._unary_op(_method_wrapper(name)))
    inject_reduce_methods(cls)
