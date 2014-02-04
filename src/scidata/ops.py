import operator

import numpy as np


UNARY_OPS = ['neg', 'pos', 'abs', 'invert']
CMP_BINARY_OPS = ['lt', 'le', 'eq', 'ne', 'ge', 'gt']
NUM_BINARY_OPS = ['add', 'sub', 'mul', 'div', 'truediv', 'floordiv', 'mod',
                  'pow', 'and', 'xor', 'or']
NUMPY_COLLAPSE_METHODS = ['all', 'any', 'argmax', 'argmin', 'cumprod',
                          'cumsum', 'max', 'mean', 'min', 'prod', 'ptp', 'std',
                          'sum', 'var']


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
    for name in NUMPY_COLLAPSE_METHODS:
        setattr(cls, name, cls._collapse_method(getattr(np, name),
                                                name, 'numpy'))
