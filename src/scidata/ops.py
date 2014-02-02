import operator


# NUMPY_COLLAPSE_METHODS = ['argmax', 'min', 'argmin', 'ptp', 'sum', 'cumsum',
#                           'mean', 'var', 'std', 'prod', 'cumprod', 'all',
#                           'any']

# def wrap_numpy_collapse_method(f):
#     def func(self, dimension=None, axis=None):
#         if dimension is not None:
#             if axis is None:
#                 axis = self.dimensions.index(dim)
#             else:
#                 raise ValueError("cannot specify both 'axis' and 'dimension'")
#         # dims = tuple(dim for dim in self.dimension is dim != dimension)
#         if axis is not None:
#             dims = tuple(dim for i, dim in enumerate(self.dimension)
#                          if i not in [axis, axis + self.ndim])
#         else:
#             dims = (),
#         data = f(self.data, axis=axis)
#         return Variable(dims, data, self.attributes)


UNARY_OPS = ['neg', 'pos', 'abs', 'invert']
CMP_BINARY_OPS = ['lt', 'le', 'eq', 'ne', 'ge', 'gt']
NUM_BINARY_OPS = ['add', 'sub', 'mul', 'div', 'truediv', 'floordiv', 'mod',
                  'pow', 'and', 'xor', 'or']


def inject_special_operations(cls, unary_op, binary_op, inplace_binary_op,
                              priority=50):
    # priortize our operations over those of numpy.ndarray (priority=1)
    # and numpy.matrix (priority=10)
    cls.__array_priority__ = priority
    op_str = lambda name: '__%s__' % name
    op = lambda name: getattr(operator, op_str(name))
    # patch in standard special operations
    for op_names, op_wrap in [(UNARY_OPS, unary_op),
                              (CMP_BINARY_OPS + NUM_BINARY_OPS, binary_op)]:
        for name in op_names:
            setattr(cls, op_str(name), op_wrap(op(name)))
    # only numeric operations have in-place and reflexive variants
    for name in NUM_BINARY_OPS:
        setattr(cls, op_str('r' + name), binary_op(op(name), reflexive=True))
        setattr(cls, op_str('i' + name), inplace_binary_op(op('i' + name)))
