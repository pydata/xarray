__all__ = []

__array_api_version__ = "2023.12"

__all__ += ["__array_api_version__"]

from xarray.namedarray.core import NamedArray as Array

__all__ += ["Array"]

from xarray.namedarray._array_api.constants import e, inf, nan, newaxis, pi

__all__ += ["e", "inf", "nan", "newaxis", "pi"]

from xarray.namedarray._array_api.creation_functions import (
    arange,
    asarray,
    empty,
    empty_like,
    # eye,
    full,
    full_like,
    linspace,
    # meshgrid,
    ones,
    ones_like,
    # tril,
    # triu,
    zeros,
    zeros_like,
)

__all__ += [
    "arange",
    "asarray",
    "empty",
    "empty_like",
    # "eye",
    "full",
    "full_like",
    "linspace",
    # "meshgrid",
    "ones",
    "ones_like",
    # "tril",
    # "triu",
    "zeros",
    "zeros_like",
]

from xarray.namedarray._array_api.data_type_functions import (
    astype,
    can_cast,
    finfo,
    iinfo,
    isdtype,
    result_type,
)

__all__ += [
    "astype",
    "can_cast",
    "finfo",
    "iinfo",
    "isdtype",
    "result_type",
]

from xarray.namedarray._array_api.dtypes import (
    bool,
    complex64,
    complex128,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)

__all__ += [
    "bool",
    "complex64",
    "complex128",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]

from xarray.namedarray._array_api.elementwise_functions import (
    abs,
    acos,
    acosh,
    add,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    bitwise_and,
    bitwise_invert,
    bitwise_left_shift,
    bitwise_or,
    bitwise_right_shift,
    bitwise_xor,
    ceil,
    conj,
    cos,
    cosh,
    divide,
    equal,
    exp,
    expm1,
    floor,
    floor_divide,
    greater,
    greater_equal,
    imag,
    isfinite,
    isinf,
    isnan,
    less,
    less_equal,
    log,
    log1p,
    log2,
    log10,
    logaddexp,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    multiply,
    negative,
    not_equal,
    positive,
    pow,
    real,
    remainder,
    round,
    sign,
    sin,
    sinh,
    sqrt,
    square,
    subtract,
    tan,
    tanh,
    trunc,
)

__all__ += [
    "abs",
    "acos",
    "acosh",
    "add",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_invert",
    "bitwise_left_shift",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "ceil",
    "conj",
    "cos",
    "cosh",
    "divide",
    "equal",
    "exp",
    "expm1",
    "floor",
    "floor_divide",
    "greater",
    "greater_equal",
    "imag",
    "isfinite",
    "isinf",
    "isnan",
    "less",
    "less_equal",
    "log",
    "log1p",
    "log2",
    "log10",
    "logaddexp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "multiply",
    "negative",
    "not_equal",
    "positive",
    "pow",
    "real",
    "remainder",
    "round",
    "sign",
    "sin",
    "sinh",
    "sqrt",
    "square",
    "subtract",
    "tan",
    "tanh",
    "trunc",
]

# from xarray.namedarray._array_api.indexing_functions import take

# __all__ += ["take"]

# from xarray.namedarray._array_api.linear_algebra_functions import (
#     matmul,
#     matrix_transpose,
#     outer,
#     tensordot,
#     vecdot,
# )

# __all__ += ["matmul", "matrix_transpose", "outer", "tensordot", "vecdot"]

from xarray.namedarray._array_api.manipulation_functions import (
    # broadcast_arrays,
    # broadcast_to,
    # concat,
    expand_dims,
    # flip,
    # moveaxis,
    permute_dims,
    reshape,
    # roll,
    # squeeze,
    # stack,
)

__all__ += [
    # "broadcast_arrays",
    # "broadcast_to",
    # "concat",
    "expand_dims",
    # "flip",
    # "moveaxis",
    "permute_dims",
    "reshape",
    # "roll",
    # "squeeze",
    # "stack",
]

# from xarray.namedarray._array_api.searching_functions import (
#     argmax,
#     argmin,
#     where,
# )

# __all__ += [
#     "argmax",
#     "argmin",
#     "where",
# ]

from xarray.namedarray._array_api.statistical_functions import (
    max,
    mean,
    min,
    prod,
    sum,
)

__all__ += [
    "max",
    "mean",
    "min",
    "prod",
    "sum",
]

from xarray.namedarray._array_api.utility_functions import (
    all,
    any,
)

__all__ += [
    "all",
    "any",
]
