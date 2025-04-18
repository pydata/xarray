__all__ = []

__array_api_version__ = "2024.12"

__all__ += ["__array_api_version__"]

# from xarray.namedarray.core import NamedArray as Array

# __all__ += ["Array"]

from xarray.namedarray._array_api._constants import e, inf, nan, newaxis, pi

__all__ += ["e", "inf", "nan", "newaxis", "pi"]

from xarray.namedarray._array_api._creation_functions import (
    arange,
    asarray,
    empty,
    empty_like,
    eye,
    from_dlpack,
    full,
    full_like,
    linspace,
    meshgrid,
    ones,
    ones_like,
    tril,
    triu,
    zeros,
    zeros_like,
)

__all__ += [
    "arange",
    "asarray",
    "empty",
    "empty_like",
    "eye",
    "from_dlpack",
    "full",
    "full_like",
    "linspace",
    "meshgrid",
    "ones",
    "ones_like",
    "tril",
    "triu",
    "zeros",
    "zeros_like",
]

from xarray.namedarray._array_api._data_type_functions import (
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

from xarray.namedarray._array_api._dtypes import (
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

from xarray.namedarray._array_api._elementwise_functions import (
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
    clip,
    conj,
    copysign,
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
    hypot,
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
    maximum,
    minimum,
    multiply,
    negative,
    nextafter,
    not_equal,
    positive,
    pow,
    real,
    reciprocal,
    remainder,
    round,
    sign,
    signbit,
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
    "clip",
    "conj",
    "copysign",
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
    "hypot",
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
    "maximum",
    "minimum",
    "multiply",
    "negative",
    "nextafter",
    "not_equal",
    "positive",
    "pow",
    "real",
    "reciprocal",
    "remainder",
    "round",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "sqrt",
    "square",
    "subtract",
    "tan",
    "tanh",
    "trunc",
]

import xarray.namedarray._array_api._fft as fft

__all__ = ["fft"]

from xarray.namedarray._array_api._indexing_functions import take

__all__ += ["take"]

from xarray.namedarray._array_api._info import __array_namespace_info__

__all__ += [
    "__array_namespace_info__",
]

import xarray.namedarray._array_api._linalg as linalg

__all__ = ["linalg"]


from xarray.namedarray._array_api._linear_algebra_functions import (
    matmul,
    matrix_transpose,
    tensordot,
    vecdot,
)

__all__ += [
    "matmul",
    "matrix_transpose",
    "tensordot",
    "vecdot",
]

from xarray.namedarray._array_api._manipulation_functions import (
    broadcast_arrays,
    broadcast_to,
    concat,
    expand_dims,
    flip,
    moveaxis,
    permute_dims,
    repeat,
    reshape,
    roll,
    squeeze,
    stack,
    tile,
    unstack,
)

__all__ += [
    "broadcast_arrays",
    "broadcast_to",
    "concat",
    "expand_dims",
    "flip",
    "moveaxis",
    "permute_dims",
    "repeat",
    "reshape",
    "roll",
    "squeeze",
    "stack",
    "tile",
    "unstack",
]

from xarray.namedarray._array_api._searching_functions import (
    argmax,
    argmin,
    count_nonzero,
    nonzero,
    searchsorted,
    where,
)

__all__ += [
    "argmax",
    "argmin",
    "count_nonzero",
    "nonzero",
    "searchsorted",
    "where",
]

from xarray.namedarray._array_api._set_functions import (
    unique_all,
    unique_counts,
    unique_inverse,
    unique_values,
)

__all__ += [
    "unique_all",
    "unique_counts",
    "unique_inverse",
    "unique_values",
]


from xarray.namedarray._array_api._sorting_functions import argsort, sort

__all__ += ["argsort", "sort"]

from xarray.namedarray._array_api._statistical_functions import (
    cumulative_prod,
    cumulative_sum,
    max,
    mean,
    min,
    prod,
    std,
    sum,
    var,
)

__all__ += [
    "cumulative_prod",
    "cumulative_sum",
    "max",
    "mean",
    "min",
    "prod",
    "std",
    "sum",
    "var",
]

from xarray.namedarray._array_api._utility_functions import (
    all,
    any,
    diff,
)

__all__ += [
    "all",
    "any",
    "diff",
]
