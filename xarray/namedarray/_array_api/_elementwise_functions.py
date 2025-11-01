from __future__ import annotations

from typing import Any

import numpy as np

from xarray.namedarray._array_api._data_type_functions import _promote_scalars
from xarray.namedarray._array_api._manipulation_functions import _arithmetic_broadcast
from xarray.namedarray._array_api._utils import (
    _get_data_namespace,
)
from xarray.namedarray._typing import (
    _DType,
    _ScalarType,
    _ShapeType,
    _SupportsImag,
    _SupportsReal,
)
from xarray.namedarray.core import NamedArray


def abs(x: NamedArray[_ShapeType, _DType], /) -> NamedArray[_ShapeType, _DType]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.abs(x._data)
    return x._new(_dims, _data)


def acos(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.acos(x._data)
    return x._new(_dims, _data)


def acosh(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.acosh(x._data)
    return x._new(_dims, _data)


def add(
    x1: NamedArray[Any, Any] | bool | int | float | complex,
    x2: NamedArray[Any, Any] | bool | int | float | complex,
    /,
) -> NamedArray[Any, Any]:
    """
    Calculates the sum for each element x1_i of the input array x1 with the
    respective element x2_i of the input array x2.

    Examples
    --------
    >>> import numpy as np
    >>> x1 = NamedArray(("x",), np.array([1, 2, 3]))
    >>> x2 = 2
    >>> add(x1, x2)
    <xarray.NamedArray (x: 3)> Size: 24B
    array([3, 4, 5])
    >>> add(x2, x1)
    <xarray.NamedArray (x: 3)> Size: 24B
    array([3, 4, 5])
    """
    x1_arr, x2_arr = _promote_scalars(x1, x2, "numeric", "add")
    xp = _get_data_namespace(x1_arr)
    x1_new, x2_new = _arithmetic_broadcast(x1_arr, x2_arr)
    _dims = x1_new.dims
    _data = xp.add(x1_new._data, x2_new._data)
    return x1_arr._new(_dims, _data)


def asin(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.asin(x._data)
    return x._new(_dims, _data)


def asinh(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.asinh(x._data)
    return x._new(_dims, _data)


def atan(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.atan(x._data)
    return x._new(_dims, _data)


def atan2(
    x1: NamedArray[Any, Any] | bool | int | float | complex,
    x2: NamedArray[Any, Any] | bool | int | float | complex,
    /,
) -> NamedArray[Any, Any]:
    x1_arr, x2_arr = _promote_scalars(x1, x2, "real floating-point", "atan2")
    xp = _get_data_namespace(x1_arr)
    x1_new, x2_new = _arithmetic_broadcast(x1_arr, x2_arr)
    _dims = x1_new.dims
    _data = xp.atan2(x1_new._data, x2_new._data)
    return x1_arr._new(_dims, _data)


def atanh(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.atanh(x._data)
    return x._new(_dims, _data)


def bitwise_and(
    x1: NamedArray[Any, Any] | bool | int | float | complex,
    x2: NamedArray[Any, Any] | bool | int | float | complex,
    /,
) -> NamedArray[Any, Any]:
    x1_arr, x2_arr = _promote_scalars(x1, x2, "integer or boolean", "bitwise_and")
    xp = _get_data_namespace(x1_arr)
    x1_new, x2_new = _arithmetic_broadcast(x1_arr, x2_arr)
    _dims = x1_new.dims
    _data = xp.bitwise_and(x1_new._data, x2_new._data)
    return x1_arr._new(_dims, _data)


def bitwise_invert(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.bitwise_invert(x._data)
    return x._new(_dims, _data)


def bitwise_left_shift(
    x1: NamedArray[Any, Any] | bool | int | float | complex,
    x2: NamedArray[Any, Any] | bool | int | float | complex,
    /,
) -> NamedArray[Any, Any]:
    x1_arr, x2_arr = _promote_scalars(x1, x2, "integer", "bitwise_left_shift")
    xp = _get_data_namespace(x1_arr)
    x1_new, x2_new = _arithmetic_broadcast(x1_arr, x2_arr)
    _dims = x1_new.dims
    _data = xp.bitwise_left_shift(x1_new._data, x2_new._data)
    return x1_arr._new(_dims, _data)


def bitwise_or(
    x1: NamedArray[Any, Any] | bool | int | float | complex,
    x2: NamedArray[Any, Any] | bool | int | float | complex,
    /,
) -> NamedArray[Any, Any]:
    x1_arr, x2_arr = _promote_scalars(x1, x2, "integer or boolean", "bitwise_or")
    xp = _get_data_namespace(x1_arr)
    x1_new, x2_new = _arithmetic_broadcast(x1_arr, x2_arr)
    _dims = x1_new.dims
    _data = xp.bitwise_or(x1_new._data, x2_new._data)
    return x1_arr._new(_dims, _data)


def bitwise_right_shift(
    x1: NamedArray[Any, Any] | bool | int | float | complex,
    x2: NamedArray[Any, Any] | bool | int | float | complex,
    /,
) -> NamedArray[Any, Any]:
    x1_arr, x2_arr = _promote_scalars(x1, x2, "integer", "bitwise_right_shift")
    xp = _get_data_namespace(x1_arr)
    x1_new, x2_new = _arithmetic_broadcast(x1_arr, x2_arr)
    _dims = x1_new.dims
    _data = xp.bitwise_right_shift(x1_new._data, x2_new._data)
    return x1_arr._new(_dims, _data)


def bitwise_xor(
    x1: NamedArray[Any, Any] | bool | int | float | complex,
    x2: NamedArray[Any, Any] | bool | int | float | complex,
    /,
) -> NamedArray[Any, Any]:
    x1_arr, x2_arr = _promote_scalars(x1, x2, "integer or boolean", "bitwise_xor")
    xp = _get_data_namespace(x1_arr)
    x1_new, x2_new = _arithmetic_broadcast(x1_arr, x2_arr)
    _dims = x1_new.dims
    _data = xp.bitwise_xor(x1_new._data, x2_new._data)
    return x1_arr._new(_dims, _data)


def ceil(x: NamedArray[_ShapeType, _DType], /) -> NamedArray[_ShapeType, _DType]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.ceil(x._data)
    return x._new(_dims, _data)


def clip(
    x: NamedArray[Any, Any],
    /,
    min: int | float | NamedArray[Any, Any] | None = None,
    max: int | float | NamedArray[Any, Any] | None = None,
) -> NamedArray[Any, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.clip(x._data, min=min, max=max)
    return x._new(_dims, _data)


def conj(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.conj(x._data)
    return x._new(_dims, _data)


def copysign(
    x1: NamedArray[Any, Any] | bool | int | float | complex,
    x2: NamedArray[Any, Any] | bool | int | float | complex,
    /,
) -> NamedArray[Any, Any]:
    x1_arr, x2_arr = _promote_scalars(x1, x2, "real floating-point", "copysign")
    xp = _get_data_namespace(x1_arr)
    x1_new, x2_new = _arithmetic_broadcast(x1_arr, x2_arr)
    _dims = x1_new.dims
    _data = xp.copysign(x1_new._data, x2_new._data)
    return x1_arr._new(_dims, _data)


def cos(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.cos(x._data)
    return x._new(_dims, _data)


def cosh(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.cosh(x._data)
    return x._new(_dims, _data)


def divide(
    x1: NamedArray[Any, Any] | bool | int | float | complex,
    x2: NamedArray[Any, Any] | bool | int | float | complex,
    /,
) -> NamedArray[Any, Any]:
    x1_arr, x2_arr = _promote_scalars(x1, x2, "floating-point", "divide")
    xp = _get_data_namespace(x1_arr)
    x1_new, x2_new = _arithmetic_broadcast(x1_arr, x2_arr)
    _dims = x1_new.dims
    _data = xp.divide(x1_new._data, x2_new._data)
    return x1_arr._new(_dims, _data)


def exp(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.exp(x._data)
    return x._new(_dims, _data)


def expm1(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.expm1(x._data)
    return x._new(_dims, _data)


def equal(
    x1: NamedArray[Any, Any] | bool | int | float | complex,
    x2: NamedArray[Any, Any] | bool | int | float | complex,
    /,
) -> NamedArray[Any, Any]:
    x1_arr, x2_arr = _promote_scalars(x1, x2, "all", "equal")
    xp = _get_data_namespace(x1_arr)
    x1_new, x2_new = _arithmetic_broadcast(x1_arr, x2_arr)
    _dims = x1_new.dims
    _data = xp.equal(x1_new._data, x2_new._data)
    return x1_arr._new(_dims, _data)


def floor(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.floor(x._data)
    return x._new(_dims, _data)


def floor_divide(
    x1: NamedArray[Any, Any] | bool | int | float | complex,
    x2: NamedArray[Any, Any] | bool | int | float | complex,
    /,
) -> NamedArray[Any, Any]:
    x1_arr, x2_arr = _promote_scalars(x1, x2, "real numeric", "floor_divide")
    xp = _get_data_namespace(x1_arr)
    x1_new, x2_new = _arithmetic_broadcast(x1_arr, x2_arr)
    _dims = x1_new.dims
    _data = xp.floor_divide(x1_new._data, x2_new._data)
    return x1_arr._new(_dims, _data)


def greater(
    x1: NamedArray[Any, Any] | bool | int | float | complex,
    x2: NamedArray[Any, Any] | bool | int | float | complex,
    /,
) -> NamedArray[Any, Any]:
    x1_arr, x2_arr = _promote_scalars(x1, x2, "real numeric", "greater")
    xp = _get_data_namespace(x1_arr)
    x1_new, x2_new = _arithmetic_broadcast(x1_arr, x2_arr)
    _dims = x1_new.dims
    _data = xp.greater(x1_new._data, x2_new._data)
    return x1_arr._new(_dims, _data)


def greater_equal(
    x1: NamedArray[Any, Any] | bool | int | float | complex,
    x2: NamedArray[Any, Any] | bool | int | float | complex,
    /,
) -> NamedArray[Any, Any]:
    x1_arr, x2_arr = _promote_scalars(x1, x2, "real numeric", "greater_equal")
    xp = _get_data_namespace(x1_arr)
    x1_new, x2_new = _arithmetic_broadcast(x1_arr, x2_arr)
    _dims = x1_new.dims
    _data = xp.greater_equal(x1_new._data, x2_new._data)
    return x1_arr._new(_dims, _data)


def hypot(
    x1: NamedArray[Any, Any] | bool | int | float | complex,
    x2: NamedArray[Any, Any] | bool | int | float | complex,
    /,
) -> NamedArray[Any, Any]:
    x1_arr, x2_arr = _promote_scalars(x1, x2, "real floating-point", "hypot")
    xp = _get_data_namespace(x1_arr)
    x1_new, x2_new = _arithmetic_broadcast(x1_arr, x2_arr)
    _dims = x1_new.dims
    _data = xp.hypot(x1_new._data, x2_new._data)
    return x1_arr._new(_dims, _data)


def imag(
    x: NamedArray[_ShapeType, np.dtype[_SupportsImag[_ScalarType]]],
    /,  # type: ignore[type-var]
) -> NamedArray[_ShapeType, np.dtype[_ScalarType]]:
    """
    Returns the imaginary component of a complex number for each element x_i of the
    input array x.

    Parameters
    ----------
    x : NamedArray
        Input array. Should have a complex floating-point data type.

    Returns
    -------
    out : NamedArray
        An array containing the element-wise results. The returned array must have a
        floating-point data type with the same floating-point precision as x
        (e.g., if x is complex64, the returned array must have the floating-point
        data type float32).

    Examples
    --------
    >>> narr = NamedArray(("x",), np.asarray([1.0 + 2j, 2 + 4j]))
    >>> imag(narr)
    <xarray.NamedArray (x: 2)> Size: 16B
    array([2., 4.])
    """
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.imag(x._data)
    return x._new(_dims, _data)


def isfinite(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.isfinite(x._data)
    return x._new(_dims, _data)


def isinf(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.isinf(x._data)
    return x._new(_dims, _data)


def isnan(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.isnan(x._data)
    return x._new(_dims, _data)


def less(
    x1: NamedArray[Any, Any] | bool | int | float | complex,
    x2: NamedArray[Any, Any] | bool | int | float | complex,
    /,
) -> NamedArray[Any, Any]:
    x1_arr, x2_arr = _promote_scalars(x1, x2, "real numeric", "less")
    xp = _get_data_namespace(x1_arr)
    x1_new, x2_new = _arithmetic_broadcast(x1_arr, x2_arr)
    _dims = x1_new.dims
    _data = xp.less(x1_new._data, x2_new._data)
    return x1_arr._new(_dims, _data)


def less_equal(
    x1: NamedArray[Any, Any] | bool | int | float | complex,
    x2: NamedArray[Any, Any] | bool | int | float | complex,
    /,
) -> NamedArray[Any, Any]:
    x1_arr, x2_arr = _promote_scalars(x1, x2, "real numeric", "less_equal")
    xp = _get_data_namespace(x1_arr)
    x1_new, x2_new = _arithmetic_broadcast(x1_arr, x2_arr)
    _dims = x1_new.dims
    _data = xp.less_equal(x1_new._data, x2_new._data)
    return x1_arr._new(_dims, _data)


def log(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.log(x._data)
    return x._new(_dims, _data)


def log1p(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.log1p(x._data)
    return x._new(_dims, _data)


def log2(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.log2(x._data)
    return x._new(_dims, _data)


def log10(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.log10(x._data)
    return x._new(_dims, _data)


def logaddexp(
    x1: NamedArray[Any, Any] | bool | int | float | complex,
    x2: NamedArray[Any, Any] | bool | int | float | complex,
    /,
) -> NamedArray[Any, Any]:
    x1_arr, x2_arr = _promote_scalars(x1, x2, "real floating-point", "logaddexp")
    xp = _get_data_namespace(x1_arr)
    x1_new, x2_new = _arithmetic_broadcast(x1_arr, x2_arr)
    _dims = x1_new.dims
    _data = xp.logaddexp(x1_new._data, x2_new._data)
    return x1_arr._new(_dims, _data)


def logical_and(
    x1: NamedArray[Any, Any] | bool | int | float | complex,
    x2: NamedArray[Any, Any] | bool | int | float | complex,
    /,
) -> NamedArray[Any, Any]:
    x1_arr, x2_arr = _promote_scalars(x1, x2, "boolean", "logical_and")
    xp = _get_data_namespace(x1_arr)
    x1_new, x2_new = _arithmetic_broadcast(x1_arr, x2_arr)
    _dims = x1_new.dims
    _data = xp.logical_and(x1_new._data, x2_new._data)
    return x1_arr._new(_dims, _data)


def logical_not(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.logical_not(x._data)
    return x._new(_dims, _data)


def logical_or(
    x1: NamedArray[Any, Any] | bool | int | float | complex,
    x2: NamedArray[Any, Any] | bool | int | float | complex,
    /,
) -> NamedArray[Any, Any]:
    x1_arr, x2_arr = _promote_scalars(x1, x2, "boolean", "logical_or")
    xp = _get_data_namespace(x1_arr)
    x1_new, x2_new = _arithmetic_broadcast(x1_arr, x2_arr)
    _dims = x1_new.dims
    _data = xp.logical_or(x1_new._data, x2_new._data)
    return x1_arr._new(_dims, _data)


def logical_xor(
    x1: NamedArray[Any, Any] | bool | int | float | complex,
    x2: NamedArray[Any, Any] | bool | int | float | complex,
    /,
) -> NamedArray[Any, Any]:
    x1_arr, x2_arr = _promote_scalars(x1, x2, "boolean", "logical_xor")
    xp = _get_data_namespace(x1_arr)
    x1_new, x2_new = _arithmetic_broadcast(x1_arr, x2_arr)
    _dims = x1_new.dims
    _data = xp.logical_xor(x1_new._data, x2_new._data)
    return x1_arr._new(_dims, _data)


def maximum(
    x1: NamedArray[Any, Any] | bool | int | float | complex,
    x2: NamedArray[Any, Any] | bool | int | float | complex,
    /,
) -> NamedArray[Any, Any]:
    x1_arr, x2_arr = _promote_scalars(x1, x2, "real numeric", "maximum")
    xp = _get_data_namespace(x1_arr)
    x1_new, x2_new = _arithmetic_broadcast(x1_arr, x2_arr)
    _dims = x1_new.dims
    _data = xp.maximum(x1_new._data, x2_new._data)
    return x1_arr._new(_dims, _data)


def minimum(
    x1: NamedArray[Any, Any] | bool | int | float | complex,
    x2: NamedArray[Any, Any] | bool | int | float | complex,
    /,
) -> NamedArray[Any, Any]:
    x1_arr, x2_arr = _promote_scalars(x1, x2, "real numeric", "minimum")
    xp = _get_data_namespace(x1_arr)
    x1_new, x2_new = _arithmetic_broadcast(x1_arr, x2_arr)
    _dims = x1_new.dims
    _data = xp.minimum(x1_new._data, x2_new._data)
    return x1_arr._new(_dims, _data)


def multiply(
    x1: NamedArray[Any, Any] | bool | int | float | complex,
    x2: NamedArray[Any, Any] | bool | int | float | complex,
    /,
) -> NamedArray[Any, Any]:
    x1_arr, x2_arr = _promote_scalars(x1, x2, "numeric", "multiply")
    xp = _get_data_namespace(x1_arr)
    x1_new, x2_new = _arithmetic_broadcast(x1_arr, x2_arr)
    _dims = x1_new.dims
    _data = xp.multiply(x1_new._data, x2_new._data)
    return x1_arr._new(_dims, _data)


def negative(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.negative(x._data)
    return x._new(_dims, _data)


def nextafter(
    x1: NamedArray[_ShapeType, _DType] | int | float,
    x2: NamedArray[_ShapeType, _DType] | int | float,
    /,
) -> NamedArray[_ShapeType, _DType]:
    """
    Returns the next representable floating-point value for each element x1_i of the
    input array x1 in the direction of the respective element x2_i of the input
    array x2.

    Examples
    --------
    >>> import numpy as np
    >>> eps = np.finfo(np.float64).eps
    >>> x1 = NamedArray((), np.array(1.0))
    >>> x2 = 2.0
    >>> nextafter(x1, x2) == eps + 1
    <xarray.NamedArray ()> Size: 1B
    np.True_
    >>> x1 = NamedArray(("x",), np.array([1.0, 2.0]))
    >>> x2 = NamedArray(("x",), np.array([2.0, 1.0]))
    >>> expected = NamedArray(("x",), np.array([eps + 1, 2 - eps]))
    >>> actual = nextafter(x1, x2)
    >>> actual == expected
    <xarray.NamedArray (x: 2)> Size: 2B
    array([ True,  True])
    """
    x1_arr, x2_arr = _promote_scalars(x1, x2, "real floating-point", "nextafter")
    xp = _get_data_namespace(x1_arr)
    x1_new, x2_new = _arithmetic_broadcast(x1_arr, x2_arr)
    _dims = x1_new.dims
    _data = xp.nextafter(x1_new._data, x2_new._data)
    return x1_arr._new(_dims, _data)


def not_equal(
    x1: NamedArray[Any, Any] | bool | int | float | complex,
    x2: NamedArray[Any, Any] | bool | int | float | complex,
    /,
) -> NamedArray[Any, Any]:
    x1_arr, x2_arr = _promote_scalars(x1, x2, "all", "not_equal")
    xp = _get_data_namespace(x1_arr)
    x1_new, x2_new = _arithmetic_broadcast(x1_arr, x2_arr)
    _dims = x1_new.dims
    _data = xp.not_equal(x1_new._data, x2_new._data)
    return x1_arr._new(_dims, _data)


def positive(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.positive(x._data)
    return x._new(_dims, _data)


def pow(
    x1: NamedArray[Any, Any] | bool | int | float | complex,
    x2: NamedArray[Any, Any] | bool | int | float | complex,
    /,
) -> NamedArray[Any, Any]:
    x1_arr, x2_arr = _promote_scalars(x1, x2, "numeric", "pow")
    xp = _get_data_namespace(x1_arr)
    x1_new, x2_new = _arithmetic_broadcast(x1_arr, x2_arr)
    _dims = x1_new.dims
    _data = xp.pow(x1_new._data, x2_new._data)
    return x1_arr._new(_dims, _data)


def real(
    x: NamedArray[_ShapeType, np.dtype[_SupportsReal[_ScalarType]]],
    /,  # type: ignore[type-var]
) -> NamedArray[_ShapeType, np.dtype[_ScalarType]]:
    """
    Returns the real component of a complex number for each element x_i of the
    input array x.

    Parameters
    ----------
    x : NamedArray
        Input array. Should have a complex floating-point data type.

    Returns
    -------
    out : NamedArray
        An array containing the element-wise results. The returned array must have a
        floating-point data type with the same floating-point precision as x
        (e.g., if x is complex64, the returned array must have the floating-point
        data type float32).

    Examples
    --------
    >>> narr = NamedArray(("x",), np.asarray([1.0 + 2j, 2 + 4j]))
    >>> real(narr)
    <xarray.NamedArray (x: 2)> Size: 16B
    array([1., 2.])
    """
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.real(x._data)
    return x._new(_dims, _data)


def reciprocal(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.reciprocal(x._data)
    return x._new(_dims, _data)


def remainder(
    x1: NamedArray[Any, Any] | bool | int | float | complex,
    x2: NamedArray[Any, Any] | bool | int | float | complex,
    /,
) -> NamedArray[Any, Any]:
    x1_arr, x2_arr = _promote_scalars(x1, x2, "real numeric", "remainder")
    xp = _get_data_namespace(x1_arr)
    x1_new, x2_new = _arithmetic_broadcast(x1_arr, x2_arr)
    _dims = x1_new.dims
    _data = xp.remainder(x1_new._data, x2_new._data)
    return x1_arr._new(_dims, _data)


def round(x: NamedArray[_ShapeType, _DType], /) -> NamedArray[_ShapeType, _DType]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.round(x._data)
    return x._new(_dims, _data)


def sign(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.sign(x._data)
    return x._new(_dims, _data)


def signbit(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.signbit(x._data)
    return x._new(_dims, _data)


def sin(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.sin(x._data)
    return x._new(_dims, _data)


def sinh(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.sinh(x._data)
    return x._new(_dims, _data)


def sqrt(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.sqrt(x._data)
    return x._new(_dims, _data)


def square(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.square(x._data)
    return x._new(_dims, _data)


def subtract(
    x1: NamedArray[Any, Any] | bool | int | float | complex,
    x2: NamedArray[Any, Any] | bool | int | float | complex,
    /,
) -> NamedArray[Any, Any]:
    x1_arr, x2_arr = _promote_scalars(x1, x2, "numeric", "subtract")
    xp = _get_data_namespace(x1_arr)
    x1_new, x2_new = _arithmetic_broadcast(x1_arr, x2_arr)
    _dims = x1_new.dims
    _data = xp.subtract(x1_new._data, x2_new._data)
    return x1_arr._new(_dims, _data)


def tan(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.tan(x._data)
    return x._new(_dims, _data)


def tanh(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.tanh(x._data)
    return x._new(_dims, _data)


def trunc(x: NamedArray[_ShapeType, _DType], /) -> NamedArray[_ShapeType, _DType]:
    xp = _get_data_namespace(x)
    _dims = x.dims
    _data = xp.trunc(x._data)
    return x._new(_dims, _data)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
