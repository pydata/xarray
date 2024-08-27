from __future__ import annotations

import numpy as np

from xarray.namedarray._array_api._utils import (
    _atleast_0d,
    _get_broadcasted_dims,
    _get_data_namespace,
)
from xarray.namedarray._typing import (
    _ScalarType,
    _ShapeType,
    _SupportsImag,
    _SupportsReal,
)
from xarray.namedarray.core import NamedArray


def abs(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.abs(x._data), xp)
    return x._new(data=_data)


def acos(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.acos(x._data), xp)
    return x._new(data=_data)


def acosh(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.acosh(x._data), xp)
    return x._new(data=_data)


def add(x1: NamedArray, x2: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    _data = xp.add(x1._data, x2._data)
    return x1._new(data=_data)


def asin(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.asin(x._data), xp)
    return x._new(data=_data)


def asinh(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.asinh(x._data), xp)
    return x._new(data=_data)


def atan(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.atan(x._data), xp)
    return x._new(data=_data)


def atan2(x1: NamedArray, x2: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    _data = _atleast_0d(xp.atan2(x1._data, x2._data), xp)
    return x1._new(data=_data)


def atanh(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.atanh(x._data), xp)
    return x._new(data=_data)


def bitwise_and(x1: NamedArray, x2: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    _data = _atleast_0d(xp.bitwise_and(x1._data, x2._data), xp)
    return x1._new(data=_data)


def bitwise_invert(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.bitwise_invert(x._data), xp)
    return x._new(data=_data)


def bitwise_left_shift(x1: NamedArray, x2: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    _data = _atleast_0d(xp.bitwise_left_shift(x1._data, x2._data), xp)
    return x1._new(data=_data)


def bitwise_or(x1: NamedArray, x2: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    _data = _atleast_0d(xp.bitwise_or(x1._data, x2._data), xp)
    return x1._new(data=_data)


def bitwise_right_shift(x1: NamedArray, x2: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    _data = _atleast_0d(xp.bitwise_right_shift(x1._data, x2._data), xp)
    return x1._new(data=_data)


def bitwise_xor(x1: NamedArray, x2: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    _data = _atleast_0d(xp.bitwise_xor(x1._data, x2._data), xp)
    return x1._new(data=_data)


def ceil(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.ceil(x._data), xp)
    return x._new(data=_data)


def clip(
    x: NamedArray,
    /,
    min: int | float | NamedArray | None = None,
    max: int | float | NamedArray | None = None,
) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.clip(x._data, min=min, max=max), xp)
    return x._new(data=_data)


def conj(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.conj(x._data), xp)
    return x._new(data=_data)


def copysign(x1: NamedArray, x2: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    _data = _atleast_0d(xp.copysign(x1._data, x2._data), xp)
    return x1._new(data=_data)


def cos(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.cos(x._data), xp)
    return x._new(data=_data)


def cosh(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.cosh(x._data), xp)
    return x._new(data=_data)


def divide(x1: NamedArray, x2: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    _data = _atleast_0d(xp.divide(x1._data, x2._data), xp)
    return x1._new(data=_data)


def exp(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.exp(x._data), xp)
    return x._new(data=_data)


def expm1(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.expm1(x._data), xp)
    return x._new(data=_data)


def equal(x1: NamedArray, x2: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x1)
    _dims, _ = _get_broadcasted_dims(x1, x2)
    _data = _atleast_0d(xp.equal(x1._data, x2._data), xp)
    return NamedArray(_dims, _data)


def floor(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.floor(x._data), xp)
    return x._new(data=_data)


def floor_divide(x1: NamedArray, x2: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    _data = _atleast_0d(xp.floor_divide(x1._data, x2._data), xp)
    return x1._new(data=_data)


def greater(x1: NamedArray, x2: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    _data = _atleast_0d(xp.greater(x1._data, x2._data), xp)
    return x1._new(data=_data)


def greater_equal(x1: NamedArray, x2: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    _data = _atleast_0d(xp.greater_equal(x1._data, x2._data), xp)
    return x1._new(data=_data)


def hypot(x1: NamedArray, x2: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    _data = _atleast_0d(xp.hypot(x1._data, x2._data), xp)
    return x1._new(data=_data)


def imag(
    x: NamedArray[_ShapeType, np.dtype[_SupportsImag[_ScalarType]]], /  # type: ignore[type-var]
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
    _data = _atleast_0d(xp.imag(x._data), xp)
    return x._new(data=_data)


def isfinite(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.isfinite(x._data), xp)
    return x._new(data=_data)


def isinf(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.isinf(x._data), xp)
    return x._new(data=_data)


def isnan(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.isnan(x._data), xp)
    return x._new(data=_data)


def less(x1: NamedArray, x2: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    _data = _atleast_0d(xp.less(x1._data, x2._data), xp)
    return x1._new(data=_data)


def less_equal(x1: NamedArray, x2: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    _data = _atleast_0d(xp.less_equal(x1._data, x2._data), xp)
    return x1._new(data=_data)


def log(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.log(x._data), xp)
    return x._new(data=_data)


def log1p(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.log1p(x._data), xp)
    return x._new(data=_data)


def log2(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.log2(x._data), xp)
    return x._new(data=_data)


def log10(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.log10(x._data), xp)
    return x._new(data=_data)


def logaddexp(x1: NamedArray, x2: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    _data = _atleast_0d(xp.logaddexp(x1._data, x2._data), xp)
    return x1._new(data=_data)


def logical_and(x1: NamedArray, x2: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    _data = _atleast_0d(xp.logical_and(x1._data, x2._data), xp)
    return x1._new(data=_data)


def logical_not(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.logical_not(x._data), xp)
    return x._new(data=_data)


def logical_or(x1: NamedArray, x2: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    _data = _atleast_0d(xp.logical_or(x1._data, x2._data), xp)
    return x1._new(data=_data)


def logical_xor(x1: NamedArray, x2: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    _data = _atleast_0d(xp.logical_xor(x1._data, x2._data), xp)
    return x1._new(data=_data)


def maximum(x1: NamedArray, x2: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    _data = _atleast_0d(xp.maximum(x1._data, x2._data), xp)
    return x1._new(data=_data)


def minimum(x1: NamedArray, x2: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    _data = _atleast_0d(xp.minimum(x1._data, x2._data), xp)
    return x1._new(data=_data)


def multiply(x1: NamedArray, x2: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    _data = _atleast_0d(xp.multiply(x1._data, x2._data), xp)
    return x1._new(data=_data)


def negative(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.negative(x._data), xp)
    return x._new(data=_data)


def not_equal(x1: NamedArray, x2: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    _data = _atleast_0d(xp.not_equal(x1._data, x2._data), xp)
    return x1._new(data=_data)


def positive(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.positive(x._data), xp)
    return x._new(data=_data)


def pow(x1: NamedArray, x2: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    _data = _atleast_0d(xp.pow(x1._data, x2._data), xp)
    return x1._new(data=_data)


def real(
    x: NamedArray[_ShapeType, np.dtype[_SupportsReal[_ScalarType]]], /  # type: ignore[type-var]
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
    _data = _atleast_0d(xp.real(x._data), xp)
    return x._new(data=_data)


def remainder(x1: NamedArray, x2: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    _data = _atleast_0d(xp.remainder(x1._data, x2._data), xp)
    return x1._new(data=_data)


def round(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.round(x._data), xp)
    return x._new(data=_data)


def sign(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.sign(x._data), xp)
    return x._new(data=_data)


def signbit(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.signbit(x._data), xp)
    return x._new(data=_data)


def sin(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.sin(x._data), xp)
    return x._new(data=_data)


def sinh(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.sinh(x._data), xp)
    return x._new(data=_data)


def sqrt(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.sqrt(x._data), xp)
    return x._new(data=_data)


def square(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.square(x._data), xp)
    return x._new(data=_data)


def subtract(x1: NamedArray, x2: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    _data = _atleast_0d(xp.subtract(x1._data, x2._data), xp)
    return x1._new(data=_data)


def tan(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.tan(x._data), xp)
    return x._new(data=_data)


def tanh(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.tanh(x._data), xp)
    return x._new(data=_data)


def trunc(x, /):
    xp = _get_data_namespace(x)
    _data = _atleast_0d(xp.trunc(x._data), xp)
    return x._new(data=_data)
