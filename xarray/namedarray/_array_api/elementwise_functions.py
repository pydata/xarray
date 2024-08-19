from __future__ import annotations

from types import ModuleType
from typing import Any, overload

import numpy as np

from xarray.namedarray._array_api._utils import _get_data_namespace
from xarray.namedarray._typing import (
    Default,
    _arrayapi,
    _arrayfunction_or_api,
    _ArrayLike,
    _Axes,
    _Axis,
    _AxisLike,
    _default,
    _Dim,
    _Dims,
    _DimsLike,
    _DType,
    _dtype,
    _ScalarType,
    _Shape,
    _ShapeType,
    _SupportsImag,
    _SupportsReal,
    duckarray,
)
from xarray.namedarray.core import (
    NamedArray,
    _dims_to_axis,
    _get_remaining_dims,
)
from xarray.namedarray.utils import (
    to_0d_object_array,
)


def abs(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.abs(x._data))
    return out


def acos(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.acos(x._data))
    return out


def acosh(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.acosh(x._data))
    return out


def add(x1, x2, /):
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    out = x1._new(data=xp.add(x1._data, x2._data))
    return out


def asin(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.asin(x._data))
    return out


def asinh(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.asinh(x._data))
    return out


def atan(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.atan(x._data))
    return out


def atan2(x1, x2, /):
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    out = x1._new(data=xp.atan2(x1._data, x2._data))
    return out


def atanh(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.atanh(x._data))
    return out


def bitwise_and(x1, x2, /):
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    out = x1._new(data=xp.bitwise_and(x1._data, x2._data))
    return out


def bitwise_invert(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.bitwise_invert(x._data))
    return out


def bitwise_left_shift(x1, x2, /):
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    out = x1._new(data=xp.bitwise_left_shift(x1._data, x2._data))
    return out


def bitwise_or(x1, x2, /):
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    out = x1._new(data=xp.bitwise_or(x1._data, x2._data))
    return out


def bitwise_right_shift(x1, x2, /):
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    out = x1._new(data=xp.bitwise_right_shift(x1._data, x2._data))
    return out


def bitwise_xor(x1, x2, /):
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    out = x1._new(data=xp.bitwise_xor(x1._data, x2._data))
    return out


def ceil(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.ceil(x._data))
    return out


def conj(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.conj(x._data))
    return out


def cos(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.cos(x._data))
    return out


def cosh(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.cosh(x._data))
    return out


def divide(x1, x2, /):
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    out = x1._new(data=xp.divide(x1._data, x2._data))
    return out


def exp(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.exp(x._data))
    return out


def expm1(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.expm1(x._data))
    return out


def equal(x1, x2, /):
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    out = x1._new(data=xp.equal(x1._data, x2._data))
    return out


def floor(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.floor(x._data))
    return out


def floor_divide(x1, x2, /):
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    out = x1._new(data=xp.floor_divide(x1._data, x2._data))
    return out


def greater(x1, x2, /):
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    out = x1._new(data=xp.greater(x1._data, x2._data))
    return out


def greater_equal(x1, x2, /):
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    out = x1._new(data=xp.greater_equal(x1._data, x2._data))
    return out


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
    out = x._new(data=xp.imag(x._data))
    return out


def isfinite(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.isfinite(x._data))
    return out


def isinf(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.isinf(x._data))
    return out


def isnan(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.isnan(x._data))
    return out


def less(x1, x2, /):
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    out = x1._new(data=xp.less(x1._data, x2._data))
    return out


def less_equal(x1, x2, /):
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    out = x1._new(data=xp.less_equal(x1._data, x2._data))
    return out


def log(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.log(x._data))
    return out


def log1p(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.log1p(x._data))
    return out


def log2(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.log2(x._data))
    return out


def log10(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.log10(x._data))
    return out


def logaddexp(x1, x2, /):
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    out = x1._new(data=xp.logaddexp(x1._data, x2._data))
    return out


def logical_and(x1, x2, /):
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    out = x1._new(data=xp.logical_and(x1._data, x2._data))
    return out


def logical_not(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.logical_not(x._data))
    return out


def logical_or(x1, x2, /):
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    out = x1._new(data=xp.logical_or(x1._data, x2._data))
    return out


def logical_xor(x1, x2, /):
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    out = x1._new(data=xp.logical_xor(x1._data, x2._data))
    return out


def multiply(x1, x2, /):
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    out = x1._new(data=xp.multiply(x1._data, x2._data))
    return out


def negative(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.negative(x._data))
    return out


def not_equal(x1, x2, /):
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    out = x1._new(data=xp.not_equal(x1._data, x2._data))
    return out


def positive(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.positive(x._data))
    return out


def pow(x1, x2, /):
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    out = x1._new(data=xp.pow(x1._data, x2._data))
    return out


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
    out = x._new(data=xp.real(x._data))
    return out


def remainder(x1, x2, /):
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    out = x1._new(data=xp.remainder(x1._data, x2._data))
    return out


def round(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.round(x._data))
    return out


def sign(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.sign(x._data))
    return out


def sin(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.sin(x._data))
    return out


def sinh(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.sinh(x._data))
    return out


def sqrt(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.sqrt(x._data))
    return out


def square(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.square(x._data))
    return out


def subtract(x1, x2, /):
    xp = _get_data_namespace(x1)
    # TODO: Handle attrs? will get x1 now
    out = x1._new(data=xp.subtract(x1._data, x2._data))
    return out


def tan(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.tan(x._data))
    return out


def tanh(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.tanh(x._data))
    return out


def trunc(x, /):
    xp = _get_data_namespace(x)
    out = x._new(data=xp.trunc(x._data))
    return out
