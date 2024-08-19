from __future__ import annotations

from types import ModuleType
from typing import Any, overload

import numpy as np

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


# %% Helper functions
def _maybe_default_namespace(xp: ModuleType | None = None) -> ModuleType:
    return np if xp is None else xp


def _get_data_namespace(x: NamedArray[Any, Any]) -> ModuleType:
    if isinstance(x._data, _arrayapi):
        return x._data.__array_namespace__()

    return _maybe_default_namespace()


def _get_namespace_dtype(dtype: _dtype) -> ModuleType:
    xp = __import__(dtype.__module__)
    return xp


# %% array_api version
__array_api_version__ = "2023.12"


# %% Constants
e = np.e
inf = np.inf
nan = np.nan
newaxis = np.newaxis
pi = np.pi


# %% Creation Functions
def _infer_dims(
    shape: _Shape,
    dims: _DimsLike | Default = _default,
) -> _DimsLike:
    if dims is _default:
        return tuple(f"dim_{n}" for n in range(len(shape)))
    else:
        return dims


def arange(
    start: int | float,
    /,
    stop: int | float | None = None,
    step: int | float = 1,
    *,
    dtype: _DType | None = None,
    device=None,
) -> NamedArray[_ShapeType, _DType]:
    xp = _maybe_default_namespace()
    _data = xp.arange(start, stop=stop, step=step, dtype=dtype, device=device)
    _dims = _infer_dims(_data.shape)
    return NamedArray(_dims, _data)


@overload
def asarray(
    obj: duckarray[_ShapeType, Any],
    /,
    *,
    dtype: _DType,
    device=...,
    copy: bool | None = ...,
    dims: _DimsLike = ...,
) -> NamedArray[_ShapeType, _DType]: ...
@overload
def asarray(
    obj: _ArrayLike,
    /,
    *,
    dtype: _DType,
    device=...,
    copy: bool | None = ...,
    dims: _DimsLike = ...,
) -> NamedArray[Any, _DType]: ...
@overload
def asarray(
    obj: duckarray[_ShapeType, _DType],
    /,
    *,
    dtype: None,
    device=None,
    copy: bool | None = None,
    dims: _DimsLike = ...,
) -> NamedArray[_ShapeType, _DType]: ...
@overload
def asarray(
    obj: _ArrayLike,
    /,
    *,
    dtype: None,
    device=...,
    copy: bool | None = ...,
    dims: _DimsLike = ...,
) -> NamedArray[Any, _DType]: ...
def asarray(
    obj: duckarray[_ShapeType, _DType] | _ArrayLike,
    /,
    *,
    dtype: _DType | None = None,
    device=None,
    copy: bool | None = None,
    dims: _DimsLike = _default,
) -> NamedArray[_ShapeType, _DType] | NamedArray[Any, Any]:
    """
    Create a Named array from an array-like object.

    Parameters
    ----------
    dims : str or iterable of str
        Name(s) of the dimension(s).
    data : T_DuckArray or ArrayLike
        The actual data that populates the array. Should match the
        shape specified by `dims`.
    attrs : dict, optional
        A dictionary containing any additional information or
        attributes you want to store with the array.
        Default is None, meaning no attributes will be stored.
    """
    data = obj
    if isinstance(data, NamedArray):
        if copy:
            return data.copy()
        else:
            return data

    # TODO: dask.array.ma.MaskedArray also exists, better way?
    if isinstance(data, np.ma.MaskedArray):
        mask = np.ma.getmaskarray(data)  # type: ignore[no-untyped-call]
        if mask.any():
            # TODO: requires refactoring/vendoring xarray.core.dtypes and
            # xarray.core.duck_array_ops
            raise NotImplementedError("MaskedArray is not supported yet")

        _dims = _infer_dims(data.shape, dims)
        return NamedArray(_dims, data)

    if isinstance(data, _arrayfunction_or_api):
        _dims = _infer_dims(data.shape, dims)
        return NamedArray(_dims, data)

    if isinstance(data, tuple):
        _data = to_0d_object_array(data)
        _dims = _infer_dims(_data.shape, dims)
        return NamedArray(_dims, _data)

    # validate whether the data is valid data types.
    _data = np.asarray(data, dtype=dtype, device=device, copy=copy)
    _dims = _infer_dims(_data.shape, dims)
    return NamedArray(_dims, _data)


def full(
    shape: _Shape,
    fill_value: bool | int | float | complex,
    *,
    dtype: _DType | None = None,
    device=None,
) -> NamedArray[_ShapeType, _DType]:
    xp = _maybe_default_namespace()
    _data = xp.full(shape, fill_value, dtype=dtype, device=device)
    _dims = _infer_dims(_data.shape)
    return NamedArray(_dims, _data)


def linspace(
    start: int | float | complex,
    stop: int | float | complex,
    /,
    num: int,
    *,
    dtype: _DType | None = None,
    device=None,
    endpoint: bool = True,
) -> NamedArray[_ShapeType, _DType]:
    xp = _maybe_default_namespace()
    _data = xp.linspace(
        start, stop, num=num, dtype=dtype, device=device, endpoint=endpoint
    )
    _dims = _infer_dims(_data.shape)
    return NamedArray(_dims, _data)


def ones(
    shape: _Shape, *, dtype: _DType | None = None, device=None
) -> NamedArray[_ShapeType, _DType]:
    return full(shape, 1, dtype=dtype, device=device)


def zeros(
    shape: _Shape, *, dtype: _DType | None = None, device=None
) -> NamedArray[_ShapeType, _DType]:
    return full(shape, 0, dtype=dtype, device=device)


# %% Data types
# TODO: should delegate to underlying array? Cubed doesn't at the moment.
int8 = np.int8
int16 = np.int16
int32 = np.int32
int64 = np.int64
uint8 = np.uint8
uint16 = np.uint16
uint32 = np.uint32
uint64 = np.uint64
float32 = np.float32
float64 = np.float64
complex64 = np.complex64
complex128 = np.complex128
bool = np.bool

_all_dtypes = (
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float32,
    float64,
    complex64,
    complex128,
    bool,
)
_boolean_dtypes = (bool,)
_real_floating_dtypes = (float32, float64)
_floating_dtypes = (float32, float64, complex64, complex128)
_complex_floating_dtypes = (complex64, complex128)
_integer_dtypes = (int8, int16, int32, int64, uint8, uint16, uint32, uint64)
_signed_integer_dtypes = (int8, int16, int32, int64)
_unsigned_integer_dtypes = (uint8, uint16, uint32, uint64)
_integer_or_boolean_dtypes = (
    bool,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)
_real_numeric_dtypes = (
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
_numeric_dtypes = (
    float32,
    float64,
    complex64,
    complex128,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)

_dtype_categories = {
    "all": _all_dtypes,
    "real numeric": _real_numeric_dtypes,
    "numeric": _numeric_dtypes,
    "integer": _integer_dtypes,
    "integer or boolean": _integer_or_boolean_dtypes,
    "boolean": _boolean_dtypes,
    "real floating-point": _floating_dtypes,
    "complex floating-point": _complex_floating_dtypes,
    "floating-point": _floating_dtypes,
}

# %% Data type functions


def astype(
    x: NamedArray[_ShapeType, Any], dtype: _DType, /, *, copy: bool = True
) -> NamedArray[_ShapeType, _DType]:
    """
    Copies an array to a specified data type irrespective of Type Promotion Rules rules.

    Parameters
    ----------
    x : NamedArray
        Array to cast.
    dtype : _DType
        Desired data type.
    copy : bool, optional
        Specifies whether to copy an array when the specified dtype matches the data
        type of the input array x.
        If True, a newly allocated array must always be returned.
        If False and the specified dtype matches the data type of the input array,
        the input array must be returned; otherwise, a newly allocated array must be
        returned. Default: True.

    Returns
    -------
    out : NamedArray
        An array having the specified data type. The returned array must have the
        same shape as x.

    Examples
    --------
    >>> narr = NamedArray(("x",), np.asarray([1.5, 2.5]))
    >>> narr
    <xarray.NamedArray (x: 2)> Size: 16B
    array([1.5, 2.5])
    >>> astype(narr, np.dtype(np.int32))
    <xarray.NamedArray (x: 2)> Size: 8B
    array([1, 2], dtype=int32)
    """
    if isinstance(x._data, _arrayapi):
        xp = x._data.__array_namespace__()
        return x._new(data=xp.astype(x._data, dtype, copy=copy))

    # np.astype doesn't exist yet:
    return x._new(data=x._data.astype(dtype, copy=copy))  # type: ignore[attr-defined]


def can_cast(from_: _dtype | NamedArray, to: _dtype, /) -> bool:
    if isinstance(from_, NamedArray):
        xp = _get_data_namespace(from_)
        from_ = from_.dtype
        return xp.can_cast(from_, to)
    else:
        xp = _get_namespace_dtype(from_)
        return xp.can_cast(from_, to)


def finfo(type: _dtype | NamedArray[Any, Any], /):
    if isinstance(type, NamedArray):
        xp = _get_data_namespace(type)
        return xp.finfo(type._data)
    else:
        xp = _get_namespace_dtype(type)
        return xp.finfo(type)


def iinfo(type: _dtype | NamedArray[Any, Any], /):
    if isinstance(type, NamedArray):
        xp = _get_data_namespace(type)
        return xp.iinfo(type._data)
    else:
        xp = _get_namespace_dtype(type)
        return xp.iinfo(type)


def isdtype(dtype: _dtype, kind: _dtype | str | tuple[_dtype | str, ...]) -> bool:
    xp = _get_namespace_dtype(type)
    return xp.isdtype(dtype, kind)


def result_type(*arrays_and_dtypes: NamedArray[Any, Any] | _dtype) -> _dtype:
    # TODO: Empty arg?
    arr_or_dtype = arrays_and_dtypes[0]
    if isinstance(arr_or_dtype, NamedArray):
        xp = _get_data_namespace(arr_or_dtype)
    else:
        xp = _get_namespace_dtype(arr_or_dtype)

    return xp.result_type(
        *(a.dtype if isinstance(a, NamedArray) else a for a in arrays_and_dtypes)
    )


# %% Elementwise Functions
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


# %% Manipulation functions
def expand_dims(
    x: NamedArray[Any, _DType],
    /,
    *,
    dim: _Dim | Default = _default,
    axis: _Axis = 0,
) -> NamedArray[Any, _DType]:
    """
    Expands the shape of an array by inserting a new dimension of size one at the
    position specified by dims.

    Parameters
    ----------
    x :
        Array to expand.
    dim :
        Dimension name. New dimension will be stored in the axis position.
    axis :
        (Not recommended) Axis position (zero-based). Default is 0.

    Returns
    -------
        out :
            An expanded output array having the same data type as x.

    Examples
    --------
    >>> x = NamedArray(("x", "y"), np.asarray([[1.0, 2.0], [3.0, 4.0]]))
    >>> expand_dims(x)
    <xarray.NamedArray (dim_2: 1, x: 2, y: 2)> Size: 32B
    array([[[1., 2.],
            [3., 4.]]])
    >>> expand_dims(x, dim="z")
    <xarray.NamedArray (z: 1, x: 2, y: 2)> Size: 32B
    array([[[1., 2.],
            [3., 4.]]])
    """
    xp = _get_data_namespace(x)
    dims = x.dims
    if dim is _default:
        dim = f"dim_{len(dims)}"
    d = list(dims)
    d.insert(axis, dim)
    out = x._new(dims=tuple(d), data=xp.expand_dims(x._data, axis=axis))
    return out


def permute_dims(x: NamedArray[Any, _DType], axes: _Axes) -> NamedArray[Any, _DType]:
    """
    Permutes the dimensions of an array.

    Parameters
    ----------
    x :
        Array to permute.
    axes :
        Permutation of the dimensions of x.

    Returns
    -------
    out :
        An array with permuted dimensions. The returned array must have the same
        data type as x.

    """

    dims = x.dims
    new_dims = tuple(dims[i] for i in axes)
    if isinstance(x._data, _arrayapi):
        xp = _get_data_namespace(x)
        out = x._new(dims=new_dims, data=xp.permute_dims(x._data, axes))
    else:
        out = x._new(dims=new_dims, data=x._data.transpose(axes))  # type: ignore[attr-defined]
    return out


def reshape(x, /, shape: _Shape, *, copy: bool | None = None):
    xp = _get_data_namespace(x)
    _data = xp.reshape(x._data, shape)
    out = asarray(_data, copy=copy)
    # TODO: Have better control where the dims went.
    # TODO: If reshaping should we save the dims?
    # TODO: What's the xarray equivalent?
    return out


# %% Statistical Functions
def mean(
    x: NamedArray[Any, _DType],
    /,
    *,
    dims: _Dims | Default = _default,
    keepdims: bool = False,
    axis: _AxisLike | None = None,
) -> NamedArray[Any, _DType]:
    """
    Calculates the arithmetic mean of the input array x.

    Parameters
    ----------
    x :
        Should have a real-valued floating-point data type.
    dims :
        Dim or dims along which arithmetic means must be computed. By default,
        the mean must be computed over the entire array. If a tuple of hashables,
        arithmetic means must be computed over multiple axes.
        Default: None.
    keepdims :
        if True, the reduced axes (dimensions) must be included in the result
        as singleton dimensions, and, accordingly, the result must be compatible
        with the input array (see Broadcasting). Otherwise, if False, the
        reduced axes (dimensions) must not be included in the result.
        Default: False.
    axis :
        Axis or axes along which arithmetic means must be computed. By default,
        the mean must be computed over the entire array. If a tuple of integers,
        arithmetic means must be computed over multiple axes.
        Default: None.

    Returns
    -------
    out :
        If the arithmetic mean was computed over the entire array,
        a zero-dimensional array containing the arithmetic mean; otherwise,
        a non-zero-dimensional array containing the arithmetic means.
        The returned array must have the same data type as x.

    Examples
    --------
    >>> x = NamedArray(("x", "y"), nxp.asarray([[1.0, 2.0], [3.0, 4.0]]))
    >>> mean(x).data
    Array(2.5, dtype=float64)
    >>> mean(x, dims=("x",)).data
    Array([2., 3.], dtype=float64)

    Using keepdims:

    >>> mean(x, dims=("x",), keepdims=True)
    <xarray.NamedArray (x: 1, y: 2)>
    Array([[2., 3.]], dtype=float64)
    >>> mean(x, dims=("y",), keepdims=True)
    <xarray.NamedArray (x: 2, y: 1)>
    Array([[1.5],
           [3.5]], dtype=float64)
    """
    xp = _get_data_namespace(x)
    axis_ = _dims_to_axis(x, dims, axis)
    d = xp.mean(x._data, axis=axis_, keepdims=False)  # We fix keepdims later
    # TODO: Why do we need to do the keepdims ourselves?
    dims_, data_ = _get_remaining_dims(x, d, axis_, keepdims=keepdims)
    out = x._new(dims=dims_, data=data_)
    return out


# %% Utility functions
def all(
    x,
    /,
    *,
    dims: _Dims | Default = _default,
    keepdims: bool = False,
    axis: _AxisLike | None = None,
) -> NamedArray[Any, _DType]:
    xp = _get_data_namespace(x)
    axis_ = _dims_to_axis(x, dims, axis)
    d = xp.all(x._data, axis=axis_, keepdims=False)  # We fix keepdims later
    dims_, data_ = _get_remaining_dims(x, d, axis_, keepdims=keepdims)
    out = x._new(dims=dims_, data=data_)
    return out


def any(
    x,
    /,
    *,
    dims: _Dims | Default = _default,
    keepdims: bool = False,
    axis: _AxisLike | None = None,
) -> NamedArray[Any, _DType]:
    xp = _get_data_namespace(x)
    axis_ = _dims_to_axis(x, dims, axis)
    d = xp.any(x._data, axis=axis_, keepdims=False)  # We fix keepdims later
    dims_, data_ = _get_remaining_dims(x, d, axis_, keepdims=keepdims)
    out = x._new(dims=dims_, data=data_)
    return out
