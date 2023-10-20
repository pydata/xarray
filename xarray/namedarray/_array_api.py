import warnings
from types import ModuleType
from typing import Any

import numpy as np

from xarray.namedarray._typing import (
    _arrayapi,
    _AxisLike,
    _Dims,
    _DType,
    _ScalarType,
    _ShapeType,
    _SupportsImag,
    _SupportsReal,
    duckarray,
)
from xarray.namedarray.core import NamedArray, _dims_to_axis, _get_remaining_dims

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        r"The numpy.array_api submodule is still experimental",
        category=UserWarning,
    )
    import numpy.array_api as nxp


def _get_data_namespace(x: NamedArray[Any, Any]) -> ModuleType:
    if isinstance(x._data, _arrayapi):
        return x._data.__array_namespace__()

    return np


def _to_nxp(
    x: duckarray[_ShapeType, _DType]
) -> tuple[ModuleType, _arrayapi[_ShapeType, _DType]]:
    return nxp, nxp.asarray(x)


# %% Creation Functions


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
    >>> narr = NamedArray(("x",), nxp.asarray([1.5, 2.5]))
    >>> narr
    <xarray.NamedArray (x: 2)>
    Array([1.5, 2.5], dtype=float64)
    >>> astype(narr, np.dtype(np.int32))
    <xarray.NamedArray (x: 2)>
    Array([1, 2], dtype=int32)
    """
    if isinstance(x._data, _arrayapi):
        xp = x._data.__array_namespace__()
        return x._new(data=xp.astype(x._data, dtype, copy=copy))

    # np.astype doesn't exist yet:
    return x._new(data=x._data.astype(dtype, copy=copy))  # type: ignore[attr-defined]


# %% Elementwise Functions


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
    >>> narr = NamedArray(("x",), np.asarray([1.0 + 2j, 2 + 4j]))  # TODO: Use nxp
    >>> imag(narr)
    <xarray.NamedArray (x: 2)>
    array([2., 4.])
    """
    xp = _get_data_namespace(x)
    out = x._new(data=xp.imag(x._data))
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
    >>> narr = NamedArray(("x",), np.asarray([1.0 + 2j, 2 + 4j]))  # TODO: Use nxp
    >>> real(narr)
    <xarray.NamedArray (x: 2)>
    array([1., 2.])
    """
    xp = _get_data_namespace(x)
    out = x._new(data=xp.real(x._data))
    return out


# %% Statistical Functions


def mean(
    x: NamedArray[Any, _DType],
    /,
    *,
    dims: _Dims | None = None,
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


# if __name__ == "__main__":
#     import doctest

#     doctest.testmod()
