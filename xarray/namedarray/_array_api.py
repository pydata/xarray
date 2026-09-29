from __future__ import annotations

from types import ModuleType
from typing import Any, overload

import numpy as np

from xarray.namedarray._typing import (
    Axes,
    Axis,
    Default,
    DimType,
    DType,
    ScalarType,
    ShapeType,
    SupportsImag,
    SupportsReal,
    _default,
    arrayapi,
)
from xarray.namedarray.core import NamedArray


def _get_data_namespace(x: NamedArray[Any, Any, Any]) -> ModuleType:
    if isinstance(x._data, arrayapi):
        return x._data.__array_namespace__()

    return np


# %% Creation Functions


def astype(
    x: NamedArray[ShapeType, Any, DimType], dtype: DType, /, *, copy: bool = True
) -> NamedArray[ShapeType, DType, DimType]:
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
    if isinstance(x._data, arrayapi):
        xp = x._data.__array_namespace__()
        return x._new(data=xp.astype(x._data, dtype, copy=copy))

    # np.astype doesn't exist yet:
    return x._new(data=x._data.astype(dtype, copy=copy))  # type: ignore[attr-defined]


# %% Elementwise Functions


def imag(
    x: NamedArray[ShapeType, np.dtype[SupportsImag[ScalarType]], DimType],  # type: ignore[type-var]
    /,
) -> NamedArray[ShapeType, np.dtype[ScalarType], DimType]:
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


def real(
    x: NamedArray[ShapeType, np.dtype[SupportsReal[ScalarType]], DimType],  # type: ignore[type-var]
    /,
) -> NamedArray[ShapeType, np.dtype[ScalarType], DimType]:
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


# %% Manipulation functions


@overload
def expand_dims(
    x: NamedArray[Any, DType, DimType],
    /,
    *,
    dim: DimType,
    axis: Axis = ...,
) -> NamedArray[Any, DType, DimType]: ...


@overload
def expand_dims(
    x: NamedArray[Any, DType, DimType],
    /,
    *,
    dim: Default = ...,
    axis: Axis = ...,
) -> NamedArray[Any, DType, DimType | str]: ...


def expand_dims(
    x: NamedArray[Any, DType, DimType],
    /,
    *,
    dim: DimType | Default = _default,
    axis: Axis = 0,
) -> NamedArray[Any, DType, DimType] | NamedArray[Any, DType, DimType | str]:
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
    actual_dim: DimType | str = f"dim_{len(dims)}" if dim is _default else dim
    d: list[DimType | str] = list(dims)
    d.insert(axis, actual_dim)
    out = x._new(dims=tuple(d), data=xp.expand_dims(x._data, axis=axis))
    return out


def permute_dims(
    x: NamedArray[Any, DType, DimType], axes: Axes
) -> NamedArray[Any, DType, DimType]:
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
    if isinstance(x._data, arrayapi):
        xp = _get_data_namespace(x)
        out = x._new(dims=new_dims, data=xp.permute_dims(x._data, axes))
    else:
        out = x._new(dims=new_dims, data=x._data.transpose(axes))  # type: ignore[attr-defined]
    return out
