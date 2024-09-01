from __future__ import annotations

from typing import Any

from xarray.namedarray._array_api._utils import (
    _dims_to_axis,
    _get_data_namespace,
    _get_remaining_dims,
)
from xarray.namedarray._typing import (
    Default,
    _AxisLike,
    _default,
    _Dim,
    _Dims,
    _DType,
    _ShapeType,
)
from xarray.namedarray.core import (
    NamedArray,
)


def cumulative_sum(
    x: NamedArray[_ShapeType, _DType],
    /,
    *,
    dim: _Dim | Default = _default,
    dtype: _DType | None = None,
    include_initial: bool = False,
    axis: int | None = None,
) -> NamedArray[_ShapeType, _DType]:
    xp = _get_data_namespace(x)

    a = _dims_to_axis(x, dim, axis)
    _axis = a if a is None else a[0]
    try:
        _data = xp.cumulative_sum(
            x._data, axis=_axis, dtype=dtype, include_initial=include_initial
        )
    except AttributeError:
        # Use np.cumsum until new name is introduced:
        # np.cumsum does not support include_initial
        if include_initial:
            if axis < 0:
                axis += x.ndim
            d = xp.concat(
                [
                    xp.zeros(
                        x.shape[:axis] + (1,) + x.shape[axis + 1 :], dtype=x.dtype
                    ),
                    x._data,
                ],
                axis=axis,
            )
        else:
            d = x._data
        _data = xp.cumsum(d, axis=axis, dtype=dtype)
    return x._new(dims=x.dims, data=_data)


def max(
    x: NamedArray[Any, _DType],
    /,
    *,
    dims: _Dims | Default = _default,
    keepdims: bool = False,
    axis: _AxisLike | None = None,
) -> NamedArray[Any, _DType]:
    xp = _get_data_namespace(x)
    _axis = _dims_to_axis(x, dims, axis)
    _data = xp.max(x._data, axis=_axis, keepdims=False)
    # TODO: Why do we need to do the keepdims ourselves?
    dims_, data_ = _get_remaining_dims(x, _data, _axis, keepdims=keepdims)
    return x._new(dims=dims_, data=data_)


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
    _axis = _dims_to_axis(x, dims, axis)
    _data = xp.mean(x._data, axis=_axis, keepdims=False)
    # TODO: Why do we need to do the keepdims ourselves?
    dims_, data_ = _get_remaining_dims(x, _data, _axis, keepdims=keepdims)
    out = x._new(dims=dims_, data=data_)
    return out


def min(
    x: NamedArray[Any, _DType],
    /,
    *,
    dims: _Dims | Default = _default,
    keepdims: bool = False,
    axis: _AxisLike | None = None,
) -> NamedArray[Any, _DType]:
    xp = _get_data_namespace(x)
    _axis = _dims_to_axis(x, dims, axis)
    _data = xp.min(x._data, axis=_axis, keepdims=False)
    # TODO: Why do we need to do the keepdims ourselves?
    dims_, data_ = _get_remaining_dims(x, _data, _axis, keepdims=keepdims)
    out = x._new(dims=dims_, data=data_)
    return out


def prod(
    x: NamedArray[Any, _DType],
    /,
    *,
    dims: _Dims | Default = _default,
    dtype: _DType | None = None,
    keepdims: bool = False,
    axis: _AxisLike | None = None,
) -> NamedArray[Any, _DType]:
    xp = _get_data_namespace(x)
    _axis = _dims_to_axis(x, dims, axis)
    _data = xp.prod(x._data, axis=_axis, keepdims=False)
    # TODO: Why do we need to do the keepdims ourselves?
    dims_, data_ = _get_remaining_dims(x, _data, _axis, keepdims=keepdims)
    out = x._new(dims=dims_, data=data_)
    return out


def std(
    x: NamedArray[Any, _DType],
    /,
    *,
    dims: _Dims | Default = _default,
    correction: int | float = 0.0,
    keepdims: bool = False,
    axis: _AxisLike | None = None,
) -> NamedArray[Any, _DType]:
    xp = _get_data_namespace(x)
    _axis = _dims_to_axis(x, dims, axis)
    _data = xp.std(x._data, axis=_axis, correction=correction, keepdims=False)
    # TODO: Why do we need to do the keepdims ourselves?
    dims_, data_ = _get_remaining_dims(x, _data, _axis, keepdims=keepdims)
    out = x._new(dims=dims_, data=data_)
    return out


def sum(
    x: NamedArray[Any, _DType],
    /,
    *,
    dims: _Dims | Default = _default,
    dtype: _DType | None = None,
    keepdims: bool = False,
    axis: _AxisLike | None = None,
) -> NamedArray[Any, _DType]:
    xp = _get_data_namespace(x)
    _axis = _dims_to_axis(x, dims, axis)
    _data = xp.sum(x._data, axis=_axis, keepdims=False)
    # TODO: Why do we need to do the keepdims ourselves?
    dims_, data_ = _get_remaining_dims(x, _data, _axis, keepdims=keepdims)
    out = x._new(dims=dims_, data=data_)
    return out


def var(
    x: NamedArray[Any, _DType],
    /,
    *,
    dims: _Dims | Default = _default,
    correction: int | float = 0.0,
    keepdims: bool = False,
    axis: _AxisLike | None = None,
) -> NamedArray[Any, _DType]:
    xp = _get_data_namespace(x)
    _axis = _dims_to_axis(x, dims, axis)
    _data = xp.var(x._data, axis=_axis, correction=correction, keepdims=False)
    # TODO: Why do we need to do the keepdims ourselves?
    dims_, data_ = _get_remaining_dims(x, _data, _axis, keepdims=keepdims)
    out = x._new(dims=dims_, data=data_)
    return out
