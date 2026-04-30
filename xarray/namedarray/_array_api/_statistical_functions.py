from __future__ import annotations

from typing import Any

from xarray.namedarray._array_api._utils import (
    _dims_to_axis,
    _get_data_namespace,
    _reduce_dims,
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
from xarray.namedarray.core import NamedArray


def cumulative_prod(
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
    _axis_none: int | None
    if a is None:
        _axis_none = a
    else:
        _axis_none = a[0]

    # TODO: The standard is not clear about what should happen when x.ndim == 0.
    _axis: int
    if _axis_none is None:
        if x.ndim > 1:
            raise ValueError(
                "axis must be specified in cumulative_sum for more than one dimension"
            )
        _axis = 0
    else:
        _axis = _axis_none

    try:
        _data = xp.cumulative_prod(
            x._data, axis=_axis, dtype=dtype, include_initial=include_initial
        )
    except AttributeError:
        # Use np.cumprod until new name is introduced:
        # np.cumsum does not support include_initial
        if include_initial:
            if _axis < 0:
                _axis += x.ndim
            d = xp.concat(
                [
                    xp.ones(
                        x.shape[:_axis] + (1,) + x.shape[_axis + 1 :], dtype=x.dtype
                    ),
                    x._data,
                ],
                axis=_axis,
            )
        else:
            d = x._data
        _data = xp.cumprod(d, axis=_axis, dtype=dtype)
    return x._new(dims=x.dims, data=_data)


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
    _axis_none: int | None
    if a is None:
        _axis_none = a
    else:
        _axis_none = a[0]

    # TODO: The standard is not clear about what should happen when x.ndim == 0.
    _axis: int
    if _axis_none is None:
        if x.ndim > 1:
            raise ValueError(
                "axis must be specified in cumulative_sum for more than one dimension"
            )
        _axis = 0
    else:
        _axis = _axis_none

    try:
        _data = xp.cumulative_sum(
            x._data, axis=_axis, dtype=dtype, include_initial=include_initial
        )
    except AttributeError:
        # Use np.cumsum until new name is introduced:
        # np.cumsum does not support include_initial
        if include_initial:
            if _axis < 0:
                _axis += x.ndim
            d = xp.concat(
                [
                    xp.zeros(
                        x.shape[:_axis] + (1,) + x.shape[_axis + 1 :], dtype=x.dtype
                    ),
                    x._data,
                ],
                axis=_axis,
            )
        else:
            d = x._data
        _data = xp.cumsum(d, axis=_axis, dtype=dtype)
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
    _data = xp.max(x._data, axis=_axis, keepdims=keepdims)
    _dims = _reduce_dims(x.dims, axis=_axis, keepdims=keepdims)
    return x._new(dims=_dims, data=_data)


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
    >>> import numpy as np
    >>> x = NamedArray(("x", "y"), np.asarray([[1.0, 2.0], [3.0, 4.0]]))
    >>> mean(x)
    <xarray.NamedArray ()> Size: 8B
    np.float64(2.5)
    >>> mean(x, dims=("x",))
    <xarray.NamedArray (y: 2)> Size: 16B
    array([2., 3.])

    Using keepdims:

    >>> mean(x, dims=("x",), keepdims=True)
    <xarray.NamedArray (x: 1, y: 2)> Size: 16B
    array([[2., 3.]])
    >>> mean(x, dims=("y",), keepdims=True)
    <xarray.NamedArray (x: 2, y: 1)> Size: 16B
    array([[1.5],
           [3.5]])
    """
    xp = _get_data_namespace(x)
    _axis = _dims_to_axis(x, dims, axis)
    _data = xp.mean(x._data, axis=_axis, keepdims=keepdims)
    _dims = _reduce_dims(x.dims, axis=_axis, keepdims=keepdims)
    return x._new(dims=_dims, data=_data)


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
    _data = xp.min(x._data, axis=_axis, keepdims=keepdims)
    _dims = _reduce_dims(x.dims, axis=_axis, keepdims=keepdims)
    return x._new(dims=_dims, data=_data)


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
    _data = xp.prod(x._data, axis=_axis, dtype=dtype, keepdims=keepdims)
    _dims = _reduce_dims(x.dims, axis=_axis, keepdims=keepdims)
    return x._new(dims=_dims, data=_data)


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
    _data = xp.std(x._data, axis=_axis, correction=correction, keepdims=keepdims)
    _dims = _reduce_dims(x.dims, axis=_axis, keepdims=keepdims)
    return x._new(dims=_dims, data=_data)


def sum(
    x: NamedArray[Any, _DType],
    /,
    *,
    dims: _Dims | Default = _default,
    dtype: _DType | None = None,
    keepdims: bool = False,
    axis: _AxisLike | None = None,
) -> NamedArray[Any, _DType]:
    """
    Calculates the sum of the input array x.

    Examples
    --------
    >>> import numpy as np
    >>> sum(NamedArray(("x",), np.array([0.5, 1.5])))
    <xarray.NamedArray ()> Size: 8B
    np.float64(2.0)
    >>> sum(NamedArray(("x",), np.array([0.5, 0.7, 0.2, 1.5])), dtype=np.int32)
    <xarray.NamedArray ()> Size: 4B
    np.int32(1)
    """
    xp = _get_data_namespace(x)
    _axis = _dims_to_axis(x, dims, axis)
    _data = xp.sum(x._data, axis=_axis, dtype=dtype, keepdims=keepdims)
    _dims = _reduce_dims(x.dims, axis=_axis, keepdims=keepdims)
    return x._new(dims=_dims, data=_data)


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
    _data = xp.var(x._data, axis=_axis, correction=correction, keepdims=keepdims)
    _dims = _reduce_dims(x.dims, axis=_axis, keepdims=keepdims)
    return x._new(dims=_dims, data=_data)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
