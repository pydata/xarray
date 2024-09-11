from __future__ import annotations

from typing import Any

from xarray.namedarray._array_api._data_type_functions import result_type
from xarray.namedarray._array_api._utils import (
    _get_broadcasted_dims,
    _get_data_namespace,
    _infer_dims,
    _insert_dim,
    _dims_to_axis,
)
from xarray.namedarray._typing import (
    Default,
    _arrayapi,
    _Axes,
    _Axis,
    _AxisLike,
    _default,
    _Dim,
    _DType,
    _ShapeType,
    _Dims,
)
from xarray.namedarray.core import NamedArray


def broadcast_arrays(*arrays: NamedArray[Any, Any]) -> list[NamedArray[Any, Any]]:
    """
    Broadcasts one or more arrays against one another.

    Examples
    --------
    >>> x = xp.asarray([[1, 2, 3]])
    >>> y = xp.asarray([[4], [5]])
    >>> xp.broadcast_arrays(x, y)
    """
    x = arrays[0]
    xp = _get_data_namespace(x)
    _dims, _ = _get_broadcasted_dims(*arrays)
    _arrays = tuple(a._data for a in arrays)
    _datas = xp.broadcast_arrays(*_arrays)
    return [arr._new(_dims, _data) for arr, _data in zip(arrays, _datas)]


def broadcast_to(
    x: NamedArray[Any, _DType], /, shape: _ShapeType
) -> NamedArray[_ShapeType, _DType]:
    xp = _get_data_namespace(x)
    _data = xp.broadcast_to(x._data, shape=shape)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def concat(
    arrays: tuple[NamedArray[Any, Any], ...] | list[NamedArray[Any, Any]],
    /,
    *,
    axis: _Axis | None = 0,
) -> NamedArray[Any, Any]:
    xp = _get_data_namespace(arrays[0])
    dtype = result_type(*arrays)
    _arrays = tuple(a._data for a in arrays)
    _data = xp.concat(_arrays, axis=axis, dtype=dtype)
    _dims = _infer_dims(_data.shape)
    return NamedArray(_dims, _data)


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
    _data = xp.expand_dims(x._data, axis=axis)
    _dims = _insert_dim(x.dims, dim, axis)
    return x._new(_dims, _data)


def flip(
    x: NamedArray[_ShapeType, _DType], /, *, axis: _Axes | None = None
) -> NamedArray[_ShapeType, _DType]:
    xp = _get_data_namespace(x)
    _data = xp.flip(x._data, axis=axis)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def moveaxis(
    x: NamedArray[Any, _DType], source: _Axes, destination: _Axes, /
) -> NamedArray[Any, _DType]:
    xp = _get_data_namespace(x)
    _data = xp.moveaxis(x._data, source=source, destination=destination)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def permute_dims(
    x: NamedArray[Any, _DType],
    /,
    axes: _Axes | None = None,
    *,
    dims: _Dims | Default = _default,
) -> NamedArray[Any, _DType]:
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

    Examples
    --------
    >>> x = NamedArray(("x", "y", "z"), np.zeros((3, 4, 5)))
    >>> y = permute_dims(x, (2, 1, 0))
    >>> y.dims, y.shape
    (('z', 'y', 'x'), (5, 4, 3))
    >>> y = permute_dims(x, dims=("y", "x", "z"))
    >>> y.dims, y.shape
    (('y', 'x', 'z'), (4, 3, 5))
    """
    xp = _get_data_namespace(x)
    _axis = _dims_to_axis(x, dims, axes)
    if _axis is None:
        raise TypeError("permute_dims missing argument axes or dims")
    old_dims = x.dims
    _data = xp.permute_dims(x._data, _axis)
    _dims = tuple(old_dims[i] for i in _axis)
    return x._new(_dims, _data)


def repeat(
    x: NamedArray[Any, _DType],
    repeats: int | NamedArray[Any, Any],
    /,
    *,
    axis: _Axis | None = None,
) -> NamedArray[Any, _DType]:
    xp = _get_data_namespace(x)
    _data = xp.repeat(x._data, repeats, axis=axis)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def reshape(
    x: NamedArray[Any, _DType], /, shape: _ShapeType, *, copy: bool | None = None
) -> NamedArray[_ShapeType, _DType]:
    xp = _get_data_namespace(x)
    _data = xp.reshape(x._data, shape, copy=copy)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def roll(
    x: NamedArray[_ShapeType, _DType],
    /,
    shift: int | tuple[int, ...],
    *,
    axis: _Axes | None = None,
) -> NamedArray[_ShapeType, _DType]:
    xp = _get_data_namespace(x)
    _data = xp.roll(x._data, shift=shift, axis=axis)
    return x._new(data=_data)


def squeeze(x: NamedArray[Any, _DType], /, axis: _Axes) -> NamedArray[Any, _DType]:
    xp = _get_data_namespace(x)
    _data = xp.squeeze(x._data, axis=axis)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def stack(
    arrays: tuple[NamedArray[Any, Any], ...] | list[NamedArray[Any, Any]],
    /,
    *,
    axis: _Axis = 0,
) -> NamedArray[Any, Any]:
    x = arrays[0]
    xp = _get_data_namespace(x)
    _arrays = tuple(a._data for a in arrays)
    _data = xp.stack(_arrays, axis=axis)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def tile(
    x: NamedArray[Any, _DType], repetitions: tuple[int, ...], /
) -> NamedArray[Any, _DType]:
    xp = _get_data_namespace(x)
    _data = xp.tile(x._data, repetitions)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def unstack(
    x: NamedArray[Any, Any], /, *, axis: _Axis = 0
) -> tuple[NamedArray[Any, Any], ...]:
    xp = _get_data_namespace(x)
    _datas = xp.unstack(x._data, axis=axis)
    out: tuple[NamedArray[Any, Any], ...] = ()
    for _data in _datas:
        _dims = _infer_dims(_data.shape)  # TODO: Fix dims
        out += (x._new(_dims, _data),)
    return out
