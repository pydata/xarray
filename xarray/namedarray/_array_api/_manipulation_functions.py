from __future__ import annotations

import math
from typing import Any

from xarray.namedarray._array_api._data_type_functions import result_type
from xarray.namedarray._array_api._utils import (
    _dims_to_axis,
    _flatten_dims,
    _get_broadcasted_dims,
    _get_data_namespace,
    _infer_dims,
    _insert_dim,
    _new_unique_dim_name,
)
from xarray.namedarray._typing import (
    Default,
    _Axes,
    _Axis,
    _default,
    _Dim,
    _Dims,
    _DimsLike2,
    _DType,
    _Shape,
    _ShapeType,
)
from xarray.namedarray.core import NamedArray


def broadcast_arrays(*arrays: NamedArray[Any, Any]) -> list[NamedArray[Any, Any]]:
    """
    Broadcasts one or more arrays against one another.

    Examples
    --------
    >>> import numpy as np
    >>> x = NamedArray(("x",), np.zeros((3,)))
    >>> y = NamedArray(("y", "x"), np.zeros((2, 1)))
    >>> x_new, y_new = broadcast_arrays(x, y)
    >>> x_new.dims, x_new.shape, y_new.dims, y_new.shape
    (('y', 'x'), (2, 3), ('y', 'x'), (2, 3))

    Errors

    >>> x = NamedArray(("x",), np.zeros((3,)))
    >>> y = NamedArray(("x",), np.zeros((2)))
    >>> x_new, y_new = broadcast_arrays(x, y)
    Traceback (most recent call last):
     ...
    ValueError: operands could not be broadcast together with dims = (('x',), ('x',)) and shapes = ((3,), (2,))
    """
    x = arrays[0]
    xp = _get_data_namespace(x)
    _dims, _ = _get_broadcasted_dims(*arrays)
    _arrays = tuple(a._data for a in arrays)
    _datas = xp.broadcast_arrays(*_arrays)
    return [arr._new(_dims, _data) for arr, _data in zip(arrays, _datas, strict=False)]


def broadcast_to(
    x: NamedArray[Any, _DType],
    /,
    shape: _ShapeType,
    *,
    dims: _DimsLike2 | Default = _default,
) -> NamedArray[_ShapeType, _DType]:
    """

    Examples
    --------
    >>> import numpy as np
    >>> x = NamedArray(("x",), np.arange(0, 3))
    >>> x_new = broadcast_to(x, (1, 1, 3))
    >>> x_new.dims, x_new.shape
    (('dim_1', 'dim_0', 'x'), (1, 1, 3))

    >>> x_new = broadcast_to(x, shape=(1, 1, 3), dims=("y", "x"))
    >>> x_new.dims, x_new.shape
    (('dim_0', 'y', 'x'), (1, 1, 3))
    """
    xp = _get_data_namespace(x)
    _data = xp.broadcast_to(x._data, shape=shape)
    _dims = _infer_dims(_data.shape, x.dims if isinstance(dims, Default) else dims)
    return x._new(_dims, _data)


def concat(
    arrays: tuple[NamedArray[Any, Any], ...] | list[NamedArray[Any, Any]],
    /,
    *,
    axis: _Axis | None = 0,
) -> NamedArray[Any, Any]:
    """
    Joins a sequence of arrays along an existing axis.

    Examples
    --------
    >>> import numpy as np
    >>> x = NamedArray(("x",), np.zeros((3,)))
    >>> x1 = concat((x, 1 + x))
    >>> x1.dims, x1.shape
    (('x',), (6,))

    >>> x = NamedArray(("x", "y"), np.zeros((3, 4)))
    >>> x1 = concat((x, 1 + x))
    >>> x1.dims, x1.shape
    (('x', 'y'), (6, 4))
    """
    x = arrays[0]
    xp = _get_data_namespace(x)
    _axis = axis  # TODO: add support for dim?
    dtype = result_type(*arrays)
    _arrays = tuple(a._data for a in arrays)
    _data = xp.concat(_arrays, axis=_axis, dtype=dtype)
    return NamedArray(x.dims, _data)


def expand_dims(
    x: NamedArray[Any, _DType],
    /,
    *,
    axis: _Axis = 0,
    dim: _Dim | Default = _default,
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
        Axis position (zero-based). Default is 0.

    Returns
    -------
        out :
            An expanded output array having the same data type as x.

    Examples
    --------
    >>> import numpy as np
    >>> x = NamedArray(("x", "y"), np.asarray([[1.0, 2.0], [3.0, 4.0]]))
    >>> x_new = expand_dims(x)
    >>> x_new.dims, x_new.shape
    (('dim_2', 'x', 'y'), (1, 2, 2))
    >>> x_new = expand_dims(x, dim="z")
    >>> x_new.dims, x_new.shape
    (('z', 'x', 'y'), (1, 2, 2))
    """
    # Array Api does not support multiple axes, but maybe in the future:
    # https://github.com/data-apis/array-api/issues/760
    # xref: https://github.com/numpy/numpy/blob/3b246c6488cf246d488bbe5726ca58dc26b6ea74/numpy/lib/_shape_base_impl.py#L509C17-L509C24
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
    >>> import numpy as np
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
    """

    Examples
    --------
    >>> import numpy as np
    >>> x = NamedArray(("x",), np.zeros((3,)))
    >>> x1 = reshape(x, (-1,))
    >>> x1.dims, x1.shape
    (('x',), (3,))

    To N-dimensions

    >>> x1 = reshape(x, (1, -1, 1))
    >>> x1.dims, x1.shape
    (('dim_0', 'x', 'dim_2'), (1, 3, 1))

    >>> x = NamedArray(("x", "y"), np.zeros((3, 4)))
    >>> x1 = reshape(x, (-1,))
    >>> x1.dims, x1.shape
    ((('x', 'y'),), (12,))
    """
    xp = _get_data_namespace(x)
    _data = xp.reshape(x._data, shape, copy=copy)

    if math.prod(shape) == -1 and False:
        # Flattening operations merges all dimensions to 1:
        dims_raveled = _flatten_dims(x.dims)
        dim = dims_raveled[0]
        d = []
        for v in shape:
            d.append(dim if v == -1 else _new_unique_dim_name(tuple(d)))
        _dims = tuple(d)
    else:
        # _dims = _infer_dims(_data.shape, x.dims)
        _dims = _infer_dims(_data.shape)

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


# %% Automatic broadcasting
_OPTIONS = {}
_OPTIONS["arithmetic_broadcast"] = True


def _set_dims(
    x: NamedArray[Any, Any], dim: _Dims, shape: _Shape | None
) -> NamedArray[Any, Any]:
    """
    Return a new array with given set of dimensions.
    This method might be used to attach new dimension(s) to array.

    When possible, this operation does not copy this variable's data.

    Parameters
    ----------
    dim :
        Dimensions to include on the new variable.
    shape :
        Shape of the dimensions. If None, new dimensions are inserted with length 1.

    Examples
    --------
    >>> import numpy as np
    >>> x = NamedArray(("x",), np.asarray([1, 2, 3]))
    >>> x_new = _set_dims(x, ("y", "x"), None)
    >>> x_new.dims, x_new.shape
    (('y', 'x'), (1, 3))
    >>> x_new = _set_dims(x, ("x", "y"), None)
    >>> x_new.dims, x_new.shape
    (('x', 'y'), (3, 1))

    With shape:

    >>> x_new = _set_dims(x, ("y", "x"), (2, 3))
    >>> x_new.dims, x_new.shape
    (('y', 'x'), (2, 3))

    No operation

    >>> x_new = _set_dims(x, ("x",), None)
    >>> x_new.dims, x_new.shape
    (('x',), (3,))


    Unordered dims

    >>> x = NamedArray(("y", "x"), np.zeros((2, 3)))
    >>> x_new = _set_dims(x, ("x", "y"), None)
    >>> x_new.dims, x_new.shape
    (('x', 'y'), (3, 2))

    Errors

    >>> x = NamedArray(("x",), np.asarray([1, 2, 3]))
    >>> x_new = _set_dims(x, (), None)
    Traceback (most recent call last):
     ...
    ValueError: new dimensions () must be a superset of existing dimensions ('x',)
    """
    if x.dims == dim:
        # No operation. Don't use broadcast_to unless necessary so the result
        # remains writeable as long as possible:
        return x

    missing_dims = set(x.dims) - set(dim)
    if missing_dims:
        raise ValueError(
            f"new dimensions {dim!r} must be a superset of "
            f"existing dimensions {x.dims!r}"
        )

    extra_dims = tuple(d for d in dim if d not in x.dims)

    if shape is not None:
        # Add dimensions, with same size as shape:
        dims_map = dict(zip(dim, shape, strict=False))
        expanded_dims = extra_dims + x.dims
        tmp_shape = tuple(dims_map[d] for d in expanded_dims)
        return permute_dims(broadcast_to(x, tmp_shape, dims=expanded_dims), dims=dim)
    else:
        # Add dimensions, with size 1 only:
        out = x
        for d in extra_dims:
            out = expand_dims(out, dim=d)
        return permute_dims(out, dims=dim)


def _broadcast_arrays(*arrays: NamedArray[Any, Any]) -> NamedArray[Any, Any]:
    """
    TODO: Can this become xp.broadcast_arrays?

    Given any number of variables, return variables with matching dimensions
    and broadcast data.

    The data on the returned variables may be a view of the data on the
    corresponding original arrays but dimensions will be reordered and
    inserted so that both broadcast arrays have the same dimensions. The new
    dimensions are sorted in order of appearance in the first variable's
    dimensions followed by the second variable's dimensions.
    """
    dims, shape = _get_broadcasted_dims(*arrays)
    return tuple(_set_dims(var, dims, shape) for var in arrays)


def _broadcast_arrays_with_minimal_size(
    *arrays: NamedArray[Any, Any]
) -> NamedArray[Any, Any]:
    """
    Given any number of variables, return variables with matching dimensions.

    Unlike the result of broadcast_variables(), variables with missing dimensions
    will have them added with size 1 instead of the size of the broadcast dimension.
    """
    dims, _ = _get_broadcasted_dims(*arrays)
    return tuple(_set_dims(var, dims, None) for var in arrays)


def _arithmetic_broadcast(
    x1: NamedArray[Any, Any], x2: NamedArray[Any, Any]
) -> NamedArray[Any, Any]:
    """
    Fu

    Examples
    --------
    >>> import numpy as np
    >>> x = NamedArray(("x",), np.asarray([1, 2, 3]))
    >>> y = NamedArray(("y",), np.asarray([4, 5]))
    >>> x_new, y_new = _arithmetic_broadcast(x, y)
    >>> x_new.dims, x_new.shape, y_new.dims, y_new.shape
    (('x', 'y'), (3, 1), ('x', 'y'), (1, 2))
    """
    if not _OPTIONS["arithmetic_broadcast"]:
        if x1.dims != x2.dims:
            raise ValueError(
                "Broadcasting is necessary but automatic broadcasting is disabled "
                "via global option `'arithmetic_broadcast'`. "
                "Use `xr.set_options(arithmetic_broadcast=True)` to enable "
                "automatic broadcasting."
            )

    return _broadcast_arrays_with_minimal_size(x1, x2)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
