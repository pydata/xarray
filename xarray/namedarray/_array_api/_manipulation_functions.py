from __future__ import annotations

from typing import Any

from xarray.namedarray._array_api._utils import _get_data_namespace
from xarray.namedarray._array_api._creation_functions import asarray

from xarray.namedarray._typing import (
    Default,
    _arrayapi,
    _Axes,
    _Axis,
    _default,
    _Dim,
    _DType,
    _Shape,
)
from xarray.namedarray.core import (
    NamedArray,
)


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
