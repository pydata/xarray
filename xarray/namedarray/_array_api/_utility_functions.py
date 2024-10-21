from __future__ import annotations

from typing import Any

from xarray.namedarray._array_api._utils import (
    _dim_to_axis,
    _dims_to_axis,
    _get_data_namespace,
    _reduce_dims,
)
from xarray.namedarray._typing import (
    Default,
    _Axis,
    _AxisLike,
    _default,
    _Dim,
    _DimsLike2,
)
from xarray.namedarray.core import NamedArray


def all(
    x: NamedArray[Any, Any],
    /,
    *,
    dims: _DimsLike2 | Default = _default,
    keepdims: bool = False,
    axis: _AxisLike | None = None,
) -> NamedArray[Any, Any]:
    """
    Tests whether all input array elements evaluate to True along a specified axis.

    Examples
    --------
    >>> import numpy as np
    >>> x = NamedArray(("x", "y"), np.array([[True, False],[True, True]]))
    >>> all(x)
    <xarray.NamedArray ()> Size: 1B
    np.False_

    >>> all(x, axis=0)
    <xarray.NamedArray (y: 2)> Size: 2B
    array([ True, False])
    >>> all(x, dims="x")
    <xarray.NamedArray (y: 2)> Size: 2B
    array([ True, False])

    >>> x = NamedArray(("x",), np.array([-1, 4, 5]))
    >>> all(x)
    <xarray.NamedArray ()> Size: 1B
    np.True_
    """
    xp = _get_data_namespace(x)
    _axis = _dims_to_axis(x, dims, axis)
    _data = xp.all(x._data, axis=_axis, keepdims=keepdims)
    _dims = _reduce_dims(x.dims, axis=_axis, keepdims=keepdims)
    return x._new(dims=_dims, data=_data)


def any(
    x: NamedArray[Any, Any],
    /,
    *,
    dims: _DimsLike2 | Default = _default,
    keepdims: bool = False,
    axis: _AxisLike | None = None,
) -> NamedArray[Any, Any]:
    xp = _get_data_namespace(x)
    _axis = _dims_to_axis(x, dims, axis)
    _data = xp.any(x._data, axis=_axis, keepdims=keepdims)
    _dims = _reduce_dims(x.dims, axis=_axis, keepdims=keepdims)
    return x._new(dims=_dims, data=_data)
