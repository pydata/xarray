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
    xp = _get_data_namespace(x)
    axis_ = _dims_to_axis(x, dims, axis)
    d = xp.all(x._data, axis=axis_, keepdims=False)
    dims_, data_ = _get_remaining_dims(x, d, axis_, keepdims=keepdims)
    out = x._new(dims=dims_, data=data_)
    return out


def any(
    x: NamedArray[Any, Any],
    /,
    *,
    dims: _DimsLike2 | Default = _default,
    keepdims: bool = False,
    axis: _AxisLike | None = None,
) -> NamedArray[Any, Any]:
    xp = _get_data_namespace(x)
    axis_ = _dims_to_axis(x, dims, axis)
    d = xp.any(x._data, axis=axis_, keepdims=False)
    dims_, data_ = _get_remaining_dims(x, d, axis_, keepdims=keepdims)
    out = x._new(dims=dims_, data=data_)
    return out
