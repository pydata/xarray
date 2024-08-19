from __future__ import annotations

from typing import Any

from xarray.namedarray._array_api._utils import _get_data_namespace
from xarray.namedarray._typing import (
    Default,
    _AxisLike,
    _default,
    _Dims,
    _DType,
)
from xarray.namedarray.core import (
    NamedArray,
    _dims_to_axis,
    _get_remaining_dims,
)


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
