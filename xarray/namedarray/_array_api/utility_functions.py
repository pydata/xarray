from __future__ import annotations

from types import ModuleType
from typing import Any, overload

import numpy as np

from xarray.namedarray._array_api._utils import _get_data_namespace
from xarray.namedarray._typing import (
    Default,
    _arrayapi,
    _arrayfunction_or_api,
    _ArrayLike,
    _Axes,
    _Axis,
    _AxisLike,
    _default,
    _Dim,
    _Dims,
    _DimsLike,
    _DType,
    _dtype,
    _ScalarType,
    _Shape,
    _ShapeType,
    _SupportsImag,
    _SupportsReal,
    duckarray,
)
from xarray.namedarray.core import (
    NamedArray,
    _dims_to_axis,
    _get_remaining_dims,
)
from xarray.namedarray.utils import (
    to_0d_object_array,
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
