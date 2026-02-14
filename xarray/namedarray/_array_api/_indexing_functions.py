from __future__ import annotations

from typing import Any

from xarray.namedarray._array_api._utils import (
    _dim_to_axis,
    _dim_to_optional_axis,
    _get_data_namespace,
)
from xarray.namedarray._typing import (
    Default,
    _default,
    _Dim,
    _DType,
)
from xarray.namedarray.core import NamedArray


def take(
    x: NamedArray[Any, _DType],
    indices: NamedArray[Any, Any],
    /,
    *,
    dim: _Dim | Default = _default,
    axis: int | None = None,
) -> NamedArray[Any, _DType]:
    xp = _get_data_namespace(x)
    _axis = _dim_to_optional_axis(x, dim, axis)
    _data = xp.take(x._data, indices._data, axis=_axis)
    _dims = x.dims
    return NamedArray(_dims, _data)


def take_along_axis(
    x: NamedArray[Any, _DType],
    indices: NamedArray[Any, Any],
    /,
    *,
    dim: _Dim | Default = _default,
    axis: int = -1,
) -> NamedArray[Any, _DType]:
    xp = _get_data_namespace(x)
    _axis = _dim_to_axis(x, dim, axis, axis_default=-1)
    _data = xp.take_along_axis(x._data, indices._data, axis=_axis)
    _dims = x.dims
    return NamedArray(_dims, _data)
