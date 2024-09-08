from __future__ import annotations

from typing import Any

from xarray.namedarray._array_api._utils import _dims_to_axis, _get_data_namespace
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
    _axis = _dims_to_axis(x, dim, axis)[0]
    # TODO: Handle attrs? will get x1 now
    out = x._new(data=xp.take(x._data, indices._data, axis=_axis))
    return out
