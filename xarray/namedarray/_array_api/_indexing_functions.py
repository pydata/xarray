from __future__ import annotations

from xarray.namedarray._array_api._utils import _get_data_namespace, _dims_to_axis
from xarray.namedarray._typing import (
    Default,
    _default,
    _Dim,
)
from xarray.namedarray.core import NamedArray


def take(
    x: NamedArray,
    indices: NamedArray,
    /,
    *,
    dim: _Dim | Default = _default,
    axis: int | None = None,
) -> NamedArray:
    xp = _get_data_namespace(x)
    _axis = _dims_to_axis(x, dim, axis)
    # TODO: Handle attrs? will get x1 now
    out = x._new(data=xp.take(x._data, indices._data, axis=_axis))
    return out
