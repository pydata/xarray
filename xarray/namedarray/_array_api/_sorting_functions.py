from __future__ import annotations

from xarray.namedarray._array_api._utils import _dims_to_axis, _get_data_namespace
from xarray.namedarray._typing import (
    Default,
    _default,
    _Dim,
)
from xarray.namedarray.core import (
    NamedArray,
)


def argsort(
    x: NamedArray,
    /,
    *,
    dim: _Dim | Default = _default,
    descending: bool = False,
    stable: bool = True,
    axis: int = -1,
) -> NamedArray:
    xp = _get_data_namespace(x)
    _axis = _dims_to_axis(x, dim, axis)
    out = x._new(
        data=xp.argsort(x._data, axis=_axis, descending=descending, stable=stable)
    )
    return out


def sort(
    x: NamedArray,
    /,
    *,
    dim: _Dim | Default = _default,
    descending: bool = False,
    stable: bool = True,
    axis: int = -1,
) -> NamedArray:
    xp = _get_data_namespace(x)
    _axis = _dims_to_axis(x, dim, axis)
    out = x._new(
        data=xp.argsort(x._data, axis=_axis, descending=descending, stable=stable)
    )
    return out
