from __future__ import annotations

from typing import Any

from xarray.namedarray._array_api._utils import (
    _dim_to_axis,
    _get_data_namespace,
)
from xarray.namedarray._typing import (
    Default,
    _default,
    _Dim,
    _DType,
    _ShapeType,
)
from xarray.namedarray.core import NamedArray


def argsort(
    x: NamedArray[_ShapeType, Any],
    /,
    *,
    dim: _Dim | Default = _default,
    descending: bool = False,
    stable: bool = True,
    axis: int = -1,
) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _axis = _dim_to_axis(x, dim, axis)

    # TODO: As NumPy currently has no native descending sort, we imitate it here:
    if not descending:
        _data = xp.argsort(x._data, axis=_axis, stable=stable)
    else:
        _data = xp.flip(
            xp.argsort(xp.flip(x._data, axis=_axis), stable=stable, axis=_axis),
            axis=_axis,
        )
        # Rely on flip()/argsort() to validate axis
        normalised_axis = _axis if _axis >= 0 else x.ndim + _axis
        max_i = x.shape[normalised_axis] - 1
        _data = max_i - _data
    return x._new(data=_data)


def sort(
    x: NamedArray[_ShapeType, _DType],
    /,
    *,
    dim: _Dim | Default = _default,
    descending: bool = False,
    stable: bool = True,
    axis: int = -1,
) -> NamedArray[_ShapeType, _DType]:
    xp = _get_data_namespace(x)
    _axis = _dim_to_axis(x, dim, axis)

    _data = xp.sort(x._data, axis=_axis, stable=stable)
    # TODO: As NumPy currently has no native descending sort, we imitate it here:
    if descending:
        _data = xp.flip(_data, axis=_axis)
    return x._new(data=_data)
