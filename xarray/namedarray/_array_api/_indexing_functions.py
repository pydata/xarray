from __future__ import annotations

from typing import Any

from xarray.namedarray._array_api._creation_functions import asarray
from xarray.namedarray._array_api._utils import _get_data_namespace
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
    _dims_to_axis,
    NamedArray,
)


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
