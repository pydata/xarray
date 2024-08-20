from __future__ import annotations

from typing import Any, TYPE_CHECKING

from xarray.namedarray._array_api._utils import _get_data_namespace, _infer_dims
from xarray.namedarray._typing import (
    Default,
    _arrayfunction_or_api,
    _ArrayLike,
    _default,
    _Device,
    _DimsLike,
    _DType,
    _Dims,
    _Shape,
    _ShapeType,
    duckarray,
)
from xarray.namedarray.core import (
    NamedArray,
    _dims_to_axis,
    _get_remaining_dims,
)

if TYPE_CHECKING:
    from typing import Literal, Optional, Tuple

from xarray.namedarray._array_api._utils import _get_data_namespace


def argmax(
    x: NamedArray,
    /,
    *,
    dims: _Dims | Default = _default,
    keepdims: bool = False,
    axis: int | None = None,
) -> NamedArray:
    xp = _get_data_namespace(x)
    _axis = _dims_to_axis(x, dims, axis)
    _data = xp.argmax(x._data, axis=_axis, keepdims=False)  # We fix keepdims later
    # TODO: Why do we need to do the keepdims ourselves?
    _dims, data_ = _get_remaining_dims(x, _data, _axis, keepdims=keepdims)
    return x._new(dims=_dims, data=data_)


def argmin(
    x: NamedArray,
    /,
    *,
    dims: _Dims | Default = _default,
    keepdims: bool = False,
    axis: int | None = None,
) -> NamedArray:
    xp = _get_data_namespace(x)
    _axis = _dims_to_axis(x, dims, axis)
    _data = xp.argmin(x._data, axis=_axis, keepdims=False)  # We fix keepdims later
    # TODO: Why do we need to do the keepdims ourselves?
    _dims, data_ = _get_remaining_dims(x, _data, _axis, keepdims=keepdims)
    return x._new(dims=_dims, data=data_)


def nonzero(x: NamedArray, /) -> tuple[NamedArray, ...]:
    xp = _get_data_namespace(x)
    _datas = xp.nonzero(x._data)
    # TODO: Verify that dims and axis matches here:
    return tuple(x._new(dim, i) for dim, i in zip(x.dims, _datas))


def searchsorted(
    x1: NamedArray,
    x2: NamedArray,
    /,
    *,
    side: Literal["left", "right"] = "left",
    sorter: NamedArray | None = None,
) -> NamedArray:
    xp = _get_data_namespace(x1)
    _data = xp.searchsorted(x1._data, x2._data, side=side, sorter=sorter)
    # TODO: Check dims, probably can do it smarter:
    _dims = _infer_dims(_data.shape)
    return NamedArray(_dims, _data)


def where(condition: NamedArray, x1: NamedArray, x2: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x1)
    _data = xp.where(condition._data, x1._data, x2._data)
    return x1._new(x1.dims, _data)
