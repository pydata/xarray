from __future__ import annotations

from typing import TYPE_CHECKING, Any

from xarray.namedarray._array_api._utils import (
    _dim_to_optional_axis,
    _get_data_namespace,
    _get_remaining_dims,
    _infer_dims,
)
from xarray.namedarray._typing import (
    Default,
    _default,
    _Dim,
    _arrayapi,
)
from xarray.namedarray.core import (
    NamedArray,
)

if TYPE_CHECKING:
    from typing import Literal


def argmax(
    x: NamedArray[Any, Any],
    /,
    *,
    dim: _Dim | Default = _default,
    keepdims: bool = False,
    axis: int | None = None,
) -> NamedArray[Any, Any]:
    xp = _get_data_namespace(x)
    _axis = _dim_to_optional_axis(x, dim, axis)
    _data = xp.argmax(x._data, axis=_axis, keepdims=False)  # We fix keepdims later
    # TODO: Why do we need to do the keepdims ourselves?
    _dims, data_ = _get_remaining_dims(x, _data, _axis, keepdims=keepdims)
    return x._new(dims=_dims, data=data_)


def argmin(
    x: NamedArray[Any, Any],
    /,
    *,
    dim: _Dim | Default = _default,
    keepdims: bool = False,
    axis: int | None = None,
) -> NamedArray[Any, Any]:
    xp = _get_data_namespace(x)
    _axis = _dim_to_optional_axis(x, dim, axis)
    _data = xp.argmin(x._data, axis=_axis, keepdims=False)  # We fix keepdims later
    # TODO: Why do we need to do the keepdims ourselves?
    _dims, data_ = _get_remaining_dims(x, _data, _axis, keepdims=keepdims)
    return x._new(dims=_dims, data=data_)


def nonzero(x: NamedArray[Any, Any], /) -> tuple[NamedArray[Any, Any], ...]:
    xp = _get_data_namespace(x)
    _datas: tuple[_arrayapi[Any, Any], ...] = xp.nonzero(x._data)
    # TODO: Verify that dims and axis matches here:
    return tuple(x._new((dim,), data) for dim, data in zip(x.dims, _datas))


def searchsorted(
    x1: NamedArray[Any, Any],
    x2: NamedArray[Any, Any],
    /,
    *,
    side: Literal["left", "right"] = "left",
    sorter: NamedArray[Any, Any] | None = None,
) -> NamedArray[Any, Any]:
    xp = _get_data_namespace(x1)
    _data = xp.searchsorted(x1._data, x2._data, side=side, sorter=sorter)
    # TODO: Check dims, probably can do it smarter:
    _dims = _infer_dims(_data.shape)
    return NamedArray(_dims, _data)


def where(
    condition: NamedArray[Any, Any],
    x1: NamedArray[Any, Any],
    x2: NamedArray[Any, Any],
    /,
) -> NamedArray[Any, Any]:
    xp = _get_data_namespace(x1)
    _data = xp.where(condition._data, x1._data, x2._data)
    # TODO: Wrong, _dims should be either of the arguments. How to choose?
    _dims = _infer_dims(_data.shape)
    return NamedArray(_dims, _data)
