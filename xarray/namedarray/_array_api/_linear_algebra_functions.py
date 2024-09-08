from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from xarray.namedarray._array_api._utils import _get_data_namespace, _infer_dims
from xarray.namedarray.core import NamedArray


def matmul(
    x1: NamedArray[Any, Any], x2: NamedArray[Any, Any], /
) -> NamedArray[Any, Any]:
    xp = _get_data_namespace(x1)
    _data = xp.matmul(x1._data, x2._data)
    # TODO: Figure out a better way:
    _dims = _infer_dims(_data.shape)
    return NamedArray(_dims, _data)


def tensordot(
    x1: NamedArray[Any, Any],
    x2: NamedArray[Any, Any],
    /,
    *,
    axes: int | tuple[Sequence[int], Sequence[int]] = 2,
) -> NamedArray[Any, Any]:
    xp = _get_data_namespace(x1)
    _data = xp.tensordot(x1._data, x2._data, axes=axes)
    # TODO: Figure out a better way:
    _dims = _infer_dims(_data.shape)
    return NamedArray(_dims, _data)


def matrix_transpose(x: NamedArray[Any, Any], /) -> NamedArray[Any, Any]:
    xp = _get_data_namespace(x)
    _data = xp.matrix_transpose(x._data)
    # TODO: Figure out a better way:
    _dims = _infer_dims(_data.shape)
    return NamedArray(_dims, _data)


def vecdot(
    x1: NamedArray[Any, Any], x2: NamedArray[Any, Any], /, *, axis: int = -1
) -> NamedArray[Any, Any]:
    xp = _get_data_namespace(x1)
    _data = xp.vecdot(x1._data, x2._data, axis=axis)
    # TODO: Figure out a better way:
    _dims = _infer_dims(_data.shape)
    return NamedArray(_dims, _data)
