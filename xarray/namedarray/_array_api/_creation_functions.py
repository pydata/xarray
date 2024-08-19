from __future__ import annotations

from typing import Any, overload

import numpy as np

from xarray.namedarray._array_api._utils import (
    _get_data_namespace,
    _maybe_default_namespace,
)
from xarray.namedarray._typing import (
    Default,
    _arrayfunction_or_api,
    _ArrayLike,
    _default,
    _Device,
    _DimsLike,
    _DType,
    _Shape,
    _ShapeType,
    duckarray,
)
from xarray.namedarray.core import (
    NamedArray,
)
from xarray.namedarray.utils import (
    to_0d_object_array,
)


def _like_args(x, dtype=None, device: _Device | None = None):
    if dtype is None:
        dtype = x.dtype
    return dict(shape=x.shape, dtype=dtype, device=device)


def _infer_dims(
    shape: _Shape,
    dims: _DimsLike | Default = _default,
) -> _DimsLike:
    if dims is _default:
        return tuple(f"dim_{n}" for n in range(len(shape)))
    else:
        return dims


def arange(
    start: int | float,
    /,
    stop: int | float | None = None,
    step: int | float = 1,
    *,
    dtype: _DType | None = None,
    device: _Device | None = None,
) -> NamedArray[_ShapeType, _DType]:
    xp = _maybe_default_namespace()
    _data = xp.arange(start, stop=stop, step=step, dtype=dtype, device=device)
    _dims = _infer_dims(_data.shape)
    return NamedArray(_dims, _data)


@overload
def asarray(
    obj: duckarray[_ShapeType, Any],
    /,
    *,
    dtype: _DType,
    device: _Device | None = ...,
    copy: bool | None = ...,
    dims: _DimsLike = ...,
) -> NamedArray[_ShapeType, _DType]: ...
@overload
def asarray(
    obj: _ArrayLike,
    /,
    *,
    dtype: _DType,
    device: _Device | None = ...,
    copy: bool | None = ...,
    dims: _DimsLike = ...,
) -> NamedArray[Any, _DType]: ...
@overload
def asarray(
    obj: duckarray[_ShapeType, _DType],
    /,
    *,
    dtype: None,
    device: _Device | None = None,
    copy: bool | None = None,
    dims: _DimsLike = ...,
) -> NamedArray[_ShapeType, _DType]: ...
@overload
def asarray(
    obj: _ArrayLike,
    /,
    *,
    dtype: None,
    device: _Device | None = ...,
    copy: bool | None = ...,
    dims: _DimsLike = ...,
) -> NamedArray[Any, _DType]: ...
def asarray(
    obj: duckarray[_ShapeType, _DType] | _ArrayLike,
    /,
    *,
    dtype: _DType | None = None,
    device: _Device | None = None,
    copy: bool | None = None,
    dims: _DimsLike = _default,
) -> NamedArray[_ShapeType, _DType] | NamedArray[Any, Any]:
    """
    Create a Named array from an array-like object.

    Parameters
    ----------
    dims : str or iterable of str
        Name(s) of the dimension(s).
    data : T_DuckArray or ArrayLike
        The actual data that populates the array. Should match the
        shape specified by `dims`.
    attrs : dict, optional
        A dictionary containing any additional information or
        attributes you want to store with the array.
        Default is None, meaning no attributes will be stored.
    """
    data = obj
    if isinstance(data, NamedArray):
        if copy:
            return data.copy()
        else:
            return data

    # TODO: dask.array.ma.MaskedArray also exists, better way?
    if isinstance(data, np.ma.MaskedArray):
        mask = np.ma.getmaskarray(data)  # type: ignore[no-untyped-call]
        if mask.any():
            # TODO: requires refactoring/vendoring xarray.core.dtypes and
            # xarray.core.duck_array_ops
            raise NotImplementedError("MaskedArray is not supported yet")

        _dims = _infer_dims(data.shape, dims)
        return NamedArray(_dims, data)

    if isinstance(data, _arrayfunction_or_api):
        _dims = _infer_dims(data.shape, dims)
        return NamedArray(_dims, data)

    if isinstance(data, tuple):
        _data = to_0d_object_array(data)
        _dims = _infer_dims(_data.shape, dims)
        return NamedArray(_dims, _data)

    # validate whether the data is valid data types.
    _data = np.asarray(data, dtype=dtype, device=device, copy=copy)
    _dims = _infer_dims(_data.shape, dims)
    return NamedArray(_dims, _data)


def empty(
    shape: _ShapeType, *, dtype: _DType | None = None, device: _Device | None = None
) -> NamedArray[_ShapeType, _DType]:
    xp = _maybe_default_namespace()
    _data = xp.empty(shape, dtype=dtype, device=device)
    _dims = _infer_dims(_data.shape)
    return NamedArray(_dims, _data)


def empty_like(
    x: NamedArray[_ShapeType, _DType],
    /,
    *,
    dtype: _DType | None = None,
    device: _Device | None = None,
) -> NamedArray[_ShapeType, _DType]:
    xp = _get_data_namespace(x)
    _dtype = x.dtype if dtype is None else dtype
    _device = x.device if device is None else device
    _data = xp.empty(x.shape, dtype=_dtype, device=_device)
    return x._new(data=_data)


def full(
    shape: _Shape,
    fill_value: bool | int | float | complex,
    *,
    dtype: _DType | None = None,
    device: _Device | None = None,
) -> NamedArray[_ShapeType, _DType]:
    xp = _maybe_default_namespace()
    _data = xp.full(shape, fill_value, dtype=dtype, device=device)
    _dims = _infer_dims(_data.shape)
    return NamedArray(_dims, _data)


def full_like(
    x: NamedArray[_ShapeType, _DType],
    fill_value: bool | int | float | complex,
    /,
    *,
    dtype: _DType | None = None,
    device: _Device | None = None,
) -> NamedArray[_ShapeType, _DType]:
    xp = _get_data_namespace(x)
    _dtype = x.dtype if dtype is None else dtype
    _device = x.device if device is None else device
    _data = xp.full(x.shape, fill_value, dtype=_dtype, device=_device)
    return x._new(data=_data)


def linspace(
    start: int | float | complex,
    stop: int | float | complex,
    /,
    num: int,
    *,
    dtype: _DType | None = None,
    device: _Device | None = None,
    endpoint: bool = True,
) -> NamedArray[_ShapeType, _DType]:
    xp = _maybe_default_namespace()
    _data = xp.linspace(
        start, stop, num=num, dtype=dtype, device=device, endpoint=endpoint
    )
    _dims = _infer_dims(_data.shape)
    return NamedArray(_dims, _data)


def ones(
    shape: _Shape, *, dtype: _DType | None = None, device: _Device | None = None
) -> NamedArray[_ShapeType, _DType]:
    return full(shape, 1, dtype=dtype, device=device)


def ones_like(
    x: NamedArray[_ShapeType, _DType],
    /,
    *,
    dtype: _DType | None = None,
    device: _Device | None = None,
) -> NamedArray[_ShapeType, _DType]:
    xp = _get_data_namespace(x)
    _dtype = x.dtype if dtype is None else dtype
    _device = x.device if device is None else device
    _data = xp.ones(x.shape, dtype=_dtype, device=_device)
    return x._new(data=_data)


def zeros(
    shape: _Shape, *, dtype: _DType | None = None, device: _Device | None = None
) -> NamedArray[_ShapeType, _DType]:
    return full(shape, 0, dtype=dtype, device=device)


def zeros_like(
    x: NamedArray[_ShapeType, _DType],
    /,
    *,
    dtype: _DType | None = None,
    device: _Device | None = None,
) -> NamedArray[_ShapeType, _DType]:
    xp = _get_data_namespace(x)
    _dtype = x.dtype if dtype is None else dtype
    _device = x.device if device is None else device
    _data = xp.zeros(x.shape, dtype=_dtype, device=_device)
    return x._new(data=_data)
