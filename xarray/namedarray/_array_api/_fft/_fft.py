from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from xarray.namedarray._array_api._utils import (
    _get_data_namespace,
    _infer_dims,
    _maybe_default_namespace,
)
from xarray.namedarray.core import NamedArray

if TYPE_CHECKING:
    from xarray.namedarray._typing import _Axes, _Axis, _Device, _DType

    _Norm = Literal["backward", "ortho", "forward"]


def fft(
    x: NamedArray[Any, _DType],
    /,
    *,
    n: int | None = None,
    axis: _Axis = -1,
    norm: _Norm = "backward",
) -> NamedArray[Any, _DType]:
    xp = _get_data_namespace(x)
    _data = xp.fft.fft(x._data, n=n, axis=axis, norm=norm)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def ifft(
    x: NamedArray[Any, _DType],
    /,
    *,
    n: int | None = None,
    axis: _Axis = -1,
    norm: _Norm = "backward",
) -> NamedArray[Any, _DType]:
    xp = _get_data_namespace(x)
    _data = xp.fft.ifft(x._data, n=n, axis=axis, norm=norm)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def fftn(
    x: NamedArray[Any, _DType],
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _Norm = "backward",
) -> NamedArray[Any, _DType]:
    xp = _get_data_namespace(x)
    _data = xp.fft.fftn(x._data, s=s, axes=axes, norm=norm)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def ifftn(
    x: NamedArray[Any, _DType],
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _Norm = "backward",
) -> NamedArray[Any, _DType]:
    xp = _get_data_namespace(x)
    _data = xp.fft.ifftn(x._data, s=s, axes=axes, norm=norm)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def rfft(
    x: NamedArray[Any, Any],
    /,
    *,
    n: int | None = None,
    axis: _Axis = -1,
    norm: _Norm = "backward",
) -> NamedArray[Any, Any]:
    xp = _get_data_namespace(x)
    _data = xp.fft.rfft(x._data, n=n, axis=axis, norm=norm)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def irfft(
    x: NamedArray[Any, Any],
    /,
    *,
    n: int | None = None,
    axis: _Axis = -1,
    norm: _Norm = "backward",
) -> NamedArray[Any, Any]:
    xp = _get_data_namespace(x)
    _data = xp.fft.irfft(x._data, n=n, axis=axis, norm=norm)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def rfftn(
    x: NamedArray[Any, Any],
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _Norm = "backward",
) -> NamedArray[Any, Any]:
    xp = _get_data_namespace(x)
    _data = xp.fft.rfftn(x._data, s=s, axes=axes, norm=norm)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def irfftn(
    x: NamedArray[Any, Any],
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _Norm = "backward",
) -> NamedArray[Any, Any]:
    xp = _get_data_namespace(x)
    _data = xp.fft.irfftn(x._data, s=s, axes=axes, norm=norm)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def hfft(
    x: NamedArray[Any, Any],
    /,
    *,
    n: int | None = None,
    axis: _Axis = -1,
    norm: _Norm = "backward",
) -> NamedArray[Any, Any]:
    xp = _get_data_namespace(x)
    _data = xp.fft.hfft(x._data, n=n, axis=axis, norm=norm)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def ihfft(
    x: NamedArray[Any, Any],
    /,
    *,
    n: int | None = None,
    axis: _Axis = -1,
    norm: _Norm = "backward",
) -> NamedArray[Any, Any]:
    xp = _get_data_namespace(x)
    _data = xp.fft.ihfft(x._data, n=n, axis=axis, norm=norm)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def fftfreq(
    n: int,
    /,
    *,
    d: float = 1.0,
    dtype: _DType | None = None,
    device: _Device | None = None,
) -> NamedArray[Any, Any]:
    xp = _maybe_default_namespace()  # TODO: Can use device?
    _data = xp.fft.fftfreq(n, d=d, dtype=dtype, device=device)
    _dims = _infer_dims(_data.shape)
    return NamedArray(_dims, _data)


def rfftfreq(
    n: int,
    /,
    *,
    d: float = 1.0,
    dtype: _DType | None = None,
    device: _Device | None = None,
) -> NamedArray[Any, Any]:
    xp = _maybe_default_namespace()  # TODO: Can use device?
    _data = xp.fft.rfftfreq(n, d=d, dtype=dtype, device=device)
    _dims = _infer_dims(_data.shape)
    return NamedArray(_dims, _data)


def fftshift(
    x: NamedArray[Any, _DType], /, *, axes: _Axes | None = None
) -> NamedArray[Any, _DType]:
    xp = _get_data_namespace(x)
    _data = xp.fft.fftshift(x._data, axes=axes)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def ifftshift(
    x: NamedArray[Any, _DType], /, *, axes: _Axes | None = None
) -> NamedArray[Any, _DType]:
    xp = _get_data_namespace(x)
    _data = xp.fft.ifftshift(x._data, axes=axes)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)
