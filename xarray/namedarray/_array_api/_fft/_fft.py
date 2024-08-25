from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, NamedTuple

from xarray.namedarray._array_api._utils import (
    _get_data_namespace,
    _infer_dims,
    _maybe_default_namespace,
)
from xarray.namedarray.core import NamedArray

if TYPE_CHECKING:
    from xarray.namedarray._typing import _Axes, _Axis, _DType, _Device

    _Norm = Literal["backward", "ortho", "forward"]

from xarray.namedarray._array_api._dtypes import (
    _floating_dtypes,
    _real_floating_dtypes,
    _complex_floating_dtypes,
    float32,
    complex64,
)
from xarray.namedarray._array_api._data_type_functions import astype


def fft(
    x: NamedArray,
    /,
    *,
    n: int | None = None,
    axis: _Axis = -1,
    norm: _Norm = "backward",
) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = xp.fft.fft(x._data, n=n, axis=axis, norm=norm)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def ifft(
    x: NamedArray,
    /,
    *,
    n: int | None = None,
    axis: _Axis = -1,
    norm: _Norm = "backward",
) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = xp.fft.ifft(x._data, n=n, axis=axis, norm=norm)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def fftn(
    x: NamedArray,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _Norm = "backward",
) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = xp.fft.fftn(x._data, s=s, axes=axes, norm=norm)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def ifftn(
    x: NamedArray,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _Norm = "backward",
) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = xp.fft.ifftn(x._data, s=s, axes=axes, norm=norm)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def rfft(
    x: NamedArray,
    /,
    *,
    n: int | None = None,
    axis: _Axis = -1,
    norm: _Norm = "backward",
) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = xp.fft.rfft(x._data, n=n, axis=axis, norm=norm)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def irfft(
    x: NamedArray,
    /,
    *,
    n: int | None = None,
    axis: _Axis = -1,
    norm: _Norm = "backward",
) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = xp.fft.irfft(x._data, n=n, axis=axis, norm=norm)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def rfftn(
    x: NamedArray,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _Norm = "backward",
) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = xp.fft.rfftn(x._data, s=s, axes=axes, norm=norm)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def irfftn(
    x: NamedArray,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _Norm = "backward",
) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = xp.fft.irfftn(x._data, s=s, axes=axes, norm=norm)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def hfft(
    x: NamedArray,
    /,
    *,
    n: int | None = None,
    axis: _Axis = -1,
    norm: _Norm = "backward",
) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = xp.fft.hfft(x._data, n=n, axis=axis, norm=norm)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def ihfft(
    x: NamedArray,
    /,
    *,
    n: int | None = None,
    axis: _Axis = -1,
    norm: _Norm = "backward",
) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = xp.fft.ihfft(x._data, n=n, axis=axis, norm=norm)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def fftfreq(n: int, /, *, d: float = 1.0, device: _Device | None = None) -> NamedArray:
    xp = _maybe_default_namespace()  # TODO: Can use device?
    _data = xp.fft.fftfreq(n, d=d, device=device)
    _dims = _infer_dims(_data.shape)
    return NamedArray(_dims, _data)


def rfftfreq(n: int, /, *, d: float = 1.0, device: _Device | None = None) -> NamedArray:
    xp = _maybe_default_namespace()  # TODO: Can use device?
    _data = xp.fft.rfftfreq(n, d=d, device=device)
    _dims = _infer_dims(_data.shape)
    return NamedArray(_dims, _data)


def fftshift(x: NamedArray, /, *, axes: _Axes | None = None) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = xp.fft.fftshift(x._data, axes=axes)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def ifftshift(x: NamedArray, /, *, axes: _Axes | None = None) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = xp.fft.ifftshift(x._data, axes=axes)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)
