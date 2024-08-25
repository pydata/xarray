from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple, Literal

from xarray.namedarray._array_api._utils import _get_data_namespace, _infer_dims
from xarray.namedarray.core import NamedArray
from xarray.namedarray._array_api._dtypes import (
    _floating_dtypes,
    _numeric_dtypes,
    float32,
    complex64,
    complex128,
)
from xarray.namedarray._array_api._data_type_functions import finfo
from xarray.namedarray._array_api._manipulation_functions import reshape
from xarray.namedarray._array_api._elementwise_functions import conj


if TYPE_CHECKING:
    from xarray.namedarray._typing import _Axis, _DType, _Axes


class EighResult(NamedTuple):
    eigenvalues: NamedArray
    eigenvectors: NamedArray


class QRResult(NamedTuple):
    Q: NamedArray
    R: NamedArray


class SlogdetResult(NamedTuple):
    sign: NamedArray
    logabsdet: NamedArray


class SVDResult(NamedTuple):
    U: NamedArray
    S: NamedArray
    Vh: NamedArray


def cholesky(x: NamedArray, /, *, upper: bool = False) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = xp.linalg.cholesky(x._data, upper=upper)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


# Note: cross is the numpy top-level namespace, not np.linalg
def cross(x1: NamedArray, x2: NamedArray, /, *, axis: _Axis = -1) -> NamedArray:
    xp = _get_data_namespace(x1)
    _data = xp.linalg.cross(x1._data, x2._data, axis=axis)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x1._new(_dims, _data)


def det(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = xp.linalg.det(x._data)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def diagonal(x: NamedArray, /, *, offset: int = 0) -> NamedArray:
    # Note: diagonal is the numpy top-level namespace, not np.linalg
    xp = _get_data_namespace(x)
    _data = xp.linalg.diagonal(x._data, offset=offset)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def eigh(x: NamedArray, /) -> EighResult:
    xp = _get_data_namespace(x)
    _datas = xp.linalg.eigh(x._data)
    _dims = _infer_dims(_datas[0].shape)  # TODO: Fix dims
    return EighResult(*(x._new(_dims, _data) for _data in _datas))


def eigvalsh(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = xp.linalg.eigvalsh(x._data)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def inv(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = xp.linalg.inv(x._data)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def matrix_norm(
    x: NamedArray,
    /,
    *,
    keepdims: bool = False,
    ord: int | float | Literal["fro", "nuc"] | None = "fro",
) -> NamedArray:  # noqa: F821
    xp = _get_data_namespace(x)
    _data = xp.linalg.matrix_norm(x._data, keepdims=keepdims, ord=ord)  # ckeck xp.mean
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def matrix_power(x: NamedArray, n: int, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = xp.linalg.matrix_power(x._data, n=n)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def matrix_rank(
    x: NamedArray, /, *, rtol: float | NamedArray | None = None
) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = xp.linalg.matrix_rank(x._data, rtol=rtol)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def outer(x1: NamedArray, x2: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x1)
    _data = xp.linalg.outer(x1._data, x2._data)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x1._new(_dims, _data)


def pinv(x: NamedArray, /, *, rtol: float | NamedArray | None = None) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = xp.linalg.pinv(x._data, rtol=rtol)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def qr(
    x: NamedArray, /, *, mode: Literal["reduced", "complete"] = "reduced"
) -> QRResult:
    xp = _get_data_namespace(x)
    _datas = xp.linalg.qr(x._data)
    _dims = _infer_dims(_datas[0].shape)  # TODO: Fix dims
    return QRResult(*(x._new(_dims, _data) for _data in _datas))


def slogdet(x: NamedArray, /) -> SlogdetResult:
    xp = _get_data_namespace(x)
    _datas = xp.linalg.slogdet(x._data)
    _dims = _infer_dims(_datas[0].shape)  # TODO: Fix dims
    return SlogdetResult(*(x._new(_dims, _data) for _data in _datas))


def solve(x1: NamedArray, x2: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x1)
    _data = xp.linalg.solve(x1._data, x2._data)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x1._new(_dims, _data)


def svd(x: NamedArray, /, *, full_matrices: bool = True) -> SVDResult:
    xp = _get_data_namespace(x)
    _datas = xp.linalg.svd(x._data, full_matrices=full_matrices)
    _dims = _infer_dims(_datas[0].shape)  # TODO: Fix dims
    return SVDResult(*(x._new(_dims, _data) for _data in _datas))


def svdvals(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = xp.linalg.svdvals(x._data)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def trace(
    x: NamedArray, /, *, offset: int = 0, dtype: _DType | None = None
) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = xp.linalg.svdvals(x._data, offset=offset)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def vector_norm(
    x: NamedArray,
    /,
    *,
    axis: _Axes | None = None,
    keepdims: bool = False,
    ord: int | float | None = 2,
) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = xp.linalg.svdvals(x._data, axis=axis, keepdims=keepdims, ord=ord)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def matmul(x1: NamedArray, x2: NamedArray, /) -> NamedArray:
    from xarray.namedarray._array_api._linear_algebra_functions import matmul

    return matmul(x1, x2)


def tensordot(
    x1: NamedArray,
    x2: NamedArray,
    /,
    *,
    axes: int | tuple[Sequence[int], Sequence[int]] = 2,
) -> NamedArray:
    from xarray.namedarray._array_api._linear_algebra_functions import tensordot

    return tensordot(x1, x2, axes=axes)


def matrix_transpose(x: NamedArray, /) -> NamedArray:
    from xarray.namedarray._array_api._linear_algebra_functions import matrix_transpose

    return matrix_transpose(x)


def vecdot(x1: NamedArray, x2: NamedArray, /, *, axis: _Axis = -1) -> NamedArray:
    from xarray.namedarray._array_api._linear_algebra_functions import vecdot

    return vecdot(x1, x2, axis=axis)
