from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, NamedTuple

from xarray.namedarray._array_api._utils import _get_data_namespace, _infer_dims
from xarray.namedarray.core import NamedArray

if TYPE_CHECKING:
    from xarray.namedarray._typing import _Axes, _Axis, _DType


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
    eigvals, eigvecs = xp.linalg.eigh(x._data)
    _dims_vals = _infer_dims(eigvals.shape)  # TODO: Fix dims
    _dims_vecs = _infer_dims(eigvecs.shape)  # TODO: Fix dims
    return EighResult(
        x._new(_dims_vals, eigvals),
        x._new(_dims_vecs, eigvecs),
    )


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
    q, r = xp.linalg.qr(x._data)
    _dims_q = _infer_dims(q.shape)  # TODO: Fix dims
    _dims_r = _infer_dims(r.shape)  # TODO: Fix dims
    return QRResult(
        x._new(_dims_q, q),
        x._new(_dims_r, r),
    )


def slogdet(x: NamedArray, /) -> SlogdetResult:
    xp = _get_data_namespace(x)
    sign, logabsdet = xp.linalg.slogdet(x._data)
    _dims_sign = _infer_dims(sign.shape)  # TODO: Fix dims
    _dims_logabsdet = _infer_dims(logabsdet.shape)  # TODO: Fix dims
    return SlogdetResult(
        x._new(_dims_sign, sign),
        x._new(_dims_logabsdet, logabsdet),
    )


def solve(x1: NamedArray, x2: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x1)
    _data = xp.linalg.solve(x1._data, x2._data)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x1._new(_dims, _data)


def svd(x: NamedArray, /, *, full_matrices: bool = True) -> SVDResult:
    xp = _get_data_namespace(x)
    u, s, vh = xp.linalg.svd(x._data, full_matrices=full_matrices)
    _dims_u = _infer_dims(u.shape)  # TODO: Fix dims
    _dims_s = _infer_dims(s.shape)  # TODO: Fix dims
    _dims_vh = _infer_dims(vh.shape)  # TODO: Fix dims
    return SVDResult(
        x._new(_dims_u, u),
        x._new(_dims_s, s),
        x._new(_dims_vh, vh),
    )


def svdvals(x: NamedArray, /) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = xp.linalg.svdvals(x._data)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def trace(
    x: NamedArray, /, *, offset: int = 0, dtype: _DType | None = None
) -> NamedArray:
    xp = _get_data_namespace(x)
    _data = xp.linalg.trace(x._data, offset=offset)
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
    _data = xp.linalg.vector_norm(x._data, axis=axis, keepdims=keepdims, ord=ord)
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
