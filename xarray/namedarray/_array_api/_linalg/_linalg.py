from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, overload

from xarray.namedarray._array_api._utils import _get_data_namespace, _infer_dims
from xarray.namedarray.core import NamedArray

if TYPE_CHECKING:
    from xarray.namedarray._typing import _Axes, _Axis, _DType, _ShapeType


class EighResult(NamedTuple):
    eigenvalues: NamedArray[Any, Any]
    eigenvectors: NamedArray[Any, Any]


class QRResult(NamedTuple):
    Q: NamedArray[Any, Any]
    R: NamedArray[Any, Any]


class SlogdetResult(NamedTuple):
    sign: NamedArray[Any, Any]
    logabsdet: NamedArray[Any, Any]


class SVDResult(NamedTuple):
    U: NamedArray[Any, Any]
    S: NamedArray[Any, Any]
    Vh: NamedArray[Any, Any]


def cholesky(
    x: NamedArray[_ShapeType, Any], /, *, upper: bool = False
) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _data = xp.linalg.cholesky(x._data, upper=upper)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


# Note: cross is the numpy top-level namespace, not np.linalg
def cross(
    x1: NamedArray[Any, Any], x2: NamedArray[Any, Any], /, *, axis: _Axis = -1
) -> NamedArray[Any, Any]:
    xp = _get_data_namespace(x1)
    _data = xp.linalg.cross(x1._data, x2._data, axis=axis)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x1._new(_dims, _data)


def det(x: NamedArray[Any, _DType], /) -> NamedArray[Any, _DType]:
    xp = _get_data_namespace(x)
    _data = xp.linalg.det(x._data)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def diagonal(
    x: NamedArray[Any, _DType], /, *, offset: int = 0
) -> NamedArray[Any, _DType]:
    # Note: diagonal is the numpy top-level namespace, not np.linalg
    xp = _get_data_namespace(x)
    _data = xp.linalg.diagonal(x._data, offset=offset)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def eigh(x: NamedArray[Any, Any], /) -> EighResult:
    xp = _get_data_namespace(x)
    eigvals, eigvecs = xp.linalg.eigh(x._data)
    _dims_vals = _infer_dims(eigvals.shape)  # TODO: Fix dims
    _dims_vecs = _infer_dims(eigvecs.shape)  # TODO: Fix dims
    return EighResult(
        x._new(_dims_vals, eigvals),
        x._new(_dims_vecs, eigvecs),
    )


def eigvalsh(x: NamedArray[Any, Any], /) -> NamedArray[Any, Any]:
    xp = _get_data_namespace(x)
    _data = xp.linalg.eigvalsh(x._data)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def inv(x: NamedArray[_ShapeType, Any], /) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _data = xp.linalg.inv(x._data)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def matrix_norm(
    x: NamedArray[Any, Any],
    /,
    *,
    keepdims: bool = False,
    ord: int | float | Literal["fro", "nuc"] | None = "fro",
) -> NamedArray[Any, Any]:
    xp = _get_data_namespace(x)
    _data = xp.linalg.matrix_norm(x._data, keepdims=keepdims, ord=ord)  # check xp.mean
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def matrix_power(
    x: NamedArray[_ShapeType, Any], n: int, /
) -> NamedArray[_ShapeType, Any]:
    xp = _get_data_namespace(x)
    _data = xp.linalg.matrix_power(x._data, n=n)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def matrix_rank(
    x: NamedArray[Any, Any], /, *, rtol: float | NamedArray[Any, Any] | None = None
) -> NamedArray[Any, Any]:
    xp = _get_data_namespace(x)
    _data = xp.linalg.matrix_rank(x._data, rtol=rtol)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def matrix_transpose(x: NamedArray[Any, _DType], /) -> NamedArray[Any, _DType]:
    from xarray.namedarray._array_api._linear_algebra_functions import matrix_transpose

    return matrix_transpose(x)


def outer(
    x1: NamedArray[Any, Any], x2: NamedArray[Any, Any], /
) -> NamedArray[Any, Any]:
    xp = _get_data_namespace(x1)
    _data = xp.linalg.outer(x1._data, x2._data)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x1._new(_dims, _data)


def pinv(
    x: NamedArray[Any, Any], /, *, rtol: float | NamedArray[Any, Any] | None = None
) -> NamedArray[Any, Any]:
    xp = _get_data_namespace(x)
    _data = xp.linalg.pinv(x._data, rtol=rtol)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def qr(
    x: NamedArray[Any, Any], /, *, mode: Literal["reduced", "complete"] = "reduced"
) -> QRResult:
    xp = _get_data_namespace(x)
    q, r = xp.linalg.qr(x._data)
    _dims_q = _infer_dims(q.shape)  # TODO: Fix dims
    _dims_r = _infer_dims(r.shape)  # TODO: Fix dims
    return QRResult(
        x._new(_dims_q, q),
        x._new(_dims_r, r),
    )


def slogdet(x: NamedArray[Any, Any], /) -> SlogdetResult:
    xp = _get_data_namespace(x)
    sign, logabsdet = xp.linalg.slogdet(x._data)
    _dims_sign = _infer_dims(sign.shape)  # TODO: Fix dims
    _dims_logabsdet = _infer_dims(logabsdet.shape)  # TODO: Fix dims
    return SlogdetResult(
        x._new(_dims_sign, sign),
        x._new(_dims_logabsdet, logabsdet),
    )


def solve(
    x1: NamedArray[Any, Any], x2: NamedArray[Any, Any], /
) -> NamedArray[Any, Any]:
    xp = _get_data_namespace(x1)
    _data = xp.linalg.solve(x1._data, x2._data)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x1._new(_dims, _data)


def svd(x: NamedArray[Any, Any], /, *, full_matrices: bool = True) -> SVDResult:
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


def svdvals(x: NamedArray[Any, Any], /) -> NamedArray[Any, Any]:
    xp = _get_data_namespace(x)
    _data = xp.linalg.svdvals(x._data)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


@overload
def trace(
    x: NamedArray[Any, Any], /, *, offset: int = 0, dtype: _DType
) -> NamedArray[Any, _DType]: ...
@overload
def trace(
    x: NamedArray[Any, _DType], /, *, offset: int = 0, dtype: None
) -> NamedArray[Any, _DType]: ...
def trace(
    x: NamedArray[Any, _DType | Any], /, *, offset: int = 0, dtype: _DType | None = None
) -> NamedArray[Any, _DType]:
    xp = _get_data_namespace(x)
    _data = xp.linalg.trace(x._data, offset=offset)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)


def matmul(
    x1: NamedArray[Any, Any], x2: NamedArray[Any, Any], /
) -> NamedArray[Any, Any]:
    from xarray.namedarray._array_api._linear_algebra_functions import matmul

    return matmul(x1, x2)


def tensordot(
    x1: NamedArray[Any, Any],
    x2: NamedArray[Any, Any],
    /,
    *,
    axes: int | tuple[Sequence[int], Sequence[int]] = 2,
) -> NamedArray[Any, Any]:
    from xarray.namedarray._array_api._linear_algebra_functions import tensordot

    return tensordot(x1, x2, axes=axes)


def vecdot(
    x1: NamedArray[Any, Any], x2: NamedArray[Any, Any], /, *, axis: _Axis = -1
) -> NamedArray[Any, Any]:
    from xarray.namedarray._array_api._linear_algebra_functions import vecdot

    return vecdot(x1, x2, axis=axis)


def vector_norm(
    x: NamedArray[Any, Any],
    /,
    *,
    axis: _Axes | None = None,
    keepdims: bool = False,
    ord: int | float | None = 2,
) -> NamedArray[Any, Any]:
    xp = _get_data_namespace(x)
    _data = xp.linalg.vector_norm(x._data, axis=axis, keepdims=keepdims, ord=ord)
    _dims = _infer_dims(_data.shape)  # TODO: Fix dims
    return x._new(_dims, _data)
