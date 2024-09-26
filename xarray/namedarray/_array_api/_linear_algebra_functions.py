from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from xarray.namedarray._array_api._utils import _get_data_namespace, _infer_dims
from xarray.namedarray.core import NamedArray


def matmul(
    x1: NamedArray[Any, Any], x2: NamedArray[Any, Any], /
) -> NamedArray[Any, Any]:
    """
    Matrix product of two arrays.

    Examples
    --------
    For 2-D arrays it is the matrix product:

    >>> import numpy as np
    >>> a = NamedArray(("y", "x"), np.array([[1, 0], [0, 1]]))
    >>> b = NamedArray(("y", "x"), np.array([[4, 1], [2, 2]]))
    >>> matmul(a, b)
    <Namedarray, shape=(2, 2), dims=('y', 'x'), dtype=int64, data=[[4 1]
     [2 2]]>

    For 2-D mixed with 1-D, the result is the usual.

    >>> a = NamedArray(("y", "x"), np.array([[1, 0], [0, 1]]))
    >>> b = NamedArray(("x",), np.array([1, 2]))
    >>> matmul(a, b)

    Broadcasting is conventional for stacks of arrays

    >>> a = NamedArray(("z", "y", "x"), np.arange(2 * 2 * 4).reshape((2, 2, 4)))
    >>> b = NamedArray(("z", "y", "x"), np.arange(2 * 2 * 4).reshape((2, 4, 2)))
    >>> axb = matmul(a, b)
    >>> axb.dims, axb.shape
    """
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
    """
    Transposes a matrix (or a stack of matrices) x.

    Examples
    --------
    >>> import numpy as np
    >>> x = NamedArray(("x", "y", "z"), np.zeros((2, 3, 4)))
    >>> xT = matrix_transpose(x)
    >>> xT.dims, xT.shape
    (('x', 'z', 'y'), (2, 4, 3))

    >>> x = NamedArray(("x", "y"), np.zeros((2, 3)))
    >>> xT = matrix_transpose(x)
    >>> xT.dims, xT.shape
    (('y', 'x'), (3, 2))
    """
    xp = _get_data_namespace(x)
    _data = xp.matrix_transpose(x._data)
    d = x.dims
    _dims = d[:-2] + d[-2:][::-1]  # (..., M, N) -> (..., N, M)
    return NamedArray(_dims, _data)


def vecdot(
    x1: NamedArray[Any, Any], x2: NamedArray[Any, Any], /, *, axis: int = -1
) -> NamedArray[Any, Any]:
    xp = _get_data_namespace(x1)
    _data = xp.vecdot(x1._data, x2._data, axis=axis)
    # TODO: Figure out a better way:
    _dims = _infer_dims(_data.shape)
    return NamedArray(_dims, _data)
