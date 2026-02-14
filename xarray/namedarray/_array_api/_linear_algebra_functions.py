from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from xarray.namedarray._array_api._utils import (
    _broadcast_dims,
    _get_data_namespace,
    _infer_dims,
    _reduce_dims,
)
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
    <xarray.NamedArray (y: 2, x: 2)> Size: 32B
    array([[4, 1],
           [2, 2]])

    For 2-D mixed with 1-D, the result is the usual.

    >>> x1 = NamedArray(("n", "k"), np.array([[1, 0, 0], [0, 1, 0]]))
    >>> x2 = NamedArray(("k",), np.array([1, 2, 3]))
    >>> matmul(x1, x2)
    <xarray.NamedArray (n: 2)> Size: 16B
    array([1, 2])

    Broadcasting is conventional for stacks of arrays

    >>> a = NamedArray(("z", "y", "x"), np.arange(2 * 2 * 4).reshape((2, 2, 4)))
    >>> b = NamedArray(("z", "y", "x"), np.arange(2 * 2 * 4).reshape((2, 4, 2)))
    >>> matmul(a, b)
    <xarray.NamedArray (z: 2, y: 2, x: 2)> Size: 64B
    array([[[ 28,  34],
            [ 76,  98]],
    <BLANKLINE>
           [[428, 466],
            [604, 658]]])
    """
    xp = _get_data_namespace(x1)
    _data = xp.matmul(x1._data, x2._data)
    _dims = x1.dims[:1] + x2.dims[1:]  # (n, k),(k, m) -> (n, m)
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
    >>> x = NamedArray(("x", "y", "z"), np.zeros((1, 2, 3)))
    >>> matrix_transpose(x)
    <xarray.NamedArray (x: 1, z: 3, y: 2)> Size: 48B
    array([[[0., 0.],
            [0., 0.],
            [0., 0.]]])

    >>> x = NamedArray(("x", "y"), np.zeros((2, 3)))
    >>> matrix_transpose(x)
    <xarray.NamedArray (y: 3, x: 2)> Size: 48B
    array([[0., 0.],
           [0., 0.],
           [0., 0.]])
    """
    xp = _get_data_namespace(x)
    _data = xp.matrix_transpose(x._data)
    d = x.dims
    _dims = d[:-2] + d[-2:][::-1]  # (..., M, N) -> (..., N, M)
    return NamedArray(_dims, _data)


def vecdot(
    x1: NamedArray[Any, Any], x2: NamedArray[Any, Any], /, *, axis: int = -1
) -> NamedArray[Any, Any]:
    """
    Computes the (vector) dot product of two arrays.

    Examples
    --------
    >>> import numpy as np
    >>> v = NamedArray(
    ...     ("y", "x"),
    ...     np.array(
    ...         [[0.0, 5.0, 0.0], [0.0, 0.0, 10.0], [0.0, 6.0, 8.0], [0.0, 6.0, 8.0]]
    ...     ),
    ... )
    >>> n = NamedArray(("x",), np.array([0.0, 0.6, 0.8]))
    >>> vecdot(v, n)
    <xarray.NamedArray (y: 4)> Size: 32B
    array([ 3.,  8., 10., 10.])
    """
    xp = _get_data_namespace(x1)
    _data = xp.vecdot(x1._data, x2._data, axis=axis)
    d, _ = _broadcast_dims(x1, x2)
    _dims = _reduce_dims(d, axis=axis, keepdims=False)
    return NamedArray(_dims, _data)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
