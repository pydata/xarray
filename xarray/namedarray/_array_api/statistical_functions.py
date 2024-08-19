from __future__ import annotations

from typing import Any

from xarray.namedarray._array_api._utils import _get_data_namespace
from xarray.namedarray._typing import (
    Default,
    _AxisLike,
    _default,
    _Dims,
    _DType,
)
from xarray.namedarray.core import (
    NamedArray,
    _dims_to_axis,
    _get_remaining_dims,
)


def mean(
    x: NamedArray[Any, _DType],
    /,
    *,
    dims: _Dims | Default = _default,
    keepdims: bool = False,
    axis: _AxisLike | None = None,
) -> NamedArray[Any, _DType]:
    """
    Calculates the arithmetic mean of the input array x.

    Parameters
    ----------
    x :
        Should have a real-valued floating-point data type.
    dims :
        Dim or dims along which arithmetic means must be computed. By default,
        the mean must be computed over the entire array. If a tuple of hashables,
        arithmetic means must be computed over multiple axes.
        Default: None.
    keepdims :
        if True, the reduced axes (dimensions) must be included in the result
        as singleton dimensions, and, accordingly, the result must be compatible
        with the input array (see Broadcasting). Otherwise, if False, the
        reduced axes (dimensions) must not be included in the result.
        Default: False.
    axis :
        Axis or axes along which arithmetic means must be computed. By default,
        the mean must be computed over the entire array. If a tuple of integers,
        arithmetic means must be computed over multiple axes.
        Default: None.

    Returns
    -------
    out :
        If the arithmetic mean was computed over the entire array,
        a zero-dimensional array containing the arithmetic mean; otherwise,
        a non-zero-dimensional array containing the arithmetic means.
        The returned array must have the same data type as x.

    Examples
    --------
    >>> x = NamedArray(("x", "y"), nxp.asarray([[1.0, 2.0], [3.0, 4.0]]))
    >>> mean(x).data
    Array(2.5, dtype=float64)
    >>> mean(x, dims=("x",)).data
    Array([2., 3.], dtype=float64)

    Using keepdims:

    >>> mean(x, dims=("x",), keepdims=True)
    <xarray.NamedArray (x: 1, y: 2)>
    Array([[2., 3.]], dtype=float64)
    >>> mean(x, dims=("y",), keepdims=True)
    <xarray.NamedArray (x: 2, y: 1)>
    Array([[1.5],
           [3.5]], dtype=float64)
    """
    xp = _get_data_namespace(x)
    axis_ = _dims_to_axis(x, dims, axis)
    d = xp.mean(x._data, axis=axis_, keepdims=False)  # We fix keepdims later
    # TODO: Why do we need to do the keepdims ourselves?
    dims_, data_ = _get_remaining_dims(x, d, axis_, keepdims=keepdims)
    out = x._new(dims=dims_, data=data_)
    return out
