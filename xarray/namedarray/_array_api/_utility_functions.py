from __future__ import annotations

from typing import Any

from xarray.namedarray._array_api._utils import (
    _dim_to_axis,
    _dims_to_axis,
    _get_data_namespace,
    _reduce_dims,
)
from xarray.namedarray._typing import (
    Default,
    _arrayapi,
    _Axis,
    _AxisLike,
    _default,
    _Dim,
    _DimsLike2,
)
from xarray.namedarray.core import NamedArray


def all(
    x: NamedArray[Any, Any],
    /,
    *,
    dims: _DimsLike2 | Default = _default,
    keepdims: bool = False,
    axis: _AxisLike | None = None,
) -> NamedArray[Any, Any]:
    """
    Tests whether all input array elements evaluate to True along a specified axis.

    Examples
    --------
    >>> import numpy as np
    >>> x = NamedArray(("x", "y"), np.array([[True, False], [True, True]]))
    >>> all(x)
    <xarray.NamedArray ()> Size: 1B
    np.False_

    >>> all(x, axis=0)
    <xarray.NamedArray (y: 2)> Size: 2B
    array([ True, False])
    >>> all(x, dims="x")
    <xarray.NamedArray (y: 2)> Size: 2B
    array([ True, False])

    >>> x = NamedArray(("x",), np.array([-1, 4, 5]))
    >>> all(x)
    <xarray.NamedArray ()> Size: 1B
    np.True_
    """
    xp = _get_data_namespace(x)
    _axis = _dims_to_axis(x, dims, axis)
    _data = xp.all(x._data, axis=_axis, keepdims=keepdims)
    _dims = _reduce_dims(x.dims, axis=_axis, keepdims=keepdims)
    return NamedArray(_dims, _data)


def any(
    x: NamedArray[Any, Any],
    /,
    *,
    dims: _DimsLike2 | Default = _default,
    keepdims: bool = False,
    axis: _AxisLike | None = None,
) -> NamedArray[Any, Any]:
    """
    Tests whether any input array element evaluates to True along a specified axis.

    Examples
    --------
    >>> import numpy as np
    >>> x = NamedArray(("x", "y"), np.array([[True, False], [True, True]]))
    >>> any(x)
    <xarray.NamedArray ()> Size: 1B
    np.True_

    >>> x = NamedArray(("x", "y"), np.array([[True, False], [False, False]]))
    >>> any(x, axis=0)
    <xarray.NamedArray (y: 2)> Size: 2B
    array([ True, False])
    >>> any(x, dims="x")
    <xarray.NamedArray (y: 2)> Size: 2B
    array([ True, False])

    >>> x = NamedArray(("x",), np.array([-1, 4, 5]))
    >>> any(x)
    <xarray.NamedArray ()> Size: 1B
    np.True_
    """
    xp = _get_data_namespace(x)
    _axis = _dims_to_axis(x, dims, axis)
    _data = xp.any(x._data, axis=_axis, keepdims=keepdims)
    _dims = _reduce_dims(x.dims, axis=_axis, keepdims=keepdims)
    return NamedArray(_dims, _data)


def diff(
    x: NamedArray[Any, Any],
    /,
    *,
    dims: _Dim | Default = _default,
    n: int = 1,
    prepend: NamedArray[Any, Any] | None = None,
    append: NamedArray[Any, Any] | None = None,
    axis: _Axis = -1,
) -> NamedArray[Any, Any]:
    """
    Calculates the n-th discrete forward difference along a specified axis.

    Examples
    --------
    >>> import numpy as np
    >>> x = NamedArray(("x",), np.array([1, 2, 4, 7, 0]))
    >>> diff(x)
    <xarray.NamedArray (x: 4)> Size: 32B
    array([ 1,  2,  3, -7])
    >>> diff(x, n=2)
    <xarray.NamedArray (x: 3)> Size: 24B
    array([  1,   1, -10])

    >>> x = NamedArray(("x", "y"), np.array([[1, 3, 6, 10], [0, 5, 6, 8]]))
    >>> diff(x)
    <xarray.NamedArray (x: 2, y: 3)> Size: 48B
    array([[2, 3, 4],
           [5, 1, 2]])
    >>> diff(x, axis=0)
    <xarray.NamedArray (x: 1, y: 4)> Size: 32B
    array([[-1,  2,  0, -2]])
    >>> diff(x, dims="x")
    <xarray.NamedArray (x: 1, y: 4)> Size: 32B
    array([[-1,  2,  0, -2]])
    """
    xp = _get_data_namespace(x)
    _axis = _dim_to_axis(x, dims, axis, axis_default=-1)
    try:
        _data = xp.diff(x._data, axis=_axis, n=n, prepend=prepend, append=append)
    except TypeError as err:
        # NumPy does not support prepend=None or append=None
        kwargs: dict[str, int | _arrayapi] = {"axis": _axis, "n": n}
        if prepend is not None:
            if prepend.device != x.device:
                raise ValueError(
                    f"Arrays from two different devices ({prepend.device} and {x.device}) can not be combined."
                ) from err
            kwargs["prepend"] = prepend._data
        if append is not None:
            if append.device != x.device:
                raise ValueError(
                    f"Arrays from two different devices ({append.device} and {x.device}) can not be combined."
                ) from err
            kwargs["append"] = append._data

        _data = xp.diff(x._data, **kwargs)

    return NamedArray(x.dims, _data)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
