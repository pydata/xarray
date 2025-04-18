from __future__ import annotations

from typing import TYPE_CHECKING, Any

from xarray.namedarray._array_api._utils import (
    _dim_to_optional_axis,
    _get_data_namespace,
    _infer_dims,
    _reduce_dims,
    _dims_to_axis,
)
from xarray.namedarray._typing import (
    Default,
    _arrayapi,
    _default,
    _Dim,
    _DimsLike2,
)
from xarray.namedarray.core import (
    NamedArray,
)

if TYPE_CHECKING:
    from typing import Literal


def argmax(
    x: NamedArray[Any, Any],
    /,
    *,
    dim: _Dim | Default = _default,
    keepdims: bool = False,
    axis: int | None = None,
) -> NamedArray[Any, Any]:
    xp = _get_data_namespace(x)
    _axis = _dim_to_optional_axis(x, dim, axis)
    _data = xp.argmax(x._data, axis=_axis, keepdims=keepdims)
    _dims = _reduce_dims(x.dims, axis=_axis, keepdims=keepdims)
    return x._new(dims=_dims, data=_data)


def argmin(
    x: NamedArray[Any, Any],
    /,
    *,
    dim: _Dim | Default = _default,
    keepdims: bool = False,
    axis: int | None = None,
) -> NamedArray[Any, Any]:
    xp = _get_data_namespace(x)
    _axis = _dim_to_optional_axis(x, dim, axis)
    _data = xp.argmin(x._data, axis=_axis, keepdims=keepdims)
    _dims = _reduce_dims(x.dims, axis=_axis, keepdims=keepdims)
    return x._new(dims=_dims, data=_data)


def count_nonzero(
    x: NamedArray[Any, Any],
    /,
    *,
    dims: _DimsLike2 | Default = _default,
    keepdims: bool = False,
    axis: int | tuple[int, ...] | None = None,
) -> NamedArray[Any, Any]:
    """
    Counts the number of array elements which are non-zero.

    Examples
    --------
    >>> import numpy as np
    >>> x = NamedArray(("x", "y"), np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]))
    >>> count_nonzero(x)
    <xarray.NamedArray ()> Size: 8B
    array(5)
    >>> count_nonzero(x, dims="x")
    <xarray.NamedArray (y: 3)> Size: 24B
    array([1, 2, 2])
    """
    xp = _get_data_namespace(x)
    _axis = _dims_to_axis(x, dims, axis)
    _data = xp.count_nonzero(x._data, axis=_axis, keepdims=keepdims)
    _data = xp.asarray(_data)  # TODO: np.count_nonzero returns an int.
    _dims = _reduce_dims(x.dims, axis=_axis, keepdims=keepdims)
    return NamedArray(_dims, _data)


def nonzero(x: NamedArray[Any, Any], /) -> tuple[NamedArray[Any, Any], ...]:
    xp = _get_data_namespace(x)
    _datas: tuple[_arrayapi[Any, Any], ...] = xp.nonzero(x._data)
    # TODO: Verify that dims and axis matches here:
    return tuple(
        x._new((dim,), data) for dim, data in zip(x.dims, _datas, strict=False)
    )


def searchsorted(
    x1: NamedArray[Any, Any],
    x2: NamedArray[Any, Any],
    /,
    *,
    side: Literal["left", "right"] = "left",
    sorter: NamedArray[Any, Any] | None = None,
) -> NamedArray[Any, Any]:
    xp = _get_data_namespace(x1)
    _data = xp.searchsorted(x1._data, x2._data, side=side, sorter=sorter)
    # TODO: Check dims, probably can do it smarter:
    _dims = _infer_dims(_data.shape)
    return NamedArray(_dims, _data)


def where(
    condition: NamedArray[Any, Any],
    x1: NamedArray[Any, Any],
    x2: NamedArray[Any, Any],
    /,
) -> NamedArray[Any, Any]:
    xp = _get_data_namespace(x1)
    _data = xp.where(condition._data, x1._data, x2._data)
    # TODO: Wrong, _dims should be either of the arguments. How to choose?
    _dims = _infer_dims(_data.shape)
    return NamedArray(_dims, _data)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
