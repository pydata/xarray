from __future__ import annotations

from typing import Any, cast

from xarray.namedarray._array_api._utils import (
    _get_data_namespace,
    _get_namespace_dtype,
)
from xarray.namedarray._typing import (
    _arrayapi,
    _Device,
    _DType,
    _dtype,
    _FInfo,
    _IInfo,
    _ShapeType,
)
from xarray.namedarray.core import NamedArray


def astype(
    x: NamedArray[_ShapeType, Any],
    dtype: _DType,
    /,
    *,
    copy: bool = True,
    device: _Device | None = None,
) -> NamedArray[_ShapeType, _DType]:
    """
    Copies an array to a specified data type irrespective of Type Promotion Rules rules.

    Parameters
    ----------
    x : NamedArray
        Array to cast.
    dtype : _DType
        Desired data type.
    copy : bool, optional
        Specifies whether to copy an array when the specified dtype matches the data
        type of the input array x.
        If True, a newly allocated array must always be returned.
        If False and the specified dtype matches the data type of the input array,
        the input array must be returned; otherwise, a newly allocated array must be
        returned. Default: True.

    Returns
    -------
    out : NamedArray
        An array having the specified data type. The returned array must have the
        same shape as x.

    Examples
    --------
    >>> narr = NamedArray(("x",), np.asarray([1.5, 2.5]))
    >>> narr
    <xarray.NamedArray (x: 2)> Size: 16B
    array([1.5, 2.5])
    >>> astype(narr, np.dtype(np.int32))
    <xarray.NamedArray (x: 2)> Size: 8B
    array([1, 2], dtype=int32)
    """
    if isinstance(x._data, _arrayapi):
        xp = x._data.__array_namespace__()
        return x._new(data=xp.astype(x._data, dtype, copy=copy))

    # np.astype doesn't exist yet:
    return x._new(data=x._data.astype(dtype, copy=copy))  # type: ignore[attr-defined]


def can_cast(from_: _dtype[Any] | NamedArray[Any, Any], to: _dtype[Any], /) -> bool:
    if isinstance(from_, NamedArray):
        xp = _get_data_namespace(from_)
        from_ = from_.dtype
        return cast(bool, xp.can_cast(from_, to))  # TODO: Why is cast necessary?
    else:
        xp = _get_namespace_dtype(from_)
        return cast(bool, xp.can_cast(from_, to))  # TODO: Why is cast necessary?


def finfo(type: _dtype[Any] | NamedArray[Any, Any], /) -> _FInfo:
    if isinstance(type, NamedArray):
        xp = _get_data_namespace(type)
        return cast(_FInfo, xp.finfo(type._data))  # TODO: Why is cast necessary?
    else:
        xp = _get_namespace_dtype(type)
        return cast(_FInfo, xp.finfo(type))  # TODO: Why is cast necessary?


def iinfo(type: _dtype[Any] | NamedArray[Any, Any], /) -> _IInfo:
    if isinstance(type, NamedArray):
        xp = _get_data_namespace(type)
        return cast(_IInfo, xp.iinfo(type._data))  # TODO: Why is cast necessary?
    else:
        xp = _get_namespace_dtype(type)
        return cast(_IInfo, xp.iinfo(type))  # TODO: Why is cast necessary?


def isdtype(
    dtype: _dtype[Any], kind: _dtype[Any] | str | tuple[_dtype[Any] | str, ...]
) -> bool:
    xp = _get_namespace_dtype(dtype)
    return cast(bool, xp.isdtype(dtype, kind))  # TODO: Why is cast necessary?


def result_type(*arrays_and_dtypes: NamedArray[Any, Any] | _dtype[Any]) -> _dtype[Any]:
    # TODO: Empty arg?
    arr_or_dtype = arrays_and_dtypes[0]
    if isinstance(arr_or_dtype, NamedArray):
        xp = _get_data_namespace(arr_or_dtype)
    else:
        xp = _get_namespace_dtype(arr_or_dtype)

    return cast(
        _dtype[Any],
        xp.result_type(
            *(a.dtype if isinstance(a, NamedArray) else a for a in arrays_and_dtypes)
        ),
    )  # TODO: Why is cast necessary?
