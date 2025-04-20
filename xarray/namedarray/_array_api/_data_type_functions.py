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
    >>> import numpy as np
    >>> x = NamedArray(("x",), np.asarray([1.5, 2.5]))
    >>> x
    <xarray.NamedArray (x: 2)> Size: 16B
    array([1.5, 2.5])
    >>> astype(x, np.dtype(np.int32))
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
    """
    Machine limits for floating-point data types.

    Examples
    --------
    >>> import numpy as np
    >>> dtype = np.float32
    >>> x = NamedArray((), np.array(1, dtype=dtype))
    >>> finfo(dtype)
    finfo(resolution=1e-06, min=-3.4028235e+38, max=3.4028235e+38, dtype=float32)
    >>> finfo(x)
    finfo(resolution=1e-06, min=-3.4028235e+38, max=3.4028235e+38, dtype=float32)
    """
    if isinstance(type, NamedArray):
        xp = _get_data_namespace(type)
        try:
            return cast(_FInfo, xp.finfo(type._data))  # TODO: Why is cast necessary?
        except ValueError:
            # TODO: numpy 2.2 does not support arrays as input.
            # TODO: Why is cast necessary?
            return cast(_FInfo, xp.finfo(type._data.dtype))
    else:
        xp = _get_namespace_dtype(type)
        return cast(_FInfo, xp.finfo(type))  # TODO: Why is cast necessary?


def iinfo(type: _dtype[Any] | NamedArray[Any, Any], /) -> _IInfo:
    """
    Machine limits for integer data types.

    Examples
    --------
    >>> import numpy as np
    >>> dtype = np.int8
    >>> x = NamedArray((), np.array(1, dtype=dtype))
    >>> iinfo(dtype)
    iinfo(min=-128, max=127, dtype=int8)
    >>> iinfo(x)
    iinfo(min=-128, max=127, dtype=int8)
    """
    if isinstance(type, NamedArray):
        xp = _get_data_namespace(type)
        try:
            return cast(_IInfo, xp.iinfo(type._data))  # TODO: Why is cast necessary?
        except ValueError:
            # TODO: numpy 2.2 does not support arrays as input.
            # TODO: Why is cast necessary?
            return cast(_IInfo, xp.iinfo(type._data.dtype))
    else:
        xp = _get_namespace_dtype(type)
        return cast(_IInfo, xp.iinfo(type))  # TODO: Why is cast necessary?


def isdtype(
    dtype: _dtype[Any], kind: _dtype[Any] | str | tuple[_dtype[Any] | str, ...]
) -> bool:
    xp = _get_namespace_dtype(dtype)
    return cast(bool, xp.isdtype(dtype, kind))  # TODO: Why is cast necessary?


def result_type(
    *arrays_and_dtypes: NamedArray[Any, Any]
    | int
    | float
    | complex
    | bool
    | _dtype[Any],
) -> _dtype[Any]:
    """
    Returns the dtype that results from applying type promotion rules to the arguments.

    Examples
    --------

    Array and Array

    >>> import numpy as np
    >>> x1 = NamedArray((), np.array(-2, dtype=np.int8))
    >>> x2 = NamedArray((), np.array(3, dtype=np.int64))
    >>> result_type(x1, x2)
    dtype('int64')

    Array and DType

    >>> x1 = NamedArray((), np.array(-2, dtype=np.int8))
    >>> x2 = np.int64
    >>> result_type(x1, x2)
    dtype('int64')

    Scalar and Array

    >>> x1 = 3
    >>> x2 = NamedArray((), np.array(-2, dtype=np.int8))
    >>> result_type(x1, x2)
    dtype('int8')

    Scalar and Scalar uses the default namespace

    >>> result_type(3.0, 2)
    dtype('float64')
    """
    dtypes = []
    scalars = []
    for a in arrays_and_dtypes:
        if isinstance(a, NamedArray):
            dtypes.append(a.dtype)
        elif isinstance(a, (bool, int, float, complex)):
            scalars.append(a)
        else:
            dtypes.append(a)

    # if not dtypes:
    #     # Need at least 1 array or dtype to retrieve namespace otherwise need to use
    #     # the default namespace.
    #     raise ValueError("at least one array or dtype is required")

    xp = _get_namespace_dtype(next(iter(dtypes), None))

    # TODO: Why is cast necessary?
    return cast(_dtype[Any], xp.result_type(*dtypes + scalars))


# %% NamedArray helpers
def _promote_scalars(
    x1: NamedArray[Any, Any] | bool | int | float | complex,
    x2: NamedArray[Any, Any] | bool | int | float | complex,
    dtype_category: str | None = None,
    func_name: str = "",
) -> tuple[NamedArray[Any, Any], NamedArray[Any, Any]]:
    """
    Promote at most one of x1 or x2 to an array from a Python scalar

    Examples
    --------
    >>> import numpy as np
    >>> x = NamedArray((), np.array(-2, dtype=np.int8))
    >>> _promote_scalars(x, 3)
    (<xarray.NamedArray ()> Size: 1B
    array(-2, dtype=int8), <xarray.NamedArray ()> Size: 1B
    array(3, dtype=int8))

    """
    x1_is_scalar = isinstance(x1, bool | int | float | complex)
    x2_is_scalar = isinstance(x2, bool | int | float | complex)
    if x1_is_scalar and x2_is_scalar:
        raise TypeError(f"At least one of x1 and x2 must be an array in {func_name}")
    elif x1_is_scalar:
        x1 = x2._promote_scalar(x1, dtype_category, func_name)
    elif x2_is_scalar:
        x2 = x1._promote_scalar(x2, dtype_category, func_name)
    return x1, x2


# %% Test
if __name__ == "__main__":
    import doctest

    doctest.testmod()
