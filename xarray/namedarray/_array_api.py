from types import ModuleType
from typing import Any

import numpy as np

from xarray.namedarray._typing import (
    _arrayapi,
    _DType,
    _ScalarType,
    _ShapeType,
    _SupportsImag,
    _SupportsReal,
)
from xarray.namedarray.core import NamedArray, _new


def _get_data_namespace(x: NamedArray[Any, Any]) -> ModuleType:
    if isinstance(x._data, _arrayapi):
        return x._data.__array_namespace__()
    else:
        return np


def astype(
    x: NamedArray[Any, Any], dtype: _DType, /, *, copy: bool = True
) -> NamedArray[Any, _DType]:
    if isinstance(x._data, _arrayapi):
        xp = x._data.__array_namespace__()

        return _new(x, x._dims, xp.astype(x, dtype, copy=copy), x._attrs)  # type: ignore[no-any-return]

    # np.astype doesn't exist yet:
    return _new(x, data=x.astype(dtype, copy=copy))  # type: ignore[no-any-return, attr-defined]


def imag(
    x: NamedArray[_ShapeType, np.dtype[_SupportsImag[_ScalarType]]], /  # type: ignore[type-var]
) -> NamedArray[_ShapeType, np.dtype[_ScalarType]]:
    xp = _get_data_namespace(x)
    return _new(x, data=xp.imag(x._data))


def real(
    x: NamedArray[_ShapeType, np.dtype[_SupportsReal[_ScalarType]]], /  # type: ignore[type-var]
) -> NamedArray[_ShapeType, np.dtype[_ScalarType]]:
    xp = _get_data_namespace(x)
    return _new(x, xp.real(x._data))


a = NamedArray(("x",), np.array([1 + 3j, 2 + 2j, 3 + 3j], dtype=np.complex64))
reveal_type(a)
reveal_type(a.data)


xp = _get_data_namespace(a)
reveal_type(xp)

# def _real
# _func: Callable[
#     NamedArray[_ShapeType, np.dtype[_SupportsReal[_ScalarType]]],
#     NamedArray[_ShapeType, np.dtype[_ScalarType]],
# ] = xp.imag
# reveal_type(_func)

b = real(a)
reveal_type(b)
reveal_type(b.data)

b = imag(a)
reveal_type(b)
reveal_type(b.data)
