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

        return _new(x, x._dims, xp.astype(x, dtype, copy=copy), x._attrs)

    # np.astype doesn't exist yet:
    return _new(x, data=x.astype(dtype, copy=copy))  # type: ignore[attr-defined]


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
