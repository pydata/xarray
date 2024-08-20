from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Any

from xarray.namedarray._typing import _arrayapi, _dtype

if TYPE_CHECKING:
    from xarray.namedarray.core import NamedArray


def _maybe_default_namespace(xp: ModuleType | None = None) -> ModuleType:
    if xp is None:
        import array_api_strict as xpd

        # import numpy as xpd

        return xpd
    else:
        return xp


def _get_data_namespace(x: NamedArray[Any, Any]) -> ModuleType:
    if isinstance(x._data, _arrayapi):
        return x._data.__array_namespace__()

    return _maybe_default_namespace()


def _get_namespace_dtype(dtype: _dtype) -> ModuleType:
    xp = __import__(dtype.__module__)
    return xp
