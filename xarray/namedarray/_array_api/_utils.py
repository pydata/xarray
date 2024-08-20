from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Any

from xarray.namedarray._typing import (
    Default,
    _arrayapi,
    _default,
    _DimsLike,
    _dtype,
    _Shape,
)

if TYPE_CHECKING:
    from xarray.namedarray.core import NamedArray


def _maybe_default_namespace(xp: ModuleType | None = None) -> ModuleType:
    if xp is None:
        # import array_api_strict as xpd
        import array_api_compat.numpy as xpd

        # import numpy as xpd

        return xpd
    else:
        return xp


def _get_data_namespace(x: NamedArray[Any, Any]) -> ModuleType:
    if isinstance(x._data, _arrayapi):
        return x._data.__array_namespace__()

    return _maybe_default_namespace()


def _get_namespace_dtype(dtype: _dtype | None = None) -> ModuleType:
    if dtype is None:
        return _maybe_default_namespace()

    try:
        xp = __import__(dtype.__module__)
    except AttributeError:
        # TODO: Fix this.
        #         FAILED array_api_tests/test_searching_functions.py::test_searchsorted - AttributeError: 'numpy.dtypes.Float64DType' object has no attribute '__module__'. Did you mean: '__mul__'?
        # Falsifying example: test_searchsorted(
        #     data=data(...),
        # )
        return _maybe_default_namespace()
    return xp


def _infer_dims(
    shape: _Shape,
    dims: _DimsLike | Default = _default,
) -> _DimsLike:
    if dims is _default:
        return tuple(f"dim_{n}" for n in range(len(shape)))
    else:
        return dims
