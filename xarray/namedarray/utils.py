from __future__ import annotations

import importlib
import sys
import typing
from collections.abc import Hashable, Iterable

import numpy as np

if typing.TYPE_CHECKING:
    if sys.version_info >= (3, 10):
        from typing import TypeGuard
    else:
        from typing_extensions import TypeGuard

    from xarray.namedarray.types import T_DuckArray


def module_available(module: str) -> bool:
    """Checks whether a module is installed without importing it.

    Use this for a lightweight check and lazy imports.

    Parameters
    ----------
    module : str
        Name of the module.

    Returns
    -------
    available : bool
        Whether the module is installed.
    """
    return importlib.util.find_spec(module) is not None


def is_dask_collection(x: typing.Any) -> bool:
    if module_available("dask"):
        from dask.base import is_dask_collection

        return is_dask_collection(x)
    return False


def is_duck_array(value: typing.Any) -> TypeGuard[T_DuckArray]:
    if isinstance(value, np.ndarray):
        return True
    return (
        hasattr(value, "ndim")
        and hasattr(value, "shape")
        and hasattr(value, "dtype")
        and (
            (hasattr(value, "__array_function__") and hasattr(value, "__array_ufunc__"))
            or hasattr(value, "__array_namespace__")
        )
    )


def is_duck_dask_array(x: typing.Any) -> bool:
    return is_duck_array(x) and is_dask_collection(x)


def to_0d_object_array(value: typing.Any) -> np.ndarray:
    """Given a value, wrap it in a 0-D numpy.ndarray with dtype=object."""
    result = np.empty((), dtype=object)
    result[()] = value
    return result


def _is_scalar(value, include_0d):
    # TODO: figure out if the following is needed
    # from xarray.core.variable import NON_NUMPY_SUPPORTED_ARRAY_TYPES
    NON_NUMPY_SUPPORTED_ARRAY_TYPES = ()

    if include_0d:
        include_0d = getattr(value, "ndim", None) == 0
    return (
        include_0d
        or isinstance(value, (str, bytes))
        or not (
            isinstance(value, (Iterable,) + NON_NUMPY_SUPPORTED_ARRAY_TYPES)
            or is_duck_array(value)
        )
    )


# See GH5624, this is a convoluted way to allow type-checking to use `TypeGuard` without
# requiring typing_extensions as a required dependency to _run_ the code (it is required
# to type-check).
try:
    if sys.version_info >= (3, 10):
        from typing import TypeGuard
    else:
        from typing_extensions import TypeGuard
except ImportError:
    if typing.TYPE_CHECKING:
        raise
    else:

        def is_scalar(value: typing.Any, include_0d: bool = True) -> bool:
            """Whether to treat a value as a scalar.

            Any non-iterable, string, or 0-D array
            """
            return _is_scalar(value, include_0d)

else:

    def is_scalar(value: typing.Any, include_0d: bool = True) -> TypeGuard[Hashable]:
        """Whether to treat a value as a scalar.

        Any non-iterable, string, or 0-D array
        """
        return _is_scalar(value, include_0d)


def is_valid_numpy_dtype(dtype: typing.Any) -> bool:
    try:
        np.dtype(dtype)
    except (TypeError, ValueError):
        return False
    else:
        return True


class ReprObject:
    """Object that prints as the given value, for use with sentinel values."""

    __slots__ = ("_value",)

    def __init__(self, value: str):
        self._value = value

    def __repr__(self) -> str:
        return self._value

    def __eq__(self, other) -> bool:
        return self._value == other._value if isinstance(other, ReprObject) else False

    def __hash__(self) -> int:
        return hash((type(self), self._value))

    def __dask_tokenize__(self):
        from dask.base import normalize_token

        return normalize_token((type(self), self._value))
