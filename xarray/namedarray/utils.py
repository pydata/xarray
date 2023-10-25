from __future__ import annotations

import sys
from collections.abc import Hashable, Iterable, Mapping
from enum import Enum
from typing import TYPE_CHECKING, Any, Final, TypeVar

import numpy as np

if TYPE_CHECKING:
    if sys.version_info >= (3, 10):
        from typing import TypeGuard
    else:
        from typing_extensions import TypeGuard

    from numpy.typing import NDArray

    from xarray.namedarray._typing import T_DuckArray, duckarray

    try:
        from dask.array.core import Array as DaskArray
        from dask.typing import DaskCollection
    except ImportError:
        DaskArray = NDArray  # type: ignore
        DaskCollection: Any = NDArray  # type: ignore

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")


# Singleton type, as per https://github.com/python/typing/pull/240
class Default(Enum):
    token: Final = 0


_default = Default.token


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
    from importlib.util import find_spec

    return find_spec(module) is not None


def is_dask_collection(x: object) -> TypeGuard[DaskCollection]:
    if module_available("dask"):
        from dask.typing import DaskCollection

        return isinstance(x, DaskCollection)
    return False


def is_duck_dask_array(x: duckarray[Any, Any]) -> TypeGuard[DaskArray]:
    return is_dask_collection(x)


def to_0d_object_array(
    value: object,
) -> NDArray[np.object_]:
    """Given a value, wrap it in a 0-D numpy.ndarray with dtype=object."""
    result = np.empty((), dtype=object)
    result[()] = value
    return result


def is_duck_array(value: Any) -> TypeGuard[T_DuckArray]:
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


# It's probably OK to give this as a TypeGuard; though it's not perfectly robust.
def is_dict_like(value: Any) -> TypeGuard[Mapping]:
    return hasattr(value, "keys") and hasattr(value, "__getitem__")


def _is_scalar(value, include_0d):
    # from xarray.core.variable import NON_NUMPY_SUPPORTED_ARRAY_TYPES
    NON_NUMPY_SUPPORTED_ARRAY_TYPES = tuple()

    if include_0d:
        include_0d = getattr(value, "ndim", None) == 0
    return (
        include_0d
        or isinstance(value, (str, bytes))
        or not (
            isinstance(value, (Iterable,) + NON_NUMPY_SUPPORTED_ARRAY_TYPES)
            or hasattr(value, "__array_function__")
            or hasattr(value, "__array_namespace__")
        )
    )


def is_scalar(value: Any, include_0d: bool = True) -> TypeGuard[Hashable]:
    """Whether to treat a value as a scalar.

    Any non-iterable, string, or 0-D array
    """
    return _is_scalar(value, include_0d)


def is_0d_dask_array(x):
    return is_duck_dask_array(x) and is_scalar(x)


class ReprObject:
    """Object that prints as the given value, for use with sentinel values."""

    __slots__ = ("_value",)

    _value: str

    def __init__(self, value: str):
        self._value = value

    def __repr__(self) -> str:
        return self._value

    def __eq__(self, other: ReprObject | Any) -> bool:
        # TODO: What type can other be? ArrayLike?
        return self._value == other._value if isinstance(other, ReprObject) else False

    def __hash__(self) -> int:
        return hash((type(self), self._value))

    def __dask_tokenize__(self) -> Hashable:
        from dask.base import normalize_token

        return normalize_token((type(self), self._value))  # type: ignore[no-any-return]
