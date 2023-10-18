from __future__ import annotations

import sys
from collections.abc import Hashable, Mapping
from enum import Enum
from types import ModuleType
from typing import TYPE_CHECKING, Any, Final, Protocol, TypeVar, cast

import numpy as np

if TYPE_CHECKING:
    if sys.version_info >= (3, 10):
        from typing import TypeGuard
    else:
        from typing_extensions import TypeGuard

    from numpy.typing import NDArray

    from xarray.namedarray._typing import (
        duckarray,
    )

    try:
        from dask.array.core import Array as DaskArray
        from dask.typing import DaskCollection
    except ImportError:
        DaskArray = NDArray  # type: ignore
        DaskCollection: Any = NDArray  # type: ignore

T = TypeVar("T")

# https://stackoverflow.com/questions/74633074/how-to-type-hint-a-generic-numpy-array
T_DType_co = TypeVar("T_DType_co", bound=np.dtype[np.generic], covariant=True)
T_DType = TypeVar("T_DType", bound=np.dtype[np.generic])


class _Array(Protocol[T_DType_co]):
    @property
    def dtype(self) -> T_DType_co:
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        ...

    # TODO: numpy doesn't use any inputs:
    # https://github.com/numpy/numpy/blob/v1.24.3/numpy/_typing/_array_like.py#L38
    def __array__(self) -> np.ndarray[Any, T_DType_co]:
        ...


class _ChunkedArray(_Array[T_DType_co], Protocol[T_DType_co]):
    @property
    def chunks(self) -> tuple[tuple[int, ...], ...]:
        ...


# temporary placeholder for indicating an array api compliant type.
# hopefully in the future we can narrow this down more
T_DuckArray = TypeVar("T_DuckArray", bound=_Array[np.dtype[np.generic]])
T_ChunkedArray = TypeVar("T_ChunkedArray", bound=_ChunkedArray[np.dtype[np.generic]])


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


# %% Array API functions
def get_array_namespace(x: _Array[Any]) -> ModuleType:
    if hasattr(x, "__array_namespace__"):
        return x.__array_namespace__()  # type: ignore[no-any-return]
    else:
        return np


def astype(x: _Array[Any], dtype: T_DType, /, *, copy: bool = True) -> _Array[T_DType]:
    if hasattr(x, "__array_namespace__"):
        xp = x.__array_namespace__()
        return xp.astype(x, dtype, copy=copy)  # type: ignore[no-any-return]

    # np.astype doesn't exist yet:
    return x.astype(dtype, copy=copy)  # type: ignore[no-any-return, attr-defined]


def imag(x: _Array[Any], /) -> _Array[Any]:
    xp = get_array_namespace(x)
    return xp.imag(x)  # type: ignore[no-any-return]


def real(x: _Array[Any], /) -> _Array[Any]:
    xp = get_array_namespace(x)
    return xp.real(x)  # type: ignore[no-any-return]


# It's probably OK to give this as a TypeGuard; though it's not perfectly robust.
def is_dict_like(value: Any) -> TypeGuard[Mapping]:
    return hasattr(value, "keys") and hasattr(value, "__getitem__")


def either_dict_or_kwargs(
    pos_kwargs: Mapping[Any, T] | None,
    kw_kwargs: Mapping[str, T],
    func_name: str,
) -> Mapping[Hashable, T]:
    if pos_kwargs is None or pos_kwargs == {}:
        # Need an explicit cast to appease mypy due to invariance; see
        # https://github.com/python/mypy/issues/6228
        return cast(Mapping[Hashable, T], kw_kwargs)

    if not is_dict_like(pos_kwargs):
        raise ValueError(f"the first argument to .{func_name} must be a dictionary")
    if kw_kwargs:
        raise ValueError(
            f"cannot specify both keyword and positional arguments to .{func_name}"
        )
    return pos_kwargs


def consolidate_dask_from_array_kwargs(
    from_array_kwargs: dict,
    name: str | None = None,
    lock: bool | None = None,
    inline_array: bool | None = None,
) -> dict:
    """
    Merge dask-specific kwargs with arbitrary from_array_kwargs dict.

    Temporary function, to be deleted once explicitly passing dask-specific kwargs to .chunk() is deprecated.
    """

    from_array_kwargs = _resolve_doubly_passed_kwarg(
        from_array_kwargs,
        kwarg_name="name",
        passed_kwarg_value=name,
        default=None,
        err_msg_dict_name="from_array_kwargs",
    )
    from_array_kwargs = _resolve_doubly_passed_kwarg(
        from_array_kwargs,
        kwarg_name="lock",
        passed_kwarg_value=lock,
        default=False,
        err_msg_dict_name="from_array_kwargs",
    )
    from_array_kwargs = _resolve_doubly_passed_kwarg(
        from_array_kwargs,
        kwarg_name="inline_array",
        passed_kwarg_value=inline_array,
        default=False,
        err_msg_dict_name="from_array_kwargs",
    )

    return from_array_kwargs


def _resolve_doubly_passed_kwarg(
    kwargs_dict: dict,
    kwarg_name: str,
    passed_kwarg_value: str | bool | None,
    default: bool | None,
    err_msg_dict_name: str,
) -> dict:
    # if in kwargs_dict but not passed explicitly then just pass kwargs_dict through unaltered
    if kwarg_name in kwargs_dict and passed_kwarg_value is None:
        pass
    # if passed explicitly but not in kwargs_dict then use that
    elif kwarg_name not in kwargs_dict and passed_kwarg_value is not None:
        kwargs_dict[kwarg_name] = passed_kwarg_value
    # if in neither then use default
    elif kwarg_name not in kwargs_dict and passed_kwarg_value is None:
        kwargs_dict[kwarg_name] = default
    # if in both then raise
    else:
        raise ValueError(
            f"argument {kwarg_name} cannot be passed both as a keyword argument and within "
            f"the {err_msg_dict_name} dictionary"
        )

    return kwargs_dict
