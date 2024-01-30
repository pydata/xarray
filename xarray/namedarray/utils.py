from __future__ import annotations

import sys
import warnings
from collections.abc import Hashable, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np

from xarray.namedarray._typing import (
    ErrorOptionsWithWarn,
    _arrayfunction_or_api,
    _DimsLike,
)

if TYPE_CHECKING:
    if sys.version_info >= (3, 10):
        from typing import TypeGuard
    else:
        from typing_extensions import TypeGuard

    from numpy.typing import NDArray

    from xarray.namedarray._typing import _Dim, duckarray

    try:
        from dask.array.core import Array as DaskArray
        from dask.typing import DaskCollection
    except ImportError:
        DaskArray = NDArray  # type: ignore
        DaskCollection: Any = NDArray  # type: ignore


K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")


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
    return isinstance(x, _arrayfunction_or_api) and is_dask_collection(x)


def to_0d_object_array(
    value: object,
) -> NDArray[np.object_]:
    """Given a value, wrap it in a 0-D numpy.ndarray with dtype=object."""
    result = np.empty((), dtype=object)
    result[()] = value
    return result


def is_dict_like(value: Any) -> TypeGuard[Mapping[Any, Any]]:
    return hasattr(value, "keys") and hasattr(value, "__getitem__")


def drop_missing_dims(
    supplied_dims: Iterable[_Dim],
    dims: Iterable[_Dim],
    missing_dims: ErrorOptionsWithWarn,
) -> _DimsLike:
    """Depending on the setting of missing_dims, drop any dimensions from supplied_dims that
    are not present in dims.

    Parameters
    ----------
    supplied_dims : Iterable of Hashable
    dims : Iterable of Hashable
    missing_dims : {"raise", "warn", "ignore"}
    """

    if missing_dims == "raise":
        supplied_dims_set = {val for val in supplied_dims if val is not ...}
        if invalid := supplied_dims_set - set(dims):
            raise ValueError(
                f"Dimensions {invalid} do not exist. Expected one or more of {dims}"
            )

        return supplied_dims

    elif missing_dims == "warn":
        if invalid := set(supplied_dims) - set(dims):
            warnings.warn(
                f"Dimensions {invalid} do not exist. Expected one or more of {dims}"
            )

        return [val for val in supplied_dims if val in dims or val is ...]

    elif missing_dims == "ignore":
        return [val for val in supplied_dims if val in dims or val is ...]

    else:
        raise ValueError(
            f"Unrecognised option {missing_dims} for missing_dims argument"
        )


def infix_dims(
    dims_supplied: Iterable[_Dim],
    dims_all: Iterable[_Dim],
    missing_dims: ErrorOptionsWithWarn = "raise",
) -> Iterator[_Dim]:
    """
    Resolves a supplied list containing an ellipsis representing other items, to
    a generator with the 'realized' list of all items
    """
    if ... in dims_supplied:
        dims_all_list = list(dims_all)
        if len(set(dims_all)) != len(dims_all_list):
            raise ValueError("Cannot use ellipsis with repeated dims")
        if list(dims_supplied).count(...) > 1:
            raise ValueError("More than one ellipsis supplied")
        other_dims = [d for d in dims_all if d not in dims_supplied]
        existing_dims = drop_missing_dims(dims_supplied, dims_all, missing_dims)
        for d in existing_dims:
            if d is ...:
                yield from other_dims
            else:
                yield d
    else:
        existing_dims = drop_missing_dims(dims_supplied, dims_all, missing_dims)
        if set(existing_dims) ^ set(dims_all):
            raise ValueError(
                f"{dims_supplied} must be a permuted list of {dims_all}, unless `...` is included"
            )
        yield from existing_dims


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


def consolidate_dask_from_array_kwargs(
    from_array_kwargs: dict[Any, Any],
    name: str | None = None,
    lock: bool | None = None,
    inline_array: bool | None = None,
) -> dict[Any, Any]:
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
    kwargs_dict: dict[Any, Any],
    kwarg_name: str,
    passed_kwarg_value: str | bool | None,
    default: bool | None,
    err_msg_dict_name: str,
) -> dict[Any, Any]:
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
