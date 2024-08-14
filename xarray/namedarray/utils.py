from __future__ import annotations

import importlib
import sys
import warnings
from collections.abc import Hashable, Iterable, Iterator, Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np
from packaging.version import Version

from xarray.namedarray._typing import ErrorOptionsWithWarn, _DimsLike, _NormalizedChunks

if TYPE_CHECKING:
    if sys.version_info >= (3, 10):
        from typing import TypeGuard
    else:
        from typing_extensions import TypeGuard

    from numpy.typing import NDArray

    try:
        from dask.array.core import Array as DaskArray
        from dask.typing import DaskCollection
    except ImportError:
        DaskArray = NDArray  # type: ignore
        DaskCollection: Any = NDArray  # type: ignore

    from xarray.namedarray._typing import (
        T_Chunks,
        _Dim,
        _Dims,
        _DType,
        _Shape,
        duckarray,
    )


K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")


@lru_cache
def module_available(module: str, minversion: str | None = None) -> bool:
    """Checks whether a module is installed without importing it.

    Use this for a lightweight check and lazy imports.

    Parameters
    ----------
    module : str
        Name of the module.
    minversion : str, optional
        Minimum version of the module

    Returns
    -------
    available : bool
        Whether the module is installed.
    """
    if importlib.util.find_spec(module) is None:
        return False

    if minversion is not None:
        version = importlib.metadata.version(module)

        return Version(version) >= Version(minversion)

    return True


def is_dask_collection(x: object) -> TypeGuard[DaskCollection]:
    if module_available("dask"):
        from dask.base import is_dask_collection

        # use is_dask_collection function instead of dask.typing.DaskCollection
        # see https://github.com/pydata/xarray/pull/8241#discussion_r1476276023
        return is_dask_collection(x)
    return False


def is_duck_array(value: Any) -> TypeGuard[duckarray[Any, Any]]:
    # TODO: replace is_duck_array with runtime checks via _arrayfunction_or_api protocol on
    # python 3.12 and higher (see https://github.com/pydata/xarray/issues/8696#issuecomment-1924588981)
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


def is_duck_dask_array(x: duckarray[Any, Any]) -> TypeGuard[DaskArray]:
    return is_duck_array(x) and is_dask_collection(x)


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

    def __dask_tokenize__(self) -> object:
        from dask.base import normalize_token

        return normalize_token((type(self), self._value))


def normalize_chunks_to_tuples(
    chunks: T_Chunks,
    dims: _Dims,
    shape: _Shape,
    dtype: _DType,
    previous_chunks: _NormalizedChunks | None = None,
) -> _NormalizedChunks:
    """
    Converts any specification of chunking to a tuple-of-tuple of ints along every axis.

    Handles:
      tuples or lists of repeated chunk lengths
      tuples of tuples of individual chunk lengths
      dicts mapping dim name to chunk lengths
      chunks passed as 'auto'
      chunks passed as -1

    If a chunk axis is not specified it will fallback to using `previous_chunks` if given, else the array shape (i.e. one chunk per axis).
    """

    if previous_chunks is None:
        # default to using array shape, i.e. one chunk per axis
        _previous_chunks: _NormalizedChunks = tuple((lc,) for lc in shape)
    else:
        _previous_chunks = previous_chunks

    if is_dict_like(chunks):
        # turns dict[str, tuple[in, ..]] -> dict[int, tuple[int, ...]]
        # This method of iteration allows for duplicated dimension names, GH8579
        chunks = {
            dim_number: chunks[dim]
            for dim_number, dim in enumerate(dims)
            if dim in chunks
        }

        # (everything below here is vendored from dask)
        from xarray.vendor.dask.array.utils import validate_axis

        # validate that chunk lengths are valid choices
        ndim = len(dims)
        chunks = {validate_axis(c, ndim): v for c, v in chunks.items()}

        # fill in any missing dimensions in the dict
        for i in range(ndim):
            if i not in chunks:
                chunks[i] = _previous_chunks[i]
            elif chunks[i] is None:
                chunks[i] = _previous_chunks[i]

    # coerce list-like iterables to tuple-of-tuples
    if isinstance(chunks, (tuple, list)):
        chunks = tuple(
            lc if lc is not None else rc for lc, rc in zip(chunks, _previous_chunks)
        )

    # TODO vendor the normalize_chunks function and remove it from the ChunkManager
    from xarray.vendor.dask.array.core import normalize_chunks

    # supports the 'auto' option, using previous_chunks as a fallback
    return cast(
        _NormalizedChunks,
        normalize_chunks(  # type: ignore[no-untyped-call]
            chunks, shape, dtype=dtype, previous_chunks=_previous_chunks
        ),
    )
