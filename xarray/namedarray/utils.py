from __future__ import annotations

import importlib
import sys
import warnings
from collections.abc import Hashable, Iterable, Iterator, Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np
from packaging.version import Version

from xarray.namedarray._typing import ErrorOptionsWithWarn, _DimsLike

if TYPE_CHECKING:
    from typing import TypeGuard

    from numpy.typing import NDArray

    try:
        from dask.array.core import Array as DaskArray
        from dask.typing import DaskCollection
    except ImportError:
        DaskArray = NDArray  # type: ignore[assignment, misc]
        DaskCollection: Any = NDArray  # type: ignore[no-redef]

    from xarray.core.variable import Variable
    from xarray.namedarray._typing import DuckArray, _Dim, _DType, duckarray
    from xarray.namedarray.parallelcompat import T_ChunkedArray


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
                f"Dimensions {invalid} do not exist. Expected one or more of {dims}",
                stacklevel=2,
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


def fake_target_chunksize(
    data: DuckArray[Any] | T_ChunkedArray | Variable,
    target_chunksize: int,
) -> tuple[int, _DType]:
    """
    Naughty trick - let's get the ratio of our cftime_nbytes, and then compute
    the ratio of that size to a np.float64. Then we can just adjust our target_chunksize
    and use the default dask chunking algorithm to get a reasonable chunk size.

    ? I don't think T_chunkedArray or Variable should be necessary, but the calls
    ? to this in daskmanager.py requires it to be that. I still need to wrap my head
    ? around the typing here a bit more.
    """
    import numpy as np

    from xarray.core.formatting import first_n_items

    output_dtype: _DType = np.dtype(np.float64)  # type: ignore[assignment]

    if data.dtype == object:
        nbytes_approx: int = sys.getsizeof(first_n_items(data, 1))  # type: ignore[no-untyped-call]
    else:
        nbytes_approx = data[0].itemsize

    f64_nbytes = output_dtype.itemsize  # Should be 8 bytes

    target_chunksize = int(target_chunksize * (f64_nbytes / nbytes_approx))

    return target_chunksize, output_dtype


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
