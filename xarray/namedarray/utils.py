from __future__ import annotations

import sys
from collections.abc import Hashable, Iterable, Mapping, Sequence
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Final,
    Literal,
    ModuleType,
    Protocol,
    SupportsIndex,
    TypeVar,
    Union,
    runtime_checkable,
)

import numpy as np

if TYPE_CHECKING:
    if sys.version_info >= (3, 10):
        from typing import TypeGuard
    else:
        from typing_extensions import TypeGuard

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

    from dask.array.core import Array as DaskArray
    from dask.typing import DaskCollection
    from numpy.typing import DTypeLike, NDArray

    # try:
    #     from dask.array.core import Array as DaskArray
    #     from dask.typing import DaskCollection
    # except ImportError:
    #     DaskArray = NDArray  # type: ignore
    #     DaskCollection: Any = NDArray  # type: ignore


# https://stackoverflow.com/questions/74633074/how-to-type-hint-a-generic-numpy-array
_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)

_DType = TypeVar("_DType", bound=np.dtype[Any])
_DType_co = TypeVar("_DType_co", covariant=True, bound=np.dtype[Any])
_ScalarType = TypeVar("_ScalarType", bound=np.generic)
_ScalarType_co = TypeVar("_ScalarType_co", bound=np.generic, covariant=True)

# For unknown shapes Dask uses np.nan, array_api uses None:
_IntOrUnknown = int
_Shape = tuple[_IntOrUnknown, ...]
_ShapeLike = Union[SupportsIndex, Sequence[SupportsIndex]]
_ShapeType = TypeVar("_ShapeType", bound=Any)
_ShapeType_co = TypeVar("_ShapeType_co", bound=Any, covariant=True)

_Chunks = tuple[_Shape, ...]

_Dim = Hashable
_Dims = tuple[_Dim, ...]

_DimsLike = Union[str, Iterable[_Dim]]
_AttrsLike = Union[Mapping[Any, Any], None]

_dtype = np.dtype


class _SupportsReal(Protocol[_T_co]):
    @property
    def real(self) -> _T_co:
        ...


class _SupportsImag(Protocol[_T_co]):
    @property
    def imag(self) -> _T_co:
        ...


@runtime_checkable
class _array(Protocol[_ShapeType_co, _DType_co]):
    """
    Minimal duck array.

    Corresponds to np.ndarray.
    """

    @property
    def shape(self) -> _Shape:
        ...

    @property
    def real(self) -> _array[Any, _DType_co]:
        ...

    @property
    def imag(self) -> Self[Any, _DType_co]:
        ...

    # @property
    # def imag(
    #     self(: _array[_ShapeType, np.dtype[_SupportsImag[_ScalarType]]],  # type: ignore[type-var]
    # ) -> _array[_ShapeType, _dtype[_ScalarType]]:)
    #     ...

    def astype(self, dtype: DTypeLike) -> Self:
        ...

    # Keep `dtype` at the bottom to avoid name conflicts with `np.dtype`
    @property
    def dtype(self) -> _DType_co:
        ...

    # def to_numpy(self) -> NDArray[_ScalarType_co]:
    #     ...

    # # TODO: numpy doesn't use any inputs:
    # # https://github.com/numpy/numpy/blob/v1.24.3/numpy/_typing/_array_like.py#L38
    # def __array__(self) -> NDArray[_ScalarType_co]:
    #     ...


# Corresponds to np.typing.NDArray:
_Array = _array[Any, np.dtype[_ScalarType_co]]


@runtime_checkable
class _arrayfunction(
    _array[_ShapeType_co, _DType_co], Protocol[_ShapeType_co, _DType_co]
):
    """
    Duck array supporting NEP 18.

    Corresponds to np.ndarray.
    """

    def __array_ufunc__(
        self,
        ufunc: Callable[..., Any],
        method: Literal[
            "__call__", "reduce", "reduceat", "accumulate", "outer", "inner"
        ],
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        ...

    def __array_function__(
        self,
        func: Callable[..., Any],
        types: Iterable[type],
        args: Iterable[Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        ...


# Corresponds to np.typing.NDArray:
_ArrayFunction = _arrayfunction[Any, np.dtype[_ScalarType_co]]


@runtime_checkable
class _arrayapi(_array[_ShapeType_co, _DType_co], Protocol[_ShapeType_co, _DType_co]):
    """
    Duck array supporting NEP 47.

    Corresponds to np.ndarray.
    """

    def __array_namespace__(self) -> ModuleType:
        ...


# Corresponds to np.typing.NDArray:
_ArrayAPI = _arrayapi[Any, np.dtype[_ScalarType_co]]


@runtime_checkable
class _chunkedarray(
    _array[_ShapeType_co, _DType_co], Protocol[_ShapeType_co, _DType_co]
):
    """
    Minimal chunked duck array.

    Corresponds to np.ndarray.
    """

    @property
    def chunks(self) -> _Chunks:
        ...


# Corresponds to np.typing.NDArray:
_ChunkedArray = _chunkedarray[Any, np.dtype[_ScalarType_co]]


@runtime_checkable
class _chunkedarrayfunction(
    _arrayfunction[_ShapeType_co, _DType_co], Protocol[_ShapeType_co, _DType_co]
):
    """
    Minimal chunked duck array.

    Corresponds to np.ndarray.
    """

    @property
    def chunks(self) -> _Chunks:
        ...


# Corresponds to np.typing.NDArray:
_ChunkedArrayFunction = _chunkedarrayfunction[Any, np.dtype[_ScalarType_co]]


@runtime_checkable
class _chunkedarrayapi(
    _arrayapi[_ShapeType_co, _DType_co], Protocol[_ShapeType_co, _DType_co]
):
    """
    Minimal chunked duck array.

    Corresponds to np.ndarray.
    """

    @property
    def chunks(self) -> _Chunks:
        ...


# Corresponds to np.typing.NDArray:
_ChunkedArrayAPI = _chunkedarrayapi[Any, np.dtype[_ScalarType_co]]


@runtime_checkable
class _sparsearray(
    _array[_ShapeType_co, _DType_co], Protocol[_ShapeType_co, _DType_co]
):
    """
    Minimal sparse duck array.

    Corresponds to np.ndarray.
    """

    def todense(self) -> NDArray[_ScalarType_co]:
        ...


# Corresponds to np.typing.NDArray:
_SparseArray = _sparsearray[Any, np.dtype[_ScalarType_co]]


# temporary placeholder for indicating an array api compliant type.
# hopefully in the future we can narrow this down more
T_DuckArray = TypeVar("T_DuckArray", bound=_Array[np.generic])
T_ChunkedArray = TypeVar("T_ChunkedArray", bound=_ChunkedArray[np.generic])


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


# def is_duck_array(value: _T) -> TypeGuard[_T]:
#     # if isinstance(value, np.ndarray):
#     #     return True
#     return isinstance(value, _array) and (
#         (hasattr(value, "__array_function__") and hasattr(value, "__array_ufunc__"))
#         or hasattr(value, "__array_namespace__")
#     )


def is_duck_dask_array(x: _Array[np.generic]) -> TypeGuard[DaskArray]:
    return is_dask_collection(x)


def is_chunked_duck_array(
    x: _Array[np.generic],
) -> TypeGuard[_ChunkedArray[np.generic]]:
    return hasattr(x, "chunks")


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
