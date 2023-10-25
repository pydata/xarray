from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Protocol,
    SupportsIndex,
    TypeVar,
    Union,
    overload,
    runtime_checkable,
)

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


# https://stackoverflow.com/questions/74633074/how-to-type-hint-a-generic-numpy-array
_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)


_DType = TypeVar("_DType", bound=np.dtype[Any])
_DType_co = TypeVar("_DType_co", covariant=True, bound=np.dtype[Any])
# A subset of `npt.DTypeLike` that can be parametrized w.r.t. `np.generic`

_ScalarType = TypeVar("_ScalarType", bound=np.generic)
_ScalarType_co = TypeVar("_ScalarType_co", bound=np.generic, covariant=True)


# A protocol for anything with the dtype attribute
@runtime_checkable
class _SupportsDType(Protocol[_DType_co]):
    @property
    def dtype(self) -> _DType_co:
        ...


_DTypeLike = Union[
    np.dtype[_ScalarType],
    type[_ScalarType],
    _SupportsDType[np.dtype[_ScalarType]],
]

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
    Minimal duck array named array uses.

    Corresponds to np.ndarray.
    """

    @property
    def shape(self) -> _Shape:
        ...

    @property
    def dtype(self) -> _DType_co:
        ...

    @overload
    def __array__(self, dtype: None = ..., /) -> np.ndarray[Any, _DType_co]:
        ...

    @overload
    def __array__(self, dtype: _DType, /) -> np.ndarray[Any, _DType]:
        ...


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

    # TODO: Should return the same subclass but with a new dtype generic.
    # https://github.com/python/typing/issues/548
    def __array_ufunc__(
        self,
        ufunc: Any,
        method: Any,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        ...

    # TODO: Should return the same subclass but with a new dtype generic.
    # https://github.com/python/typing/issues/548
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

# NamedArray can most likely use both __array_function__ and __array_namespace__:
_arrayfunction_or_api = (_arrayfunction, _arrayapi)
# _ArrayFunctionOrAPI = Union[
#     _arrayfunction[_ShapeType_co, _DType_co], _arrayapi[_ShapeType_co, _DType_co]
# ]

duckarray = Union[
    _arrayfunction[_ShapeType_co, _DType_co], _arrayapi[_ShapeType_co, _DType_co]
]
DuckArray = _arrayfunction[Any, np.dtype[_ScalarType_co]]
T_DuckArray = TypeVar("T_DuckArray", bound=_arrayfunction[Any, Any])


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
    Chunked duck array supporting NEP 18.

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
    Chunked duck array supporting NEP 47.

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


@runtime_checkable
class _sparsearrayfunction(
    _arrayfunction[_ShapeType_co, _DType_co], Protocol[_ShapeType_co, _DType_co]
):
    """
    Sparse duck array supporting NEP 18.

    Corresponds to np.ndarray.
    """

    def todense(self) -> NDArray[_ScalarType_co]:
        ...


# Corresponds to np.typing.NDArray:
_SparseArrayFunction = _sparsearrayfunction[Any, np.dtype[_ScalarType_co]]


@runtime_checkable
class _sparsearrayapi(
    _arrayapi[_ShapeType_co, _DType_co], Protocol[_ShapeType_co, _DType_co]
):
    """
    Sparse duck array supporting NEP 47.

    Corresponds to np.ndarray.
    """

    def todense(self) -> NDArray[_ScalarType_co]:
        ...


# Corresponds to np.typing.NDArray:
_SparseArrayAPI = _sparsearrayapi[Any, np.dtype[_ScalarType_co]]

# NamedArray can most likely use both __array_function__ and __array_namespace__:
_sparsearrayfunction_or_api = (_sparsearrayfunction, _sparsearrayapi)
_SparseArrayFunctionOrAPI = Union[
    _SparseArrayFunction[np.generic], _SparseArrayAPI[np.generic]
]


# Temporary placeholder for indicating an array api compliant type.
# hopefully in the future we can narrow this down more
# T_DuckArray = TypeVar("T_DuckArray", bound=_ArrayFunctionOrAPI)

# The chunked arrays like dask or cubed:
_ChunkedArrayFunctionOrAPI = Union[
    _ChunkedArrayFunction[np.generic], _ChunkedArrayAPI[np.generic]
]
T_ChunkedArray = TypeVar("T_ChunkedArray", bound=_ChunkedArrayFunctionOrAPI)
