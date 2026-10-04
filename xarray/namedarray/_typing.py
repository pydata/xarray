from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from enum import Enum
from types import EllipsisType, ModuleType
from typing import (
    Any,
    Final,
    Literal,
    Protocol,
    SupportsIndex,
    TypeVar,
    Union,
    overload,
    runtime_checkable,
)

import numpy as np


# Singleton type, as per https://github.com/python/typing/pull/240
class Default(Enum):
    token: Final = 0


_default = Default.token

# Type variables of NamedArray. It does not use PEP 695 type parameters, because
# their variance would be inferred as invariant, see the comment there.
DType_co = TypeVar("DType_co", covariant=True, bound=np.dtype[Any])
ShapeType_co = TypeVar("ShapeType_co", bound=Any, covariant=True)
DimType_co = TypeVar("DimType_co", bound=Hashable, covariant=True)

dtype = np.dtype


# A protocol for anything with the dtype attribute
@runtime_checkable
class SupportsDType[DType: np.dtype[Any]](Protocol):
    @property
    def dtype(self) -> DType: ...


# A subset of `npt.DTypeLike` that can be parametrized w.r.t. `np.generic`
type DTypeLike[ScalarType: np.generic] = (
    np.dtype[ScalarType] | type[ScalarType] | SupportsDType[np.dtype[ScalarType]]
)

# For unknown shapes Dask uses np.nan, array_api uses None:
IntOrUnknown = int
Shape = tuple[IntOrUnknown, ...]
ShapeLike = Union[SupportsIndex, Sequence[SupportsIndex]]

Axis = int
Axes = tuple[Axis, ...]
AxisLike = Union[Axis, Axes]

Chunks = tuple[Shape, ...]
NormalizedChunks = tuple[tuple[int, ...], ...]
# FYI in some cases we don't allow `None`, which this doesn't take account of.
# # FYI the `str` is for a size string, e.g. "16MB", supported by dask.
type T_ChunkDim = str | int | Literal["auto"] | tuple[int, ...] | None  # noqa: PYI051
# We allow the tuple form of this (though arguably we could transition to named dims only)
type T_Chunks = T_ChunkDim | Mapping[Any, T_ChunkDim] | tuple[T_ChunkDim, ...]

# single str is also allowed, but luckily str = Iterable[str]
type DimsLike[DimType: Hashable] = Iterable[DimType] | EllipsisType | None

# https://data-apis.org/array-api/latest/API_specification/indexing.html
# TODO: np.array_api was bugged and didn't allow (None,), but should!
# https://github.com/numpy/numpy/pull/25022
# https://github.com/data-apis/array-api/pull/674
IndexKey = Union[int, slice, EllipsisType]
IndexKeys = tuple[IndexKey, ...]  #  tuple[Union[_IndexKey, None], ...]
IndexKeyLike = Union[IndexKey, IndexKeys]

AttrsLike = Union[Mapping[Any, Any], None]


class SupportsReal[T](Protocol):
    @property
    def real(self) -> T: ...


class SupportsImag[T](Protocol):
    @property
    def imag(self) -> T: ...


@runtime_checkable
class array[ShapeType, DType: np.dtype[Any]](Protocol):
    """
    Minimal duck array named array uses.

    Corresponds to np.ndarray.
    """

    @property
    def shape(self) -> Shape: ...

    @property
    def dtype(self) -> DType: ...


@runtime_checkable
class arrayfunction[ShapeType, DType: np.dtype[Any]](array[ShapeType, DType], Protocol):
    """
    Duck array supporting NEP 18.

    Corresponds to np.ndarray.
    """

    @overload
    def __getitem__(
        self, key: arrayfunction[Any, Any] | tuple[arrayfunction[Any, Any], ...], /
    ) -> arrayfunction[Any, DType]: ...

    @overload
    def __getitem__(self, key: IndexKeyLike, /) -> Any: ...

    def __getitem__(
        self,
        key: (
            IndexKeyLike | arrayfunction[Any, Any] | tuple[arrayfunction[Any, Any], ...]
        ),
        /,
    ) -> arrayfunction[Any, DType] | Any: ...

    @overload
    def __array__(
        self, dtype: None = ..., /, *, copy: bool | None = ...
    ) -> np.ndarray[Any, DType]: ...

    @overload
    def __array__[DType2: np.dtype[Any]](
        self, dtype: DType2, /, *, copy: bool | None = ...
    ) -> np.ndarray[Any, DType2]: ...

    def __array__[DType2: np.dtype[Any]](
        self, dtype: DType2 | None = ..., /, *, copy: bool | None = ...
    ) -> np.ndarray[Any, DType2] | np.ndarray[Any, DType]: ...

    # TODO: Should return the same subclass but with a new dtype generic.
    # https://github.com/python/typing/issues/548
    def __array_ufunc__(
        self,
        ufunc: Any,
        method: Any,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any: ...

    # TODO: Should return the same subclass but with a new dtype generic.
    # https://github.com/python/typing/issues/548
    def __array_function__(
        self,
        func: Callable[..., Any],
        types: Iterable[type],
        args: Iterable[Any],
        kwargs: Mapping[str, Any],
    ) -> Any: ...

    @property
    def imag(self) -> arrayfunction[ShapeType, Any]: ...

    @property
    def real(self) -> arrayfunction[ShapeType, Any]: ...


@runtime_checkable
class arrayapi[ShapeType, DType: np.dtype[Any]](array[ShapeType, DType], Protocol):
    """
    Duck array supporting NEP 47.

    Corresponds to np.ndarray.
    """

    def __getitem__(
        self,
        key: (
            IndexKeyLike | Any
        ),  # TODO: Any should be _arrayapi[Any, _dtype[np.integer]]
        /,
    ) -> arrayapi[Any, Any]: ...

    def __array_namespace__(self) -> ModuleType: ...


# NamedArray can most likely use both __array_function__ and __array_namespace__:
_arrayfunction_or_api = (arrayfunction, arrayapi)

type duckarray[ShapeType, DType: np.dtype[Any]] = (  # noqa: PYI042
    arrayfunction[ShapeType, DType] | arrayapi[ShapeType, DType]
)

# Corresponds to np.typing.NDArray:
type DuckArray[ScalarType: np.generic] = arrayfunction[Any, np.dtype[ScalarType]]


@runtime_checkable
class chunkedarray[ShapeType, DType: np.dtype[Any]](array[ShapeType, DType], Protocol):
    """
    Minimal chunked duck array.

    Corresponds to np.ndarray.
    """

    @property
    def chunks(self) -> Chunks: ...


@runtime_checkable
class chunkedarrayfunction[ShapeType, DType: np.dtype[Any]](
    arrayfunction[ShapeType, DType], Protocol
):
    """
    Chunked duck array supporting NEP 18.

    Corresponds to np.ndarray.
    """

    @property
    def chunks(self) -> Chunks: ...


@runtime_checkable
class chunkedarrayapi[ShapeType, DType: np.dtype[Any]](
    arrayapi[ShapeType, DType], Protocol
):
    """
    Chunked duck array supporting NEP 47.

    Corresponds to np.ndarray.
    """

    @property
    def chunks(self) -> Chunks: ...


# NamedArray can most likely use both __array_function__ and __array_namespace__:
_chunkedarrayfunction_or_api = (chunkedarrayfunction, chunkedarrayapi)
type chunkedduckarray[ShapeType, DType: np.dtype[Any]] = (  # noqa: PYI042
    chunkedarrayfunction[ShapeType, DType] | chunkedarrayapi[ShapeType, DType]
)


@runtime_checkable
class sparsearray[ShapeType, DType: np.dtype[Any]](array[ShapeType, DType], Protocol):
    """
    Minimal sparse duck array.

    Corresponds to np.ndarray.
    """

    def todense(self) -> np.ndarray[Any, DType]: ...


@runtime_checkable
class sparsearrayfunction[ShapeType, DType: np.dtype[Any]](
    arrayfunction[ShapeType, DType], Protocol
):
    """
    Sparse duck array supporting NEP 18.

    Corresponds to np.ndarray.
    """

    def todense(self) -> np.ndarray[Any, DType]: ...


@runtime_checkable
class sparsearrayapi[ShapeType, DType: np.dtype[Any]](
    arrayapi[ShapeType, DType], Protocol
):
    """
    Sparse duck array supporting NEP 47.

    Corresponds to np.ndarray.
    """

    def todense(self) -> np.ndarray[Any, DType]: ...


# NamedArray can most likely use both __array_function__ and __array_namespace__:
_sparsearrayfunction_or_api = (sparsearrayfunction, sparsearrayapi)
type sparseduckarray[ShapeType, DType: np.dtype[Any]] = (  # noqa: PYI042
    sparsearrayfunction[ShapeType, DType] | sparsearrayapi[ShapeType, DType]
)

ErrorHandling = Literal["raise", "ignore"]
ErrorHandlingWithWarn = Literal["raise", "warn", "ignore"]
