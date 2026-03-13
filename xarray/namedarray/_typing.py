from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from enum import Enum
from types import EllipsisType, ModuleType
from typing import (
    TYPE_CHECKING,
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

try:
    from typing import TypeAlias
except ImportError:
    if TYPE_CHECKING:
        raise
    else:
        Self: Any = None


# Singleton type, as per https://github.com/python/typing/pull/240
class Default(Enum):
    token: Final = 0


_default = Default.token

# https://stackoverflow.com/questions/74633074/how-to-type-hint-a-generic-numpy-array
_T_co = TypeVar("_T_co", covariant=True)

dtype: TypeAlias = np.dtype  # noqa: PYI042
DType = TypeVar("DType", bound=np.dtype[Any])
DType_co = TypeVar("DType_co", covariant=True, bound=np.dtype[Any])
# A subset of `npt.DTypeLike` that can be parametrized w.r.t. `np.generic`

ScalarType = TypeVar("ScalarType", bound=np.generic)
ScalarType_co = TypeVar("ScalarType_co", bound=np.generic, covariant=True)


# A protocol for anything with the dtype attribute
@runtime_checkable
class SupportsDType(Protocol[DType_co]):
    @property
    def dtype(self) -> DType_co: ...


DTypeLike: TypeAlias = Union[
    np.dtype[ScalarType],
    type[ScalarType],
    SupportsDType[np.dtype[ScalarType]],
]

# For unknown shapes Dask uses np.nan, array_api uses None:
IntOrUnknown: TypeAlias = int
Shape: TypeAlias = tuple[IntOrUnknown, ...]
ShapeLike: TypeAlias = Union[SupportsIndex, Sequence[SupportsIndex]]
ShapeType = TypeVar("ShapeType", bound=Any)
ShapeType_co = TypeVar("ShapeType_co", bound=Any, covariant=True)


Axis: TypeAlias = int
Axes: TypeAlias = tuple[Axis, ...]
AxisLike = Union[Axis, Axes]

Chunks: TypeAlias = tuple[Shape, ...]
NormalizedChunks: TypeAlias = tuple[tuple[int, ...], ...]
# FYI in some cases we don't allow `None`, which this doesn't take account of.
# # FYI the `str` is for a size string, e.g. "16MB", supported by dask.
T_ChunkDim: TypeAlias = str | int | Literal["auto"] | tuple[int, ...] | None  # noqa: PYI051
# We allow the tuple form of this (though arguably we could transition to named dims only)
T_Chunks: TypeAlias = T_ChunkDim | Mapping[Any, T_ChunkDim]

DimType = TypeVar("DimType", bound=Hashable)
DimType_co = TypeVar("DimType_co", bound=Hashable, covariant=True)
DimsLike: TypeAlias = Union[
    Iterable[DimType_co], None, EllipsisType
]  # single str is also allowed, but luckily str = Iterable[str]

# https://data-apis.org/array-api/latest/API_specification/indexing.html
# TODO: np.array_api was bugged and didn't allow (None,), but should!
# https://github.com/numpy/numpy/pull/25022
# https://github.com/data-apis/array-api/pull/674
IndexKey: TypeAlias = Union[int, slice, EllipsisType]
IndexKeys: TypeAlias = tuple[IndexKey, ...]  #  tuple[Union[_IndexKey, None], ...]
IndexKeyLike: TypeAlias = Union[IndexKey, IndexKeys]

AttrsLike: TypeAlias = Union[Mapping[Any, Any], None]


class SupportsReal(Protocol[_T_co]):
    @property
    def real(self) -> _T_co: ...


class SupportsImag(Protocol[_T_co]):
    @property
    def imag(self) -> _T_co: ...


@runtime_checkable
class array(Protocol[ShapeType_co, DType_co]):
    """
    Minimal duck array named array uses.

    Corresponds to np.ndarray.
    """

    @property
    def shape(self) -> Shape: ...

    @property
    def dtype(self) -> DType_co: ...


@runtime_checkable
class arrayfunction(array[ShapeType_co, DType_co], Protocol[ShapeType_co, DType_co]):
    """
    Duck array supporting NEP 18.

    Corresponds to np.ndarray.
    """

    @overload
    def __getitem__(
        self, key: arrayfunction[Any, Any] | tuple[arrayfunction[Any, Any], ...], /
    ) -> arrayfunction[Any, DType_co]: ...

    @overload
    def __getitem__(self, key: IndexKeyLike, /) -> Any: ...

    def __getitem__(
        self,
        key: (
            IndexKeyLike | arrayfunction[Any, Any] | tuple[arrayfunction[Any, Any], ...]
        ),
        /,
    ) -> arrayfunction[Any, DType_co] | Any: ...

    @overload
    def __array__(
        self, dtype: None = ..., /, *, copy: bool | None = ...
    ) -> np.ndarray[Any, DType_co]: ...

    @overload
    def __array__(
        self, dtype: DType, /, *, copy: bool | None = ...
    ) -> np.ndarray[Any, DType]: ...

    def __array__(
        self, dtype: DType | None = ..., /, *, copy: bool | None = ...
    ) -> np.ndarray[Any, DType] | np.ndarray[Any, DType_co]: ...

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
    def imag(self) -> arrayfunction[ShapeType_co, Any]: ...

    @property
    def real(self) -> arrayfunction[ShapeType_co, Any]: ...


@runtime_checkable
class arrayapi(array[ShapeType_co, DType_co], Protocol[ShapeType_co, DType_co]):
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

duckarray: TypeAlias = Union[  # noqa: PYI042
    arrayfunction[ShapeType_co, DType_co], arrayapi[ShapeType_co, DType_co]
]

# Corresponds to np.typing.NDArray:
DuckArray: TypeAlias = arrayfunction[Any, np.dtype[ScalarType_co]]


@runtime_checkable
class chunkedarray(array[ShapeType_co, DType_co], Protocol[ShapeType_co, DType_co]):
    """
    Minimal chunked duck array.

    Corresponds to np.ndarray.
    """

    @property
    def chunks(self) -> Chunks: ...


@runtime_checkable
class chunkedarrayfunction(
    arrayfunction[ShapeType_co, DType_co], Protocol[ShapeType_co, DType_co]
):
    """
    Chunked duck array supporting NEP 18.

    Corresponds to np.ndarray.
    """

    @property
    def chunks(self) -> Chunks: ...


@runtime_checkable
class chunkedarrayapi(
    arrayapi[ShapeType_co, DType_co], Protocol[ShapeType_co, DType_co]
):
    """
    Chunked duck array supporting NEP 47.

    Corresponds to np.ndarray.
    """

    @property
    def chunks(self) -> Chunks: ...


# NamedArray can most likely use both __array_function__ and __array_namespace__:
_chunkedarrayfunction_or_api = (chunkedarrayfunction, chunkedarrayapi)
chunkedduckarray: TypeAlias = Union[  # noqa: PYI042
    chunkedarrayfunction[ShapeType_co, DType_co],
    chunkedarrayapi[ShapeType_co, DType_co],
]


@runtime_checkable
class sparsearray(array[ShapeType_co, DType_co], Protocol[ShapeType_co, DType_co]):
    """
    Minimal sparse duck array.

    Corresponds to np.ndarray.
    """

    def todense(self) -> np.ndarray[Any, DType_co]: ...


@runtime_checkable
class sparsearrayfunction(
    arrayfunction[ShapeType_co, DType_co], Protocol[ShapeType_co, DType_co]
):
    """
    Sparse duck array supporting NEP 18.

    Corresponds to np.ndarray.
    """

    def todense(self) -> np.ndarray[Any, DType_co]: ...


@runtime_checkable
class sparsearrayapi(
    arrayapi[ShapeType_co, DType_co], Protocol[ShapeType_co, DType_co]
):
    """
    Sparse duck array supporting NEP 47.

    Corresponds to np.ndarray.
    """

    def todense(self) -> np.ndarray[Any, DType_co]: ...


# NamedArray can most likely use both __array_function__ and __array_namespace__:
_sparsearrayfunction_or_api = (sparsearrayfunction, sparsearrayapi)
sparseduckarray: TypeAlias = Union[  # noqa: PYI042
    sparsearrayfunction[ShapeType_co, DType_co],
    sparsearrayapi[ShapeType_co, DType_co],
]

ErrorHandling: TypeAlias = Literal["raise", "ignore"]
ErrorHandlingWithWarn: TypeAlias = Literal["raise", "warn", "ignore"]
