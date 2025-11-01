from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from types import EllipsisType, ModuleType
from typing import (
    Any,
    Final,
    Literal,
    Never,
    Protocol,
    SupportsIndex,
    TypeAlias,
    TypedDict,
    TypeVar,
    Union,
    overload,
    runtime_checkable,
)

import numpy as np
from numpy.typing import ArrayLike as _npArrayLike

_ArrayLike = _npArrayLike


class Default(list[Never]):
    """
    Non-Hashable default value.

    A replacement value for Optional None since it is Hashable.
    Same idea as https://github.com/python/typing/pull/240

    Examples
    --------

    Runtime checks:

    >>> _default = Default()
    >>> isinstance(_default, Hashable)
    False
    >>> _default == _default
    True
    >>> _default is _default
    True

    Typing usage:

    >>> x: Hashable | Default = _default
    >>> if isinstance(x, Default):
    ...     y: Default = x
    ... else:
    ...     h: Hashable = x
    ...

    TODO: if x is _default does not narrow typing, use isinstance check instead.
    """

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"


_default: Final[Default] = Default()

_T = TypeVar("_T")

# https://stackoverflow.com/questions/74633074/how-to-type-hint-a-generic-numpy-array
_T_co = TypeVar("_T_co", covariant=True)

_ScalarType = TypeVar("_ScalarType", bound=np.generic)
_ScalarType_co = TypeVar("_ScalarType_co", bound=np.generic, covariant=True)

_dtype = np.dtype
_DType = TypeVar("_DType", bound=np.dtype[Any])
_DType_co = TypeVar("_DType_co", covariant=True, bound=np.dtype[Any])
# A subset of `npt.DTypeLike` that can be parametrized w.r.t. `np.generic`


# A protocol for anything with the dtype attribute
@runtime_checkable
class _SupportsDType(Protocol[_DType_co]):
    @property
    def dtype(self) -> _DType_co: ...


class _SupportsReal(Protocol[_T_co]):
    @property
    def real(self) -> _T_co: ...


class _SupportsImag(Protocol[_T_co]):
    @property
    def imag(self) -> _T_co: ...


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
_Shape1D = tuple[int]

_Axis = int
_Axes = tuple[_Axis, ...]
_AxisLike = Union[_Axis, _Axes]

_Chunks = tuple[_Shape, ...]
_NormalizedChunks = tuple[tuple[int, ...], ...]
# FYI in some cases we don't allow `None`, which this doesn't take account of.
# # FYI the `str` is for a size string, e.g. "16MB", supported by dask.
T_ChunkDim: TypeAlias = str | int | Literal["auto"] | tuple[int, ...] | None  # noqa: PYI051
# We allow the tuple form of this (though arguably we could transition to named dims only)
T_Chunks: TypeAlias = T_ChunkDim | Mapping[Any, T_ChunkDim]

_Dim = Hashable
_Dims = tuple[_Dim, ...]
_DimsLike2 = Union[_Dim, _Dims]
_DimsLike = Union[str, Iterable[_Dim]]

# https://data-apis.org/array-api/latest/API_specification/indexing.html
# TODO: np.array_api was bugged and didn't allow (None,), but should!
# https://github.com/numpy/numpy/pull/25022
# https://github.com/data-apis/array-api/pull/674
_IndexKeyNoEllipsis = Union[int, slice, None]
_IndexKey = Union[_IndexKeyNoEllipsis, EllipsisType]
_IndexKeysNoEllipsis = tuple[_IndexKeyNoEllipsis, ...]
_IndexKeys = tuple[_IndexKey, ...]  #  tuple[Union[_IndexKey, None], ...]
_IndexKeysDims = tuple[Union[_IndexKey, _Dims], ...]
_IndexKeyLike = Union[_IndexKey, _IndexKeys]

_AttrsLike = Union[Mapping[Any, Any], None]

_Device = Any


class _IInfo(Protocol):
    bits: int
    max: int
    min: int
    dtype: _dtype[Any]


class _FInfo(Protocol):
    bits: int
    eps: float
    max: float
    min: float
    smallest_normal: float
    dtype: _dtype[Any]


_Capabilities = TypedDict(
    "_Capabilities",
    {
        "boolean indexing": bool,
        "data-dependent shapes": bool,
        "max dimensions": int | None,
    },
)

_DefaultDataTypes = TypedDict(
    "_DefaultDataTypes",
    {
        "real floating": _dtype[Any],
        "complex floating": _dtype[Any],
        "integral": _dtype[Any],
        "indexing": _dtype[Any],
    },
)


class _DataTypes(TypedDict, total=False):
    bool: _dtype[Any]
    float32: _dtype[Any]
    float64: _dtype[Any]
    complex64: _dtype[Any]
    complex128: _dtype[Any]
    int8: _dtype[Any]
    int16: _dtype[Any]
    int32: _dtype[Any]
    int64: _dtype[Any]
    uint8: _dtype[Any]
    uint16: _dtype[Any]
    uint32: _dtype[Any]
    uint64: _dtype[Any]


@runtime_checkable
class _array(Protocol[_ShapeType_co, _DType_co]):
    """
    Minimal duck array named array uses.

    Corresponds to np.ndarray.
    """

    @property
    def shape(self) -> _Shape: ...

    @property
    def dtype(self) -> _DType_co: ...


@runtime_checkable
class _arrayfunction(
    _array[_ShapeType_co, _DType_co], Protocol[_ShapeType_co, _DType_co]
):
    """
    Duck array supporting NEP 18.

    Corresponds to np.ndarray.
    """

    @overload
    def __getitem__(
        self, key: _arrayfunction[Any, Any] | tuple[_arrayfunction[Any, Any], ...], /
    ) -> _arrayfunction[Any, _DType_co]: ...

    @overload
    def __getitem__(self, key: _IndexKeyLike, /) -> Any: ...

    def __getitem__(
        self,
        key: (
            _IndexKeyLike
            | _arrayfunction[Any, Any]
            | tuple[_arrayfunction[Any, Any], ...]
        ),
        /,
    ) -> _arrayfunction[Any, _DType_co] | Any: ...

    @overload
    def __array__(
        self, dtype: None = ..., /, *, copy: bool | None = ...
    ) -> np.ndarray[Any, _DType_co]: ...
    @overload
    def __array__(
        self, dtype: _DType, /, *, copy: bool | None = ...
    ) -> np.ndarray[Any, _DType]: ...

    def __array__(
        self, dtype: _DType | None = ..., /, *, copy: bool | None = ...
    ) -> np.ndarray[Any, _DType] | np.ndarray[Any, _DType_co]: ...

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
    def imag(self) -> _arrayfunction[_ShapeType_co, Any]: ...

    @property
    def real(self) -> _arrayfunction[_ShapeType_co, Any]: ...


@runtime_checkable
class _arrayapi(_array[_ShapeType_co, _DType_co], Protocol[_ShapeType_co, _DType_co]):
    """
    Duck array supporting NEP 47.

    Corresponds to np.ndarray.
    """

    def __getitem__(
        self,
        key: (
            _IndexKeyLike | Any
        ),  # TODO: Any should be _arrayapi[Any, _dtype[np.integer]]
        /,
    ) -> _arrayapi[Any, Any]: ...

    def __array_namespace__(self) -> ModuleType: ...

    def to_device(
        self, device: _Device, /, stream: None = None
    ) -> _arrayapi[_ShapeType_co, _DType_co]: ...

    @property
    def device(self) -> _Device: ...

    @property
    def mT(self) -> _arrayapi[Any, _DType_co]: ...


# NamedArray can most likely use both __array_function__ and __array_namespace__:
_arrayfunction_or_api = (_arrayfunction, _arrayapi)

duckarray = Union[
    _arrayfunction[_ShapeType_co, _DType_co], _arrayapi[_ShapeType_co, _DType_co]
]

# Corresponds to np.typing.NDArray:
DuckArray = _arrayfunction[Any, np.dtype[_ScalarType_co]]


@runtime_checkable
class _chunkedarray(
    _array[_ShapeType_co, _DType_co], Protocol[_ShapeType_co, _DType_co]
):
    """
    Minimal chunked duck array.

    Corresponds to np.ndarray.
    """

    @property
    def chunks(self) -> _Chunks: ...


@runtime_checkable
class _chunkedarrayfunction(
    _arrayfunction[_ShapeType_co, _DType_co], Protocol[_ShapeType_co, _DType_co]
):
    """
    Chunked duck array supporting NEP 18.

    Corresponds to np.ndarray.
    """

    @property
    def chunks(self) -> _Chunks: ...


@runtime_checkable
class _chunkedarrayapi(
    _arrayapi[_ShapeType_co, _DType_co], Protocol[_ShapeType_co, _DType_co]
):
    """
    Chunked duck array supporting NEP 47.

    Corresponds to np.ndarray.
    """

    @property
    def chunks(self) -> _Chunks: ...


# NamedArray can most likely use both __array_function__ and __array_namespace__:
_chunkedarrayfunction_or_api = (_chunkedarrayfunction, _chunkedarrayapi)
chunkedduckarray = Union[
    _chunkedarrayfunction[_ShapeType_co, _DType_co],
    _chunkedarrayapi[_ShapeType_co, _DType_co],
]


@runtime_checkable
class _sparsearray(
    _array[_ShapeType_co, _DType_co], Protocol[_ShapeType_co, _DType_co]
):
    """
    Minimal sparse duck array.

    Corresponds to np.ndarray.
    """

    def todense(self) -> np.ndarray[Any, _DType_co]: ...


@runtime_checkable
class _sparsearrayfunction(
    _arrayfunction[_ShapeType_co, _DType_co], Protocol[_ShapeType_co, _DType_co]
):
    """
    Sparse duck array supporting NEP 18.

    Corresponds to np.ndarray.
    """

    def todense(self) -> np.ndarray[Any, _DType_co]: ...


@runtime_checkable
class _sparsearrayapi(
    _arrayapi[_ShapeType_co, _DType_co], Protocol[_ShapeType_co, _DType_co]
):
    """
    Sparse duck array supporting NEP 47.

    Corresponds to np.ndarray.
    """

    def todense(self) -> np.ndarray[Any, _DType_co]: ...


# NamedArray can most likely use both __array_function__ and __array_namespace__:
_sparsearrayfunction_or_api = (_sparsearrayfunction, _sparsearrayapi)
sparseduckarray = Union[
    _sparsearrayfunction[_ShapeType_co, _DType_co],
    _sparsearrayapi[_ShapeType_co, _DType_co],
]

ErrorOptions = Literal["raise", "ignore"]
ErrorOptionsWithWarn = Literal["raise", "warn", "ignore"]
