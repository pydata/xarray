# Vendored from https://github.com/data-apis/array-api/pull/589
"""
Types for type annotations used in the array API standard.

The type variables should be replaced with the actual types for a given
library, e.g., for NumPy TypeVar('array') would be replaced with ndarray.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    Protocol,
)
from enum import Enum


if TYPE_CHECKING:
    from xarray._array_api.array_object import Array
    from xarray._array_api.data_types import DType

array = TypeVar("array", bound="Array")
device = TypeVar("device")
dtype = TypeVar("dtype", bound="DType")
SupportsDLPack = TypeVar("SupportsDLPack")
SupportsBufferProtocol = TypeVar("SupportsBufferProtocol")
PyCapsule = TypeVar("PyCapsule")
# ellipsis cannot actually be imported from anywhere, so include a dummy here
# to keep pyflakes happy. https://github.com/python/typeshed/issues/3556
ellipsis = TypeVar("ellipsis")


@dataclass
class finfo_object:
    """Dataclass returned by `finfo`."""

    bits: int
    eps: float
    max: float
    min: float
    smallest_normal: float


@dataclass
class iinfo_object:
    """Dataclass returned by `iinfo`."""

    bits: int
    max: int
    min: int


_T_co = TypeVar("_T_co", covariant=True)


class NestedSequence(Protocol[_T_co]):
    def __getitem__(self, key: int, /) -> Union[_T_co, NestedSequence[_T_co]]:
        ...

    def __len__(self, /) -> int:
        ...


__all__ = [
    "Any",
    "List",
    "Literal",
    "NestedSequence",
    "Optional",
    "PyCapsule",
    "SupportsBufferProtocol",
    "SupportsDLPack",
    "Tuple",
    "Union",
    "Sequence",
    "array",
    "device",
    "dtype",
    "ellipsis",
    "finfo_object",
    "iinfo_object",
    "Enum",
]
