from __future__ import annotations

import typing
from collections.abc import Hashable, Iterable

import numpy as np

if typing.TYPE_CHECKING:
    from xarray.namedarray.core import NamedArray

try:
    from dask.array import Array as DaskArray
except ImportError:
    DaskArray = np.ndarray  # type: ignore

T_NamedArray = typing.TypeVar("T_NamedArray", bound="NamedArray")
DimsInput = typing.Union[str, Iterable[Hashable]]
Dims = tuple[Hashable, ...]


# temporary placeholder for indicating an array api compliant type.
# hopefully in the future we can narrow this down more
T_DuckArray = typing.TypeVar("T_DuckArray", bound=typing.Any)

ScalarOrArray = typing.Union[np.generic, np.ndarray, DaskArray]
NamedArrayCompatible = typing.Union[T_NamedArray, ScalarOrArray]
