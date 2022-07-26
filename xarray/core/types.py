from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Protocol,
    Sequence,
    TypeVar,
    Union,
)

import numpy as np
from numpy.typing._dtype_like import _DTypeLikeNested, _ShapeLike

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from .common import DataWithCoords
    from .dataarray import DataArray
    from .dataset import Dataset
    from .groupby import DataArrayGroupBy, GroupBy
    from .indexes import Index
    from .variable import Variable

    try:
        from dask.array import Array as DaskArray
    except ImportError:
        DaskArray = np.ndarray  # type: ignore


T_Dataset = TypeVar("T_Dataset", bound="Dataset")
T_DataArray = TypeVar("T_DataArray", bound="DataArray")
T_Variable = TypeVar("T_Variable", bound="Variable")
T_Index = TypeVar("T_Index", bound="Index")

T_DataArrayOrSet = TypeVar("T_DataArrayOrSet", bound=Union["Dataset", "DataArray"])

# Maybe we rename this to T_Data or something less Fortran-y?
T_Xarray = TypeVar("T_Xarray", "DataArray", "Dataset")
T_DataWithCoords = TypeVar("T_DataWithCoords", bound="DataWithCoords")

ScalarOrArray = Union["ArrayLike", np.generic, np.ndarray, "DaskArray"]
DsCompatible = Union["Dataset", "DataArray", "Variable", "GroupBy", "ScalarOrArray"]
DaCompatible = Union["DataArray", "Variable", "DataArrayGroupBy", "ScalarOrArray"]
VarCompatible = Union["Variable", "ScalarOrArray"]
GroupByIncompatible = Union["Variable", "GroupBy"]

ErrorOptions = Literal["raise", "ignore"]
ErrorOptionsWithWarn = Literal["raise", "warn", "ignore"]

CompatOptions = Literal[
    "identical", "equals", "broadcast_equals", "no_conflicts", "override", "minimal"
]
ConcatOptions = Literal["all", "minimal", "different"]
CombineAttrsOptions = Union[
    Literal["drop", "identical", "no_conflicts", "drop_conflicts", "override"],
    Callable[..., Any],
]
JoinOptions = Literal["outer", "inner", "left", "right", "exact", "override"]

Interp1dOptions = Literal[
    "linear", "nearest", "zero", "slinear", "quadratic", "cubic", "polynomial"
]
InterpolantOptions = Literal["barycentric", "krog", "pchip", "spline", "akima"]
InterpOptions = Union[Interp1dOptions, InterpolantOptions]

DatetimeUnitOptions = Literal[
    "Y", "M", "W", "D", "h", "m", "s", "ms", "us", "μs", "ns", "ps", "fs", "as", None
]

QueryEngineOptions = Literal["python", "numexpr", None]
QueryParserOptions = Literal["pandas", "python"]

ReindexMethodOptions = Literal["nearest", "pad", "ffill", "backfill", "bfill", None]

PadModeOptions = Literal[
    "constant",
    "edge",
    "linear_ramp",
    "maximum",
    "mean",
    "median",
    "minimum",
    "reflect",
    "symmetric",
    "wrap",
]
PadReflectOptions = Literal["even", "odd", None]

CFCalendar = Literal[
    "standard",
    "gregorian",
    "proleptic_gregorian",
    "noleap",
    "365_day",
    "360_day",
    "julian",
    "all_leap",
    "366_day",
]

CoarsenBoundaryOptions = Literal["exact", "trim", "pad"]
SideOptions = Literal["left", "right"]

# TODO: Wait until mypy supports recursive objects in combination with typevars
_T = TypeVar("_T")
NestedSequence = Union[
    _T,
    Sequence[_T],
    Sequence[Sequence[_T]],
    Sequence[Sequence[Sequence[_T]]],
    Sequence[Sequence[Sequence[Sequence[_T]]]],
]


# once NumPy 1.21 is minimum version, use NumPys definition directly
# 1.20 uses a non-generic Protocol (like we define here for simplicity)
class _SupportsDType(Protocol):
    @property
    def dtype(self) -> np.dtype:
        ...


# Xarray requires a Mapping[Hashable, dtype] in many places which
# conflics with numpys own DTypeLike (with dtypes for fields).
# https://numpy.org/devdocs/reference/typing.html#numpy.typing.DTypeLike
# This is a copy of this DTypeLike that allows only non-Mapping dtypes.
DTypeLikeSave = Union[
    np.dtype,
    # default data type (float64)
    None,
    # array-scalar types and generic types
    type[Any],
    # character codes, type strings or comma-separated fields, e.g., 'float64'
    str,
    # (flexible_dtype, itemsize)
    tuple[_DTypeLikeNested, int],
    # (fixed_dtype, shape)
    tuple[_DTypeLikeNested, _ShapeLike],
    # (base_dtype, new_dtype)
    tuple[_DTypeLikeNested, _DTypeLikeNested],
    # because numpy does the same?
    list[Any],
    # anything with a dtype attribute
    _SupportsDType,
]
