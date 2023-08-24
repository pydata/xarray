from __future__ import annotations

import datetime
import sys
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Protocol,
    SupportsIndex,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd

try:
    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self
except ImportError:
    if TYPE_CHECKING:
        raise
    else:
        Self: Any = None

if TYPE_CHECKING:
    from numpy._typing import _SupportsDType
    from numpy.typing import ArrayLike

    from xarray.backends.common import BackendEntrypoint
    from xarray.core.alignment import Aligner
    from xarray.core.common import AbstractArray, DataWithCoords
    from xarray.core.coordinates import Coordinates
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset
    from xarray.core.groupby import DataArrayGroupBy, GroupBy
    from xarray.core.indexes import Index, Indexes
    from xarray.core.utils import Frozen
    from xarray.core.variable import Variable

    try:
        from dask.array import Array as DaskArray
    except ImportError:
        DaskArray = np.ndarray  # type: ignore

    try:
        from cubed import Array as CubedArray
    except ImportError:
        CubedArray = np.ndarray

    try:
        from zarr.core import Array as ZarrArray
    except ImportError:
        ZarrArray = np.ndarray

    # Anything that can be coerced to a shape tuple
    _ShapeLike = Union[SupportsIndex, Sequence[SupportsIndex]]
    _DTypeLikeNested = Any  # TODO: wait for support for recursive types

    # Xarray requires a Mapping[Hashable, dtype] in many places which
    # conflics with numpys own DTypeLike (with dtypes for fields).
    # https://numpy.org/devdocs/reference/typing.html#numpy.typing.DTypeLike
    # This is a copy of this DTypeLike that allows only non-Mapping dtypes.
    DTypeLikeSave = Union[
        np.dtype[Any],
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
        _SupportsDType[np.dtype[Any]],
    ]
    try:
        from cftime import datetime as CFTimeDatetime
    except ImportError:
        CFTimeDatetime = Any
    DatetimeLike = Union[pd.Timestamp, datetime.datetime, np.datetime64, CFTimeDatetime]
else:
    DTypeLikeSave: Any = None


class Alignable(Protocol):
    """Represents any Xarray type that supports alignment.

    It may be ``Dataset``, ``DataArray`` or ``Coordinates``. This protocol class
    is needed since those types do not all have a common base class.

    """

    @property
    def dims(self) -> Frozen[Hashable, int] | tuple[Hashable, ...]:
        ...

    @property
    def sizes(self) -> Frozen[Hashable, int]:
        ...

    @property
    def xindexes(self) -> Indexes[Index]:
        ...

    def _reindex_callback(
        self,
        aligner: Aligner,
        dim_pos_indexers: dict[Hashable, Any],
        variables: dict[Hashable, Variable],
        indexes: dict[Hashable, Index],
        fill_value: Any,
        exclude_dims: frozenset[Hashable],
        exclude_vars: frozenset[Hashable],
    ) -> Self:
        ...

    def _overwrite_indexes(
        self,
        indexes: Mapping[Any, Index],
        variables: Mapping[Any, Variable] | None = None,
    ) -> Self:
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[Hashable]:
        ...

    def copy(
        self,
        deep: bool = False,
    ) -> Self:
        ...


T_Backend = TypeVar("T_Backend", bound="BackendEntrypoint")
T_Dataset = TypeVar("T_Dataset", bound="Dataset")
T_DataArray = TypeVar("T_DataArray", bound="DataArray")
T_Variable = TypeVar("T_Variable", bound="Variable")
T_Coordinates = TypeVar("T_Coordinates", bound="Coordinates")
T_Array = TypeVar("T_Array", bound="AbstractArray")
T_Index = TypeVar("T_Index", bound="Index")

T_DataArrayOrSet = TypeVar("T_DataArrayOrSet", bound=Union["Dataset", "DataArray"])

# Maybe we rename this to T_Data or something less Fortran-y?
T_Xarray = TypeVar("T_Xarray", "DataArray", "Dataset")
T_DataWithCoords = TypeVar("T_DataWithCoords", bound="DataWithCoords")
T_Alignable = TypeVar("T_Alignable", bound="Alignable")

ScalarOrArray = Union["ArrayLike", np.generic, np.ndarray, "DaskArray"]
DsCompatible = Union["Dataset", "DataArray", "Variable", "GroupBy", "ScalarOrArray"]
DaCompatible = Union["DataArray", "Variable", "DataArrayGroupBy", "ScalarOrArray"]
VarCompatible = Union["Variable", "ScalarOrArray"]
GroupByIncompatible = Union["Variable", "GroupBy"]

Dims = Union[str, Iterable[Hashable], "ellipsis", None]
OrderedDims = Union[str, Sequence[Union[Hashable, "ellipsis"]], "ellipsis", None]

T_Chunks = Union[int, dict[Any, Any], Literal["auto"], None]
T_NormalizedChunks = tuple[tuple[int, ...], ...]

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
InclusiveOptions = Literal["both", "neither", "left", "right"]

ScaleOptions = Literal["linear", "symlog", "log", "logit", None]
HueStyleOptions = Literal["continuous", "discrete", None]
AspectOptions = Union[Literal["auto", "equal"], float, None]
ExtendOptions = Literal["neither", "both", "min", "max", None]

# TODO: Wait until mypy supports recursive objects in combination with typevars
_T = TypeVar("_T")
NestedSequence = Union[
    _T,
    Sequence[_T],
    Sequence[Sequence[_T]],
    Sequence[Sequence[Sequence[_T]]],
    Sequence[Sequence[Sequence[Sequence[_T]]]],
]


QuantileMethods = Literal[
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "linear",
    "median_unbiased",
    "normal_unbiased",
    "lower",
    "higher",
    "midpoint",
    "nearest",
]
