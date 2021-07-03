from typing import TypeVar, Union

from .dataarray import DataArray
from .dataset import Dataset
from .groupby import DataArrayGroupBy, GroupBy
from .npcompat import ArrayLike
from .variable import Variable

import numpy as np

try:
    from dask.array import Array as DaskArray
except ImportError:
    DaskArray = np.ndarray

T_Dataset = TypeVar("T_Dataset", bound="Dataset")
T_DataArray = TypeVar("T_DataArray", bound="DataArray")
T_Variable = TypeVar("T_Variable", bound="Variable")
# Maybe we rename this to T_Data or something less Fortran-y?
T_DSorDA = TypeVar("T_DSorDA", "DataArray", Dataset)

ScalarOrArray = Union[ArrayLike, np.generic, np.ndarray, DaskArray]
DsCompatible = Union[Dataset, DataArray, Variable, GroupBy, ScalarOrArray]
DaCompatible = Union[DataArray, Variable, DataArrayGroupBy, ScalarOrArray]
VarCompatible = Union[Variable, ScalarOrArray]
GroupByIncompatible = Union[Variable, GroupBy]
