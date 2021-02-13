"""Stub file for arithmetic operators of various xarray classes.

This file was generated using xarray.util.stubgen_ops. Do not edit manually."""

from typing import NoReturn, TypeVar, Union, overload

import numpy as np

from .dataarray import DataArray
from .dataset import Dataset
from .groupby import DataArrayGroupBy, DatasetGroupBy, GroupBy
from .variable import Variable

try:
    from dask.array import Array as DaskArray
except ImportError:
    DaskArray = np.ndarray

T_Dataset = TypeVar("T_Dataset", bound=Dataset)
T_DataArray = TypeVar("T_DataArray", bound=DataArray)
T_Variable = TypeVar("T_Variable", bound=Variable)
T_Self = TypeVar("T_Self")

# Note: T_Other (and types involving T_Other) is to be used last in overloads,
# since nd.ndarray is typed as Any for older versions of numpy.
T_Other = Union[complex, bytes, str, np.ndarray, np.generic, DaskArray]
T_DsOther = Union[Dataset, DataArray, Variable, T_Other, GroupBy]
T_DaOther = Union[DataArray, Variable, T_Other]
T_VarOther = Union[Variable, T_Other]
T_GroupByIncompatible = Union[Variable, GroupBy, T_Other]

class TypedDatasetOps:
    def __neg__(self: T_Self) -> T_Self: ...
    def __pos__(self: T_Self) -> T_Self: ...
    def __abs__(self: T_Self) -> T_Self: ...
    def __invert__(self: T_Self) -> T_Self: ...
    def __eq__(self: T_Dataset, other: T_DsOther) -> T_Dataset: ...  # type: ignore[override, misc]
    def __ne__(self: T_Dataset, other: T_DsOther) -> T_Dataset: ...  # type: ignore[override, misc]
    def __lt__(self: T_Dataset, other: T_DsOther) -> T_Dataset: ...  # type: ignore[misc]
    def __le__(self: T_Dataset, other: T_DsOther) -> T_Dataset: ...  # type: ignore[misc]
    def __gt__(self: T_Dataset, other: T_DsOther) -> T_Dataset: ...  # type: ignore[misc]
    def __ge__(self: T_Dataset, other: T_DsOther) -> T_Dataset: ...  # type: ignore[misc]
    def __add__(self: T_Dataset, other: T_DsOther) -> T_Dataset: ...  # type: ignore[misc]
    def __sub__(self: T_Dataset, other: T_DsOther) -> T_Dataset: ...  # type: ignore[misc]
    def __mul__(self: T_Dataset, other: T_DsOther) -> T_Dataset: ...  # type: ignore[misc]
    def __pow__(self: T_Dataset, other: T_DsOther) -> T_Dataset: ...  # type: ignore[misc]
    def __truediv__(self: T_Dataset, other: T_DsOther) -> T_Dataset: ...  # type: ignore[misc]
    def __floordiv__(self: T_Dataset, other: T_DsOther) -> T_Dataset: ...  # type: ignore[misc]
    def __mod__(self: T_Dataset, other: T_DsOther) -> T_Dataset: ...  # type: ignore[misc]
    def __radd__(self: T_Dataset, other: T_DsOther) -> T_Dataset: ...  # type: ignore[misc]
    def __rsub__(self: T_Dataset, other: T_DsOther) -> T_Dataset: ...  # type: ignore[misc]
    def __rmul__(self: T_Dataset, other: T_DsOther) -> T_Dataset: ...  # type: ignore[misc]
    def __rpow__(self: T_Dataset, other: T_DsOther) -> T_Dataset: ...  # type: ignore[misc]
    def __rtruediv__(self: T_Dataset, other: T_DsOther) -> T_Dataset: ...  # type: ignore[misc]
    def __rfloordiv__(self: T_Dataset, other: T_DsOther) -> T_Dataset: ...  # type: ignore[misc]
    def __rmod__(self: T_Dataset, other: T_DsOther) -> T_Dataset: ...  # type: ignore[misc]
    def __and__(self: T_Dataset, other: T_DsOther) -> T_Dataset: ...  # type: ignore[misc]
    def __xor__(self: T_Dataset, other: T_DsOther) -> T_Dataset: ...  # type: ignore[misc]
    def __or__(self: T_Dataset, other: T_DsOther) -> T_Dataset: ...  # type: ignore[misc]
    def __rand__(self: T_Dataset, other: T_DsOther) -> T_Dataset: ...  # type: ignore[misc]
    def __rxor__(self: T_Dataset, other: T_DsOther) -> T_Dataset: ...  # type: ignore[misc]
    def __ror__(self: T_Dataset, other: T_DsOther) -> T_Dataset: ...  # type: ignore[misc]

class TypedDataArrayOps:
    def __neg__(self: T_Self) -> T_Self: ...
    def __pos__(self: T_Self) -> T_Self: ...
    def __abs__(self: T_Self) -> T_Self: ...
    def __invert__(self: T_Self) -> T_Self: ...
    @overload  # type: ignore[override]
    def __eq__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __eq__(self, other: DatasetGroupBy) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __eq__(self: T_DataArray, other: DataArrayGroupBy) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __eq__(self: T_DataArray, other: T_DaOther) -> T_DataArray: ...  # type: ignore[misc]
    @overload  # type: ignore[override]
    def __ne__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __ne__(self, other: DatasetGroupBy) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __ne__(self: T_DataArray, other: DataArrayGroupBy) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __ne__(self: T_DataArray, other: T_DaOther) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __lt__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __lt__(self, other: DatasetGroupBy) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __lt__(self: T_DataArray, other: DataArrayGroupBy) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __lt__(self: T_DataArray, other: T_DaOther) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __le__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __le__(self, other: DatasetGroupBy) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __le__(self: T_DataArray, other: DataArrayGroupBy) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __le__(self: T_DataArray, other: T_DaOther) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __gt__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __gt__(self, other: DatasetGroupBy) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __gt__(self: T_DataArray, other: DataArrayGroupBy) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __gt__(self: T_DataArray, other: T_DaOther) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __ge__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __ge__(self, other: DatasetGroupBy) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __ge__(self: T_DataArray, other: DataArrayGroupBy) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __ge__(self: T_DataArray, other: T_DaOther) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __add__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __add__(self, other: DatasetGroupBy) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __add__(self: T_DataArray, other: DataArrayGroupBy) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __add__(self: T_DataArray, other: T_DaOther) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __sub__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __sub__(self, other: DatasetGroupBy) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __sub__(self: T_DataArray, other: DataArrayGroupBy) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __sub__(self: T_DataArray, other: T_DaOther) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __mul__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __mul__(self, other: DatasetGroupBy) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __mul__(self: T_DataArray, other: DataArrayGroupBy) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __mul__(self: T_DataArray, other: T_DaOther) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __pow__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __pow__(self, other: DatasetGroupBy) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __pow__(self: T_DataArray, other: DataArrayGroupBy) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __pow__(self: T_DataArray, other: T_DaOther) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __truediv__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __truediv__(self, other: DatasetGroupBy) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __truediv__(self: T_DataArray, other: DataArrayGroupBy) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __truediv__(self: T_DataArray, other: T_DaOther) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __floordiv__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __floordiv__(self, other: DatasetGroupBy) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __floordiv__(self: T_DataArray, other: DataArrayGroupBy) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __floordiv__(self: T_DataArray, other: T_DaOther) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __mod__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __mod__(self, other: DatasetGroupBy) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __mod__(self: T_DataArray, other: DataArrayGroupBy) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __mod__(self: T_DataArray, other: T_DaOther) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __radd__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __radd__(self, other: DatasetGroupBy) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __radd__(self: T_DataArray, other: DataArrayGroupBy) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __radd__(self: T_DataArray, other: T_DaOther) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __rsub__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __rsub__(self, other: DatasetGroupBy) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: T_DataArray, other: DataArrayGroupBy) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: T_DataArray, other: T_DaOther) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __rmul__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __rmul__(self, other: DatasetGroupBy) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: T_DataArray, other: DataArrayGroupBy) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: T_DataArray, other: T_DaOther) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __rpow__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __rpow__(self, other: DatasetGroupBy) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __rpow__(self: T_DataArray, other: DataArrayGroupBy) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __rpow__(self: T_DataArray, other: T_DaOther) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __rtruediv__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __rtruediv__(self, other: DatasetGroupBy) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __rtruediv__(self: T_DataArray, other: DataArrayGroupBy) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __rtruediv__(self: T_DataArray, other: T_DaOther) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __rfloordiv__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __rfloordiv__(self, other: DatasetGroupBy) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __rfloordiv__(self: T_DataArray, other: DataArrayGroupBy) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __rfloordiv__(self: T_DataArray, other: T_DaOther) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __rmod__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __rmod__(self, other: DatasetGroupBy) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __rmod__(self: T_DataArray, other: DataArrayGroupBy) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __rmod__(self: T_DataArray, other: T_DaOther) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __and__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __and__(self, other: DatasetGroupBy) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __and__(self: T_DataArray, other: DataArrayGroupBy) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __and__(self: T_DataArray, other: T_DaOther) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __xor__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __xor__(self, other: DatasetGroupBy) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __xor__(self: T_DataArray, other: DataArrayGroupBy) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __xor__(self: T_DataArray, other: T_DaOther) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __or__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __or__(self, other: DatasetGroupBy) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __or__(self: T_DataArray, other: DataArrayGroupBy) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __or__(self: T_DataArray, other: T_DaOther) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __rand__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __rand__(self, other: DatasetGroupBy) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __rand__(self: T_DataArray, other: DataArrayGroupBy) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __rand__(self: T_DataArray, other: T_DaOther) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __rxor__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __rxor__(self, other: DatasetGroupBy) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __rxor__(self: T_DataArray, other: DataArrayGroupBy) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __rxor__(self: T_DataArray, other: T_DaOther) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __ror__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __ror__(self, other: DatasetGroupBy) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __ror__(self: T_DataArray, other: DataArrayGroupBy) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __ror__(self: T_DataArray, other: T_DaOther) -> T_DataArray: ...  # type: ignore[misc]

class TypedVariableOps:
    def __neg__(self: T_Self) -> T_Self: ...
    def __pos__(self: T_Self) -> T_Self: ...
    def __abs__(self: T_Self) -> T_Self: ...
    def __invert__(self: T_Self) -> T_Self: ...
    @overload  # type: ignore[override]
    def __eq__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __eq__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __eq__(self: T_Variable, other: T_VarOther) -> T_Variable: ...  # type: ignore[misc]
    @overload  # type: ignore[override]
    def __ne__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __ne__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __ne__(self: T_Variable, other: T_VarOther) -> T_Variable: ...  # type: ignore[misc]
    @overload
    def __lt__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __lt__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __lt__(self: T_Variable, other: T_VarOther) -> T_Variable: ...  # type: ignore[misc]
    @overload
    def __le__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __le__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __le__(self: T_Variable, other: T_VarOther) -> T_Variable: ...  # type: ignore[misc]
    @overload
    def __gt__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __gt__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __gt__(self: T_Variable, other: T_VarOther) -> T_Variable: ...  # type: ignore[misc]
    @overload
    def __ge__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __ge__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __ge__(self: T_Variable, other: T_VarOther) -> T_Variable: ...  # type: ignore[misc]
    @overload
    def __add__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __add__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __add__(self: T_Variable, other: T_VarOther) -> T_Variable: ...  # type: ignore[misc]
    @overload
    def __sub__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __sub__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __sub__(self: T_Variable, other: T_VarOther) -> T_Variable: ...  # type: ignore[misc]
    @overload
    def __mul__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __mul__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __mul__(self: T_Variable, other: T_VarOther) -> T_Variable: ...  # type: ignore[misc]
    @overload
    def __pow__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __pow__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __pow__(self: T_Variable, other: T_VarOther) -> T_Variable: ...  # type: ignore[misc]
    @overload
    def __truediv__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __truediv__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __truediv__(self: T_Variable, other: T_VarOther) -> T_Variable: ...  # type: ignore[misc]
    @overload
    def __floordiv__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __floordiv__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __floordiv__(self: T_Variable, other: T_VarOther) -> T_Variable: ...  # type: ignore[misc]
    @overload
    def __mod__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __mod__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __mod__(self: T_Variable, other: T_VarOther) -> T_Variable: ...  # type: ignore[misc]
    @overload
    def __radd__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __radd__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __radd__(self: T_Variable, other: T_VarOther) -> T_Variable: ...  # type: ignore[misc]
    @overload
    def __rsub__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __rsub__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: T_Variable, other: T_VarOther) -> T_Variable: ...  # type: ignore[misc]
    @overload
    def __rmul__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __rmul__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: T_Variable, other: T_VarOther) -> T_Variable: ...  # type: ignore[misc]
    @overload
    def __rpow__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __rpow__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __rpow__(self: T_Variable, other: T_VarOther) -> T_Variable: ...  # type: ignore[misc]
    @overload
    def __rtruediv__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __rtruediv__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __rtruediv__(self: T_Variable, other: T_VarOther) -> T_Variable: ...  # type: ignore[misc]
    @overload
    def __rfloordiv__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __rfloordiv__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __rfloordiv__(self: T_Variable, other: T_VarOther) -> T_Variable: ...  # type: ignore[misc]
    @overload
    def __rmod__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __rmod__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __rmod__(self: T_Variable, other: T_VarOther) -> T_Variable: ...  # type: ignore[misc]
    @overload
    def __and__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __and__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __and__(self: T_Variable, other: T_VarOther) -> T_Variable: ...  # type: ignore[misc]
    @overload
    def __xor__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __xor__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __xor__(self: T_Variable, other: T_VarOther) -> T_Variable: ...  # type: ignore[misc]
    @overload
    def __or__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __or__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __or__(self: T_Variable, other: T_VarOther) -> T_Variable: ...  # type: ignore[misc]
    @overload
    def __rand__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __rand__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __rand__(self: T_Variable, other: T_VarOther) -> T_Variable: ...  # type: ignore[misc]
    @overload
    def __rxor__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __rxor__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __rxor__(self: T_Variable, other: T_VarOther) -> T_Variable: ...  # type: ignore[misc]
    @overload
    def __ror__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __ror__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __ror__(self: T_Variable, other: T_VarOther) -> T_Variable: ...  # type: ignore[misc]

class TypedDatasetGroupByOps:
    @overload  # type: ignore[override]
    def __eq__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __eq__(self, other: DataArray) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __eq__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload  # type: ignore[override]
    def __ne__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __ne__(self, other: DataArray) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __ne__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __lt__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __lt__(self, other: DataArray) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __lt__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __le__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __le__(self, other: DataArray) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __le__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __gt__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __gt__(self, other: DataArray) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __gt__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __ge__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __ge__(self, other: DataArray) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __ge__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __add__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __add__(self, other: DataArray) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __add__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __sub__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __sub__(self, other: DataArray) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __sub__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __mul__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __mul__(self, other: DataArray) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __mul__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __pow__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __pow__(self, other: DataArray) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __pow__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __truediv__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __truediv__(self, other: DataArray) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __truediv__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __floordiv__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __floordiv__(self, other: DataArray) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __floordiv__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __mod__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __mod__(self, other: DataArray) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __mod__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __radd__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __radd__(self, other: DataArray) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __radd__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __rsub__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __rsub__(self, other: DataArray) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __rsub__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __rmul__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __rmul__(self, other: DataArray) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __rmul__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __rpow__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __rpow__(self, other: DataArray) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __rpow__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __rtruediv__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __rtruediv__(self, other: DataArray) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __rtruediv__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __rfloordiv__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __rfloordiv__(self, other: DataArray) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __rfloordiv__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __rmod__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __rmod__(self, other: DataArray) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __rmod__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __and__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __and__(self, other: DataArray) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __and__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __xor__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __xor__(self, other: DataArray) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __xor__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __or__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __or__(self, other: DataArray) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __or__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __rand__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __rand__(self, other: DataArray) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __rand__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __rxor__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __rxor__(self, other: DataArray) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __rxor__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __ror__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __ror__(self, other: DataArray) -> Dataset: ...  # type: ignore[misc]
    @overload
    def __ror__(self, other: T_GroupByIncompatible) -> NoReturn: ...

class TypedDataArrayGroupByOps:
    @overload  # type: ignore[override]
    def __eq__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __eq__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __eq__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload  # type: ignore[override]
    def __ne__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __ne__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __ne__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __lt__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __lt__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __lt__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __le__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __le__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __le__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __gt__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __gt__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __gt__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __ge__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __ge__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __ge__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __add__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __add__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __add__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __sub__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __sub__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __sub__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __mul__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __mul__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __mul__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __pow__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __pow__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __pow__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __truediv__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __truediv__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __truediv__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __floordiv__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __floordiv__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __floordiv__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __mod__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __mod__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __mod__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __radd__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __radd__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __radd__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __rsub__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __rsub__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __rsub__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __rmul__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __rmul__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __rmul__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __rpow__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __rpow__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __rpow__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __rtruediv__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __rtruediv__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __rtruediv__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __rfloordiv__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __rfloordiv__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __rfloordiv__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __rmod__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __rmod__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __rmod__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __and__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __and__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __and__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __xor__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __xor__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __xor__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __or__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __or__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __or__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __rand__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __rand__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __rand__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __rxor__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __rxor__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __rxor__(self, other: T_GroupByIncompatible) -> NoReturn: ...
    @overload
    def __ror__(self, other: T_Dataset) -> T_Dataset: ...  # type: ignore[misc]
    @overload
    def __ror__(self, other: T_DataArray) -> T_DataArray: ...  # type: ignore[misc]
    @overload
    def __ror__(self, other: T_GroupByIncompatible) -> NoReturn: ...
