from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Union

import numpy as np

if TYPE_CHECKING:
    from .common import DataWithCoords
    from .dataarray import DataArray
    from .dataset import Dataset
    from .groupby import DataArrayGroupBy, GroupBy
    from .npcompat import ArrayLike
    from .variable import Variable

    try:
        from dask.array import Array as DaskArray
    except ImportError:
        DaskArray = np.ndarray

    T_Dataset = TypeVar("T_Dataset", bound=Dataset)
    T_DataArray = TypeVar("T_DataArray", bound=DataArray)
    T_Variable = TypeVar("T_Variable", bound=Variable)
    # Maybe we rename this to T_Data or something less Fortran-y?
    T_DSorDA = TypeVar("T_DSorDA", "DataArray", Dataset)
    T_DataWithCoords = TypeVar("T_DataWithCoords", bound=DataWithCoords)

    ScalarOrArray = ArrayLike | np.generic | np.ndarray | DaskArray
    DsCompatible = Dataset | DataArray | Variable | GroupBy | ScalarOrArray
    DaCompatible = DataArray | Variable | DataArrayGroupBy | ScalarOrArray
    VarCompatible = Variable | ScalarOrArray
    GroupByIncompatible = Variable | GroupBy
