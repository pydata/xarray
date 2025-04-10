"""
Public typing utilities for use by external libraries.
"""

from typing import (
    TYPE_CHECKING,
)


if TYPE_CHECKING:
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset
    from xarray.computation.rolling import (
        DataArrayCoarsen,
        DataArrayRolling,
        DatasetRolling,
    )
    from xarray.computation.weighted import Weighted, DataArrayWeighted, DatasetWeighted
    from xarray.core.groupby import DataArrayGroupBy
    from xarray.core.resample import DataArrayResample

__all__ = [
    "DataArray",
    "Dataset",
    "DataArrayCoarsen",
    "DataArrayRolling",
    "DatasetRolling",
    "Weighted",
    "DataArrayWeighted",
    "DatasetWeighted",
    "DataArrayGroupBy",
    "DataArrayResample",
]
