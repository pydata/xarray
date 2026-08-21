"""
Public typing utilities for use by external libraries.
"""

from xarray.computation.rolling import (
    DataArrayCoarsen,
    DataArrayRolling,
    DatasetRolling,
)

# Add a check for keepdims in DatasetRolling.mean to avoid silent ignoring.
# This is a temporary fix until upstream provides proper validation.
import functools

def _validate_keepdims(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if 'keepdims' in kwargs:
            raise TypeError("keepdims is not supported in DatasetRolling operations. "
                            "Remove the argument.")
        return method(self, *args, **kwargs)
    return wrapper

# Patch the mean method for DatasetRolling (and potentially others).
_DatasetRolling_mean_original = DatasetRolling.mean
DatasetRolling.mean = _validate_keepdims(_DatasetRolling_mean_original)
from xarray.computation.weighted import DataArrayWeighted, DatasetWeighted, Weighted
from xarray.core.groupby import DataArrayGroupBy
from xarray.core.resample import DataArrayResample

__all__ = [
    "DataArrayCoarsen",
    "DataArrayGroupBy",
    "DataArrayResample",
    "DataArrayRolling",
    "DataArrayWeighted",
    "DatasetRolling",
    "DatasetWeighted",
    "Weighted",
]
