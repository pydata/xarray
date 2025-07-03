"""Xarray index objects for label-based selection and alignment of Dataset /
DataArray objects.

"""

from xarray.core.indexes import (
    Index,
    PandasIndex,
    PandasMultiIndex,
)
from xarray.indexes.cf_interval_index import CFIntervalIndex
from xarray.indexes.range_index import RangeIndex

__all__ = ["CFIntervalIndex", "Index", "PandasIndex", "PandasMultiIndex", "RangeIndex"]
