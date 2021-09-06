import numpy as np
import pandas as pd

import xarray as xr

from . import parameterized, requires_dask


class GroupBy:
    def setup(self, *args, **kwargs):
        self.ds = xr.Dataset({
            "a": xr.DataArray(np.r_[np.arange(500.), np.arange(500.)]),
            "b": xr.DataArray(np.arange(1000.)),
        })

    @parameterized(["method"], [("sum", "mean")])
    def agg(self, method):
        getattr(self.ds.groupby("a"), method)()


class GroupByDask(GroupBy):
    def setup(self, *args, **kwargs):
        requires_dask()
        super().setup(**kwargs)
        self.ds = self.ds.chunk({"t": 50})


class GroupByDataFrame(GroupBy):
    def setup(self, *args, **kwargs):
        super().setup(**kwargs)
        self.ds = self.ds.to_dataframe()


class GroupByDaskDataFrame(GroupBy):
    def setup(self, *args, **kwargs):
        requires_dask()
        super().setup(**kwargs)
        self.ds = self.ds.chunk({"t": 50}).to_dataframe()
