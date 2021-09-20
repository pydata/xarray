import numpy as np

import xarray as xr

from . import parameterized, requires_dask


class GroupBy:
    def setup(self, *args, **kwargs):
        self.ds = xr.Dataset(
            {
                "a": xr.DataArray(np.r_[np.arange(500.0), np.arange(500.0)]),
                "b": xr.DataArray(np.arange(1000.0)),
            }
        )

    @parameterized(["method"], [("sum", "mean")])
    def time_agg(self, method):
        return getattr(self.ds.groupby("a"), method)()


class GroupByDask(GroupBy):
    def setup(self, *args, **kwargs):
        requires_dask()
        super().setup(**kwargs)
        self.ds = self.ds.chunk({"dim_0": 50})


class GroupByDataFrame(GroupBy):
    def setup(self, *args, **kwargs):
        super().setup(**kwargs)
        self.ds = self.ds.to_dataframe()


class GroupByDaskDataFrame(GroupBy):
    def setup(self, *args, **kwargs):
        requires_dask()
        super().setup(**kwargs)
        self.ds = self.ds.chunk({"dim_0": 50}).to_dataframe()
