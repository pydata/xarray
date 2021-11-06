import numpy as np
import pandas as pd

import xarray as xr

from . import parameterized, requires_dask


class GroupBy:
    def setup(self, *args, **kwargs):
        self.ds1d = xr.Dataset(
            {
                "a": xr.DataArray(np.r_[np.arange(300), np.arange(300)]),
                "b": xr.DataArray(np.arange(500)),
            }
        )
        self.ds2d = self.ds1d.expand_dims(z=10)

    @parameterized(["ndim"], [(1, 2)])
    def time_init(self, ndim):
        getattr(self, f"ds{ndim}d").groupby("b")

    @parameterized(["method", "ndim"], [("sum", "mean"), (1, 2)])
    def time_agg_small_num_groups(self, method, ndim):
        ds = getattr(self, f"ds{ndim}d")
        getattr(ds.groupby("a"), method)()

    @parameterized(["method", "ndim"], [("sum", "mean"), (1, 2)])
    def time_agg_large_num_groups(self, method, ndim):
        ds = getattr(self, f"ds{ndim}d")
        getattr(ds.groupby("b"), method)()


class GroupByDask(GroupBy):
    def setup(self, *args, **kwargs):
        requires_dask()
        super().setup(**kwargs)
        self.ds1d = self.ds1d.sel(dim_0=slice(250)).chunk({"dim_0": 50})
        self.ds2d = self.ds2d.sel(dim_0=slice(250)).chunk({"dim_0": 50, "z": 4})


class GroupByDataFrame(GroupBy):
    def setup(self, *args, **kwargs):
        super().setup(**kwargs)
        self.ds1d = self.ds1d.to_dataframe()


class GroupByDaskDataFrame(GroupBy):
    def setup(self, *args, **kwargs):
        requires_dask()
        super().setup(**kwargs)
        self.ds1d = self.ds1d.chunk({"dim_0": 50}).to_dataframe()


class Resample:
    def setup(self, *args, **kwargs):
        self.ds1d = xr.Dataset(
            {
                "b": ("time", np.arange(365.0 * 24)),
            },
            coords={"time": pd.date_range("2001-01-01", freq="H", periods=365 * 24)},
        )
        self.ds2d = self.ds1d.expand_dims(z=10)

    @parameterized(["ndim"], [(1, 2)])
    def time_init(self, ndim):
        getattr(self, f"ds{ndim}d").resample(time="D")

    @parameterized(["method", "ndim"], [("sum", "mean"), (1, 2)])
    def time_agg_small_num_groups(self, method, ndim):
        ds = getattr(self, f"ds{ndim}d")
        getattr(ds.resample(time="3M"), method)()

    @parameterized(["method", "ndim"], [("sum", "mean"), (1, 2)])
    def time_agg_large_num_groups(self, method, ndim):
        ds = getattr(self, f"ds{ndim}d")
        getattr(ds.resample(time="12H"), method)()


class ResampleDask(Resample):
    def setup(self, *args, **kwargs):
        requires_dask()
        super().setup(**kwargs)
        self.ds1d = self.ds1d.chunk({"time": 50})
        self.ds2d = self.ds2d.chunk({"time": 50, "z": 4})
