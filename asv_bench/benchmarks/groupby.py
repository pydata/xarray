import numpy as np
import pandas as pd

import xarray as xr

from . import _skip_slow, parameterized, requires_dask


class GroupBy:
    def setup(self, *args, **kwargs):
        self.n = 100
        self.ds1d = xr.Dataset(
            {
                "a": xr.DataArray(np.r_[np.repeat(1, self.n), np.repeat(2, self.n)]),
                "b": xr.DataArray(np.arange(2 * self.n)),
                "c": xr.DataArray(np.arange(2 * self.n)),
            }
        )
        self.ds2d = self.ds1d.expand_dims(z=10)
        self.ds1d_mean = self.ds1d.groupby("b").mean()
        self.ds2d_mean = self.ds2d.groupby("b").mean()

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

    def time_groupby_binary_op_1d(self):
        self.ds1d - self.ds1d_mean

    def time_groupby_binary_op_2d(self):
        self.ds2d - self.ds2d_mean

    def peakmem_groupby_binary_op_1d(self):
        self.ds1d - self.ds1d_mean

    def peakmem_groupby_binary_op_2d(self):
        self.ds2d - self.ds2d_mean


class GroupByDask(GroupBy):
    def setup(self, *args, **kwargs):
        requires_dask()
        super().setup(**kwargs)

        self.ds1d = self.ds1d.sel(dim_0=slice(None, None, 2))
        self.ds1d["c"] = self.ds1d["c"].chunk({"dim_0": 50})
        self.ds2d = self.ds2d.sel(dim_0=slice(None, None, 2))
        self.ds2d["c"] = self.ds2d["c"].chunk({"dim_0": 50, "z": 5})
        self.ds1d_mean = self.ds1d.groupby("b").mean()
        self.ds2d_mean = self.ds2d.groupby("b").mean()


class GroupByPandasDataFrame(GroupBy):
    """Run groupby tests using pandas DataFrame."""

    def setup(self, *args, **kwargs):
        # Skip testing in CI as it won't ever change in a commit:
        _skip_slow()

        super().setup(**kwargs)
        self.ds1d = self.ds1d.to_dataframe()
        self.ds1d_mean = self.ds1d.groupby("b").mean()

    def time_groupby_binary_op_2d(self):
        raise NotImplementedError

    def peakmem_groupby_binary_op_2d(self):
        raise NotImplementedError


class GroupByDaskDataFrame(GroupBy):
    """Run groupby tests using dask DataFrame."""

    def setup(self, *args, **kwargs):
        # Skip testing in CI as it won't ever change in a commit:
        _skip_slow()

        requires_dask()
        super().setup(**kwargs)
        self.ds1d = self.ds1d.chunk({"dim_0": 50}).to_dataframe()
        self.ds1d_mean = self.ds1d.groupby("b").mean()

    def time_groupby_binary_op_2d(self):
        raise NotImplementedError

    def peakmem_groupby_binary_op_2d(self):
        raise NotImplementedError


class Resample:
    def setup(self, *args, **kwargs):
        self.ds1d = xr.Dataset(
            {
                "b": ("time", np.arange(365.0 * 24)),
            },
            coords={"time": pd.date_range("2001-01-01", freq="H", periods=365 * 24)},
        )
        self.ds2d = self.ds1d.expand_dims(z=10)
        self.ds1d_mean = self.ds1d.resample(time="48H").mean()
        self.ds2d_mean = self.ds2d.resample(time="48H").mean()

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
        getattr(ds.resample(time="48H"), method)()

    def time_groupby_binary_op_1d(self):
        self.ds1d - self.ds1d_mean

    def time_groupby_binary_op_2d(self):
        self.ds2d - self.ds2d_mean

    def peakmem_groupby_binary_op_1d(self):
        self.ds1d - self.ds1d_mean

    def peakmem_groupby_binary_op_2d(self):
        self.ds2d - self.ds2d_mean


class ResampleDask(Resample):
    def setup(self, *args, **kwargs):
        requires_dask()
        super().setup(**kwargs)
        self.ds1d = self.ds1d.chunk({"time": 50})
        self.ds2d = self.ds2d.chunk({"time": 50, "z": 4})
