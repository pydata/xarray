# import flox to avoid the cost of first import
import cftime
import flox.xarray  # noqa: F401
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
        self.ds2d = self.ds1d.expand_dims(z=10).copy()
        self.ds1d_mean = self.ds1d.groupby("b").mean()
        self.ds2d_mean = self.ds2d.groupby("b").mean()

    @parameterized(["ndim"], [(1, 2)])
    def time_init(self, ndim):
        getattr(self, f"ds{ndim}d").groupby("b")

    @parameterized(
        ["method", "ndim", "use_flox"], [("sum", "mean"), (1, 2), (True, False)]
    )
    def time_agg_small_num_groups(self, method, ndim, use_flox):
        ds = getattr(self, f"ds{ndim}d")
        with xr.set_options(use_flox=use_flox):
            getattr(ds.groupby("a"), method)().compute()

    @parameterized(
        ["method", "ndim", "use_flox"], [("sum", "mean"), (1, 2), (True, False)]
    )
    def time_agg_large_num_groups(self, method, ndim, use_flox):
        ds = getattr(self, f"ds{ndim}d")
        with xr.set_options(use_flox=use_flox):
            getattr(ds.groupby("b"), method)().compute()

    def time_binary_op_1d(self):
        (self.ds1d.groupby("b") - self.ds1d_mean).compute()

    def time_binary_op_2d(self):
        (self.ds2d.groupby("b") - self.ds2d_mean).compute()

    def peakmem_binary_op_1d(self):
        (self.ds1d.groupby("b") - self.ds1d_mean).compute()

    def peakmem_binary_op_2d(self):
        (self.ds2d.groupby("b") - self.ds2d_mean).compute()


class GroupByDask(GroupBy):
    def setup(self, *args, **kwargs):
        requires_dask()
        super().setup(**kwargs)

        self.ds1d = self.ds1d.sel(dim_0=slice(None, None, 2))
        self.ds1d["c"] = self.ds1d["c"].chunk({"dim_0": 50})
        self.ds2d = self.ds2d.sel(dim_0=slice(None, None, 2))
        self.ds2d["c"] = self.ds2d["c"].chunk({"dim_0": 50, "z": 5})
        self.ds1d_mean = self.ds1d.groupby("b").mean().compute()
        self.ds2d_mean = self.ds2d.groupby("b").mean().compute()


# TODO: These don't work now because we are calling `.compute` explicitly.
class GroupByPandasDataFrame(GroupBy):
    """Run groupby tests using pandas DataFrame."""

    def setup(self, *args, **kwargs):
        # Skip testing in CI as it won't ever change in a commit:
        _skip_slow()

        super().setup(**kwargs)
        self.ds1d = self.ds1d.to_dataframe()
        self.ds1d_mean = self.ds1d.groupby("b").mean()

    def time_binary_op_2d(self):
        raise NotImplementedError

    def peakmem_binary_op_2d(self):
        raise NotImplementedError


class GroupByDaskDataFrame(GroupBy):
    """Run groupby tests using dask DataFrame."""

    def setup(self, *args, **kwargs):
        # Skip testing in CI as it won't ever change in a commit:
        _skip_slow()

        requires_dask()
        super().setup(**kwargs)
        self.ds1d = self.ds1d.chunk({"dim_0": 50}).to_dask_dataframe()
        self.ds1d_mean = self.ds1d.groupby("b").mean().compute()

    def time_binary_op_2d(self):
        raise NotImplementedError

    def peakmem_binary_op_2d(self):
        raise NotImplementedError


class Resample:
    def setup(self, *args, **kwargs):
        self.ds1d = xr.Dataset(
            {
                "b": ("time", np.arange(365.0 * 24)),
            },
            coords={"time": pd.date_range("2001-01-01", freq="h", periods=365 * 24)},
        )
        self.ds2d = self.ds1d.expand_dims(z=10)
        self.ds1d_mean = self.ds1d.resample(time="48h").mean()
        self.ds2d_mean = self.ds2d.resample(time="48h").mean()

    @parameterized(["ndim"], [(1, 2)])
    def time_init(self, ndim):
        getattr(self, f"ds{ndim}d").resample(time="D")

    @parameterized(
        ["method", "ndim", "use_flox"], [("sum", "mean"), (1, 2), (True, False)]
    )
    def time_agg_small_num_groups(self, method, ndim, use_flox):
        ds = getattr(self, f"ds{ndim}d")
        with xr.set_options(use_flox=use_flox):
            getattr(ds.resample(time="3ME"), method)().compute()

    @parameterized(
        ["method", "ndim", "use_flox"], [("sum", "mean"), (1, 2), (True, False)]
    )
    def time_agg_large_num_groups(self, method, ndim, use_flox):
        ds = getattr(self, f"ds{ndim}d")
        with xr.set_options(use_flox=use_flox):
            getattr(ds.resample(time="48h"), method)().compute()


class ResampleDask(Resample):
    def setup(self, *args, **kwargs):
        requires_dask()
        super().setup(**kwargs)
        self.ds1d = self.ds1d.chunk({"time": 50})
        self.ds2d = self.ds2d.chunk({"time": 50, "z": 4})


class ResampleCFTime(Resample):
    def setup(self, *args, **kwargs):
        self.ds1d = xr.Dataset(
            {
                "b": ("time", np.arange(365.0 * 24)),
            },
            coords={
                "time": xr.date_range(
                    "2001-01-01", freq="h", periods=365 * 24, calendar="noleap"
                )
            },
        )
        self.ds2d = self.ds1d.expand_dims(z=10)
        self.ds1d_mean = self.ds1d.resample(time="48h").mean()
        self.ds2d_mean = self.ds2d.resample(time="48h").mean()


@parameterized(["use_cftime", "use_flox"], [[True, False], [True, False]])
class GroupByLongTime:
    def setup(self, use_cftime, use_flox):
        arr = np.random.randn(10, 10, 365 * 30)
        time = xr.date_range("2000", periods=30 * 365, use_cftime=use_cftime)

        # GH9426 - deep-copying CFTime object arrays is weirdly slow
        asda = xr.DataArray(time)
        labeled_time = []
        for year, month in zip(asda.dt.year, asda.dt.month, strict=True):
            labeled_time.append(cftime.datetime(year, month, 1))

        self.da = xr.DataArray(
            arr,
            dims=("y", "x", "time"),
            coords={"time": time, "time2": ("time", labeled_time)},
        )

    def time_setup(self, use_cftime, use_flox):
        self.da.groupby("time.month")

    def time_mean(self, use_cftime, use_flox):
        with xr.set_options(use_flox=use_flox):
            self.da.groupby("time.year").mean()
