import numpy as np
import pandas as pd

import xarray as xr

from . import parameterized, randn, requires_dask

nx = 300
long_nx = 30000
ny = 200
nt = 100
window = 20

randn_xy = randn((nx, ny), frac_nan=0.1)
randn_xt = randn((nx, nt))
randn_t = randn((nt,))
randn_long = randn((long_nx,), frac_nan=0.1)


class Rolling:
    def setup(self, *args, **kwargs):
        self.ds = xr.Dataset(
            {
                "var1": (("x", "y"), randn_xy),
                "var2": (("x", "t"), randn_xt),
                "var3": (("t",), randn_t),
            },
            coords={
                "x": np.arange(nx),
                "y": np.linspace(0, 1, ny),
                "t": pd.date_range("1970-01-01", periods=nt, freq="D"),
                "x_coords": ("x", np.linspace(1.1, 2.1, nx)),
            },
        )
        self.da_long = xr.DataArray(
            randn_long, dims="x", coords={"x": np.arange(long_nx) * 0.1}
        )

    @parameterized(
        ["func", "center", "use_bottleneck"],
        (["mean", "count"], [True, False], [True, False]),
    )
    def time_rolling(self, func, center, use_bottleneck):
        with xr.set_options(use_bottleneck=use_bottleneck):
            getattr(self.ds.rolling(x=window, center=center), func)().load()

    @parameterized(
        ["func", "pandas", "use_bottleneck"],
        (["mean", "count"], [True, False], [True, False]),
    )
    def time_rolling_long(self, func, pandas, use_bottleneck):
        if pandas:
            se = self.da_long.to_series()
            getattr(se.rolling(window=window, min_periods=window), func)()
        else:
            with xr.set_options(use_bottleneck=use_bottleneck):
                getattr(
                    self.da_long.rolling(x=window, min_periods=window), func
                )().load()

    @parameterized(
        ["window_", "min_periods", "use_bottleneck"], ([20, 40], [5, 5], [True, False])
    )
    def time_rolling_np(self, window_, min_periods, use_bottleneck):
        with xr.set_options(use_bottleneck=use_bottleneck):
            self.ds.rolling(x=window_, center=False, min_periods=min_periods).reduce(
                getattr(np, "nansum")
            ).load()

    @parameterized(
        ["center", "stride", "use_bottleneck"], ([True, False], [1, 1], [True, False])
    )
    def time_rolling_construct(self, center, stride, use_bottleneck):
        with xr.set_options(use_bottleneck=use_bottleneck):
            self.ds.rolling(x=window, center=center).construct(
                "window_dim", stride=stride
            ).sum(dim="window_dim").load()


class RollingDask(Rolling):
    def setup(self, *args, **kwargs):
        requires_dask()
        super().setup(**kwargs)
        self.ds = self.ds.chunk({"x": 100, "y": 50, "t": 50})
        self.da_long = self.da_long.chunk({"x": 10000})


class RollingMemory:
    def setup(self, *args, **kwargs):
        self.ds = xr.Dataset(
            {
                "var1": (("x", "y"), randn_xy),
                "var2": (("x", "t"), randn_xt),
                "var3": (("t",), randn_t),
            },
            coords={
                "x": np.arange(nx),
                "y": np.linspace(0, 1, ny),
                "t": pd.date_range("1970-01-01", periods=nt, freq="D"),
                "x_coords": ("x", np.linspace(1.1, 2.1, nx)),
            },
        )


class DataArrayRollingMemory(RollingMemory):
    @parameterized(["func", "use_bottleneck"], (["sum", "max", "mean"], [True, False]))
    def peakmem_ndrolling_reduce(self, func, use_bottleneck):
        with xr.set_options(use_bottleneck=use_bottleneck):
            roll = self.ds.var1.rolling(x=10, y=4)
            getattr(roll, func)()

    @parameterized(["func", "use_bottleneck"], (["sum", "max", "mean"], [True, False]))
    def peakmem_1drolling_reduce(self, func, use_bottleneck):
        with xr.set_options(use_bottleneck=use_bottleneck):
            roll = self.ds.var3.rolling(t=100)
            getattr(roll, func)()


class DatasetRollingMemory(RollingMemory):
    @parameterized(["func", "use_bottleneck"], (["sum", "max", "mean"], [True, False]))
    def peakmem_ndrolling_reduce(self, func, use_bottleneck):
        with xr.set_options(use_bottleneck=use_bottleneck):
            roll = self.ds.rolling(x=10, y=4)
            getattr(roll, func)()

    @parameterized(["func", "use_bottleneck"], (["sum", "max", "mean"], [True, False]))
    def peakmem_1drolling_reduce(self, func, use_bottleneck):
        with xr.set_options(use_bottleneck=use_bottleneck):
            roll = self.ds.rolling(t=100)
            getattr(roll, func)()
