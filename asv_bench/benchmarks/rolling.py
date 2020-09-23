import numpy as np
import pandas as pd

import xarray as xr

from . import parameterized, randn, requires_dask

nx = 3000
long_nx = 30000000
ny = 2000
nt = 1000
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

    @parameterized(["func", "center"], (["mean", "count"], [True, False]))
    def time_rolling(self, func, center):
        getattr(self.ds.rolling(x=window, center=center), func)().load()

    @parameterized(["func", "pandas"], (["mean", "count"], [True, False]))
    def time_rolling_long(self, func, pandas):
        if pandas:
            se = self.da_long.to_series()
            getattr(se.rolling(window=window), func)()
        else:
            getattr(self.da_long.rolling(x=window), func)().load()

    @parameterized(["window_", "min_periods"], ([20, 40], [5, None]))
    def time_rolling_np(self, window_, min_periods):
        self.ds.rolling(x=window_, center=False, min_periods=min_periods).reduce(
            getattr(np, "nanmean")
        ).load()

    @parameterized(["center", "stride"], ([True, False], [1, 200]))
    def time_rolling_construct(self, center, stride):
        self.ds.rolling(x=window, center=center).construct(
            "window_dim", stride=stride
        ).mean(dim="window_dim").load()


class RollingDask(Rolling):
    def setup(self, *args, **kwargs):
        requires_dask()
        super().setup(**kwargs)
        self.ds = self.ds.chunk({"x": 100, "y": 50, "t": 50})
        self.da_long = self.da_long.chunk({"x": 10000})
