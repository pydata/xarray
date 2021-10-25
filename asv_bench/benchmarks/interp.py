import numpy as np
import pandas as pd

import xarray as xr

from . import parameterized, randn, requires_dask

nx = 1500
ny = 1000
nt = 500

randn_xy = randn((nx, ny), frac_nan=0.1)
randn_xt = randn((nx, nt))
randn_t = randn((nt,))

new_x_short = np.linspace(0.3 * nx, 0.7 * nx, 100)
new_x_long = np.linspace(0.3 * nx, 0.7 * nx, 500)
new_y_long = np.linspace(0.1, 0.9, 500)


class Interpolation:
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

    @parameterized(["method", "is_short"], (["linear", "cubic"], [True, False]))
    def time_interpolation(self, method, is_short):
        new_x = new_x_short if is_short else new_x_long
        self.ds.interp(x=new_x, method=method).load()

    @parameterized(["method"], (["linear", "nearest"]))
    def time_interpolation_2d(self, method):
        self.ds.interp(x=new_x_long, y=new_y_long, method=method).load()


class InterpolationDask(Interpolation):
    def setup(self, *args, **kwargs):
        requires_dask()
        super().setup(**kwargs)
        self.ds = self.ds.chunk({"t": 50})
