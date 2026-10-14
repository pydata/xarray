from typing import Any

import numpy as np
import pandas as pd
from asv_runner.benchmarks.mark import parameterize

import xarray as xr

from . import randn, requires_dask

nx = 1500
ny = 1000
nt = 500

randn_xy = randn((nx, ny), frac_nan=0.1)
randn_xt = randn((nx, nt))
randn_t = randn((nt,))

new_x_short = np.linspace(0.3 * nx, 0.7 * nx, 100)
new_x_long = np.linspace(0.3 * nx, 0.7 * nx, 500)
new_y_long = np.linspace(0.1, 0.9, 500)


class InterpolationSplineND:
    # spline interpolation over multiple dimensions (GH11342)
    def setup(self, *args: Any, **kwargs: Any) -> None:
        nt, ny, nx = 50, 40, 50
        self.da = xr.DataArray(
            randn((nt, ny, nx)),
            dims=("t", "y", "x"),
            coords={
                "t": np.arange(nt),
                "y": np.linspace(0, 1, ny),
                "x": np.linspace(0, 1, nx),
            },
        )
        self.destinations = {
            "point": {"t": 25.5, "y": 0.45, "x": 0.12},
            "grid": {
                "t": np.linspace(5, 45, 20),
                "y": np.linspace(0.1, 0.9, 20),
                "x": np.linspace(0.1, 0.9, 20),
            },
            "pointwise": {
                dim: xr.DataArray(np.linspace(lo, hi, 100), dims="p")
                for dim, lo, hi in [("t", 5, 45), ("y", 0.1, 0.9), ("x", 0.9, 0.1)]
            },
        }

    @parameterized(
        ["method", "destination"],
        (["cubic", "quintic"], ["point", "grid", "pointwise"]),
    )
    def time_interpolation_spline_nd(self, method: str, destination: str) -> None:
        self.da.interp(self.destinations[destination], method=method).compute()

    @parameterized(["method"], (["cubic", "quintic"]))
    def peakmem_interpolation_spline_nd(self, method: str) -> None:
        self.da.interp(self.destinations["point"], method=method).compute()


class InterpolationSplineNDDask(InterpolationSplineND):
    def setup(self, *args: Any, **kwargs: Any) -> None:
        requires_dask()
        super().setup(**kwargs)
        self.da = self.da.chunk({"t": 10})


class Interpolation:
    def setup(self, *args, **kwargs):
        self.ds = xr.Dataset(
            {
                "var1": (("x", "y"), randn_xy),
                "var2": (("x", "t"), randn_xt),
                "var3": (("t",), randn_t),
                "var4": (("z",), np.array(["text"])),
                "var5": (("k",), np.array(["a", "b", "c"])),
            },
            coords={
                "x": np.arange(nx),
                "y": np.linspace(0, 1, ny),
                "t": pd.date_range("1970-01-01", periods=nt, freq="D"),
                "x_coords": ("x", np.linspace(1.1, 2.1, nx)),
                "z": np.array([1]),
                "k": np.linspace(0, nx, 3),
            },
        )

    @parameterize({"method": ["linear", "cubic"], "is_short": [True, False]})
    def time_interpolation_numeric_1d(self, method, is_short):
        new_x = new_x_short if is_short else new_x_long
        self.ds.interp(x=new_x, method=method).compute()

    @parameterize({"method": ["linear", "nearest"]})
    def time_interpolation_numeric_2d(self, method):
        self.ds.interp(x=new_x_long, y=new_y_long, method=method).compute()

    @parameterize({"is_short": [True, False]})
    def time_interpolation_string_scalar(self, is_short):
        new_z = new_x_short if is_short else new_x_long
        self.ds.interp(z=new_z).compute()

    @parameterize({"is_short": [True, False]})
    def time_interpolation_string_1d(self, is_short):
        new_k = new_x_short if is_short else new_x_long
        self.ds.interp(k=new_k).compute()


class InterpolationDask(Interpolation):
    def setup(self, *args, **kwargs):
        requires_dask()
        super().setup(**kwargs)
        self.ds = self.ds.chunk({"t": 50})
