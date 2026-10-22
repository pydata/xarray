from typing import Any

import numpy as np
import pandas as pd
from asv_runner.benchmarks.mark import parameterize

import xarray as xr
from xarray.core.types import InterpOptions

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


class InterpolationVectorized:
    # the destination coordinates vary along a dimension of the source (GH10683)
    def setup(self, *args, **kwargs):
        nt, nr, nz = 100, 50, 50
        self.da = xr.DataArray(
            randn((nt, nr, nz)),
            dims=("t", "r", "z"),
            coords={
                "t": np.arange(nt),
                "r": np.arange(nr),
                "z": np.linspace(0, 1, nz),
            },
        )
        self.new_z = xr.DataArray(np.linspace(0.1, 0.9, nt), dims="t")

    @parameterize({"method": ["linear", "cubic"]})
    def time_interpolation_vectorized(self, method):
        self.da.interp(z=self.new_z, method=method).compute()


class InterpolationVectorizedDask(InterpolationVectorized):
    def setup(self, *args, **kwargs):
        requires_dask()
        super().setup(**kwargs)
        self.da = self.da.chunk({"t": 25, "r": 25})


class InterpolationDaskChunkedCoords:
    # scipy casts integer data to float, which copies the whole block unless
    # it is localized to the range of the (chunked) destination coordinates
    def setup(self, *args: Any, **kwargs: Any) -> None:
        requires_dask()
        import dask.array

        ny, nx = 16, 2_000_000
        chunks = (8, nx)  # 122 MiB blocks, generated lazily
        self.data = {
            "int": dask.array.random.randint(0, 100, (ny, nx), chunks=chunks),
            "float": dask.array.random.random((ny, nx), chunks=chunks),
        }
        self.x = np.arange(nx)
        self.new_x = xr.DataArray(
            dask.array.linspace(1000, 1500, 400, chunks=100), dims="z"
        )

    @parameterize({"method": ["linear", "nearest"], "dtype": ["int", "float"]})
    def peakmem_interpolation_chunked_coords(
        self, method: InterpOptions, dtype: str
    ) -> None:
        import dask

        da = xr.DataArray(self.data[dtype], dims=("y", "x"), coords={"x": self.x})
        with dask.config.set(scheduler="synchronous"):
            da.interp(x=self.new_x, method=method).compute()
