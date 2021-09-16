import pandas as pd

import xarray as xr

from . import randn, parameterized, requires_dask

try:
    import dask  # noqa: F401
except ImportError:
    pass


def make_bench_data(shape, frac_nan, chunks):
    vals = randn(shape, frac_nan)
    coords = {"time": pd.date_range("2000-01-01", freq="D", periods=shape[0])}
    da = xr.DataArray(vals, dims=("time", "x", "y"), coords=coords)

    if chunks is not None:
        da = da.chunk(chunks)

    return da


class DataArrayMissing:
    def setup(self, shape, chunks, method, limitt):
        requires_dask()
        self.da = make_bench_data(shape, 0.1, chunks)

    @parameterized(
        ["shape", "chunks", "method", "limit"],
        (
            [(100, 25, 25)],
            [None, {"x": 25, "y": 25}],
            ["linear", "spline", "quadratic", "cubic"],
            [None, 3],
        ),
    )
    def time_interpolate_na(self, shape, chunks, method, limit):
        actual = self.da.interpolate_na(dim="time", method="linear", limit=limit)

        if chunks is not None:
            actual = actual.compute()

    @parameterized(
        ["shape", "chunks", "limit"],
        (
            [(100, 25, 25)],
            [None, {"x": 25, "y": 25}],
            [None, 3],
        ),
    )
    def time_ffill(self, shape, chunks, limit):
        actual = self.da.ffill(dim="time", limit=limit)

        if chunks is not None:
            actual = actual.compute()

    @parameterized(
        ["shape", "chunks", "limit"],
        (
            [(100, 25, 25)],
            [None, {"x": 25, "y": 25}],
            [None, 3],
        ),
    )
    def time_bfill(self, shape, chunks, limit):
        actual = self.da.ffill(dim="time", limit=limit)

        if chunks is not None:
            actual = actual.compute()
