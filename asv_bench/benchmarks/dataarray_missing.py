import pandas as pd
from asv_runner.benchmarks.mark import parameterize

import xarray as xr

from . import randn, requires_dask


def make_bench_data(shape, frac_nan, chunks):
    vals = randn(shape, frac_nan)
    coords = {"time": pd.date_range("2000-01-01", freq="D", periods=shape[0])}
    da = xr.DataArray(vals, dims=("time", "x", "y"), coords=coords)

    if chunks is not None:
        da = da.chunk(chunks)

    return da


class DataArrayMissingInterpolateNA:
    def setup(self, shape, chunks, limit):
        if chunks is not None:
            requires_dask()
        self.da = make_bench_data(shape, 0.1, chunks)

    @parameterize(
        {
            "shape": [(365, 75, 75)],
            "chunks": [None, {"x": 25, "y": 25}],
            "limit": [None, 3],
        }
    )
    def time_interpolate_na(self, shape, chunks, limit):
        actual = self.da.interpolate_na(dim="time", method="linear", limit=limit)

        if chunks is not None:
            actual = actual.compute()


class DataArrayMissingBottleneck:
    def setup(self, shape, chunks, limit):
        if chunks is not None:
            requires_dask()
        self.da = make_bench_data(shape, 0.1, chunks)

    @parameterize(
        {
            "shape": [(365, 75, 75)],
            "chunks": [None, {"x": 25, "y": 25}],
            "limit": [None, 3],
        }
    )
    def time_ffill(self, shape, chunks, limit):
        actual = self.da.ffill(dim="time", limit=limit)

        if chunks is not None:
            actual = actual.compute()

    @parameterize(
        {
            "shape": [(365, 75, 75)],
            "chunks": [None, {"x": 25, "y": 25}],
            "limit": [None, 3],
        }
    )
    def time_bfill(self, shape, chunks, limit):
        actual = self.da.bfill(dim="time", limit=limit)

        if chunks is not None:
            actual = actual.compute()
