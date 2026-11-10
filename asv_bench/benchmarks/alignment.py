import numpy as np
from asv_runner.benchmarks.mark import parameterize

import xarray as xr

from . import requires_dask

ntime = 365 * 30
nx = 50
ny = 50

rng = np.random.default_rng(0)


class Align:
    def setup(self, *args, **kwargs):
        data = rng.standard_normal((ntime, nx, ny))
        self.ds = xr.Dataset(
            {"temperature": (("time", "x", "y"), data)},
            coords={
                "time": xr.date_range("2000", periods=ntime),
                "x": np.arange(nx),
                "y": np.arange(ny),
            },
        )
        self.year = self.ds.time.dt.year
        self.idx = np.unique(rng.integers(low=0, high=ntime, size=ntime // 2))
        self.year_subset = self.year.isel(time=self.idx)

    @parameterize({"join": ("outer", "inner", "left", "right", "exact", "override")})
    def time_already_aligned(self, join):
        xr.align(self.ds, self.year, join=join)

    @parameterize({"join": ("outer", "inner", "left", "right")})
    def time_not_aligned(self, join):
        xr.align(self.ds, self.year[-100:], join=join)

    @parameterize({"join": ("outer", "inner", "left", "right")})
    def time_not_aligned_random_integers(self, join):
        xr.align(self.ds, self.year_subset, join=join)


class AlignCFTime(Align):
    def setup(self, *args, **kwargs):
        super().setup()
        self.ds["time"] = xr.date_range("2000", periods=ntime, calendar="noleap")
        self.year = self.ds.time.dt.year
        self.year_subset = self.year.isel(time=self.idx)


class AlignDask(Align):
    def setup(self, *args, **kwargs):
        requires_dask()
        super().setup()
        self.ds = self.ds.chunk({"time": 100})
