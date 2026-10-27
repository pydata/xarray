import numpy as np
from asv_runner.benchmarks.mark import parameterize

import xarray as xr

NTIME = 365 * 30


@parameterize({"calendar": ("standard", "noleap")})
class DateTimeAccessor:
    def setup(self, calendar):
        np.random.randn(NTIME)
        time = xr.date_range("2000", periods=30 * 365, calendar=calendar)
        data = np.ones((NTIME,))
        self.da = xr.DataArray(data, dims="time", coords={"time": time})

    def time_dayofyear(self, calendar):
        _ = self.da.time.dt.dayofyear

    def time_year(self, calendar):
        _ = self.da.time.dt.year

    def time_floor(self, calendar):
        _ = self.da.time.dt.floor("D")


NSTRINGS = 100_000


@parameterize({"dtype": ("U", "S", "O")})
class StringAccessor:
    def setup(self, dtype):
        rng = np.random.default_rng(0)
        chars = np.array(list("abcdefgh XYZ"))
        strings = rng.choice(chars, size=(NSTRINGS, 12)).view("U12").ravel()
        self.da = xr.DataArray(strings.astype(dtype), dims="x")

    def time_len(self, dtype):
        self.da.str.len()

    def time_isalpha(self, dtype):
        self.da.str.isalpha()

    def time_startswith(self, dtype):
        self.da.str.startswith("ab")

    def time_find(self, dtype):
        self.da.str.find("ab")

    def time_contains(self, dtype):
        self.da.str.contains("ab", regex=False)

    def time_strip(self, dtype):
        self.da.str.strip()

    def time_center(self, dtype):
        self.da.str.center(16)

    def time_replace(self, dtype):
        self.da.str.replace("ab", "-", regex=False)

    def time_repeat(self, dtype):
        self.da.str.repeat(2)
