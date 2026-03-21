import numpy as np

import xarray as xr

from . import parameterized

NTIME = 365 * 30


@parameterized(["calendar"], [("standard", "noleap")])
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
