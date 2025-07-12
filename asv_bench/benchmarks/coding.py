import numpy as np

import xarray as xr

from . import parameterized


@parameterized(["calendar"], [("standard", "noleap")])
class EncodeCFDatetime:
    def setup(self, calendar):
        self.units = "days since 2000-01-01"
        self.dtype = np.dtype("int64")
        self.times = xr.date_range(
            "2000", freq="D", periods=10000, calendar=calendar
        ).values

    def time_encode_cf_datetime(self, calendar):
        xr.coding.times.encode_cf_datetime(self.times, self.units, calendar, self.dtype)
