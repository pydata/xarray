import pandas as pd

import xarray as xr


class ReprMultiIndex:
    def setup(self):
        index = pd.MultiIndex.from_product(
            [range(10000), range(10000)], names=("level_0", "level_1")
        )
        series = pd.Series(range(100000000), index=index)
        self.da = xr.DataArray(series)

    def time_repr(self):
        repr(self.da)

    def time_repr_html(self):
        self.da._repr_html_()
