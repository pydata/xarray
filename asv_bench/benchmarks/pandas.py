import numpy as np
import pandas as pd

from . import parameterized


class MultiIndexSeries:
    def setup(self, dtype, subset):
        data = np.random.rand(100000).astype(dtype)
        index = pd.MultiIndex.from_product(
            [
                list("abcdefhijk"),
                list("abcdefhijk"),
                pd.date_range(start="2000-01-01", periods=1000, freq="B"),
            ]
        )
        series = pd.Series(data, index)
        if subset:
            series = series[::3]
        self.series = series

    @parameterized(["dtype", "subset"], ([int, float], [True, False]))
    def time_to_xarray(self, dtype, subset):
        self.series.to_xarray()
