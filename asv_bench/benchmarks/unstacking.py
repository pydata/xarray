import numpy as np

import xarray as xr

from . import requires_dask


class Unstacking:
    def setup(self):
        # data = np.random.RandomState(0).randn(1, 1000, 500)
        data = np.random.RandomState(0).randn(1000, 500)
        self.da = xr.DataArray(data, dims=list("ab")).stack(c=[...])
        self.da_slow = self.da[::-1]
        self.df = self.da.to_pandas()
        self.df_slow = self.da_slow.to_pandas()

    def time_unstack_fast(self):
        self.da.unstack("c")

    def time_unstack_slow(self):
        self.da_slow.unstack("c")

    def time_unstack_pandas_fast(self):
        # As comparison
        self.df.unstack()

    def time_unstack_pandas_slow(self):
        # As comparison
        self.df_slow.unstack()


class UnstackingDask(Unstacking):
    def setup(self, *args, **kwargs):
        requires_dask()
        super().setup(**kwargs)
        self.da = self.da.chunk({"c": 50})
