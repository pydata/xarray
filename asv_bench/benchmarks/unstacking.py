import numpy as np

import xarray as xr

from . import requires_dask


class Unstacking:
    def setup(self):
        data = np.random.RandomState(0).randn(500, 1000)
        self.da_full = xr.DataArray(data, dims=list("ab")).stack(flat_dim=[...])
        self.da_missing = self.da_full[:-1]
        self.df_missing = self.da_missing.to_pandas()

    def time_unstack_fast(self):
        self.da_full.unstack("flat_dim")

    def time_unstack_slow(self):
        self.da_missing.unstack("flat_dim")

    def time_unstack_pandas_slow(self):
        self.df_missing.unstack()


class UnstackingDask(Unstacking):
    def setup(self, *args, **kwargs):
        requires_dask()
        super().setup(**kwargs)
        self.da_full = self.da_full.chunk({"flat_dim": 50})
