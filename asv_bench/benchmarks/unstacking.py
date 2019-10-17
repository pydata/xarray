import numpy as np

import xarray as xr

from . import requires_dask


class Unstacking:
    def setup(self):
        data = np.random.RandomState(0).randn(1, 1000, 500)
        self.ds = xr.DataArray(data).stack(flat_dim=["dim_1", "dim_2"])

    def time_unstack_fast(self):
        self.ds.unstack("flat_dim")

    def time_unstack_slow(self):
        self.ds[:, ::-1].unstack("flat_dim")


class UnstackingDask(Unstacking):
    def setup(self, *args, **kwargs):
        requires_dask()
        super().setup(**kwargs)
        self.ds = self.ds.chunk({"flat_dim": 50})
