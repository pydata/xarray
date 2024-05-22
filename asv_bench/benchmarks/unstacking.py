import numpy as np
import pandas as pd

import xarray as xr

from . import requires_dask, requires_sparse


class Unstacking:
    def setup(self):
        data = np.random.RandomState(0).randn(250, 500)
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
        self.da_full = self.da_full.chunk({"flat_dim": 25})


class UnstackingSparse(Unstacking):
    def setup(self, *args, **kwargs):
        requires_sparse()

        import sparse

        data = sparse.random((500, 1000), random_state=0, fill_value=0)
        self.da_full = xr.DataArray(data, dims=list("ab")).stack(flat_dim=[...])
        self.da_missing = self.da_full[:-1]

        mindex = pd.MultiIndex.from_arrays([np.arange(100), np.arange(100)])
        self.da_eye_2d = xr.DataArray(np.ones((100,)), dims="z", coords={"z": mindex})
        self.da_eye_3d = xr.DataArray(
            np.ones((100, 50)),
            dims=("z", "foo"),
            coords={"z": mindex, "foo": np.arange(50)},
        )

    def time_unstack_to_sparse_2d(self):
        self.da_eye_2d.unstack(sparse=True)

    def time_unstack_to_sparse_3d(self):
        self.da_eye_3d.unstack(sparse=True)

    def peakmem_unstack_to_sparse_2d(self):
        self.da_eye_2d.unstack(sparse=True)

    def peakmem_unstack_to_sparse_3d(self):
        self.da_eye_3d.unstack(sparse=True)

    def time_unstack_pandas_slow(self):
        pass
