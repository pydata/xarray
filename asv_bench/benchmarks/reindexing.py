import numpy as np

import xarray as xr

from . import requires_dask

ntime = 500
nx = 50
ny = 50


class Reindex:
    def setup(self):
        data = np.random.RandomState(0).randn(ntime, nx, ny)
        self.ds = xr.Dataset(
            {"temperature": (("time", "x", "y"), data)},
            coords={"time": np.arange(ntime), "x": np.arange(nx), "y": np.arange(ny)},
        )

    def time_1d_coarse(self):
        self.ds.reindex(time=np.arange(0, ntime, 5)).load()

    def time_1d_fine_all_found(self):
        self.ds.reindex(time=np.arange(0, ntime, 0.5), method="nearest").load()

    def time_1d_fine_some_missing(self):
        self.ds.reindex(
            time=np.arange(0, ntime, 0.5), method="nearest", tolerance=0.1
        ).load()

    def time_2d_coarse(self):
        self.ds.reindex(x=np.arange(0, nx, 2), y=np.arange(0, ny, 2)).load()

    def time_2d_fine_all_found(self):
        self.ds.reindex(
            x=np.arange(0, nx, 0.5), y=np.arange(0, ny, 0.5), method="nearest"
        ).load()

    def time_2d_fine_some_missing(self):
        self.ds.reindex(
            x=np.arange(0, nx, 0.5),
            y=np.arange(0, ny, 0.5),
            method="nearest",
            tolerance=0.1,
        ).load()


class ReindexDask(Reindex):
    def setup(self):
        requires_dask()
        super().setup()
        self.ds = self.ds.chunk({"time": 100})
