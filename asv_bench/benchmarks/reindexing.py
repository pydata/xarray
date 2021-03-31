import numpy as np

import xarray as xr

from . import requires_dask


class Reindex:
    def setup(self):
        data = np.random.RandomState(0).randn(1000, 100, 100)
        self.ds = xr.Dataset(
            {"temperature": (("time", "x", "y"), data)},
            coords={"time": np.arange(1000), "x": np.arange(100), "y": np.arange(100)},
        )

    def time_1d_coarse(self):
        self.ds.reindex(time=np.arange(0, 1000, 5)).load()

    def time_1d_fine_all_found(self):
        self.ds.reindex(time=np.arange(0, 1000, 0.5), method="nearest").load()

    def time_1d_fine_some_missing(self):
        self.ds.reindex(
            time=np.arange(0, 1000, 0.5), method="nearest", tolerance=0.1
        ).load()

    def time_2d_coarse(self):
        self.ds.reindex(x=np.arange(0, 100, 2), y=np.arange(0, 100, 2)).load()

    def time_2d_fine_all_found(self):
        self.ds.reindex(
            x=np.arange(0, 100, 0.5), y=np.arange(0, 100, 0.5), method="nearest"
        ).load()

    def time_2d_fine_some_missing(self):
        self.ds.reindex(
            x=np.arange(0, 100, 0.5),
            y=np.arange(0, 100, 0.5),
            method="nearest",
            tolerance=0.1,
        ).load()


class ReindexDask(Reindex):
    def setup(self):
        requires_dask()
        super().setup()
        self.ds = self.ds.chunk({"time": 100})
