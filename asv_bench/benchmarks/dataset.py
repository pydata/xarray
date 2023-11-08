import numpy as np

from xarray import Dataset

from . import requires_dask


class DatasetBinaryOp:
    def setup(self):
        self.ds = Dataset(
            {
                "a": (("x", "y"), np.ones((300, 400))),
                "b": (("x", "y"), np.ones((300, 400))),
            }
        )
        self.mean = self.ds.mean()
        self.std = self.ds.std()

    def time_normalize(self):
        (self.ds - self.mean) / self.std


class DatasetChunk:
    def setup(self):
        requires_dask()
        self.ds = Dataset()
        array = np.ones(1000)
        for i in range(250):
            self.ds[f"var{i}"] = ("x", array)

    def time_chunk(self):
        self.ds.chunk(x=(1,) * 1000)
