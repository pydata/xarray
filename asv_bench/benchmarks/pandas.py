import numpy as np
import pandas as pd

import xarray as xr

from . import parameterized, requires_dask


class MultiIndexSeries:
    def setup(self, dtype, subset):
        data = np.random.rand(100000).astype(dtype)
        index = pd.MultiIndex.from_product(
            [
                list("abcdefhijk"),
                list("abcdefhijk"),
                pd.date_range(start="2000-01-01", periods=1000, freq="D"),
            ]
        )
        series = pd.Series(data, index)
        if subset:
            series = series[::3]
        self.series = series

    @parameterized(["dtype", "subset"], ([int, float], [True, False]))
    def time_from_series(self, dtype, subset):
        xr.DataArray.from_series(self.series)


class ToDataFrame:
    def setup(self, *args, **kwargs):
        xp = kwargs.get("xp", np)
        nvars = kwargs.get("nvars", 1)
        random_kws = kwargs.get("random_kws", {})
        method = kwargs.get("method", "to_dataframe")

        dim1 = 10_000
        dim2 = 10_000

        var = xr.Variable(
            dims=("dim1", "dim2"), data=xp.random.random((dim1, dim2), **random_kws)
        )
        data_vars = {f"long_name_{v}": (("dim1", "dim2"), var) for v in range(nvars)}

        ds = xr.Dataset(
            data_vars, coords={"dim1": np.arange(0, dim1), "dim2": np.arange(0, dim2)}
        )
        self.to_frame = getattr(ds, method)

    def time_to_dataframe(self):
        self.to_frame()

    def peakmem_to_dataframe(self):
        self.to_frame()


class ToDataFrameDask(ToDataFrame):
    def setup(self, *args, **kwargs):
        requires_dask()

        import dask.array as da

        super().setup(
            xp=da, random_kws=dict(chunks=5000), method="to_dask_dataframe", nvars=500
        )
