import numpy as np

import xarray as xr

from . import requires_dask


class Combine1d:
    """Benchmark concatenating and merging large datasets"""

    def setup(self) -> None:
        """Create 2 datasets with two different variables"""

        t_size = 8000
        t = np.arange(t_size)
        data = np.random.randn(t_size)

        self.dsA0 = xr.Dataset({"A": xr.DataArray(data, coords={"T": t}, dims=("T"))})
        self.dsA1 = xr.Dataset(
            {"A": xr.DataArray(data, coords={"T": t + t_size}, dims=("T"))}
        )

    def time_combine_by_coords(self) -> None:
        """Also has to load and arrange t coordinate"""
        datasets = [self.dsA0, self.dsA1]

        xr.combine_by_coords(datasets)


class Combine1dDask(Combine1d):
    """Benchmark concatenating and merging large datasets"""

    def setup(self) -> None:
        """Create 2 datasets with two different variables"""
        requires_dask()

        t_size = 8000
        t = np.arange(t_size)
        var = xr.Variable(dims=("T",), data=np.random.randn(t_size)).chunk()

        data_vars = {f"long_name_{v}": ("T", var) for v in range(500)}

        self.dsA0 = xr.Dataset(data_vars, coords={"T": t})
        self.dsA1 = xr.Dataset(data_vars, coords={"T": t + t_size})


class Combine3d:
    """Benchmark concatenating and merging large datasets"""

    def setup(self):
        """Create 4 datasets with two different variables"""

        t_size, x_size, y_size = 50, 450, 400
        t = np.arange(t_size)
        data = np.random.randn(t_size, x_size, y_size)

        self.dsA0 = xr.Dataset(
            {"A": xr.DataArray(data, coords={"T": t}, dims=("T", "X", "Y"))}
        )
        self.dsA1 = xr.Dataset(
            {"A": xr.DataArray(data, coords={"T": t + t_size}, dims=("T", "X", "Y"))}
        )
        self.dsB0 = xr.Dataset(
            {"B": xr.DataArray(data, coords={"T": t}, dims=("T", "X", "Y"))}
        )
        self.dsB1 = xr.Dataset(
            {"B": xr.DataArray(data, coords={"T": t + t_size}, dims=("T", "X", "Y"))}
        )

    def time_combine_nested(self):
        datasets = [[self.dsA0, self.dsA1], [self.dsB0, self.dsB1]]

        xr.combine_nested(datasets, concat_dim=[None, "T"])

    def time_combine_by_coords(self):
        """Also has to load and arrange t coordinate"""
        datasets = [self.dsA0, self.dsA1, self.dsB0, self.dsB1]

        xr.combine_by_coords(datasets)
