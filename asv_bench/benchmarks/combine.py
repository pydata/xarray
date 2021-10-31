import numpy as np

import xarray as xr


class Combine:
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
