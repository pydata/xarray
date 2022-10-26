import xarray as xr

from . import parameterized


class Creation:
    def setup(self, elements):
        self.datasets = {}
        # Dictionary insertion is fast(er) than xarray.Dataser insertion
        d = {}
        for i in range(elements):
            d[f"var{i}"] = i
        self.dataset = xr.merge([d])

    @parameterized(["elements"], [(0, 10, 100, 1000)])
    def time_dataset_creation(self, elements):
        dataset = self.dataset
        for i in range(5):
            dataset[f"new_var{i}"] = i
