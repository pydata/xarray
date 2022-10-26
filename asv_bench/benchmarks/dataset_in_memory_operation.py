import xarray as xr


class DatasetInMemoryOperations:
    params = [0, 10, 100, 1000]

    def setup(self, elements):
        self.datasets = {}
        # Dictionary insertion is fast(er) than xarray.Dataser insertion
        d = {}
        for i in range(elements):
            d[f"var{i}"] = i
        self.dataset = xr.merge([d])

    def time_dataset_insertion(self, elements):
        dataset = self.dataset
        for i in range(5):
            dataset[f"new_var{i}"] = i
