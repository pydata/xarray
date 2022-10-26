import xarray as xr


class DatasetAddVariable:
    params = [0, 10, 100, 1000]

    def setup(self, elements):
        self.datasets = {}
        # Dictionary insertion is fast(er) than xarray.Dataset insertion
        d = {}
        for i in range(elements):
            d[f"var{i}"] = i
        self.dataset = xr.merge([d])

    def time_variable_insertion(self):
        dataset = self.dataset
        for i in range(5):
            dataset[f"new_var{i}"] = i
