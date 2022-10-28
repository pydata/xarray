import xarray as xr


class DatasetAddVariable:
    param_names = ["existing_elements"]
    params = [[0, 10, 100, 1000]]

    def setup(self, existing_elements):
        self.datasets = {}
        # Dictionary insertion is fast(er) than xarray.Dataset insertion
        d = {}
        for i in range(existing_elements):
            d[f"var{i}"] = i
        self.dataset = xr.merge([d])

    def time_variable_insertion(self, existin_elements):
        dataset = self.dataset
        dataset["new_var"] = 0
