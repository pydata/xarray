import xarray as xr


class Creation:
    def setup(self):
        # Everybody is lazy loading these days
        # so lets force modules to get instantiated here, instead of
        # in the benchmark
        dummy_dataset = xr.Dataset()
        dummy_dataset["a"] = 1
        dummy_dataset["b"] = 1

        self.dataset = xr.Dataset()

    def time_dataset_creation(self):
        dataset = self.dataset
        for i in range(100):
            dataset[f"var{i}"] = i
