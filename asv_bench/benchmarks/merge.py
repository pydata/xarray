import numpy as np

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

        d = {"set_2_{i}": i for i in range(existing_elements)}
        self.dataset2 = xr.merge([d])

    def time_variable_insertion(self, existing_elements):
        dataset = self.dataset
        dataset["new_var"] = 0

    def time_merge_two_datasets(self, existing_elements):
        xr.merge([self.dataset, self.dataset2])


class DatasetCreation:
    param_names = ["strategy", "count"]
    params = [
        ["dict_of_DataArrays", "dict_of_Variables", "dict_of_Tuples"],
        [0, 1, 10, 100, 1000],
    ]

    def setup(self, strategy, count):
        self.dataset_value = np.array(["0", "b"], dtype=str)
        self.dataset_coords = dict(time=np.array([0, 1]))

    def time_dataset_creation(self, strategy, count):
        # The idea here is to time how long it takes to go from numpy
        # and python data types, to a full dataset
        # See discussion
        # https://github.com/pydata/xarray/issues/7224#issuecomment-1292216344
        if strategy == "dict_of_DataArrays":
            data_vars = {
                f"long_variable_name_{i}": xr.DataArray(
                    data=self.dataset_value, dims=("time")
                )
                for i in range(count)
            }
        elif strategy == "dict_of_Variables":
            data_vars = {
                f"long_variable_name_{i}": xr.Variable("time", self.dataset_value)
                for i in range(count)
            }
        elif strategy == "dict_of_Tuples":
            data_vars = {
                f"long_variable_name_{i}": ("time", self.dataset_value)
                for i in range(count)
            }

        xr.Dataset(data_vars=data_vars, coords=self.dataset_coords)
