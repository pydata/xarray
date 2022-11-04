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

        d = {f"set_2_{i}": i for i in range(existing_elements)}
        self.dataset2 = xr.merge([d])

    def time_variable_insertion(self, existing_elements):
        dataset = self.dataset
        dataset["new_var"] = 0

    def time_merge_two_datasets(self, existing_elements):
        xr.merge([self.dataset, self.dataset2])


class DatasetCreation:
    # The idea here is to time how long it takes to go from numpy
    # and python data types, to a full dataset
    # See discussion
    # https://github.com/pydata/xarray/issues/7224#issuecomment-1292216344
    param_names = ["strategy", "count"]
    params = [
        ["dict_of_DataArrays", "dict_of_Variables", "dict_of_Tuples"],
        [0, 1, 10, 100, 1000],
    ]

    def setup(self, strategy, count):
        data = np.array(["0", "b"], dtype=str)
        self.dataset_coords = dict(time=np.array([0, 1]))
        self.dataset_attrs = dict(description="Test data")
        attrs = dict(units="Celcius")
        if strategy == "dict_of_DataArrays":

            def create_data_vars():
                return {
                    f"long_variable_name_{i}": xr.DataArray(
                        data=data, dims=("time"), attrs=attrs
                    )
                    for i in range(count)
                }

        elif strategy == "dict_of_Variables":

            def create_data_vars():
                return {
                    f"long_variable_name_{i}": xr.Variable("time", data, attrs=attrs)
                    for i in range(count)
                }

        elif strategy == "dict_of_Tuples":

            def create_data_vars():
                return {
                    f"long_variable_name_{i}": ("time", data, attrs)
                    for i in range(count)
                }

        self.create_data_vars = create_data_vars

    def time_dataset_creation(self, strategy, count):
        data_vars = self.create_data_vars()
        xr.Dataset(
            data_vars=data_vars, coords=self.dataset_coords, attrs=self.dataset_attrs
        )
