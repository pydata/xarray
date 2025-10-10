import xarray as xr
from xarray.core.datatree import DataTree


class Datatree:
    def setup(self):
        run1 = DataTree.from_dict({"run1": xr.Dataset({"a": 1})})
        self.d_few = {"run1": run1}
        self.d_many = {f"run{i}": xr.Dataset({"a": 1}) for i in range(100)}

    def time_from_dict_few(self):
        DataTree.from_dict(self.d_few)

    def time_from_dict_many(self):
        DataTree.from_dict(self.d_many)
