import xarray as xr
from xarray.core.datatree import DataTree


class Datatree:
    def setup(self):
        run1 = DataTree.from_dict({"run1": xr.Dataset({"a": 1})})
        self.d = {"run1": run1}

    def time_from_dict(self):
        DataTree.from_dict(self.d)
