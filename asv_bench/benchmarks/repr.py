import numpy as np
import pandas as pd

import xarray as xr


class Repr:
    def setup(self):
        a = np.arange(0, 100)
        data_vars = dict()
        for i in a:
            data_vars[f"long_variable_name_{i}"] = xr.DataArray(
                name=f"long_variable_name_{i}",
                data=np.arange(0, 20),
                dims=[f"long_coord_name_{i}_x"],
                coords={f"long_coord_name_{i}_x": np.arange(0, 20) * 2},
            )
        self.ds = xr.Dataset(data_vars)
        self.ds.attrs = {f"attr_{k}": 2 for k in a}

    def time_repr(self):
        repr(self.ds)

    def time_repr_html(self):
        self.ds._repr_html_()


class ReprDataTree:
    def setup(self):
        # construct a datatree with 500 nodes
        number_of_files = 20
        number_of_groups = 25
        tree_dict = {}
        for f in range(number_of_files):
            for g in range(number_of_groups):
                tree_dict[f"file_{f}/group_{g}"] = xr.Dataset({"g": f * g})

        self.dt = xr.DataTree.from_dict(tree_dict)

    def time_repr(self):
        repr(self.dt)

    def time_repr_html(self):
        self.dt._repr_html_()


class ReprMultiIndex:
    def setup(self):
        index = pd.MultiIndex.from_product(
            [range(1000), range(1000)], names=("level_0", "level_1")
        )
        series = pd.Series(range(1000 * 1000), index=index)
        self.da = xr.DataArray(series)

    def time_repr(self):
        repr(self.da)

    def time_repr_html(self):
        self.da._repr_html_()
