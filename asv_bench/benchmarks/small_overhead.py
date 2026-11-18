import numpy as np

import xarray as xr


class SmallOverhead:
    """Operations on small in-memory arrays.

    The computation itself is cheap for arrays of this size, so these
    benchmarks measure the per-call overhead of xarray.
    """

    def __init__(self):
        self.nx = 22
        self.ny = 4
        self.nvars = 4

    def setup(self):
        rng = np.random.default_rng(0)
        coords = {
            "x": np.arange(self.nx),
            "y": np.arange(self.ny),
            "t": ("x", np.linspace(0, 1, self.nx)),
        }
        self.arrays = {
            f"v{i}": xr.DataArray(
                rng.standard_normal((self.nx, self.ny)),
                dims=("x", "y"),
                coords=coords,
            )
            for i in range(self.nvars)
        }
        self.ds = xr.Dataset(self.arrays)
        self.da = self.ds["v0"]
        self.da2 = self.ds["v1"]
        self.mask = self.da > 0
        self.var = self.da.variable

    def time_dataset_from_dataarrays(self):
        xr.Dataset(self.arrays)

    def time_dataset_getitem(self):
        self.ds["v0"]

    def time_dataset_isel_int(self):
        self.ds.isel(x=3)

    def time_dataarray_isel_slice(self):
        self.da.isel(x=slice(5, 15))

    def time_dataarray_sel(self):
        self.da.sel(x=3)

    def time_apply_ufunc(self):
        xr.apply_ufunc(np.abs, self.da)

    def time_apply_ufunc_core_dims(self):
        xr.apply_ufunc(np.sum, self.da, input_core_dims=[["y"]], kwargs={"axis": -1})

    def time_align_exact(self):
        xr.align(self.da, self.da2, join="exact")

    def time_align_inner(self):
        xr.align(self.da, self.da2, join="inner")

    def time_align_inner_no_copy(self):
        xr.align(self.da, self.da2, join="inner", copy=False)

    def time_broadcast(self):
        xr.broadcast(self.da, self.da2)

    def time_merge(self):
        xr.merge([self.da.rename("a"), self.da2.rename("b")])

    def time_dataarray_where(self):
        self.da.where(self.mask)

    def time_where(self):
        xr.where(self.mask, self.da, self.da2)

    def time_concat_new_dim(self):
        xr.concat([self.da, self.da2], dim="new")

    def time_dataarray_mean(self):
        self.da.mean("x")

    def time_dataarray_count(self):
        self.da.count("x")

    def time_dataset_mean(self):
        self.ds.mean("x")

    def time_dataarray_add_scalar(self):
        self.da + 1

    def time_dataarray_add_dataarray(self):
        self.da + self.da2

    def time_dataset_add_dataset(self):
        self.ds + self.ds

    def time_dataset_assign(self):
        self.ds.assign(new=self.da)

    def time_dataarray_assign_coords(self):
        self.da.assign_coords(c=1)

    def time_dataset_copy(self):
        self.ds.copy(deep=False)

    def time_dataarray_copy(self):
        self.da.copy(deep=False)

    def time_variable_copy(self):
        self.var.copy(deep=False)
