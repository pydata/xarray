import os

import numpy as np
import pandas as pd

import xarray as xr

from . import parameterized, randint, randn, requires_dask

nx = 2000
ny = 1000
nt = 500

basic_indexes = {
    "1scalar": {"x": 0},
    "1slice": {"x": slice(0, 3)},
    "1slice-1scalar": {"x": 0, "y": slice(None, None, 3)},
    "2slicess-1scalar": {"x": slice(3, -3, 3), "y": 1, "t": slice(None, -3, 3)},
}

basic_assignment_values = {
    "1scalar": 0,
    "1slice": xr.DataArray(randn((3, ny), frac_nan=0.1), dims=["x", "y"]),
    "1slice-1scalar": xr.DataArray(randn(int(ny / 3) + 1, frac_nan=0.1), dims=["y"]),
    "2slicess-1scalar": xr.DataArray(
        randn(np.empty(nx)[slice(3, -3, 3)].size, frac_nan=0.1), dims=["x"]
    ),
}

outer_indexes = {
    "1d": {"x": randint(0, nx, 400)},
    "2d": {"x": randint(0, nx, 500), "y": randint(0, ny, 400)},
    "2d-1scalar": {"x": randint(0, nx, 100), "y": 1, "t": randint(0, nt, 400)},
}

outer_assignment_values = {
    "1d": xr.DataArray(randn((400, ny), frac_nan=0.1), dims=["x", "y"]),
    "2d": xr.DataArray(randn((500, 400), frac_nan=0.1), dims=["x", "y"]),
    "2d-1scalar": xr.DataArray(randn(100, frac_nan=0.1), dims=["x"]),
}

vectorized_indexes = {
    "1-1d": {"x": xr.DataArray(randint(0, nx, 400), dims="a")},
    "2-1d": {
        "x": xr.DataArray(randint(0, nx, 400), dims="a"),
        "y": xr.DataArray(randint(0, ny, 400), dims="a"),
    },
    "3-2d": {
        "x": xr.DataArray(randint(0, nx, 400).reshape(4, 100), dims=["a", "b"]),
        "y": xr.DataArray(randint(0, ny, 400).reshape(4, 100), dims=["a", "b"]),
        "t": xr.DataArray(randint(0, nt, 400).reshape(4, 100), dims=["a", "b"]),
    },
}

vectorized_assignment_values = {
    "1-1d": xr.DataArray(randn((400, ny)), dims=["a", "y"], coords={"a": randn(400)}),
    "2-1d": xr.DataArray(randn(400), dims=["a"], coords={"a": randn(400)}),
    "3-2d": xr.DataArray(
        randn((4, 100)), dims=["a", "b"], coords={"a": randn(4), "b": randn(100)}
    ),
}


class Base:
    def setup(self, key):
        self.ds = xr.Dataset(
            {
                "var1": (("x", "y"), randn((nx, ny), frac_nan=0.1)),
                "var2": (("x", "t"), randn((nx, nt))),
                "var3": (("t",), randn(nt)),
            },
            coords={
                "x": np.arange(nx),
                "y": np.linspace(0, 1, ny),
                "t": pd.date_range("1970-01-01", periods=nt, freq="D"),
                "x_coords": ("x", np.linspace(1.1, 2.1, nx)),
            },
        )
        # Benchmark how indexing is slowed down by adding many scalar variable
        # to the dataset
        # https://github.com/pydata/xarray/pull/9003
        self.ds_large = self.ds.merge({f"extra_var{i}": i for i in range(400)})


class Indexing(Base):
    @parameterized(["key"], [list(basic_indexes.keys())])
    def time_indexing_basic(self, key):
        self.ds.isel(**basic_indexes[key]).load()

    @parameterized(["key"], [list(outer_indexes.keys())])
    def time_indexing_outer(self, key):
        self.ds.isel(**outer_indexes[key]).load()

    @parameterized(["key"], [list(vectorized_indexes.keys())])
    def time_indexing_vectorized(self, key):
        self.ds.isel(**vectorized_indexes[key]).load()

    @parameterized(["key"], [list(basic_indexes.keys())])
    def time_indexing_basic_ds_large(self, key):
        # https://github.com/pydata/xarray/pull/9003
        self.ds_large.isel(**basic_indexes[key]).load()


class Assignment(Base):
    @parameterized(["key"], [list(basic_indexes.keys())])
    def time_assignment_basic(self, key):
        ind = basic_indexes[key]
        val = basic_assignment_values[key]
        self.ds["var1"][ind.get("x", slice(None)), ind.get("y", slice(None))] = val

    @parameterized(["key"], [list(outer_indexes.keys())])
    def time_assignment_outer(self, key):
        ind = outer_indexes[key]
        val = outer_assignment_values[key]
        self.ds["var1"][ind.get("x", slice(None)), ind.get("y", slice(None))] = val

    @parameterized(["key"], [list(vectorized_indexes.keys())])
    def time_assignment_vectorized(self, key):
        ind = vectorized_indexes[key]
        val = vectorized_assignment_values[key]
        self.ds["var1"][ind.get("x", slice(None)), ind.get("y", slice(None))] = val


class IndexingDask(Indexing):
    def setup(self, key):
        requires_dask()
        super().setup(key)
        self.ds = self.ds.chunk({"x": 100, "y": 50, "t": 50})


class BooleanIndexing:
    # https://github.com/pydata/xarray/issues/2227
    def setup(self):
        self.ds = xr.Dataset(
            {"a": ("time", np.arange(10_000_000))},
            coords={"time": np.arange(10_000_000)},
        )
        self.time_filter = self.ds.time > 50_000

    def time_indexing(self):
        self.ds.isel(time=self.time_filter)


class HugeAxisSmallSliceIndexing:
    # https://github.com/pydata/xarray/pull/4560
    def setup(self):
        self.filepath = "test_indexing_huge_axis_small_slice.nc"
        if not os.path.isfile(self.filepath):
            xr.Dataset(
                {"a": ("x", np.arange(10_000_000))},
                coords={"x": np.arange(10_000_000)},
            ).to_netcdf(self.filepath, format="NETCDF4")

        self.ds = xr.open_dataset(self.filepath)

    def time_indexing(self):
        self.ds.isel(x=slice(100))

    def cleanup(self):
        self.ds.close()


class AssignmentOptimized:
    # https://github.com/pydata/xarray/pull/7382
    def setup(self):
        self.ds = xr.Dataset(coords={"x": np.arange(500_000)})
        self.da = xr.DataArray(np.arange(500_000), dims="x")

    def time_assign_no_reindex(self):
        # assign with non-indexed DataArray of same dimension size
        self.ds.assign(foo=self.da)

    def time_assign_identical_indexes(self):
        # fastpath index comparison (same index object)
        self.ds.assign(foo=self.ds.x)
