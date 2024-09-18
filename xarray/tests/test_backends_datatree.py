from __future__ import annotations

import re
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

import xarray as xr
from xarray.backends.api import open_datatree, open_groups
from xarray.core.datatree import DataTree
from xarray.testing import assert_equal, assert_identical
from xarray.tests import (
    requires_h5netcdf,
    requires_netCDF4,
    requires_zarr,
)

if TYPE_CHECKING:
    from xarray.core.datatree_io import T_DataTreeNetcdfEngine

try:
    import netCDF4 as nc4
except ImportError:
    pass


@pytest.fixture(scope="module")
def unaligned_datatree_nc(tmp_path_factory):
    """Creates a test netCDF4 file with the following unaligned structure, writes it to a /tmp directory
    and returns the file path of the netCDF4 file.

    Group: /
    │   Dimensions:        (lat: 1, lon: 2)
    │   Dimensions without coordinates: lat, lon
    │   Data variables:
    │       root_variable  (lat, lon) float64 16B ...
    └── Group: /Group1
        │   Dimensions:      (lat: 1, lon: 2)
        │   Dimensions without coordinates: lat, lon
        │   Data variables:
        │       group_1_var  (lat, lon) float64 16B ...
        └── Group: /Group1/subgroup1
                Dimensions:        (lat: 2, lon: 2)
                Dimensions without coordinates: lat, lon
                Data variables:
                    subgroup1_var  (lat, lon) float64 32B ...
    """
    filepath = tmp_path_factory.mktemp("data") / "unaligned_subgroups.nc"
    with nc4.Dataset(filepath, "w", format="NETCDF4") as root_group:
        group_1 = root_group.createGroup("/Group1")
        subgroup_1 = group_1.createGroup("/subgroup1")

        root_group.createDimension("lat", 1)
        root_group.createDimension("lon", 2)
        root_group.createVariable("root_variable", np.float64, ("lat", "lon"))

        group_1_var = group_1.createVariable("group_1_var", np.float64, ("lat", "lon"))
        group_1_var[:] = np.array([[0.1, 0.2]])
        group_1_var.units = "K"
        group_1_var.long_name = "air_temperature"

        subgroup_1.createDimension("lat", 2)

        subgroup1_var = subgroup_1.createVariable(
            "subgroup1_var", np.float64, ("lat", "lon")
        )
        subgroup1_var[:] = np.array([[0.1, 0.2]])

    yield filepath


@pytest.fixture(scope="module")
def unaligned_datatree_zarr(tmp_path_factory):
    """Creates a zarr store with the following unaligned group hierarchy:
    Group: /
    │   Dimensions:  (y: 3, x: 2)
    │   Dimensions without coordinates: y, x
    │   Data variables:
    │       a        (y) int64 24B ...
    │       set0     (x) int64 16B ...
    └── Group: /Group1
    │   │   Dimensions:  ()
    │   │   Data variables:
    │   │       a        int64 8B ...
    │   │       b        int64 8B ...
    │   └── /Group1/subgroup1
    │           Dimensions:  ()
    │           Data variables:
    │               a        int64 8B ...
    │               b        int64 8B ...
    └── Group: /Group2
            Dimensions:  (y: 2, x: 2)
            Dimensions without coordinates: y, x
            Data variables:
                a        (y) int64 16B ...
                b        (x) float64 16B ...
    """
    filepath = tmp_path_factory.mktemp("data") / "unaligned_simple_datatree.zarr"
    root_data = xr.Dataset({"a": ("y", [6, 7, 8]), "set0": ("x", [9, 10])})
    set1_data = xr.Dataset({"a": 0, "b": 1})
    set2_data = xr.Dataset({"a": ("y", [2, 3]), "b": ("x", [0.1, 0.2])})
    root_data.to_zarr(filepath)
    set1_data.to_zarr(filepath, group="/Group1", mode="a")
    set2_data.to_zarr(filepath, group="/Group2", mode="a")
    set1_data.to_zarr(filepath, group="/Group1/subgroup1", mode="a")
    yield filepath


class DatatreeIOBase:
    engine: T_DataTreeNetcdfEngine | None = None

    def test_to_netcdf(self, tmpdir, simple_datatree):
        filepath = tmpdir / "test.nc"
        original_dt = simple_datatree
        original_dt.to_netcdf(filepath, engine=self.engine)

        roundtrip_dt = open_datatree(filepath, engine=self.engine)
        assert_equal(original_dt, roundtrip_dt)

    def test_to_netcdf_inherited_coords(self, tmpdir):
        filepath = tmpdir / "test.nc"
        original_dt = DataTree.from_dict(
            {
                "/": xr.Dataset({"a": (("x",), [1, 2])}, coords={"x": [3, 4]}),
                "/sub": xr.Dataset({"b": (("x",), [5, 6])}),
            }
        )
        original_dt.to_netcdf(filepath, engine=self.engine)

        roundtrip_dt = open_datatree(filepath, engine=self.engine)
        assert_equal(original_dt, roundtrip_dt)
        subtree = cast(DataTree, roundtrip_dt["/sub"])
        assert "x" not in subtree.to_dataset(inherited=False).coords

    def test_netcdf_encoding(self, tmpdir, simple_datatree):
        filepath = tmpdir / "test.nc"
        original_dt = simple_datatree

        # add compression
        comp = dict(zlib=True, complevel=9)
        enc = {"/set2": {var: comp for var in original_dt["/set2"].dataset.data_vars}}

        original_dt.to_netcdf(filepath, encoding=enc, engine=self.engine)
        roundtrip_dt = open_datatree(filepath, engine=self.engine)

        assert roundtrip_dt["/set2/a"].encoding["zlib"] == comp["zlib"]
        assert roundtrip_dt["/set2/a"].encoding["complevel"] == comp["complevel"]

        enc["/not/a/group"] = {"foo": "bar"}  # type: ignore[dict-item]
        with pytest.raises(ValueError, match="unexpected encoding group.*"):
            original_dt.to_netcdf(filepath, encoding=enc, engine=self.engine)


@requires_netCDF4
class TestNetCDF4DatatreeIO(DatatreeIOBase):
    engine: T_DataTreeNetcdfEngine | None = "netcdf4"

    def test_open_datatree(self, unaligned_datatree_nc) -> None:
        """Test if `open_datatree` fails to open a netCDF4 with an unaligned group hierarchy."""

        with pytest.raises(
            ValueError,
            match=(
                re.escape(
                    "group '/Group1/subgroup1' is not aligned with its parents:\nGroup:\n"
                )
                + ".*"
            ),
        ):
            open_datatree(unaligned_datatree_nc)

    def test_open_groups(self, unaligned_datatree_nc) -> None:
        """Test `open_groups` with a netCDF4 file with an unaligned group hierarchy."""
        unaligned_dict_of_datasets = open_groups(unaligned_datatree_nc)

        # Check that group names are keys in the dictionary of `xr.Datasets`
        assert "/" in unaligned_dict_of_datasets.keys()
        assert "/Group1" in unaligned_dict_of_datasets.keys()
        assert "/Group1/subgroup1" in unaligned_dict_of_datasets.keys()
        # Check that group name returns the correct datasets
        assert_identical(
            unaligned_dict_of_datasets["/"],
            xr.open_dataset(unaligned_datatree_nc, group="/"),
        )
        assert_identical(
            unaligned_dict_of_datasets["/Group1"],
            xr.open_dataset(unaligned_datatree_nc, group="Group1"),
        )
        assert_identical(
            unaligned_dict_of_datasets["/Group1/subgroup1"],
            xr.open_dataset(unaligned_datatree_nc, group="/Group1/subgroup1"),
        )

    def test_open_groups_to_dict(self, tmpdir) -> None:
        """Create an aligned netCDF4 with the following structure to test `open_groups`
        and `DataTree.from_dict`.
        Group: /
        │   Dimensions:        (lat: 1, lon: 2)
        │   Dimensions without coordinates: lat, lon
        │   Data variables:
        │       root_variable  (lat, lon) float64 16B ...
        └── Group: /Group1
            │   Dimensions:      (lat: 1, lon: 2)
            │   Dimensions without coordinates: lat, lon
            │   Data variables:
            │       group_1_var  (lat, lon) float64 16B ...
            └── Group: /Group1/subgroup1
                    Dimensions:        (lat: 1, lon: 2)
                    Dimensions without coordinates: lat, lon
                    Data variables:
                        subgroup1_var  (lat, lon) float64 16B ...
        """
        filepath = tmpdir + "/all_aligned_child_nodes.nc"
        with nc4.Dataset(filepath, "w", format="NETCDF4") as root_group:
            group_1 = root_group.createGroup("/Group1")
            subgroup_1 = group_1.createGroup("/subgroup1")

            root_group.createDimension("lat", 1)
            root_group.createDimension("lon", 2)
            root_group.createVariable("root_variable", np.float64, ("lat", "lon"))

            group_1_var = group_1.createVariable(
                "group_1_var", np.float64, ("lat", "lon")
            )
            group_1_var[:] = np.array([[0.1, 0.2]])
            group_1_var.units = "K"
            group_1_var.long_name = "air_temperature"

            subgroup1_var = subgroup_1.createVariable(
                "subgroup1_var", np.float64, ("lat", "lon")
            )
            subgroup1_var[:] = np.array([[0.1, 0.2]])

        aligned_dict_of_datasets = open_groups(filepath)
        aligned_dt = DataTree.from_dict(aligned_dict_of_datasets)

        assert open_datatree(filepath).identical(aligned_dt)


@requires_h5netcdf
class TestH5NetCDFDatatreeIO(DatatreeIOBase):
    engine: T_DataTreeNetcdfEngine | None = "h5netcdf"


@requires_zarr
class TestZarrDatatreeIO:
    engine = "zarr"

    def test_to_zarr(self, tmpdir, simple_datatree):
        filepath = tmpdir / "test.zarr"
        original_dt = simple_datatree
        original_dt.to_zarr(filepath)

        roundtrip_dt = open_datatree(filepath, engine="zarr")
        assert_equal(original_dt, roundtrip_dt)

    def test_zarr_encoding(self, tmpdir, simple_datatree):
        import zarr

        filepath = tmpdir / "test.zarr"
        original_dt = simple_datatree

        comp = {"compressor": zarr.Blosc(cname="zstd", clevel=3, shuffle=2)}
        enc = {"/set2": {var: comp for var in original_dt["/set2"].dataset.data_vars}}
        original_dt.to_zarr(filepath, encoding=enc)
        roundtrip_dt = open_datatree(filepath, engine="zarr")

        print(roundtrip_dt["/set2/a"].encoding)
        assert roundtrip_dt["/set2/a"].encoding["compressor"] == comp["compressor"]

        enc["/not/a/group"] = {"foo": "bar"}  # type: ignore[dict-item]
        with pytest.raises(ValueError, match="unexpected encoding group.*"):
            original_dt.to_zarr(filepath, encoding=enc, engine="zarr")

    def test_to_zarr_zip_store(self, tmpdir, simple_datatree):
        from zarr.storage import ZipStore

        filepath = tmpdir / "test.zarr.zip"
        original_dt = simple_datatree
        store = ZipStore(filepath)
        original_dt.to_zarr(store)

        roundtrip_dt = open_datatree(store, engine="zarr")
        assert_equal(original_dt, roundtrip_dt)

    def test_to_zarr_not_consolidated(self, tmpdir, simple_datatree):
        filepath = tmpdir / "test.zarr"
        zmetadata = filepath / ".zmetadata"
        s1zmetadata = filepath / "set1" / ".zmetadata"
        filepath = str(filepath)  # casting to str avoids a pathlib bug in xarray
        original_dt = simple_datatree
        original_dt.to_zarr(filepath, consolidated=False)
        assert not zmetadata.exists()
        assert not s1zmetadata.exists()

        with pytest.warns(RuntimeWarning, match="consolidated"):
            roundtrip_dt = open_datatree(filepath, engine="zarr")
        assert_equal(original_dt, roundtrip_dt)

    def test_to_zarr_default_write_mode(self, tmpdir, simple_datatree):
        import zarr

        simple_datatree.to_zarr(tmpdir)

        # with default settings, to_zarr should not overwrite an existing dir
        with pytest.raises(zarr.errors.ContainsGroupError):
            simple_datatree.to_zarr(tmpdir)

    def test_to_zarr_inherited_coords(self, tmpdir):
        original_dt = DataTree.from_dict(
            {
                "/": xr.Dataset({"a": (("x",), [1, 2])}, coords={"x": [3, 4]}),
                "/sub": xr.Dataset({"b": (("x",), [5, 6])}),
            }
        )
        filepath = tmpdir / "test.zarr"
        original_dt.to_zarr(filepath)

        roundtrip_dt = open_datatree(filepath, engine="zarr")
        assert_equal(original_dt, roundtrip_dt)
        subtree = cast(DataTree, roundtrip_dt["/sub"])
        assert "x" not in subtree.to_dataset(inherited=False).coords

    def test_open_groups_round_trip(self, tmpdir, simple_datatree) -> None:
        """Test `open_groups` opens a zarr store with the `simple_datatree` structure."""
        filepath = tmpdir / "test.zarr"
        original_dt = simple_datatree
        original_dt.to_zarr(filepath)

        roundtrip_dict = open_groups(filepath, engine="zarr")
        roundtrip_dt = DataTree.from_dict(roundtrip_dict)

        assert open_datatree(filepath, engine="zarr").identical(roundtrip_dt)

    def test_open_datatree(self, unaligned_datatree_zarr) -> None:
        """Test if `open_datatree` fails to open a zarr store with an unaligned group hierarchy."""
        with pytest.raises(
            ValueError,
            match=(
                re.escape("group '/Group2' is not aligned with its parents:") + ".*"
            ),
        ):
            open_datatree(unaligned_datatree_zarr, engine="zarr")

    def test_open_groups(self, unaligned_datatree_zarr) -> None:
        """Test `open_groups` with a zarr store of an unaligned group hierarchy."""

        unaligned_dict_of_datasets = open_groups(unaligned_datatree_zarr, engine="zarr")

        assert "/" in unaligned_dict_of_datasets.keys()
        assert "/Group1" in unaligned_dict_of_datasets.keys()
        assert "/Group1/subgroup1" in unaligned_dict_of_datasets.keys()
        assert "/Group2" in unaligned_dict_of_datasets.keys()
        # Check that group name returns the correct datasets
        assert_identical(
            unaligned_dict_of_datasets["/"],
            xr.open_dataset(unaligned_datatree_zarr, group="/", engine="zarr"),
        )
        assert_identical(
            unaligned_dict_of_datasets["/Group1"],
            xr.open_dataset(unaligned_datatree_zarr, group="Group1", engine="zarr"),
        )
        assert_identical(
            unaligned_dict_of_datasets["/Group1/subgroup1"],
            xr.open_dataset(
                unaligned_datatree_zarr, group="/Group1/subgroup1", engine="zarr"
            ),
        )
        assert_identical(
            unaligned_dict_of_datasets["/Group2"],
            xr.open_dataset(unaligned_datatree_zarr, group="/Group2", engine="zarr"),
        )
