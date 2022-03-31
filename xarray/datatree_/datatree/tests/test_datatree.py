import textwrap

import pytest
import xarray as xr
import xarray.testing as xrt
from anytree.resolver import ChildResolverError

from datatree import DataTree
from datatree.io import open_datatree
from datatree.testing import assert_equal
from datatree.tests import requires_h5netcdf, requires_netCDF4, requires_zarr


def create_test_datatree(modify=lambda ds: ds):
    """
    Create a test datatree with this structure:

    <xtree.DataTree>
    |-- set1
    |   |-- <xarray.Dataset>
    |   |   Dimensions:  ()
    |   |   Data variables:
    |   |       a        int64 0
    |   |       b        int64 1
    |   |-- set1
    |   |-- set2
    |-- set2
    |   |-- <xarray.Dataset>
    |   |   Dimensions:  (x: 2)
    |   |   Data variables:
    |   |       a        (x) int64 2, 3
    |   |       b        (x) int64 0.1, 0.2
    |   |-- set1
    |-- set3
    |-- <xarray.Dataset>
    |   Dimensions:  (x: 2, y: 3)
    |   Data variables:
    |       a        (y) int64 6, 7, 8
    |       set0     (x) int64 9, 10

    The structure has deliberately repeated names of tags, variables, and
    dimensions in order to better check for bugs caused by name conflicts.
    """
    set1_data = modify(xr.Dataset({"a": 0, "b": 1}))
    set2_data = modify(xr.Dataset({"a": ("x", [2, 3]), "b": ("x", [0.1, 0.2])}))
    root_data = modify(xr.Dataset({"a": ("y", [6, 7, 8]), "set0": ("x", [9, 10])}))

    # Avoid using __init__ so we can independently test it
    root = DataTree(name="root", data=root_data)
    set1 = DataTree(name="set1", parent=root, data=set1_data)
    DataTree(name="set1", parent=set1)
    DataTree(name="set2", parent=set1)
    set2 = DataTree(name="set2", parent=root, data=set2_data)
    DataTree(name="set1", parent=set2)
    DataTree(name="set3", parent=root)

    return root


class TestStoreDatasets:
    def test_create_DataTree(self):
        dat = xr.Dataset({"a": 0})
        john = DataTree("john", data=dat)
        assert john.ds is dat
        with pytest.raises(TypeError):
            DataTree("mary", parent=john, data="junk")

    def test_set_data(self):
        john = DataTree("john")
        dat = xr.Dataset({"a": 0})
        john.ds = dat
        assert john.ds is dat
        with pytest.raises(TypeError):
            john.ds = "junk"

    def test_has_data(self):
        john = DataTree("john", data=xr.Dataset({"a": 0}))
        assert john.has_data

        john = DataTree("john", data=None)
        assert not john.has_data


class TestVariablesChildrenNameCollisions:
    def test_parent_already_has_variable_with_childs_name(self):
        dt = DataTree("root", data=xr.Dataset({"a": [0], "b": 1}))
        with pytest.raises(KeyError, match="already contains a data variable named a"):
            DataTree("a", data=None, parent=dt)

        with pytest.raises(KeyError, match="already contains a data variable named a"):
            dt.add_child(DataTree("a", data=None))

    def test_assign_when_already_child_with_variables_name(self):
        dt = DataTree("root", data=None)
        DataTree("a", data=None, parent=dt)
        with pytest.raises(KeyError, match="already has a child named a"):
            dt.ds = xr.Dataset({"a": 0})

        dt.ds = xr.Dataset()
        with pytest.raises(KeyError, match="already has a child named a"):
            dt.ds = dt.ds.assign(a=xr.DataArray(0))

    @pytest.mark.xfail
    def test_update_when_already_child_with_variables_name(self):
        # See issue https://github.com/xarray-contrib/datatree/issues/38
        dt = DataTree("root", data=None)
        DataTree("a", data=None, parent=dt)
        with pytest.raises(KeyError, match="already has a child named a"):
            dt.ds["a"] = xr.DataArray(0)


class TestGetItems:
    def test_get_node(self):
        folder1 = DataTree("folder1")
        results = DataTree("results", parent=folder1)
        highres = DataTree("highres", parent=results)
        assert folder1["results"] is results
        assert folder1["results/highres"] is highres
        assert folder1[("results", "highres")] is highres

    def test_get_single_data_variable(self):
        data = xr.Dataset({"temp": [0, 50]})
        results = DataTree("results", data=data)
        xrt.assert_identical(results["temp"], data["temp"])

    def test_get_single_data_variable_from_node(self):
        data = xr.Dataset({"temp": [0, 50]})
        folder1 = DataTree("folder1")
        results = DataTree("results", parent=folder1)
        DataTree("highres", parent=results, data=data)
        xrt.assert_identical(folder1["results/highres/temp"], data["temp"])
        xrt.assert_identical(folder1[("results", "highres", "temp")], data["temp"])

    def test_get_nonexistent_node(self):
        folder1 = DataTree("folder1")
        DataTree("results", parent=folder1)
        with pytest.raises(ChildResolverError):
            folder1["results/highres"]

    def test_get_nonexistent_variable(self):
        data = xr.Dataset({"temp": [0, 50]})
        results = DataTree("results", data=data)
        with pytest.raises(ChildResolverError):
            results["pressure"]

    def test_get_multiple_data_variables(self):
        data = xr.Dataset({"temp": [0, 50], "p": [5, 8, 7]})
        results = DataTree("results", data=data)
        xrt.assert_identical(results[["temp", "p"]], data[["temp", "p"]])

    def test_dict_like_selection_access_to_dataset(self):
        data = xr.Dataset({"temp": [0, 50]})
        results = DataTree("results", data=data)
        xrt.assert_identical(results[{"temp": 1}], data[{"temp": 1}])


class TestSetItems:
    # TODO test tuple-style access too
    def test_set_new_child_node(self):
        john = DataTree("john")
        mary = DataTree("mary")
        john["/"] = mary
        assert john["mary"] is mary

    def test_set_new_grandchild_node(self):
        john = DataTree("john")
        DataTree("mary", parent=john)
        rose = DataTree("rose")
        john["mary/"] = rose
        assert john["mary/rose"] is rose

    def test_set_new_empty_node(self):
        john = DataTree("john")
        john["mary"] = None
        mary = john["mary"]
        assert isinstance(mary, DataTree)
        xrt.assert_identical(mary.ds, xr.Dataset())

    def test_overwrite_data_in_node_with_none(self):
        john = DataTree("john")
        mary = DataTree("mary", parent=john, data=xr.Dataset())
        john["mary"] = None
        xrt.assert_identical(mary.ds, xr.Dataset())

        john.ds = xr.Dataset()
        john["/"] = None
        xrt.assert_identical(john.ds, xr.Dataset())

    def test_set_dataset_on_this_node(self):
        data = xr.Dataset({"temp": [0, 50]})
        results = DataTree("results")
        results["/"] = data
        assert results.ds is data

    def test_set_dataset_as_new_node(self):
        data = xr.Dataset({"temp": [0, 50]})
        folder1 = DataTree("folder1")
        folder1["results"] = data
        assert folder1["results"].ds is data

    def test_set_dataset_as_new_node_requiring_intermediate_nodes(self):
        data = xr.Dataset({"temp": [0, 50]})
        folder1 = DataTree("folder1")
        folder1["results/highres"] = data
        assert folder1["results/highres"].ds is data

    def test_set_named_dataarray_as_new_node(self):
        data = xr.DataArray(name="temp", data=[0, 50])
        folder1 = DataTree("folder1")
        folder1["results"] = data
        xrt.assert_identical(folder1["results"].ds, data.to_dataset())

    def test_set_unnamed_dataarray(self):
        data = xr.DataArray([0, 50])
        folder1 = DataTree("folder1")
        with pytest.raises(ValueError, match="unable to convert"):
            folder1["results"] = data

    def test_add_new_variable_to_empty_node(self):
        results = DataTree("results")
        results["/"] = xr.DataArray(name="pressure", data=[2, 3])
        assert "pressure" in results.ds

        # What if there is a path to traverse first?
        results = DataTree("results")
        results["highres/"] = xr.DataArray(name="pressure", data=[2, 3])
        assert "pressure" in results["highres"].ds

    def test_dataarray_replace_existing_node(self):
        t = xr.Dataset({"temp": [0, 50]})
        results = DataTree("results", data=t)
        p = xr.DataArray(name="pressure", data=[2, 3])
        results["/"] = p
        xrt.assert_identical(results.ds, p.to_dataset())


class TestTreeCreation:
    def test_empty(self):
        dt = DataTree()
        assert dt.name == "root"
        assert dt.parent is None
        assert dt.children == ()
        xrt.assert_identical(dt.ds, xr.Dataset())

    def test_data_in_root(self):
        dat = xr.Dataset()
        dt = DataTree.from_dict({"root": dat})
        assert dt.name == "root"
        assert dt.parent is None
        assert dt.children == ()
        assert dt.ds is dat

    def test_one_layer(self):
        dat1, dat2 = xr.Dataset({"a": 1}), xr.Dataset({"b": 2})
        dt = DataTree.from_dict({"run1": dat1, "run2": dat2})
        xrt.assert_identical(dt.ds, xr.Dataset())
        assert dt["run1"].ds is dat1
        assert dt["run1"].children == ()
        assert dt["run2"].ds is dat2
        assert dt["run2"].children == ()

    def test_two_layers(self):
        dat1, dat2 = xr.Dataset({"a": 1}), xr.Dataset({"a": [1, 2]})
        dt = DataTree.from_dict({"highres/run": dat1, "lowres/run": dat2})
        assert "highres" in [c.name for c in dt.children]
        assert "lowres" in [c.name for c in dt.children]
        highres_run = dt.get_node("highres/run")
        assert highres_run.ds is dat1

    def test_full(self):
        dt = create_test_datatree()
        paths = list(node.pathstr for node in dt.subtree)
        assert paths == [
            "root",
            "root/set1",
            "root/set1/set1",
            "root/set1/set2",
            "root/set2",
            "root/set2/set1",
            "root/set3",
        ]


class TestBrowsing:
    ...


class TestRestructuring:
    ...


class TestRepr:
    def test_print_empty_node(self):
        dt = DataTree("root")
        printout = dt.__str__()
        assert printout == "DataTree('root', parent=None)"

    def test_print_empty_node_with_attrs(self):
        dat = xr.Dataset(attrs={"note": "has attrs"})
        dt = DataTree("root", data=dat)
        printout = dt.__str__()
        assert printout == textwrap.dedent(
            """\
            DataTree('root', parent=None)
            Dimensions:  ()
            Data variables:
                *empty*
            Attributes:
                note:     has attrs"""
        )

    def test_print_node_with_data(self):
        dat = xr.Dataset({"a": [0, 2]})
        dt = DataTree("root", data=dat)
        printout = dt.__str__()
        expected = [
            "DataTree('root', parent=None)",
            "Dimensions",
            "Coordinates",
            "a",
            "Data variables",
            "*empty*",
        ]
        for expected_line, printed_line in zip(expected, printout.splitlines()):
            assert expected_line in printed_line

    def test_nested_node(self):
        dat = xr.Dataset({"a": [0, 2]})
        root = DataTree("root")
        DataTree("results", data=dat, parent=root)
        printout = root.__str__()
        assert printout.splitlines()[2].startswith("    ")

    def test_print_datatree(self):
        dt = create_test_datatree()
        print(dt)
        print(dt.descendants)

        # TODO work out how to test something complex like this

    def test_repr_of_node_with_data(self):
        dat = xr.Dataset({"a": [0, 2]})
        dt = DataTree("root", data=dat)
        assert "Coordinates" in repr(dt)


class TestIO:
    @requires_netCDF4
    def test_to_netcdf(self, tmpdir):
        filepath = str(
            tmpdir / "test.nc"
        )  # casting to str avoids a pathlib bug in xarray
        original_dt = create_test_datatree()
        original_dt.to_netcdf(filepath, engine="netcdf4")

        roundtrip_dt = open_datatree(filepath)
        assert_equal(original_dt, roundtrip_dt)

    @requires_h5netcdf
    def test_to_h5netcdf(self, tmpdir):
        filepath = str(
            tmpdir / "test.nc"
        )  # casting to str avoids a pathlib bug in xarray
        original_dt = create_test_datatree()
        original_dt.to_netcdf(filepath, engine="h5netcdf")

        roundtrip_dt = open_datatree(filepath)
        assert_equal(original_dt, roundtrip_dt)

    @requires_zarr
    def test_to_zarr(self, tmpdir):
        filepath = str(
            tmpdir / "test.zarr"
        )  # casting to str avoids a pathlib bug in xarray
        original_dt = create_test_datatree()
        original_dt.to_zarr(filepath)

        roundtrip_dt = open_datatree(filepath, engine="zarr")
        assert_equal(original_dt, roundtrip_dt)

    @requires_zarr
    def test_to_zarr_not_consolidated(self, tmpdir):
        filepath = tmpdir / "test.zarr"
        zmetadata = filepath / ".zmetadata"
        s1zmetadata = filepath / "set1" / ".zmetadata"
        filepath = str(filepath)  # casting to str avoids a pathlib bug in xarray
        original_dt = create_test_datatree()
        original_dt.to_zarr(filepath, consolidated=False)
        assert not zmetadata.exists()
        assert not s1zmetadata.exists()

        with pytest.warns(RuntimeWarning, match="consolidated"):
            roundtrip_dt = open_datatree(filepath, engine="zarr")
        assert_equal(original_dt, roundtrip_dt)
