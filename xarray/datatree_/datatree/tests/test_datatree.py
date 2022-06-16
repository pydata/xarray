from copy import copy, deepcopy

import numpy as np
import pytest
import xarray as xr
import xarray.testing as xrt
from xarray.tests import create_test_data, source_ndarray

import datatree.testing as dtt
from datatree import DataTree


class TestTreeCreation:
    def test_empty(self):
        dt = DataTree(name="root")
        assert dt.name == "root"
        assert dt.parent is None
        assert dt.children == {}
        xrt.assert_identical(dt.to_dataset(), xr.Dataset())

    def test_unnamed(self):
        dt = DataTree()
        assert dt.name is None

    def test_bad_names(self):
        with pytest.raises(TypeError):
            DataTree(name=5)

        with pytest.raises(ValueError):
            DataTree(name="folder/data")


class TestFamilyTree:
    def test_setparent_unnamed_child_node_fails(self):
        john = DataTree(name="john")
        with pytest.raises(ValueError, match="unnamed"):
            DataTree(parent=john)

    def test_create_two_children(self):
        root_data = xr.Dataset({"a": ("y", [6, 7, 8]), "set0": ("x", [9, 10])})
        set1_data = xr.Dataset({"a": 0, "b": 1})

        root = DataTree(data=root_data)
        set1 = DataTree(name="set1", parent=root, data=set1_data)
        DataTree(name="set1", parent=root)
        DataTree(name="set2", parent=set1)

    def test_create_full_tree(self, simple_datatree):
        root_data = xr.Dataset({"a": ("y", [6, 7, 8]), "set0": ("x", [9, 10])})
        set1_data = xr.Dataset({"a": 0, "b": 1})
        set2_data = xr.Dataset({"a": ("x", [2, 3]), "b": ("x", [0.1, 0.2])})

        root = DataTree(data=root_data)
        set1 = DataTree(name="set1", parent=root, data=set1_data)
        DataTree(name="set1", parent=set1)
        DataTree(name="set2", parent=set1)
        set2 = DataTree(name="set2", parent=root, data=set2_data)
        DataTree(name="set1", parent=set2)
        DataTree(name="set3", parent=root)

        expected = simple_datatree
        assert root.identical(expected)


class TestStoreDatasets:
    def test_create_with_data(self):
        dat = xr.Dataset({"a": 0})
        john = DataTree(name="john", data=dat)
        xrt.assert_identical(john.to_dataset(), dat)

        with pytest.raises(TypeError):
            DataTree(name="mary", parent=john, data="junk")  # noqa

    def test_set_data(self):
        john = DataTree(name="john")
        dat = xr.Dataset({"a": 0})
        john.ds = dat
        xrt.assert_identical(john.to_dataset(), dat)
        with pytest.raises(TypeError):
            john.ds = "junk"

    def test_has_data(self):
        john = DataTree(name="john", data=xr.Dataset({"a": 0}))
        assert john.has_data

        john = DataTree(name="john", data=None)
        assert not john.has_data


class TestVariablesChildrenNameCollisions:
    def test_parent_already_has_variable_with_childs_name(self):
        dt = DataTree(data=xr.Dataset({"a": [0], "b": 1}))
        with pytest.raises(KeyError, match="already contains a data variable named a"):
            DataTree(name="a", data=None, parent=dt)

    def test_assign_when_already_child_with_variables_name(self):
        dt = DataTree(data=None)
        DataTree(name="a", data=None, parent=dt)
        with pytest.raises(KeyError, match="names would collide"):
            dt.ds = xr.Dataset({"a": 0})

        dt.ds = xr.Dataset()
        new_ds = dt.to_dataset().assign(a=xr.DataArray(0))
        with pytest.raises(KeyError, match="names would collide"):
            dt.ds = new_ds


class TestGet:
    ...


class TestGetItem:
    def test_getitem_node(self):
        folder1 = DataTree(name="folder1")
        results = DataTree(name="results", parent=folder1)
        highres = DataTree(name="highres", parent=results)
        assert folder1["results"] is results
        assert folder1["results/highres"] is highres

    def test_getitem_self(self):
        dt = DataTree()
        assert dt["."] is dt

    def test_getitem_single_data_variable(self):
        data = xr.Dataset({"temp": [0, 50]})
        results = DataTree(name="results", data=data)
        xrt.assert_identical(results["temp"], data["temp"])

    def test_getitem_single_data_variable_from_node(self):
        data = xr.Dataset({"temp": [0, 50]})
        folder1 = DataTree(name="folder1")
        results = DataTree(name="results", parent=folder1)
        DataTree(name="highres", parent=results, data=data)
        xrt.assert_identical(folder1["results/highres/temp"], data["temp"])

    def test_getitem_nonexistent_node(self):
        folder1 = DataTree(name="folder1")
        DataTree(name="results", parent=folder1)
        with pytest.raises(KeyError):
            folder1["results/highres"]

    def test_getitem_nonexistent_variable(self):
        data = xr.Dataset({"temp": [0, 50]})
        results = DataTree(name="results", data=data)
        with pytest.raises(KeyError):
            results["pressure"]

    @pytest.mark.xfail(reason="Should be deprecated in favour of .subset")
    def test_getitem_multiple_data_variables(self):
        data = xr.Dataset({"temp": [0, 50], "p": [5, 8, 7]})
        results = DataTree(name="results", data=data)
        xrt.assert_identical(results[["temp", "p"]], data[["temp", "p"]])

    @pytest.mark.xfail(reason="Indexing needs to return whole tree (GH https://github.com/xarray-contrib/datatree/issues/77)")
    def test_getitem_dict_like_selection_access_to_dataset(self):
        data = xr.Dataset({"temp": [0, 50]})
        results = DataTree(name="results", data=data)
        xrt.assert_identical(results[{"temp": 1}], data[{"temp": 1}])


class TestUpdate:
    def test_update_new_named_dataarray(self):
        da = xr.DataArray(name="temp", data=[0, 50])
        folder1 = DataTree(name="folder1")
        folder1.update({"results": da})
        expected = da.rename("results")
        xrt.assert_equal(folder1["results"], expected)


class TestCopy:
    def test_copy(self, create_test_datatree):
        dt = create_test_datatree()

        for node in dt.root.subtree:
            node.attrs["Test"] = [1, 2, 3]

        for copied in [dt.copy(deep=False), copy(dt)]:
            dtt.assert_identical(dt, copied)

            for node, copied_node in zip(dt.root.subtree, copied.root.subtree):

                assert node.encoding == copied_node.encoding
                # Note: IndexVariable objects with string dtype are always
                # copied because of xarray.core.util.safe_cast_to_index.
                # Limiting the test to data variables.
                for k in node.data_vars:
                    v0 = node.variables[k]
                    v1 = copied_node.variables[k]
                    assert source_ndarray(v0.data) is source_ndarray(v1.data)
                copied_node["foo"] = xr.DataArray(data=np.arange(5), dims="z")
                assert "foo" not in node

                copied_node.attrs["foo"] = "bar"
                assert "foo" not in node.attrs
                assert node.attrs["Test"] is copied_node.attrs["Test"]

    def test_deepcopy(self, create_test_datatree):
        dt = create_test_datatree()

        for node in dt.root.subtree:
            node.attrs["Test"] = [1, 2, 3]

        for copied in [dt.copy(deep=True), deepcopy(dt)]:
            dtt.assert_identical(dt, copied)

            for node, copied_node in zip(dt.root.subtree, copied.root.subtree):
                assert node.encoding == copied_node.encoding
                # Note: IndexVariable objects with string dtype are always
                # copied because of xarray.core.util.safe_cast_to_index.
                # Limiting the test to data variables.
                for k in node.data_vars:
                    v0 = node.variables[k]
                    v1 = copied_node.variables[k]
                    assert source_ndarray(v0.data) is not source_ndarray(v1.data)
                copied_node["foo"] = xr.DataArray(data=np.arange(5), dims="z")
                assert "foo" not in node

                copied_node.attrs["foo"] = "bar"
                assert "foo" not in node.attrs
                assert node.attrs["Test"] is not copied_node.attrs["Test"]

    @pytest.mark.xfail(reason="data argument not yet implemented")
    def test_copy_with_data(self, create_test_datatree):
        orig = create_test_datatree()
        # TODO use .data_vars once that property is available
        data_vars = {
            k: v for k, v in orig.variables.items() if k not in orig._coord_names
        }
        new_data = {k: np.random.randn(*v.shape) for k, v in data_vars.items()}
        actual = orig.copy(data=new_data)

        expected = orig.copy()
        for k, v in new_data.items():
            expected[k].data = v
        dtt.assert_identical(expected, actual)

        # TODO test parents and children?


class TestSetItem:
    def test_setitem_new_child_node(self):
        john = DataTree(name="john")
        mary = DataTree(name="mary")
        john["Mary"] = mary
        assert john["Mary"] is mary

    def test_setitem_unnamed_child_node_becomes_named(self):
        john2 = DataTree(name="john2")
        john2["sonny"] = DataTree()
        assert john2["sonny"].name == "sonny"

    @pytest.mark.xfail(reason="bug with name overwriting")
    def test_setitem_child_node_keeps_name(self):
        john = DataTree(name="john")
        r2d2 = DataTree(name="R2D2")
        john["Mary"] = r2d2
        assert r2d2.name == "R2D2"

    def test_setitem_new_grandchild_node(self):
        john = DataTree(name="john")
        DataTree(name="mary", parent=john)
        rose = DataTree(name="rose")
        john["Mary/Rose"] = rose
        assert john["Mary/Rose"] is rose

    def test_setitem_new_empty_node(self):
        john = DataTree(name="john")
        john["mary"] = DataTree()
        mary = john["mary"]
        assert isinstance(mary, DataTree)
        xrt.assert_identical(mary.to_dataset(), xr.Dataset())

    def test_setitem_overwrite_data_in_node_with_none(self):
        john = DataTree(name="john")
        mary = DataTree(name="mary", parent=john, data=xr.Dataset())
        john["mary"] = DataTree()
        xrt.assert_identical(mary.to_dataset(), xr.Dataset())

        john.ds = xr.Dataset()
        with pytest.raises(ValueError, match="has no name"):
            john["."] = DataTree()

    @pytest.mark.xfail(reason="assigning Datasets doesn't yet create new nodes")
    def test_setitem_dataset_on_this_node(self):
        data = xr.Dataset({"temp": [0, 50]})
        results = DataTree(name="results")
        results["."] = data
        xrt.assert_identical(results.to_dataset(), data)

    @pytest.mark.xfail(reason="assigning Datasets doesn't yet create new nodes")
    def test_setitem_dataset_as_new_node(self):
        data = xr.Dataset({"temp": [0, 50]})
        folder1 = DataTree(name="folder1")
        folder1["results"] = data
        xrt.assert_identical(folder1["results"].to_dataset(), data)

    @pytest.mark.xfail(reason="assigning Datasets doesn't yet create new nodes")
    def test_setitem_dataset_as_new_node_requiring_intermediate_nodes(self):
        data = xr.Dataset({"temp": [0, 50]})
        folder1 = DataTree(name="folder1")
        folder1["results/highres"] = data
        xrt.assert_identical(folder1["results/highres"].to_dataset(), data)

    def test_setitem_named_dataarray(self):
        da = xr.DataArray(name="temp", data=[0, 50])
        folder1 = DataTree(name="folder1")
        folder1["results"] = da
        expected = da.rename("results")
        xrt.assert_equal(folder1["results"], expected)

    def test_setitem_unnamed_dataarray(self):
        data = xr.DataArray([0, 50])
        folder1 = DataTree(name="folder1")
        folder1["results"] = data
        xrt.assert_equal(folder1["results"], data)

    def test_setitem_add_new_variable_to_empty_node(self):
        results = DataTree(name="results")
        results["pressure"] = xr.DataArray(data=[2, 3])
        assert "pressure" in results.ds
        results["temp"] = xr.Variable(data=[10, 11], dims=["x"])
        assert "temp" in results.ds

        # What if there is a path to traverse first?
        results = DataTree(name="results")
        results["highres/pressure"] = xr.DataArray(data=[2, 3])
        assert "pressure" in results["highres"].ds
        results["highres/temp"] = xr.Variable(data=[10, 11], dims=["x"])
        assert "temp" in results["highres"].ds

    def test_setitem_dataarray_replace_existing_node(self):
        t = xr.Dataset({"temp": [0, 50]})
        results = DataTree(name="results", data=t)
        p = xr.DataArray(data=[2, 3])
        results["pressure"] = p
        expected = t.assign(pressure=p)
        xrt.assert_identical(results.to_dataset(), expected)


class TestDictionaryInterface:
    ...


class TestTreeFromDict:
    def test_data_in_root(self):
        dat = xr.Dataset()
        dt = DataTree.from_dict({"/": dat})
        assert dt.name is None
        assert dt.parent is None
        assert dt.children == {}
        xrt.assert_identical(dt.to_dataset(), dat)

    def test_one_layer(self):
        dat1, dat2 = xr.Dataset({"a": 1}), xr.Dataset({"b": 2})
        dt = DataTree.from_dict({"run1": dat1, "run2": dat2})
        xrt.assert_identical(dt.to_dataset(), xr.Dataset())
        assert dt.name is None
        xrt.assert_identical(dt["run1"].to_dataset(), dat1)
        assert dt["run1"].children == {}
        xrt.assert_identical(dt["run2"].to_dataset(), dat2)
        assert dt["run2"].children == {}

    def test_two_layers(self):
        dat1, dat2 = xr.Dataset({"a": 1}), xr.Dataset({"a": [1, 2]})
        dt = DataTree.from_dict({"highres/run": dat1, "lowres/run": dat2})
        assert "highres" in dt.children
        assert "lowres" in dt.children
        highres_run = dt["highres/run"]
        xrt.assert_identical(highres_run.to_dataset(), dat1)

    def test_nones(self):
        dt = DataTree.from_dict({"d": None, "d/e": None})
        assert [node.name for node in dt.subtree] == [None, "d", "e"]
        assert [node.path for node in dt.subtree] == ["/", "/d", "/d/e"]
        xrt.assert_identical(dt["d/e"].to_dataset(), xr.Dataset())

    def test_full(self, simple_datatree):
        dt = simple_datatree
        paths = list(node.path for node in dt.subtree)
        assert paths == [
            "/",
            "/set1",
            "/set1/set1",
            "/set1/set2",
            "/set2",
            "/set2/set1",
            "/set3",
        ]

    def test_roundtrip(self, simple_datatree):
        dt = simple_datatree
        roundtrip = DataTree.from_dict(dt.to_dict())
        assert roundtrip.equals(dt)

    @pytest.mark.xfail
    def test_roundtrip_unnamed_root(self, simple_datatree):
        # See GH81

        dt = simple_datatree
        dt.name = "root"
        roundtrip = DataTree.from_dict(dt.to_dict())
        assert roundtrip.equals(dt)


class TestDatasetView:
    def test_view_contents(self):
        ds = create_test_data()
        dt = DataTree(data=ds)
        assert ds.identical(
            dt.ds
        )  # this only works because Dataset.identical doesn't check types
        assert isinstance(dt.ds, xr.Dataset)

    def test_immutability(self):
        # See issue https://github.com/xarray-contrib/datatree/issues/38
        dt = DataTree(name="root", data=None)
        DataTree(name="a", data=None, parent=dt)

        with pytest.raises(
            AttributeError, match="Mutation of the DatasetView is not allowed"
        ):
            dt.ds["a"] = xr.DataArray(0)

        with pytest.raises(
            AttributeError, match="Mutation of the DatasetView is not allowed"
        ):
            dt.ds.update({"a": 0})

        # TODO are there any other ways you can normally modify state (in-place)?
        # (not attribute-like assignment because that doesn't work on Dataset anyway)

    def test_methods(self):
        ds = create_test_data()
        dt = DataTree(data=ds)
        assert ds.mean().identical(dt.ds.mean())
        assert type(dt.ds.mean()) == xr.Dataset

    def test_arithmetic(self, create_test_datatree):
        dt = create_test_datatree()
        expected = create_test_datatree(modify=lambda ds: 10.0 * ds)["set1"]
        result = 10.0 * dt["set1"].ds
        assert result.identical(expected)


class TestRestructuring:
    ...
