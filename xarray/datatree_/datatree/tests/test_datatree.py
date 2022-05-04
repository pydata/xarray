import pytest
import xarray as xr
import xarray.testing as xrt

from datatree import DataTree


def create_test_datatree(modify=lambda ds: ds):
    """
    Create a test datatree with this structure:

    <datatree.DataTree>
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
    root = DataTree(data=root_data)
    set1 = DataTree(name="set1", parent=root, data=set1_data)
    DataTree(name="set1", parent=set1)
    DataTree(name="set2", parent=set1)
    set2 = DataTree(name="set2", parent=root, data=set2_data)
    DataTree(name="set1", parent=set2)
    DataTree(name="set3", parent=root)

    return root


class TestTreeCreation:
    def test_empty(self):
        dt = DataTree(name="root")
        assert dt.name == "root"
        assert dt.parent is None
        assert dt.children == {}
        xrt.assert_identical(dt.ds, xr.Dataset())

    def test_unnamed(self):
        dt = DataTree()
        assert dt.name is None


class TestFamilyTree:
    def test_setparent_unnamed_child_node_fails(self):
        john = DataTree(name="john")
        with pytest.raises(ValueError, match="unnamed"):
            DataTree(parent=john)


class TestStoreDatasets:
    def test_create_with_data(self):
        dat = xr.Dataset({"a": 0})
        john = DataTree(name="john", data=dat)
        assert john.ds is dat

        with pytest.raises(TypeError):
            DataTree(name="mary", parent=john, data="junk")  # noqa

    def test_set_data(self):
        john = DataTree(name="john")
        dat = xr.Dataset({"a": 0})
        john.ds = dat
        assert john.ds is dat
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
        with pytest.raises(KeyError, match="already has a child named a"):
            dt.ds = xr.Dataset({"a": 0})

        dt.ds = xr.Dataset()
        with pytest.raises(KeyError, match="already has a child named a"):
            dt.ds = dt.ds.assign(a=xr.DataArray(0))

    @pytest.mark.xfail
    def test_update_when_already_child_with_variables_name(self):
        # See issue https://github.com/xarray-contrib/datatree/issues/38
        dt = DataTree(name="root", data=None)
        DataTree(name="a", data=None, parent=dt)
        with pytest.raises(KeyError, match="already has a child named a"):
            dt.ds["a"] = xr.DataArray(0)


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
    ...


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
        xrt.assert_identical(mary.ds, xr.Dataset())

    def test_setitem_overwrite_data_in_node_with_none(self):
        john = DataTree(name="john")
        mary = DataTree(name="mary", parent=john, data=xr.Dataset())
        john["mary"] = DataTree()
        xrt.assert_identical(mary.ds, xr.Dataset())

        john.ds = xr.Dataset()
        with pytest.raises(ValueError, match="has no name"):
            john["."] = DataTree()

    @pytest.mark.xfail(reason="assigning Datasets doesn't yet create new nodes")
    def test_setitem_dataset_on_this_node(self):
        data = xr.Dataset({"temp": [0, 50]})
        results = DataTree(name="results")
        results["."] = data
        assert results.ds is data

    @pytest.mark.xfail(reason="assigning Datasets doesn't yet create new nodes")
    def test_setitem_dataset_as_new_node(self):
        data = xr.Dataset({"temp": [0, 50]})
        folder1 = DataTree(name="folder1")
        folder1["results"] = data
        assert folder1["results"].ds is data

    @pytest.mark.xfail(reason="assigning Datasets doesn't yet create new nodes")
    def test_setitem_dataset_as_new_node_requiring_intermediate_nodes(self):
        data = xr.Dataset({"temp": [0, 50]})
        folder1 = DataTree(name="folder1")
        folder1["results/highres"] = data
        assert folder1["results/highres"].ds is data

    def test_setitem_named_dataarray(self):
        data = xr.DataArray(name="temp", data=[0, 50])
        folder1 = DataTree(name="folder1")
        folder1["results"] = data
        expected = data.rename("results")
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
        xrt.assert_identical(results.ds, expected)


class TestDictionaryInterface:
    ...


class TestTreeFromDict:
    def test_data_in_root(self):
        dat = xr.Dataset()
        dt = DataTree.from_dict({"/": dat})
        assert dt.name is None
        assert dt.parent is None
        assert dt.children == {}
        assert dt.ds is dat

    def test_one_layer(self):
        dat1, dat2 = xr.Dataset({"a": 1}), xr.Dataset({"b": 2})
        dt = DataTree.from_dict({"run1": dat1, "run2": dat2})
        xrt.assert_identical(dt.ds, xr.Dataset())
        assert dt.name is None
        assert dt["run1"].ds is dat1
        assert dt["run1"].children == {}
        assert dt["run2"].ds is dat2
        assert dt["run2"].children == {}

    def test_two_layers(self):
        dat1, dat2 = xr.Dataset({"a": 1}), xr.Dataset({"a": [1, 2]})
        dt = DataTree.from_dict({"highres/run": dat1, "lowres/run": dat2})
        assert "highres" in dt.children
        assert "lowres" in dt.children
        highres_run = dt["highres/run"]
        assert highres_run.ds is dat1

    def test_nones(self):
        dt = DataTree.from_dict({"d": None, "d/e": None})
        assert [node.name for node in dt.subtree] == [None, "d", "e"]
        assert [node.path for node in dt.subtree] == ["/", "/d", "/d/e"]
        xrt.assert_equal(dt["d/e"].ds, xr.Dataset())

    def test_full(self):
        dt = create_test_datatree()
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


class TestBrowsing:
    ...


class TestRestructuring:
    ...
