from copy import copy, deepcopy

import numpy as np
import pytest

import xarray as xr
import xarray.datatree_.datatree.testing as dtt
import xarray.testing as xrt
from xarray.core.datatree import DataTree
from xarray.core.treenode import NotFoundInTreeError
from xarray.tests import create_test_data, source_ndarray


class TestTreeCreation:
    def test_empty(self):
        dt: DataTree = DataTree(name="root")
        assert dt.name == "root"
        assert dt.parent is None
        assert dt.children == {}
        xrt.assert_identical(dt.to_dataset(), xr.Dataset())

    def test_unnamed(self):
        dt: DataTree = DataTree()
        assert dt.name is None

    def test_bad_names(self):
        with pytest.raises(TypeError):
            DataTree(name=5)  # type: ignore[arg-type]

        with pytest.raises(ValueError):
            DataTree(name="folder/data")


class TestFamilyTree:
    def test_setparent_unnamed_child_node_fails(self):
        john: DataTree = DataTree(name="john")
        with pytest.raises(ValueError, match="unnamed"):
            DataTree(parent=john)

    def test_create_two_children(self):
        root_data = xr.Dataset({"a": ("y", [6, 7, 8]), "set0": ("x", [9, 10])})
        set1_data = xr.Dataset({"a": 0, "b": 1})

        root: DataTree = DataTree(data=root_data)
        set1: DataTree = DataTree(name="set1", parent=root, data=set1_data)
        DataTree(name="set1", parent=root)
        DataTree(name="set2", parent=set1)

    def test_create_full_tree(self, simple_datatree):
        root_data = xr.Dataset({"a": ("y", [6, 7, 8]), "set0": ("x", [9, 10])})
        set1_data = xr.Dataset({"a": 0, "b": 1})
        set2_data = xr.Dataset({"a": ("x", [2, 3]), "b": ("x", [0.1, 0.2])})

        root: DataTree = DataTree(data=root_data)
        set1: DataTree = DataTree(name="set1", parent=root, data=set1_data)
        DataTree(name="set1", parent=set1)
        DataTree(name="set2", parent=set1)
        set2: DataTree = DataTree(name="set2", parent=root, data=set2_data)
        DataTree(name="set1", parent=set2)
        DataTree(name="set3", parent=root)

        expected = simple_datatree
        assert root.identical(expected)


class TestNames:
    def test_child_gets_named_on_attach(self):
        sue: DataTree = DataTree()
        mary: DataTree = DataTree(children={"Sue": sue})  # noqa
        assert sue.name == "Sue"


class TestPaths:
    def test_path_property(self):
        sue: DataTree = DataTree()
        mary: DataTree = DataTree(children={"Sue": sue})
        john: DataTree = DataTree(children={"Mary": mary})
        assert sue.path == "/Mary/Sue"
        assert john.path == "/"

    def test_path_roundtrip(self):
        sue: DataTree = DataTree()
        mary: DataTree = DataTree(children={"Sue": sue})
        john: DataTree = DataTree(children={"Mary": mary})
        assert john[sue.path] is sue

    def test_same_tree(self):
        mary: DataTree = DataTree()
        kate: DataTree = DataTree()
        john: DataTree = DataTree(children={"Mary": mary, "Kate": kate})  # noqa
        assert mary.same_tree(kate)

    def test_relative_paths(self):
        sue: DataTree = DataTree()
        mary: DataTree = DataTree(children={"Sue": sue})
        annie: DataTree = DataTree()
        john: DataTree = DataTree(children={"Mary": mary, "Annie": annie})

        result = sue.relative_to(john)
        assert result == "Mary/Sue"
        assert john.relative_to(sue) == "../.."
        assert annie.relative_to(sue) == "../../Annie"
        assert sue.relative_to(annie) == "../Mary/Sue"
        assert sue.relative_to(sue) == "."

        evil_kate: DataTree = DataTree()
        with pytest.raises(
            NotFoundInTreeError, match="nodes do not lie within the same tree"
        ):
            sue.relative_to(evil_kate)


class TestStoreDatasets:
    def test_create_with_data(self):
        dat = xr.Dataset({"a": 0})
        john: DataTree = DataTree(name="john", data=dat)

        xrt.assert_identical(john.to_dataset(), dat)

        with pytest.raises(TypeError):
            DataTree(name="mary", parent=john, data="junk")  # type: ignore[arg-type]

    def test_set_data(self):
        john: DataTree = DataTree(name="john")
        dat = xr.Dataset({"a": 0})
        john.ds = dat  # type: ignore[assignment]

        xrt.assert_identical(john.to_dataset(), dat)

        with pytest.raises(TypeError):
            john.ds = "junk"  # type: ignore[assignment]

    def test_has_data(self):
        john: DataTree = DataTree(name="john", data=xr.Dataset({"a": 0}))
        assert john.has_data

        john_no_data: DataTree = DataTree(name="john", data=None)
        assert not john_no_data.has_data

    def test_is_hollow(self):
        john: DataTree = DataTree(data=xr.Dataset({"a": 0}))
        assert john.is_hollow

        eve: DataTree = DataTree(children={"john": john})
        assert eve.is_hollow

        eve.ds = xr.Dataset({"a": 1})  # type: ignore[assignment]
        assert not eve.is_hollow


class TestVariablesChildrenNameCollisions:
    def test_parent_already_has_variable_with_childs_name(self):
        dt: DataTree = DataTree(data=xr.Dataset({"a": [0], "b": 1}))
        with pytest.raises(KeyError, match="already contains a data variable named a"):
            DataTree(name="a", data=None, parent=dt)

    def test_assign_when_already_child_with_variables_name(self):
        dt: DataTree = DataTree(data=None)
        DataTree(name="a", data=None, parent=dt)
        with pytest.raises(KeyError, match="names would collide"):
            dt.ds = xr.Dataset({"a": 0})  # type: ignore[assignment]

        dt.ds = xr.Dataset()  # type: ignore[assignment]

        new_ds = dt.to_dataset().assign(a=xr.DataArray(0))
        with pytest.raises(KeyError, match="names would collide"):
            dt.ds = new_ds  # type: ignore[assignment]


class TestGet: ...


class TestGetItem:
    def test_getitem_node(self):
        folder1: DataTree = DataTree(name="folder1")
        results: DataTree = DataTree(name="results", parent=folder1)
        highres: DataTree = DataTree(name="highres", parent=results)
        assert folder1["results"] is results
        assert folder1["results/highres"] is highres

    def test_getitem_self(self):
        dt: DataTree = DataTree()
        assert dt["."] is dt

    def test_getitem_single_data_variable(self):
        data = xr.Dataset({"temp": [0, 50]})
        results: DataTree = DataTree(name="results", data=data)
        xrt.assert_identical(results["temp"], data["temp"])

    def test_getitem_single_data_variable_from_node(self):
        data = xr.Dataset({"temp": [0, 50]})
        folder1: DataTree = DataTree(name="folder1")
        results: DataTree = DataTree(name="results", parent=folder1)
        DataTree(name="highres", parent=results, data=data)
        xrt.assert_identical(folder1["results/highres/temp"], data["temp"])

    def test_getitem_nonexistent_node(self):
        folder1: DataTree = DataTree(name="folder1")
        DataTree(name="results", parent=folder1)
        with pytest.raises(KeyError):
            folder1["results/highres"]

    def test_getitem_nonexistent_variable(self):
        data = xr.Dataset({"temp": [0, 50]})
        results: DataTree = DataTree(name="results", data=data)
        with pytest.raises(KeyError):
            results["pressure"]

    @pytest.mark.xfail(reason="Should be deprecated in favour of .subset")
    def test_getitem_multiple_data_variables(self):
        data = xr.Dataset({"temp": [0, 50], "p": [5, 8, 7]})
        results: DataTree = DataTree(name="results", data=data)
        xrt.assert_identical(results[["temp", "p"]], data[["temp", "p"]])  # type: ignore[index]

    @pytest.mark.xfail(
        reason="Indexing needs to return whole tree (GH https://github.com/xarray-contrib/datatree/issues/77)"
    )
    def test_getitem_dict_like_selection_access_to_dataset(self):
        data = xr.Dataset({"temp": [0, 50]})
        results: DataTree = DataTree(name="results", data=data)
        xrt.assert_identical(results[{"temp": 1}], data[{"temp": 1}])  # type: ignore[index]


class TestUpdate:
    def test_update(self):
        dt: DataTree = DataTree()
        dt.update({"foo": xr.DataArray(0), "a": DataTree()})
        expected = DataTree.from_dict({"/": xr.Dataset({"foo": 0}), "a": None})
        print(dt)
        print(dt.children)
        print(dt._children)
        print(dt["a"])
        print(expected)
        dtt.assert_equal(dt, expected)

    def test_update_new_named_dataarray(self):
        da = xr.DataArray(name="temp", data=[0, 50])
        folder1: DataTree = DataTree(name="folder1")
        folder1.update({"results": da})
        expected = da.rename("results")
        xrt.assert_equal(folder1["results"], expected)

    def test_update_doesnt_alter_child_name(self):
        dt: DataTree = DataTree()
        dt.update({"foo": xr.DataArray(0), "a": DataTree(name="b")})
        assert "a" in dt.children
        child = dt["a"]
        assert child.name == "a"

    def test_update_overwrite(self):
        actual = DataTree.from_dict({"a": DataTree(xr.Dataset({"x": 1}))})
        actual.update({"a": DataTree(xr.Dataset({"x": 2}))})

        expected = DataTree.from_dict({"a": DataTree(xr.Dataset({"x": 2}))})

        print(actual)
        print(expected)

        dtt.assert_equal(actual, expected)


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

    def test_copy_subtree(self):
        dt = DataTree.from_dict({"/level1/level2/level3": xr.Dataset()})

        actual = dt["/level1/level2"].copy()
        expected = DataTree.from_dict({"/level3": xr.Dataset()}, name="level2")

        dtt.assert_identical(actual, expected)

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
        john: DataTree = DataTree(name="john")
        mary: DataTree = DataTree(name="mary")
        john["mary"] = mary

        grafted_mary = john["mary"]
        assert grafted_mary.parent is john
        assert grafted_mary.name == "mary"

    def test_setitem_unnamed_child_node_becomes_named(self):
        john2: DataTree = DataTree(name="john2")
        john2["sonny"] = DataTree()
        assert john2["sonny"].name == "sonny"

    def test_setitem_new_grandchild_node(self):
        john: DataTree = DataTree(name="john")
        mary: DataTree = DataTree(name="mary", parent=john)
        rose: DataTree = DataTree(name="rose")
        john["mary/rose"] = rose

        grafted_rose = john["mary/rose"]
        assert grafted_rose.parent is mary
        assert grafted_rose.name == "rose"

    def test_grafted_subtree_retains_name(self):
        subtree: DataTree = DataTree(name="original_subtree_name")
        root: DataTree = DataTree(name="root")
        root["new_subtree_name"] = subtree  # noqa
        assert subtree.name == "original_subtree_name"

    def test_setitem_new_empty_node(self):
        john: DataTree = DataTree(name="john")
        john["mary"] = DataTree()
        mary = john["mary"]
        assert isinstance(mary, DataTree)
        xrt.assert_identical(mary.to_dataset(), xr.Dataset())

    def test_setitem_overwrite_data_in_node_with_none(self):
        john: DataTree = DataTree(name="john")
        mary: DataTree = DataTree(name="mary", parent=john, data=xr.Dataset())
        john["mary"] = DataTree()
        xrt.assert_identical(mary.to_dataset(), xr.Dataset())

        john.ds = xr.Dataset()  # type: ignore[assignment]
        with pytest.raises(ValueError, match="has no name"):
            john["."] = DataTree()

    @pytest.mark.xfail(reason="assigning Datasets doesn't yet create new nodes")
    def test_setitem_dataset_on_this_node(self):
        data = xr.Dataset({"temp": [0, 50]})
        results: DataTree = DataTree(name="results")
        results["."] = data
        xrt.assert_identical(results.to_dataset(), data)

    @pytest.mark.xfail(reason="assigning Datasets doesn't yet create new nodes")
    def test_setitem_dataset_as_new_node(self):
        data = xr.Dataset({"temp": [0, 50]})
        folder1: DataTree = DataTree(name="folder1")
        folder1["results"] = data
        xrt.assert_identical(folder1["results"].to_dataset(), data)

    @pytest.mark.xfail(reason="assigning Datasets doesn't yet create new nodes")
    def test_setitem_dataset_as_new_node_requiring_intermediate_nodes(self):
        data = xr.Dataset({"temp": [0, 50]})
        folder1: DataTree = DataTree(name="folder1")
        folder1["results/highres"] = data
        xrt.assert_identical(folder1["results/highres"].to_dataset(), data)

    def test_setitem_named_dataarray(self):
        da = xr.DataArray(name="temp", data=[0, 50])
        folder1: DataTree = DataTree(name="folder1")
        folder1["results"] = da
        expected = da.rename("results")
        xrt.assert_equal(folder1["results"], expected)

    def test_setitem_unnamed_dataarray(self):
        data = xr.DataArray([0, 50])
        folder1: DataTree = DataTree(name="folder1")
        folder1["results"] = data
        xrt.assert_equal(folder1["results"], data)

    def test_setitem_variable(self):
        var = xr.Variable(data=[0, 50], dims="x")
        folder1: DataTree = DataTree(name="folder1")
        folder1["results"] = var
        xrt.assert_equal(folder1["results"], xr.DataArray(var))

    def test_setitem_coerce_to_dataarray(self):
        folder1: DataTree = DataTree(name="folder1")
        folder1["results"] = 0
        xrt.assert_equal(folder1["results"], xr.DataArray(0))

    def test_setitem_add_new_variable_to_empty_node(self):
        results: DataTree = DataTree(name="results")
        results["pressure"] = xr.DataArray(data=[2, 3])
        assert "pressure" in results.ds
        results["temp"] = xr.Variable(data=[10, 11], dims=["x"])
        assert "temp" in results.ds

        # What if there is a path to traverse first?
        results_with_path: DataTree = DataTree(name="results")
        results_with_path["highres/pressure"] = xr.DataArray(data=[2, 3])
        assert "pressure" in results_with_path["highres"].ds
        results_with_path["highres/temp"] = xr.Variable(data=[10, 11], dims=["x"])
        assert "temp" in results_with_path["highres"].ds

    def test_setitem_dataarray_replace_existing_node(self):
        t = xr.Dataset({"temp": [0, 50]})
        results: DataTree = DataTree(name="results", data=t)
        p = xr.DataArray(data=[2, 3])
        results["pressure"] = p
        expected = t.assign(pressure=p)
        xrt.assert_identical(results.to_dataset(), expected)


class TestDictionaryInterface: ...


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

    def test_datatree_values(self):
        dat1: DataTree = DataTree(data=xr.Dataset({"a": 1}))
        expected: DataTree = DataTree()
        expected["a"] = dat1

        actual = DataTree.from_dict({"a": dat1})

        dtt.assert_identical(actual, expected)

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
        dt: DataTree = DataTree(data=ds)
        assert ds.identical(
            dt.ds
        )  # this only works because Dataset.identical doesn't check types
        assert isinstance(dt.ds, xr.Dataset)

    def test_immutability(self):
        # See issue https://github.com/xarray-contrib/datatree/issues/38
        dt: DataTree = DataTree(name="root", data=None)
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
        dt: DataTree = DataTree(data=ds)
        assert ds.mean().identical(dt.ds.mean())
        assert type(dt.ds.mean()) == xr.Dataset

    def test_arithmetic(self, create_test_datatree):
        dt = create_test_datatree()
        expected = create_test_datatree(modify=lambda ds: 10.0 * ds)["set1"]
        result = 10.0 * dt["set1"].ds
        assert result.identical(expected)

    def test_init_via_type(self):
        # from datatree GH issue https://github.com/xarray-contrib/datatree/issues/188
        # xarray's .weighted is unusual because it uses type() to create a Dataset/DataArray

        a = xr.DataArray(
            np.random.rand(3, 4, 10),
            dims=["x", "y", "time"],
            coords={"area": (["x", "y"], np.random.rand(3, 4))},
        ).to_dataset(name="data")
        dt: DataTree = DataTree(data=a)

        def weighted_mean(ds):
            return ds.weighted(ds.area).mean(["x", "y"])

        weighted_mean(dt.ds)


class TestAccess:
    def test_attribute_access(self, create_test_datatree):
        dt = create_test_datatree()

        # vars / coords
        for key in ["a", "set0"]:
            xrt.assert_equal(dt[key], getattr(dt, key))
            assert key in dir(dt)

        # dims
        xrt.assert_equal(dt["a"]["y"], getattr(dt.a, "y"))
        assert "y" in dir(dt["a"])

        # children
        for key in ["set1", "set2", "set3"]:
            dtt.assert_equal(dt[key], getattr(dt, key))
            assert key in dir(dt)

        # attrs
        dt.attrs["meta"] = "NASA"
        assert dt.attrs["meta"] == "NASA"
        assert "meta" in dir(dt)

    def test_ipython_key_completions(self, create_test_datatree):
        dt = create_test_datatree()
        key_completions = dt._ipython_key_completions_()

        node_keys = [node.path[1:] for node in dt.subtree]
        assert all(node_key in key_completions for node_key in node_keys)

        var_keys = list(dt.variables.keys())
        assert all(var_key in key_completions for var_key in var_keys)

    def test_operation_with_attrs_but_no_data(self):
        # tests bug from xarray-datatree GH262
        xs = xr.Dataset({"testvar": xr.DataArray(np.ones((2, 3)))})
        dt = DataTree.from_dict({"node1": xs, "node2": xs})
        dt.attrs["test_key"] = 1  # sel works fine without this line
        dt.sel(dim_0=0)


class TestRestructuring:
    def test_drop_nodes(self):
        sue = DataTree.from_dict({"Mary": None, "Kate": None, "Ashley": None})

        # test drop just one node
        dropped_one = sue.drop_nodes(names="Mary")
        assert "Mary" not in dropped_one.children

        # test drop multiple nodes
        dropped = sue.drop_nodes(names=["Mary", "Kate"])
        assert not set(["Mary", "Kate"]).intersection(set(dropped.children))
        assert "Ashley" in dropped.children

        # test raise
        with pytest.raises(KeyError, match="nodes {'Mary'} not present"):
            dropped.drop_nodes(names=["Mary", "Ashley"])

        # test ignore
        childless = dropped.drop_nodes(names=["Mary", "Ashley"], errors="ignore")
        assert childless.children == {}

    def test_assign(self):
        dt: DataTree = DataTree()
        expected = DataTree.from_dict({"/": xr.Dataset({"foo": 0}), "/a": None})

        # kwargs form
        result = dt.assign(foo=xr.DataArray(0), a=DataTree())
        dtt.assert_equal(result, expected)

        # dict form
        result = dt.assign({"foo": xr.DataArray(0), "a": DataTree()})
        dtt.assert_equal(result, expected)


class TestPipe:
    def test_noop(self, create_test_datatree):
        dt = create_test_datatree()

        actual = dt.pipe(lambda tree: tree)
        assert actual.identical(dt)

    def test_params(self, create_test_datatree):
        dt = create_test_datatree()

        def f(tree, **attrs):
            return tree.assign(arr_with_attrs=xr.Variable("dim0", [], attrs=attrs))

        attrs = {"x": 1, "y": 2, "z": 3}

        actual = dt.pipe(f, **attrs)
        assert actual["arr_with_attrs"].attrs == attrs

    def test_named_self(self, create_test_datatree):
        dt = create_test_datatree()

        def f(x, tree, y):
            tree.attrs.update({"x": x, "y": y})
            return tree

        attrs = {"x": 1, "y": 2}

        actual = dt.pipe((f, "tree"), **attrs)

        assert actual is dt and actual.attrs == attrs


class TestSubset:
    def test_match(self):
        # TODO is this example going to cause problems with case sensitivity?
        dt = DataTree.from_dict(
            {
                "/a/A": None,
                "/a/B": None,
                "/b/A": None,
                "/b/B": None,
            }
        )
        result = dt.match("*/B")
        expected = DataTree.from_dict(
            {
                "/a/B": None,
                "/b/B": None,
            }
        )
        dtt.assert_identical(result, expected)

    def test_filter(self):
        simpsons = DataTree.from_dict(
            d={
                "/": xr.Dataset({"age": 83}),
                "/Herbert": xr.Dataset({"age": 40}),
                "/Homer": xr.Dataset({"age": 39}),
                "/Homer/Bart": xr.Dataset({"age": 10}),
                "/Homer/Lisa": xr.Dataset({"age": 8}),
                "/Homer/Maggie": xr.Dataset({"age": 1}),
            },
            name="Abe",
        )
        expected = DataTree.from_dict(
            d={
                "/": xr.Dataset({"age": 83}),
                "/Herbert": xr.Dataset({"age": 40}),
                "/Homer": xr.Dataset({"age": 39}),
            },
            name="Abe",
        )
        elders = simpsons.filter(lambda node: node["age"].item() > 18)
        dtt.assert_identical(elders, expected)
