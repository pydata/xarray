import pytest

import xarray as xr
from xarray.testing import assert_identical

from anytree.resolver import ChildResolverError

from datatree import DataTree
from datatree.datatree import DatasetNode


def create_test_datatree():
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
    |   |       b        (x) int64 'foo', 'bar'
    |   |-- set1
    |-- set3
    |-- <xarray.Dataset>
    |   Dimensions:  (x: 2, y: 3)
    |   Data variables:
    |       a        (y) int64 6, 7, 8
    |       set1     (x) int64 9, 10

    The structure has deliberately repeated names of tags, variables, and
    dimensions in order to better check for bugs caused by name conflicts.
    """
    set1_data = xr.Dataset({'a': 0, 'b': 1})
    set2_data = xr.Dataset({'a': ('x', [2, 3]), 'b': ('x', ['foo', 'bar'])})
    root_data = xr.Dataset({'a': ('y', [6, 7, 8]), 'set1': ('x', [9, 10])})

    # Avoid using __init__ so we can independently test it
    root = DataTree(data_objects={'root': root_data})
    set1 = DatasetNode(name="set1", parent=root, data=set1_data)
    set1_set1 = DatasetNode(name="set1", parent=set1)
    set1_set2 = DatasetNode(name="set2", parent=set1)
    set2 = DatasetNode(name="set2", parent=root, data=set2_data)
    set2_set1 = DatasetNode(name="set1", parent=set2)
    set3 = DatasetNode(name="set3", parent=root)

    return root


class TestStoreDatasets:
    def test_create_datanode(self):
        dat = xr.Dataset({'a': 0})
        john = DatasetNode("john", data=dat)
        assert john.ds is dat
        with pytest.raises(TypeError):
            DatasetNode("mary", parent=john, data="junk")

    def test_set_data(self):
        john = DatasetNode("john")
        dat = xr.Dataset({'a': 0})
        john.ds = dat
        assert john.ds is dat
        with pytest.raises(TypeError):
            john.ds = "junk"

    def test_has_data(self):
        john = DatasetNode("john", data=xr.Dataset({'a': 0}))
        assert john.has_data

        john = DatasetNode("john", data=None)
        assert not john.has_data


class TestGetItems:
    def test_get_node(self):
        folder1 = DatasetNode("folder1")
        results = DatasetNode("results", parent=folder1)
        highres = DatasetNode("highres", parent=results)
        assert folder1["results"] is results
        assert folder1["results/highres"] is highres
        assert folder1[("results", "highres")] is highres

    def test_get_single_data_variable(self):
        data = xr.Dataset({"temp": [0, 50]})
        results = DatasetNode("results", data=data)
        assert_identical(results["temp"], data["temp"])

    def test_get_single_data_variable_from_node(self):
        data = xr.Dataset({"temp": [0, 50]})
        folder1 = DatasetNode("folder1")
        results = DatasetNode("results", parent=folder1)
        highres = DatasetNode("highres", parent=results, data=data)
        assert_identical(folder1["results/highres/temp"], data["temp"])
        assert_identical(folder1[("results", "highres", "temp")], data["temp"])

    def test_get_nonexistent_node(self):
        folder1 = DatasetNode("folder1")
        results = DatasetNode("results", parent=folder1)
        with pytest.raises(ChildResolverError):
            folder1["results/highres"]

    def test_get_nonexistent_variable(self):
        data = xr.Dataset({"temp": [0, 50]})
        results = DatasetNode("results", data=data)
        with pytest.raises(ChildResolverError):
            results["pressure"]

    def test_get_multiple_data_variables(self):
        data = xr.Dataset({"temp": [0, 50], "p": [5, 8, 7]})
        results = DatasetNode("results", data=data)
        assert_identical(results[['temp', 'p']], data[['temp', 'p']])

    def test_dict_like_selection_access_to_dataset(self):
        data = xr.Dataset({"temp": [0, 50]})
        results = DatasetNode("results", data=data)
        assert_identical(results[{'temp': 1}], data[{'temp': 1}])


class TestSetItems:
    # TODO test tuple-style access too
    def test_set_new_child_node(self):
        john = DatasetNode("john")
        mary = DatasetNode("mary")
        john['/'] = mary
        assert john['mary'] is mary

    def test_set_new_grandchild_node(self):
        john = DatasetNode("john")
        mary = DatasetNode("mary", parent=john)
        rose = DatasetNode("rose")
        john['/mary/'] = rose
        assert john['mary/rose'] is rose

    def test_set_dataset_on_this_node(self):
        data = xr.Dataset({"temp": [0, 50]})
        results = DatasetNode("results")
        results['/'] = data
        assert results.ds is data

    def test_set_dataset_as_new_node(self):
        data = xr.Dataset({"temp": [0, 50]})
        folder1 = DatasetNode("folder1")
        folder1['results'] = data
        assert folder1['results'].ds is data

    def test_set_dataset_as_new_node_requiring_intermediate_nodes(self):
        data = xr.Dataset({"temp": [0, 50]})
        folder1 = DatasetNode("folder1")
        folder1['results/highres'] = data
        assert folder1['results/highres'].ds is data

    def test_set_named_dataarray_as_new_node(self):
        data = xr.DataArray(name='temp', data=[0, 50])
        folder1 = DatasetNode("folder1")
        folder1['results'] = data
        assert_identical(folder1['results'].ds, data.to_dataset())

    def test_set_unnamed_dataarray(self):
        data = xr.DataArray([0, 50])
        folder1 = DatasetNode("folder1")
        with pytest.raises(ValueError, match="unable to convert"):
            folder1['results'] = data

    def test_add_new_variable_to_empty_node(self):
        results = DatasetNode("results")
        results['/'] = xr.DataArray(name='pressure', data=[2, 3])
        assert 'pressure' in results.ds

        # What if there is a path to traverse first?
        results = DatasetNode("results")
        results['/highres/'] = xr.DataArray(name='pressure', data=[2, 3])
        assert 'pressure' in results['highres'].ds

    def test_dataarray_replace_existing_node(self):
        t = xr.Dataset({"temp": [0, 50]})
        results = DatasetNode("results", data=t)
        p = xr.DataArray(name='pressure', data=[2, 3])
        results['/'] = p
        assert_identical(results.ds, p.to_dataset())


class TestTreeCreation:
    def test_empty(self):
        dt = DataTree()
        assert dt.name == "root"
        assert dt.parent is None
        assert dt.children is ()
        assert dt.ds is None

    def test_data_in_root(self):
        dat = xr.Dataset()
        dt = DataTree({"root": dat})
        assert dt.name == "root"
        assert dt.parent is None
        assert dt.children is ()
        assert dt.ds is dat

    def test_one_layer(self):
        dat1, dat2 = xr.Dataset({'a': 1}), xr.Dataset({'b': 2})
        dt = DataTree({"run1": dat1, "run2": dat2})
        assert dt.ds is None
        assert dt['run1'].ds is dat1
        assert dt['run2'].ds is dat2

    def test_two_layers(self):
        dat1, dat2 = xr.Dataset({'a': 1}), xr.Dataset({'a': [1, 2]})
        dt = DataTree({"highres/run": dat1, "lowres/run": dat2})
        assert 'highres' in [c.name for c in dt.children]
        assert 'lowres' in [c.name for c in dt.children]
        highres_run = dt.get_node('highres/run')
        assert highres_run.ds is dat1

    def test_full(self):
        dt = create_test_datatree()
        paths = list(node.pathstr for node in dt.subtree_nodes)
        assert paths == ['root', 'root/set1', 'root/set1/set1',
                                              'root/set1/set2',
                                 'root/set2', 'root/set2/set1',
                                 'root/set3']


class TestBrowsing:
    ...


class TestRestructuring:
    ...


class TestRepr:
    def test_print_empty_node(self):
        dt = DatasetNode('root')
        printout = dt.__str__()
        assert printout == "DatasetNode('root')"

    def test_print_node_with_data(self):
        dat = xr.Dataset({'a': [0, 2]})
        dt = DatasetNode('root', data=dat)
        printout = dt.__str__()
        expected = ["DatasetNode('root')",
                    "Dimensions",
                    "Coordinates",
                    "a",
                    "Data variables",
                    "*empty*"]
        for expected_line, printed_line in zip(expected, printout.splitlines()):
            assert expected_line in printed_line

    def test_nested_node(self):
        dat = xr.Dataset({'a': [0, 2]})
        root = DatasetNode('root')
        DatasetNode('results', data=dat, parent=root)
        printout = root.__str__()
        assert printout.splitlines()[2].startswith("    ")

    def test_print_datatree(self):
        dt = create_test_datatree()
        print(dt)
        # TODO work out how to test something complex like this

    def test_repr_of_node_with_data(self):
        dat = xr.Dataset({'a': [0, 2]})
        dt = DatasetNode('root', data=dat)
        assert "Coordinates" in repr(dt)


class TestPropertyInheritance:
    ...


class TestMethodInheritance:
    ...


class TestUFuncs:
    ...


class TestIO:
    ...
