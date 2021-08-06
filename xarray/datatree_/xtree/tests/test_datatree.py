import pytest

import xarray as xr

from xtree.datatree import TreeNode, DatasetNode, DataTree


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
    root = DataTree(data_objects={'/': root_data})
    set1 = DatasetNode(name="set1", parent=root, data=set1_data)
    set1_set1 = DatasetNode(name="set1", parent=set1)
    set1_set2 = DatasetNode(name="set1", parent=set1)
    set2 = DatasetNode(name="set1", parent=root, data=set2_data)
    set2_set1 = DatasetNode(name="set1", parent=set2)
    set3 = DatasetNode(name="set3", parent=root)

    return root


class TestTreeNodes:
    def test_lonely(self):
        root = TreeNode("/")
        assert root.name == "/"
        assert root.parent is None
        assert root.children == []

    def test_parenting(self):
        john = TreeNode("john")
        mary = TreeNode("mary", parent=john)

        assert mary.parent == john
        assert mary in john.children

        with pytest.raises(KeyError, match="already has a child node named"):
            TreeNode("mary", parent=john)

        with pytest.raises(TypeError, match="object is not a valid parent"):
            mary.parent = "apple"

    def test_parent_swap(self):
        john = TreeNode("john")
        mary = TreeNode("mary", parent=john)

        steve = TreeNode("steve")
        mary.parent = steve
        assert mary in steve.children
        assert mary not in john.children

    def test_multi_child_family(self):
        mary = TreeNode("mary")
        kate = TreeNode("kate")
        john = TreeNode("john", children=[mary, kate])


    def test_walking_parents(self):
        ...

    def test_walking_children(self):
        ...

    def test_adoption(self):
        ...


class TestTreePlanting:
    def test_empty(self):
        dt = DataTree()
        root = DataTree()

    def test_one_layer(self):
        dt = DataTree({"run1": xr.Dataset(), "run2": xr.DataArray()})

    def test_two_layers(self):
        dt = DataTree({"highres/run1": xr.Dataset(), "highres/run2": xr.Dataset()})

        dt = DataTree({"highres/run1": xr.Dataset(), "lowres/run1": xr.Dataset()})
        assert dt.children == ...

    def test_full(self):
        dt = create_test_datatree()
        print(dt)
        assert False


class TestBrowsing:
    ...


class TestRestructuring:
    ...


class TestRepr:
    ...


class TestIO:
    ...


class TestMethodInheritance:
    ...


class TestUFuncs:
    ...
