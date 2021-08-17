import pytest

from anytree.node.exceptions import TreeError

import xarray as xr

from datatree import DataTree
from datatree.datatree import TreeNode, DatasetNode


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
        assert root.children == ()

    def test_parenting(self):
        john = TreeNode("john")
        mary = TreeNode("mary", parent=john)

        assert mary.parent == john
        assert mary in john.children

        with pytest.raises(KeyError, match="already has a child named"):
            TreeNode("mary", parent=john)

        with pytest.raises(TreeError, match="not of type 'NodeMixin'"):
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
        assert mary in john.children
        assert kate in john.children
        assert mary.parent is john
        assert kate.parent is john

    def test_disown_child(self):
        john = TreeNode("john")
        mary = TreeNode("mary", parent=john)
        mary.parent = None
        assert mary not in john.children

    def test_add_child(self):
        john = TreeNode("john")
        kate = TreeNode("kate")
        john.add_child(kate)
        assert kate in john.children
        assert kate.parent is john
        with pytest.raises(KeyError, match="already has a child named"):
            john.add_child(TreeNode("kate"))

    def test_assign_children(self):
        john = TreeNode("john")
        jack = TreeNode("jack")
        jill = TreeNode("jill")

        john.children = (jack, jill)
        assert jack in john.children
        assert jack.parent is john
        assert jill in john.children
        assert jill.parent is john

        evil_twin_jill = TreeNode("jill")
        with pytest.raises(KeyError, match="already has a child named"):
            john.children = (jack, jill, evil_twin_jill)

    def test_sibling_relationships(self):
        mary = TreeNode("mary")
        kate = TreeNode("kate")
        ashley = TreeNode("ashley")
        john = TreeNode("john", children=[mary, kate, ashley])
        assert mary in kate.siblings
        assert ashley in kate.siblings
        assert kate not in kate.siblings
        with pytest.raises(AttributeError):
            kate.siblings = john

    @pytest.mark.xfail
    def test_walking_parents(self):
        raise NotImplementedError

    @pytest.mark.xfail
    def test_walking_children(self):
        raise NotImplementedError

    @pytest.mark.xfail
    def test_adoption(self):
        raise NotImplementedError


class TestTreeCreation:
    def test_empty(self):
        dt = DataTree()
        assert dt.name == "root"
        assert dt.parent is None
        assert dt.children is ()
        assert dt.ds is None

    def test_data_in_root(self):
        dt = DataTree({"root": xr.Dataset()})
        print(dt.name)
        assert dt.name == "root"
        assert dt.parent is None
        assert dt.children is ()
        assert dt.ds is xr.Dataset()

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
    def test_render_nodetree(self):
        mary = TreeNode("mary")
        kate = TreeNode("kate")
        john = TreeNode("john", children=[mary, kate])
        sam = TreeNode("Sam", parent=mary)
        ben = TreeNode("Ben", parent=mary)
        john.render()
        assert False

    def test_render_datatree(self):
        dt = create_test_datatree()
        dt.render()


class TestMethodInheritance:
    ...


class TestUFuncs:
    ...


class TestIO:
    ...
