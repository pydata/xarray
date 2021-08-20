import pytest

import xarray as xr

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
    ...


class TestSetItems:
    def test_set_dataset(self):
        ...

    def test_set_named_dataarray(self):
        ...

    def test_set_unnamed_dataarray(self):
        ...

    def test_set_node(self):
        ...


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


@pytest.mark.xfail
class TestRepr:
    def test_render_datatree(self):
        dt = create_test_datatree()
        dt.render()


class TestPropertyInheritance:
    ...


class TestMethodInheritance:
    ...


class TestUFuncs:
    ...


class TestIO:
    ...
