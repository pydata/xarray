import pytest
import xarray as xr

from datatree import DataTree


@pytest.fixture(scope="module")
def create_test_datatree():
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

    def _create_test_datatree(modify=lambda ds: ds):
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

    return _create_test_datatree


@pytest.fixture(scope="module")
def simple_datatree(create_test_datatree):
    """
    Invoke create_test_datatree fixture (callback).

    Returns a DataTree.
    """
    return create_test_datatree()
