import xarray.testing as xrt
from xarray import Dataset
from xarray.tests import requires_datatree


@requires_datatree
def test_import_datatree():
    """Just test importing datatree package from xarray-contrib repo"""
    from xarray import DataTree

    DataTree()


@requires_datatree
def test_to_datatree():
    from xarray import DataTree

    ds = Dataset({"a": ("x", [1, 2, 3])})
    dt = ds.to_datatree(node_name="group1")

    assert isinstance(dt, DataTree)
    assert dt.name == "group1"
    xrt.assert_identical(dt.to_dataset(), ds)

    da = ds["a"]
    dt = da.to_datatree(node_name="group1")

    assert isinstance(dt, DataTree)
    assert dt.name == "group1"
    xrt.assert_identical(dt["a"], da)


@requires_datatree
def test_binary_ops():
    import datatree.testing as dtt

    from xarray import DataTree

    ds1 = Dataset({"a": [5], "b": [3]})
    ds2 = Dataset({"x": [0.1, 0.2], "y": [10, 20]})
    dt = DataTree(data=ds1)
    DataTree(name="subnode", data=ds2, parent=dt)
    other_ds = Dataset({"z": ("z", [0.1, 0.2])})

    expected = DataTree(data=ds1 * other_ds)
    DataTree(name="subnode", data=ds2 * other_ds, parent=expected)

    result = dt * other_ds
    dtt.assert_equal(result, expected)

    # This ordering won't work unless xarray.Dataset defers to DataTree.
    # See https://github.com/xarray-contrib/datatree/issues/146
    result = other_ds * dt
    dtt.assert_equal(result, expected)
