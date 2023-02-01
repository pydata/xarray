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
