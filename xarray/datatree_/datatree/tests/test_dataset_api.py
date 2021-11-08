import numpy as np
import xarray as xr
from xarray.testing import assert_equal

from datatree import DataNode

from .test_datatree import assert_tree_equal, create_test_datatree


class TestDSMethodInheritance:
    def test_dataset_method(self):
        # test root
        da = xr.DataArray(name="a", data=[1, 2, 3], dims="x")
        dt = DataNode("root", data=da)
        expected_ds = da.to_dataset().isel(x=1)
        result_ds = dt.isel(x=1).ds
        assert_equal(result_ds, expected_ds)

        # test descendant
        DataNode("results", parent=dt, data=da)
        result_ds = dt.isel(x=1)["results"].ds
        assert_equal(result_ds, expected_ds)

    def test_reduce_method(self):
        # test root
        da = xr.DataArray(name="a", data=[False, True, False], dims="x")
        dt = DataNode("root", data=da)
        expected_ds = da.to_dataset().any()
        result_ds = dt.any().ds
        assert_equal(result_ds, expected_ds)

        # test descendant
        DataNode("results", parent=dt, data=da)
        result_ds = dt.any()["results"].ds
        assert_equal(result_ds, expected_ds)

    def test_nan_reduce_method(self):
        # test root
        da = xr.DataArray(name="a", data=[1, 2, 3], dims="x")
        dt = DataNode("root", data=da)
        expected_ds = da.to_dataset().mean()
        result_ds = dt.mean().ds
        assert_equal(result_ds, expected_ds)

        # test descendant
        DataNode("results", parent=dt, data=da)
        result_ds = dt.mean()["results"].ds
        assert_equal(result_ds, expected_ds)

    def test_cum_method(self):
        # test root
        da = xr.DataArray(name="a", data=[1, 2, 3], dims="x")
        dt = DataNode("root", data=da)
        expected_ds = da.to_dataset().cumsum()
        result_ds = dt.cumsum().ds
        assert_equal(result_ds, expected_ds)

        # test descendant
        DataNode("results", parent=dt, data=da)
        result_ds = dt.cumsum()["results"].ds
        assert_equal(result_ds, expected_ds)


class TestOps:
    def test_binary_op_on_int(self):
        ds1 = xr.Dataset({"a": [5], "b": [3]})
        ds2 = xr.Dataset({"x": [0.1, 0.2], "y": [10, 20]})
        dt = DataNode("root", data=ds1)
        DataNode("subnode", data=ds2, parent=dt)

        expected_root = DataNode("root", data=ds1 * 5)
        expected_descendant = DataNode("subnode", data=ds2 * 5, parent=expected_root)
        result = dt * 5

        assert_equal(result.ds, expected_root.ds)
        assert_equal(result["subnode"].ds, expected_descendant.ds)

    def test_binary_op_on_dataset(self):
        ds1 = xr.Dataset({"a": [5], "b": [3]})
        ds2 = xr.Dataset({"x": [0.1, 0.2], "y": [10, 20]})
        dt = DataNode("root", data=ds1)
        DataNode("subnode", data=ds2, parent=dt)
        other_ds = xr.Dataset({"z": ("z", [0.1, 0.2])})

        expected_root = DataNode("root", data=ds1 * other_ds)
        expected_descendant = DataNode(
            "subnode", data=ds2 * other_ds, parent=expected_root
        )
        result = dt * other_ds

        assert_equal(result.ds, expected_root.ds)
        assert_equal(result["subnode"].ds, expected_descendant.ds)

    def test_binary_op_on_datatree(self):
        ds1 = xr.Dataset({"a": [5], "b": [3]})
        ds2 = xr.Dataset({"x": [0.1, 0.2], "y": [10, 20]})
        dt = DataNode("root", data=ds1)
        DataNode("subnode", data=ds2, parent=dt)

        expected_root = DataNode("root", data=ds1 * ds1)
        expected_descendant = DataNode("subnode", data=ds2 * ds2, parent=expected_root)
        result = dt * dt

        assert_equal(result.ds, expected_root.ds)
        assert_equal(result["subnode"].ds, expected_descendant.ds)


class TestUFuncs:
    def test_root(self):
        da = xr.DataArray(name="a", data=[1, 2, 3])
        dt = DataNode("root", data=da)
        expected_ds = np.sin(da.to_dataset())
        result_ds = np.sin(dt).ds
        assert_equal(result_ds, expected_ds)

    def test_descendants(self):
        da = xr.DataArray(name="a", data=[1, 2, 3])
        dt = DataNode("root")
        DataNode("results", parent=dt, data=da)
        expected_ds = np.sin(da.to_dataset())
        result_ds = np.sin(dt)["results"].ds
        assert_equal(result_ds, expected_ds)

    def test_tree(self):
        dt = create_test_datatree()
        expected = create_test_datatree(modify=lambda ds: np.sin(ds))
        result_tree = np.sin(dt)
        assert_tree_equal(result_tree, expected)
