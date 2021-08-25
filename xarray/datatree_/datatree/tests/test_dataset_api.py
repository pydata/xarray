import numpy as np
import pytest
import xarray as xr
from test_datatree import create_test_datatree
from xarray.testing import assert_equal

from datatree import DataNode, DataTree, map_over_subtree


class TestMapOverSubTree:
    def test_map_over_subtree(self):
        dt = create_test_datatree()

        @map_over_subtree
        def times_ten(ds):
            return 10.0 * ds

        result_tree = times_ten(dt)

        # TODO write an assert_tree_equal function
        for (
            result_node,
            original_node,
        ) in zip(result_tree.subtree, dt.subtree):
            assert isinstance(result_node, DataTree)

            if original_node.has_data:
                assert_equal(result_node.ds, original_node.ds * 10.0)
            else:
                assert not result_node.has_data

    def test_map_over_subtree_with_args_and_kwargs(self):
        dt = create_test_datatree()

        @map_over_subtree
        def multiply_then_add(ds, times, add=0.0):
            return times * ds + add

        result_tree = multiply_then_add(dt, 10.0, add=2.0)

        for (
            result_node,
            original_node,
        ) in zip(result_tree.subtree, dt.subtree):
            assert isinstance(result_node, DataTree)

            if original_node.has_data:
                assert_equal(result_node.ds, (original_node.ds * 10.0) + 2.0)
            else:
                assert not result_node.has_data

    def test_map_over_subtree_method(self):
        dt = create_test_datatree()

        def multiply_then_add(ds, times, add=0.0):
            return times * ds + add

        result_tree = dt.map_over_subtree(multiply_then_add, 10.0, add=2.0)

        for (
            result_node,
            original_node,
        ) in zip(result_tree.subtree, dt.subtree):
            assert isinstance(result_node, DataTree)

            if original_node.has_data:
                assert_equal(result_node.ds, (original_node.ds * 10.0) + 2.0)
            else:
                assert not result_node.has_data

    @pytest.mark.xfail
    def test_map_over_subtree_inplace(self):
        raise NotImplementedError


class TestDSProperties:
    def test_properties(self):
        da_a = xr.DataArray(name="a", data=[0, 2], dims=["x"])
        da_b = xr.DataArray(name="b", data=[5, 6, 7], dims=["y"])
        ds = xr.Dataset({"a": da_a, "b": da_b})
        dt = DataNode("root", data=ds)

        assert dt.attrs == dt.ds.attrs
        assert dt.encoding == dt.ds.encoding
        assert dt.dims == dt.ds.dims
        assert dt.sizes == dt.ds.sizes
        assert dt.variables == dt.ds.variables

    def test_no_data_no_properties(self):
        dt = DataNode("root", data=None)
        with pytest.raises(AttributeError):
            dt.attrs
        with pytest.raises(AttributeError):
            dt.encoding
        with pytest.raises(AttributeError):
            dt.dims
        with pytest.raises(AttributeError):
            dt.sizes
        with pytest.raises(AttributeError):
            dt.variables


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
    @pytest.mark.xfail
    def test_binary_op(self):
        ds1 = xr.Dataset({"a": [5], "b": [3]})
        ds2 = xr.Dataset({"x": [0.1, 0.2], "y": [10, 20]})
        dt = DataNode("root", data=ds1)
        DataNode("subnode", data=ds2, parent=dt)

        expected_root = DataNode("root", data=ds1 * ds1)
        expected_descendant = DataNode("subnode", data=ds2 * ds2, parent=expected_root)
        result = dt * dt

        assert_equal(result.ds, expected_root.ds)
        assert_equal(result["subnode"].ds, expected_descendant.ds)


@pytest.mark.xfail
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
