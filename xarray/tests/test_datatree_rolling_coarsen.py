from __future__ import annotations

import numpy as np
import pytest

import xarray as xr
from xarray.core.datatree import DataTree
from xarray.testing import assert_identical
from xarray.tests import requires_bottleneck


@pytest.fixture
def sample_tree() -> DataTree:
    """Create a multi-node hierarchical tree with shared and node-specific dimensions."""
    ds_root = xr.Dataset(
        {"a": ("x", np.arange(10, dtype=float))},
        coords={"x": np.arange(10)},
    )
    ds_child1 = xr.Dataset(
        {"b": ("x", np.arange(10, 20, dtype=float))},
        coords={"x": np.arange(10)},
    )
    # Child with different dimension
    ds_child2 = xr.Dataset(
        {"c": ("y", np.arange(5, dtype=float))},
        coords={"y": np.arange(5)},
    )
    return DataTree.from_dict(
        {
            "/": ds_root,
            "/group1": ds_child1,
            "/group2": ds_child2,
        }
    )


class TestDataTreeRolling:
    def test_rolling_mean(self, sample_tree: DataTree) -> None:
        dt = sample_tree
        res = dt.rolling(x=3).mean()

        # Nodes with 'x' have rolling applied
        expected_root = dt["/"].to_dataset().rolling(x=3).mean()
        assert_identical(res["/"].to_dataset(), expected_root)

        expected_child1 = dt["/group1"].to_dataset().rolling(x=3).mean()
        assert_identical(res["/group1"].to_dataset(), expected_child1)

        # Node without 'x' remains untouched
        assert_identical(res["/group2"].to_dataset(), dt["/group2"].to_dataset())

    def test_rolling_reductions(self, sample_tree: DataTree) -> None:
        dt = sample_tree
        r = dt.rolling(x=2)

        # Sum
        res_sum = r.sum()
        expected_sum = dt["/"].to_dataset().rolling(x=2).sum()
        assert_identical(res_sum["/"].to_dataset(), expected_sum)

        # Std & Var
        res_std = r.std()
        expected_std = dt["/"].to_dataset().rolling(x=2).std()
        assert_identical(res_std["/"].to_dataset(), expected_std)

        res_var = r.var()
        expected_var = dt["/"].to_dataset().rolling(x=2).var()
        assert_identical(res_var["/"].to_dataset(), expected_var)

        # Min & Max & Median
        res_min = r.min()
        expected_min = dt["/"].to_dataset().rolling(x=2).min()
        assert_identical(res_min["/"].to_dataset(), expected_min)

        res_max = r.max()
        expected_max = dt["/"].to_dataset().rolling(x=2).max()
        assert_identical(res_max["/"].to_dataset(), expected_max)

        res_median = r.median()
        expected_median = dt["/"].to_dataset().rolling(x=2).median()
        assert_identical(res_median["/"].to_dataset(), expected_median)

        # Count & Prod
        res_count = r.count()
        expected_count = dt["/"].to_dataset().rolling(x=2).count()
        assert_identical(res_count["/"].to_dataset(), expected_count)

        res_prod = r.prod()
        expected_prod = dt["/"].to_dataset().rolling(x=2).prod()
        assert_identical(res_prod["/"].to_dataset(), expected_prod)

    def test_rolling_reduce(self, sample_tree: DataTree) -> None:
        dt = sample_tree
        res = dt.rolling(x=3).reduce(np.nanmax)
        expected = dt["/"].to_dataset().rolling(x=3).reduce(np.nanmax)
        assert_identical(res["/"].to_dataset(), expected)

    def test_rolling_center_min_periods(self, sample_tree: DataTree) -> None:
        dt = sample_tree
        res = dt.rolling(x=3, center=True, min_periods=1).mean()
        expected = dt["/"].to_dataset().rolling(x=3, center=True, min_periods=1).mean()
        assert_identical(res["/"].to_dataset(), expected)

    def test_rolling_construct(self, sample_tree: DataTree) -> None:
        dt = sample_tree
        res = dt.rolling(x=3).construct(x="window")
        expected = dt["/"].to_dataset().rolling(x=3).construct(x="window")
        assert_identical(res["/"].to_dataset(), expected)
        # Check window dimension exists
        assert "window" in res["/"].to_dataset().dims
        assert "window" in res["/group1"].to_dataset().dims
        # Group2 lacks 'x', so unaffected
        assert "window" not in res["/group2"].to_dataset().dims

    def test_rolling_repr(self, sample_tree: DataTree) -> None:
        r = sample_tree.rolling(x=5)
        repr_str = repr(r)
        assert "DataTreeRolling" in repr_str
        assert "x: 5" in repr_str


class TestDataTreeCoarsen:
    def test_coarsen_mean(self, sample_tree: DataTree) -> None:
        dt = sample_tree
        res = dt.coarsen(x=2).mean()

        expected_root = dt["/"].to_dataset().coarsen(x=2).mean()
        assert_identical(res["/"].to_dataset(), expected_root)

        expected_child1 = dt["/group1"].to_dataset().coarsen(x=2).mean()
        assert_identical(res["/group1"].to_dataset(), expected_child1)

        # Data variable 'c' on node without 'x' unchanged
        assert_identical(
            res["/group2"].to_dataset()["c"], dt["/group2"].to_dataset()["c"]
        )

    def test_coarsen_reductions(self, sample_tree: DataTree) -> None:
        dt = sample_tree
        c = dt.coarsen(x=2)

        # Sum
        assert_identical(
            c.sum()["/"].to_dataset(), dt["/"].to_dataset().coarsen(x=2).sum()
        )
        # Max
        assert_identical(
            c.max()["/"].to_dataset(), dt["/"].to_dataset().coarsen(x=2).max()
        )
        # Min
        assert_identical(
            c.min()["/"].to_dataset(), dt["/"].to_dataset().coarsen(x=2).min()
        )
        # Median
        assert_identical(
            c.median()["/"].to_dataset(), dt["/"].to_dataset().coarsen(x=2).median()
        )

    def test_coarsen_boundary(self, sample_tree: DataTree) -> None:
        dt = sample_tree
        # 10 is not divisible by 3 -> boundary="trim"
        res = dt.coarsen(x=3, boundary="trim").mean()
        expected = dt["/"].to_dataset().coarsen(x=3, boundary="trim").mean()
        assert_identical(res["/"].to_dataset(), expected)

    def test_coarsen_construct(self, sample_tree: DataTree) -> None:
        dt = sample_tree
        res = dt.coarsen(x=2).construct(x=("x_new", "x_window"))
        expected = dt["/"].to_dataset().coarsen(x=2).construct(x=("x_new", "x_window"))
        assert_identical(res["/"].to_dataset(), expected)
        assert "x_window" in res["/"].to_dataset().dims
        # Data variable 'c' on child node without 'x' remains untouched
        assert_identical(
            res["/group2"].to_dataset()["c"], dt["/group2"].to_dataset()["c"]
        )
        assert res["/group2"].to_dataset()["c"].dims == ("y",)

    def test_coarsen_repr(self, sample_tree: DataTree) -> None:
        c = sample_tree.coarsen(x=2)
        repr_str = repr(c)
        assert "DataTreeCoarsen" in repr_str
        assert "x: 2" in repr_str


class TestDataTreeAdvancedMethods:
    def test_diff(self, sample_tree: DataTree) -> None:
        dt = sample_tree
        res = dt.diff(dim="x", n=1)
        expected = dt["/"].to_dataset().diff(dim="x", n=1)
        assert_identical(res["/"].to_dataset(), expected)
        # Data variable 'c' on node without 'x' untouched
        assert_identical(
            res["/group2"].to_dataset()["c"], dt["/group2"].to_dataset()["c"]
        )

    def test_expand_dims(self, sample_tree: DataTree) -> None:
        dt = sample_tree
        res = dt.expand_dims({"new_dim": 3})
        for node in res.subtree:
            assert "new_dim" in node.dims
            assert node.sizes["new_dim"] == 3

    def test_astype(self, sample_tree: DataTree) -> None:
        dt = sample_tree
        res = dt.astype(np.int32)
        assert res["/"].to_dataset()["a"].dtype == np.int32
        assert res["/group1"].to_dataset()["b"].dtype == np.int32
        assert res["/group2"].to_dataset()["c"].dtype == np.int32

    @requires_bottleneck
    def test_directional_fill(self) -> None:
        ds = xr.Dataset({"data": ("x", [np.nan, 1.0, np.nan, 2.0, np.nan])})
        dt = DataTree.from_dict({"/": ds})

        # bfill
        res_bfill = dt.bfill(dim="x")
        expected_bfill = ds.bfill(dim="x")
        assert_identical(res_bfill["/"].to_dataset(), expected_bfill)

        # ffill
        res_ffill = dt.ffill(dim="x")
        expected_ffill = ds.ffill(dim="x")
        assert_identical(res_ffill["/"].to_dataset(), expected_ffill)

    def test_interpolate_na(self) -> None:
        ds = xr.Dataset(
            {"data": ("x", [0.0, np.nan, 2.0, np.nan, 4.0])},
            coords={"x": np.arange(5)},
        )
        dt = DataTree.from_dict({"/": ds})
        res = dt.interpolate_na(dim="x", method="linear")
        expected = ds.interpolate_na(dim="x", method="linear")
        assert_identical(res["/"].to_dataset(), expected)
        assert np.array_equal(
            res["/"].to_dataset()["data"].values, [0.0, 1.0, 2.0, 3.0, 4.0]
        )

    def test_stack_unstack(self) -> None:
        ds = xr.Dataset(
            {"temp": (("x", "y"), np.ones((3, 4)))},
            coords={"x": [1, 2, 3], "y": ["a", "b", "c", "d"]},
        )
        dt = DataTree.from_dict({"/": ds})

        stacked = dt.stack(z=("x", "y"))
        assert "z" in stacked["/"].to_dataset().dims
        assert stacked["/"].to_dataset().sizes["z"] == 12

        unstacked = stacked.unstack("z")
        assert "z" not in unstacked["/"].to_dataset().dims
        assert_identical(unstacked["/"].to_dataset(), ds)
