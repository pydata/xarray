from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xarray import DataArray, Dataset, DataTree
from xarray.testing import assert_identical


@pytest.fixture
def sample_time_tree() -> DataTree:
    times = pd.date_range("2020-01-01", periods=10, freq="12h")
    ds_root = Dataset(
        {"a": (("time",), np.arange(10, dtype=float))},
        coords={"time": times},
    )
    ds_child = Dataset(
        {"b": (("time",), np.arange(10, 20, dtype=float))},
        coords={"time": times},
    )
    ds_no_time = Dataset(
        {"c": (("x",), np.arange(5, dtype=float))},
        coords={"x": np.arange(5)},
    )
    dt = DataTree.from_dict({
        "/": ds_root,
        "/child": ds_child,
        "/no_time": ds_no_time,
    })
    return dt


class TestDataTreeResample:
    def test_resample_mean(self, sample_time_tree: DataTree) -> None:
        dt = sample_time_tree
        res = dt.resample(time="1D").mean()

        expected_root = dt["/"].to_dataset().resample(time="1D").mean()
        assert_identical(res["/"].to_dataset(), expected_root)

        expected_child = dt["/child"].to_dataset().resample(time="1D").mean()
        assert_identical(res["/child"].to_dataset(), expected_child)

        # Data variable 'c' on node without time dimension remains untouched
        assert_identical(res["/no_time"].to_dataset()["c"], dt["/no_time"].to_dataset()["c"])

    def test_resample_reductions(self, sample_time_tree: DataTree) -> None:
        dt = sample_time_tree
        r = dt.resample(time="1D")

        assert_identical(
            r.sum()["/"].to_dataset(),
            dt["/"].to_dataset().resample(time="1D").sum(),
        )
        assert_identical(
            r.min()["/"].to_dataset(),
            dt["/"].to_dataset().resample(time="1D").min(),
        )
        assert_identical(
            r.max()["/"].to_dataset(),
            dt["/"].to_dataset().resample(time="1D").max(),
        )
        assert_identical(
            r.count()["/"].to_dataset(),
            dt["/"].to_dataset().resample(time="1D").count(),
        )
        assert_identical(
            r.std()["/"].to_dataset(),
            dt["/"].to_dataset().resample(time="1D").std(),
        )

    def test_resample_interpolate_and_asfreq(self, sample_time_tree: DataTree) -> None:
        dt = sample_time_tree
        res_interp = dt.resample(time="6h").interpolate("linear")
        expected_interp = dt["/"].to_dataset().resample(time="6h").interpolate("linear")
        assert_identical(res_interp["/"].to_dataset(), expected_interp)

        res_asfreq = dt.resample(time="1D").asfreq()
        expected_asfreq = dt["/"].to_dataset().resample(time="1D").asfreq()
        assert_identical(res_asfreq["/"].to_dataset(), expected_asfreq)

    def test_resample_pad_and_bfill(self, sample_time_tree: DataTree) -> None:
        dt = sample_time_tree
        res_pad = dt.resample(time="6h").pad()
        expected_pad = dt["/"].to_dataset().resample(time="6h").pad()
        assert_identical(res_pad["/"].to_dataset(), expected_pad)

        res_bfill = dt.resample(time="6h").bfill()
        expected_bfill = dt["/"].to_dataset().resample(time="6h").bfill()
        assert_identical(res_bfill["/"].to_dataset(), expected_bfill)

    def test_resample_reduce(self, sample_time_tree: DataTree) -> None:
        dt = sample_time_tree
        res = dt.resample(time="1D").reduce(np.mean)
        expected = dt["/"].to_dataset().resample(time="1D").reduce(np.mean)
        assert_identical(res["/"].to_dataset(), expected)

    def test_resample_repr(self, sample_time_tree: DataTree) -> None:
        r = sample_time_tree.resample(time="1D")
        repr_str = repr(r)
        assert "DataTreeResample" in repr_str
        assert "time: 1D" in repr_str


class TestDataTreeMapBlocks:
    def test_map_blocks_simple(self, sample_time_tree: DataTree) -> None:
        dt = sample_time_tree

        def scale_vars(ds: Dataset) -> Dataset:
            return ds * 2.0

        res = dt.map_blocks(scale_vars)
        assert np.allclose(res["/"]["a"].values, dt["/"]["a"].values * 2.0)
        assert np.allclose(res["/child"]["b"].values, dt["/child"]["b"].values * 2.0)
        assert np.allclose(res["/no_time"]["c"].values, dt["/no_time"]["c"].values * 2.0)
