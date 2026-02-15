"""Test for issue #10851: Dataset Index not included in to_dataframe when name differs from dimension."""

import numpy as np
import pandas as pd

import xarray as xr


class TestToDataFrameIndexColumn:
    """Tests for to_dataframe including index coordinates with different names."""

    def test_to_dataframe_includes_index_with_different_name(self):
        """Index coordinates with name different from dimension should be in columns."""
        ds_temp = xr.Dataset(
            data_vars=dict(temp=(["time", "pos"], np.array([[5, 10, 15, 20, 25]]))),
            coords=dict(
                pf=("pos", [1.0, 2.0, 4.2, 8.0, 10.0]),
                time=("time", [pd.to_datetime("2025-01-01")]),
            ),
        ).set_xindex("pf")

        df = ds_temp.to_dataframe()

        assert "pf" in df.columns
        assert "temp" in df.columns
        np.testing.assert_array_equal(df["pf"].values, [1.0, 2.0, 4.2, 8.0, 10.0])

    def test_to_dataframe_still_excludes_matching_dim_index(self):
        """Index coordinates where name matches dimension should not be in columns."""
        ds = xr.Dataset(
            data_vars=dict(temp=(["x"], [1, 2, 3])),
            coords=dict(x=("x", [10, 20, 30])),
        )

        df = ds.to_dataframe()

        assert "temp" in df.columns
        assert "x" not in df.columns

    def test_to_dataframe_roundtrip_with_set_xindex(self):
        """Dataset with set_xindex should roundtrip to DataFrame correctly."""
        ds = xr.Dataset(
            data_vars=dict(val=(["dim"], [100, 200, 300])),
            coords=dict(coord_idx=("dim", ["a", "b", "c"])),
        ).set_xindex("coord_idx")

        df = ds.to_dataframe()

        assert "coord_idx" in df.columns
        assert "val" in df.columns
        assert list(df["coord_idx"]) == ["a", "b", "c"]
        assert list(df["val"]) == [100, 200, 300]
