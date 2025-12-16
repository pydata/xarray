"""Test for issue #10875: Clear error message when reducing over non-existent dimension."""

import numpy as np
import pytest

import xarray as xr


class TestGroupbyDimensionError:
    """Tests for clearer error messages in groupby reduce operations."""

    def test_groupby_reduce_missing_dim_single(self):
        """Groupby reduce with single missing dimension should have clear error."""
        ds = xr.DataArray(
            np.reshape(range(27), (3, 3, 3)),
            coords=dict(
                lon=range(3),
                lat=range(3),
                time=xr.date_range("2025-10-01 00:00", "2025-10-01 02:00", freq="h"),
            ),
        )

        with pytest.raises(
            ValueError, match=r"'longitude' not found in array dimensions"
        ):
            ds.groupby("time").std(dim="longitude")

    def test_groupby_reduce_missing_dim_multiple(self):
        """Groupby reduce with multiple missing dimensions should list them."""
        ds = xr.DataArray(
            np.reshape(range(27), (3, 3, 3)),
            coords=dict(
                lon=range(3),
                lat=range(3),
                time=xr.date_range("2025-10-01 00:00", "2025-10-01 02:00", freq="h"),
            ),
        )

        with pytest.raises(ValueError, match=r"not found in array dimensions"):
            ds.groupby("time").std(dim=["longitude", "latitude"])

    def test_standard_reduce_error_matches(self):
        """Standard reduce and groupby reduce should have similar error format."""
        ds = xr.DataArray(
            np.reshape(range(27), (3, 3, 3)),
            coords=dict(
                lon=range(3),
                lat=range(3),
                time=xr.date_range("2025-10-01 00:00", "2025-10-01 02:00", freq="h"),
            ),
        )

        standard_error_msg = None
        try:
            ds.std(dim="longitude")
        except ValueError as e:
            standard_error_msg = str(e)

        groupby_error_msg = None
        try:
            ds.groupby("time").std(dim="longitude")
        except ValueError as e:
            groupby_error_msg = str(e)

        assert standard_error_msg is not None, (
            "Expected ValueError from ds.std(dim='longitude')"
        )
        assert groupby_error_msg is not None, (
            "Expected ValueError from groupby.std(dim='longitude')"
        )
        assert "longitude" in standard_error_msg
        assert "longitude" in groupby_error_msg
        assert "not found in array dimensions" in standard_error_msg
        assert "not found in array dimensions" in groupby_error_msg

    def test_groupby_reduce_valid_dim_still_works(self):
        """Ensure valid dimensions still work correctly."""
        ds = xr.DataArray(
            np.reshape(range(27), (3, 3, 3)),
            dims=["lon", "lat", "time"],
            coords=dict(
                lon=range(3),
                lat=range(3),
                time=xr.date_range("2025-10-01 00:00", "2025-10-01 02:00", freq="h"),
            ),
        )

        result = ds.groupby("time").std(dim="lon")
        assert result is not None
        assert "lon" not in result.dims
