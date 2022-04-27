import pytest

from datatree.io import open_datatree
from datatree.testing import assert_equal
from datatree.tests import requires_h5netcdf, requires_netCDF4, requires_zarr
from datatree.tests.test_datatree import create_test_datatree


class TestIO:
    @requires_netCDF4
    def test_to_netcdf(self, tmpdir):
        filepath = str(
            tmpdir / "test.nc"
        )  # casting to str avoids a pathlib bug in xarray
        original_dt = create_test_datatree()
        original_dt.to_netcdf(filepath, engine="netcdf4")

        roundtrip_dt = open_datatree(filepath)
        assert_equal(original_dt, roundtrip_dt)

    @requires_h5netcdf
    def test_to_h5netcdf(self, tmpdir):
        filepath = str(
            tmpdir / "test.nc"
        )  # casting to str avoids a pathlib bug in xarray
        original_dt = create_test_datatree()
        original_dt.to_netcdf(filepath, engine="h5netcdf")

        roundtrip_dt = open_datatree(filepath)
        assert_equal(original_dt, roundtrip_dt)

    @requires_zarr
    def test_to_zarr(self, tmpdir):
        filepath = str(
            tmpdir / "test.zarr"
        )  # casting to str avoids a pathlib bug in xarray
        original_dt = create_test_datatree()
        original_dt.to_zarr(filepath)

        roundtrip_dt = open_datatree(filepath, engine="zarr")
        assert_equal(original_dt, roundtrip_dt)

    @requires_zarr
    def test_to_zarr_not_consolidated(self, tmpdir):
        filepath = tmpdir / "test.zarr"
        zmetadata = filepath / ".zmetadata"
        s1zmetadata = filepath / "set1" / ".zmetadata"
        filepath = str(filepath)  # casting to str avoids a pathlib bug in xarray
        original_dt = create_test_datatree()
        original_dt.to_zarr(filepath, consolidated=False)
        assert not zmetadata.exists()
        assert not s1zmetadata.exists()

        with pytest.warns(RuntimeWarning, match="consolidated"):
            roundtrip_dt = open_datatree(filepath, engine="zarr")
        assert_equal(original_dt, roundtrip_dt)
