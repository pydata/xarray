from __future__ import annotations

import contextlib

import pytest
from packaging.version import Version

import xarray as xr
from xarray import DataArray, open_dataset
from xarray.backends.pydap_ import PydapDataStore
from xarray.tests import (
    assert_equal,
    mock,
    network,
    requires_dask,
    requires_pydap,
    requires_scipy_or_netCDF4,
)
from xarray.tests.backends.base import create_tmp_file, open_example_dataset


@requires_scipy_or_netCDF4
@requires_pydap
@pytest.mark.filterwarnings("ignore:The binary mode of fromstring is deprecated")
class TestPydap:
    def convert_to_pydap_dataset(self, original):
        from pydap.model import BaseType, DatasetType

        ds = DatasetType("bears", **original.attrs)
        for key, var in original.data_vars.items():
            ds[key] = BaseType(
                key, var.values, dtype=var.values.dtype.kind, dims=var.dims, **var.attrs
            )
        # check all dims are stored in ds
        for d in original.coords:
            ds[d] = BaseType(d, original[d].values, dims=(d,), **original[d].attrs)
        return ds

    @contextlib.contextmanager
    def create_datasets(self, **kwargs):
        with open_example_dataset("bears.nc") as expected:
            # print("QQ0:", expected["bears"].load())
            pydap_ds = self.convert_to_pydap_dataset(expected)
            actual = open_dataset(PydapDataStore(pydap_ds))
            # netcdf converts string to byte not unicode
            # fixed in pydap 3.5.6. https://github.com/pydap/pydap/issues/510
            actual["bears"].values = actual["bears"].values.astype("S")
            yield actual, expected

    def test_cmp_local_file(self) -> None:
        with self.create_datasets() as (actual, expected):
            assert_equal(actual, expected)

            # global attributes should be global attributes on the dataset
            assert "NC_GLOBAL" not in actual.attrs
            assert "history" in actual.attrs

            # we don't check attributes exactly with assertDatasetIdentical()
            # because the test DAP server seems to insert some extra
            # attributes not found in the netCDF file.
            assert actual.attrs.keys() == expected.attrs.keys()

        with self.create_datasets() as (actual, expected):
            assert_equal(actual[{"l": 2}], expected[{"l": 2}])

        with self.create_datasets() as (actual, expected):
            # always return arrays and not scalars
            # scalars will be promoted to unicode for numpy >= 2.3.0
            assert_equal(actual.isel(i=[0], j=[-1]), expected.isel(i=[0], j=[-1]))

        with self.create_datasets() as (actual, expected):
            assert_equal(actual.isel(j=slice(1, 2)), expected.isel(j=slice(1, 2)))

        with self.create_datasets() as (actual, expected):
            indexers = {"i": [1, 0, 0], "j": [1, 2, 0, 1]}
            assert_equal(actual.isel(**indexers), expected.isel(**indexers))

        with self.create_datasets() as (actual, expected):
            indexers2 = {
                "i": DataArray([0, 1, 0], dims="a"),
                "j": DataArray([0, 2, 1], dims="a"),
            }
            assert_equal(actual.isel(**indexers2), expected.isel(**indexers2))

    def test_compatible_to_netcdf(self) -> None:
        # make sure it can be saved as a netcdf
        with self.create_datasets() as (actual, expected):
            with create_tmp_file() as tmp_file:
                actual.to_netcdf(tmp_file)
                with open_dataset(tmp_file) as actual2:
                    assert_equal(actual2, expected)

    @requires_dask
    def test_dask(self) -> None:
        with self.create_datasets(chunks={"j": 2}) as (actual, expected):
            assert_equal(actual, expected)


@network
@requires_scipy_or_netCDF4
@requires_pydap
@pytest.mark.skip(reason="test.opendap.org is currently returning 403 Forbidden")
class TestPydapOnline(TestPydap):
    @contextlib.contextmanager
    def create_dap2_datasets(self, **kwargs):
        # in pydap 3.5.0, urls defaults to dap2.
        url = "http://test.opendap.org/opendap/data/nc/bears.nc"
        actual = open_dataset(url, engine="pydap", **kwargs)
        # pydap <3.5.6 converts to unicode dtype=|U. Not what
        # xarray expects. Thus force to bytes dtype. pydap >=3.5.6
        # does not convert to unicode. https://github.com/pydap/pydap/issues/510
        actual["bears"].values = actual["bears"].values.astype("S")
        with open_example_dataset("bears.nc") as expected:
            yield actual, expected

    def output_grid_deprecation_warning_dap2dataset(self):
        with pytest.warns(FutureWarning, match="`output_grid` is deprecated"):
            with self.create_dap2_datasets(output_grid=True) as (actual, expected):
                assert_equal(actual, expected)

    def create_dap4_dataset(self, **kwargs):
        url = "dap4://test.opendap.org/opendap/data/nc/bears.nc"
        actual = open_dataset(url, engine="pydap", **kwargs)
        with open_example_dataset("bears.nc") as expected:
            # workaround to restore string which is converted to byte
            # only needed for pydap <3.5.6 https://github.com/pydap/pydap/issues/510
            expected["bears"].values = expected["bears"].values.astype("S")
            yield actual, expected

    def test_session(self) -> None:
        from requests import Session

        session = Session()  # blank requests.Session object
        with mock.patch("pydap.client.open_url") as mock_func:
            xr.backends.PydapDataStore.open("http://test.url", session=session)
        mock_func.assert_called_with(
            url="http://test.url",
            application=None,
            session=session,
            output_grid=False,
            timeout=120,
            verify=True,
            user_charset=None,
        )


@requires_pydap
@network
@pytest.mark.parametrize("protocol", ["dap2", "dap4"])
@pytest.mark.skip(reason="test.opendap.org is currently returning 403 Forbidden")
def test_batchdap4_downloads(tmpdir, protocol) -> None:
    """Test that in dap4, all dimensions are downloaded at once"""
    import pydap
    from pydap.net import create_session

    _version_ = Version(pydap.__version__)
    # Create a session with pre-set params in pydap backend, to cache urls
    cache_name = tmpdir / "debug"
    session = create_session(use_cache=True, cache_kwargs={"cache_name": cache_name})
    session.cache.clear()
    url = "https://test.opendap.org/opendap/hyrax/data/nc/coads_climatology.nc"

    ds = open_dataset(
        url.replace("https", protocol),
        session=session,
        engine="pydap",
        decode_times=False,
    )

    if protocol == "dap4":
        if _version_ > Version("3.5.5"):
            # total downloads are:
            # 1 dmr + 1 dap (all dimensions at once)
            assert len(session.cache.urls()) == 2
            # now load the rest of the variables
            ds.load()
            # each non-dimension array is downloaded with an individual https requests
            assert len(session.cache.urls()) == 2 + 4
        else:
            assert len(session.cache.urls()) == 4
            ds.load()
            assert len(session.cache.urls()) == 4 + 4
    elif protocol == "dap2":
        # das + dds + 3 dods urls for dimensions alone
        assert len(session.cache.urls()) == 5
