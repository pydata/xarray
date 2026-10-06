from __future__ import annotations

import contextlib
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import xarray as xr
from xarray import DataArray, Dataset, backends, open_dataset, open_mfdataset
from xarray.backends.netCDF4_ import (
    NetCDF4BackendEntrypoint,
    _extract_nc4_variable_encoding,
)
from xarray.tests import (
    assert_array_equal,
    assert_equal,
    assert_identical,
    dask_array_api,
    has_netCDF4,
    requires_cftime,
    requires_dask,
    requires_h5netcdf,
    requires_h5netcdf_or_netCDF4,
    requires_netCDF4,
    requires_netCDF4_1_6_2_or_above,
    requires_netCDF4_1_7_0_or_above,
)
from xarray.tests.backends.base import (
    ON_WINDOWS,
    CFEncodedBase,
    InMemoryNetCDFWithGroups,
    NetCDF3Only,
    NetCDF4Base,
    _check_guess_can_open_and_open,
    _write_mfdataset_files,
    create_tmp_file,
)
from xarray.tests.test_dataset import create_test_data

with contextlib.suppress(ImportError):
    import netCDF4 as nc4


if TYPE_CHECKING:
    from xarray.backends.api import T_NetcdfEngine, T_NetcdfTypes


def _check_compression_codec_available(codec: str | None) -> bool:
    """Check if a compression codec is available in the netCDF4 library.

    Parameters
    ----------
    codec : str or None
        The compression codec name (e.g., 'zstd', 'blosc_lz', etc.)

    Returns
    -------
    bool
        True if the codec is available, False otherwise.
    """
    if codec is None or codec in ("zlib", "szip"):
        # These are standard and should be available
        return True

    if not has_netCDF4:
        return False

    try:
        import os

        import netCDF4

        # Try to create a file with the compression to test availability
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            nc = netCDF4.Dataset(tmp_path, "w", format="NETCDF4")
            nc.createDimension("x", 10)

            # Attempt to create a variable with the compression
            if codec and codec.startswith("blosc"):
                nc.createVariable(  # type: ignore[call-overload, unused-ignore]
                    varname="test",
                    datatype="f4",
                    dimensions=("x",),
                    compression=codec,
                    blosc_shuffle=1,
                )
            else:
                nc.createVariable(  # type: ignore[call-overload, unused-ignore]
                    varname="test", datatype="f4", dimensions=("x",), compression=codec
                )

            nc.close()
            os.unlink(tmp_path)
            return True
        except (RuntimeError, netCDF4.NetCDF4MissingFeatureException):
            # Codec not available
            if os.path.exists(tmp_path):
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)
            return False
    except Exception:
        # Any other error, assume codec is not available
        return False


@requires_netCDF4
class TestNetCDF4Data(NetCDF4Base):
    @contextlib.contextmanager
    def create_store(self):
        with create_tmp_file() as tmp_file:
            with backends.NetCDF4DataStore.open(tmp_file, mode="w") as store:
                yield store

    def test_variable_order(self) -> None:
        # doesn't work with scipy or h5py :(
        ds = Dataset()
        ds["a"] = 1
        ds["z"] = 2
        ds["b"] = 3
        ds.coords["c"] = 4

        with self.roundtrip(ds) as actual:
            assert list(ds.variables) == list(actual.variables)

    def test_unsorted_index_raises(self) -> None:
        # should be fixed in netcdf4 v1.2.1
        random_data = np.random.random(size=(4, 6))
        dim0 = [0, 1, 2, 3]
        dim1 = [0, 2, 1, 3, 5, 4]  # We will sort this in a later step
        da = xr.DataArray(
            data=random_data,
            dims=("dim0", "dim1"),
            coords={"dim0": dim0, "dim1": dim1},
            name="randovar",
        )
        ds = da.to_dataset()

        with self.roundtrip(ds) as ondisk:
            inds = np.argsort(dim1)
            ds2 = ondisk.isel(dim1=inds)
            # Older versions of NetCDF4 raise an exception here, and if so we
            # want to ensure we improve (that is, replace) the error message
            try:
                _ = ds2.randovar.values
            except IndexError as err:
                assert "first by calling .load" in str(err)

    def test_setncattr_string(self) -> None:
        list_of_strings = ["list", "of", "strings"]
        one_element_list_of_strings = ["one element"]
        one_string = "one string"
        attrs = {
            "foo": list_of_strings,
            "bar": one_element_list_of_strings,
            "baz": one_string,
        }
        ds = Dataset({"x": ("y", [1, 2, 3], attrs)}, attrs=attrs)

        with self.roundtrip(ds) as actual:
            for totest in [actual, actual["x"]]:
                assert_array_equal(list_of_strings, totest.attrs["foo"])
                assert_array_equal(one_element_list_of_strings, totest.attrs["bar"])
                assert one_string == totest.attrs["baz"]

    @pytest.mark.parametrize(
        "compression",
        [
            None,
            "zlib",
            "szip",
            pytest.param(
                "zstd",
                marks=pytest.mark.xfail(
                    not _check_compression_codec_available("zstd"),
                    reason="zstd codec not available in netCDF4 installation",
                ),
            ),
            pytest.param(
                "blosc_lz",
                marks=pytest.mark.xfail(
                    not _check_compression_codec_available("blosc_lz"),
                    reason="blosc_lz codec not available in netCDF4 installation",
                ),
            ),
            pytest.param(
                "blosc_lz4",
                marks=pytest.mark.xfail(
                    not _check_compression_codec_available("blosc_lz4"),
                    reason="blosc_lz4 codec not available in netCDF4 installation",
                ),
            ),
            pytest.param(
                "blosc_lz4hc",
                marks=pytest.mark.xfail(
                    not _check_compression_codec_available("blosc_lz4hc"),
                    reason="blosc_lz4hc codec not available in netCDF4 installation",
                ),
            ),
            pytest.param(
                "blosc_zlib",
                marks=pytest.mark.xfail(
                    not _check_compression_codec_available("blosc_zlib"),
                    reason="blosc_zlib codec not available in netCDF4 installation",
                ),
            ),
            pytest.param(
                "blosc_zstd",
                marks=pytest.mark.xfail(
                    not _check_compression_codec_available("blosc_zstd"),
                    reason="blosc_zstd codec not available in netCDF4 installation",
                ),
            ),
        ],
    )
    @requires_netCDF4_1_6_2_or_above
    @pytest.mark.xfail(ON_WINDOWS, reason="new compression not yet implemented")
    def test_compression_encoding(self, compression: str | None) -> None:
        data = create_test_data(dim_sizes=(20, 80, 10))
        encoding_params: dict[str, Any] = dict(compression=compression, blosc_shuffle=1)
        data["var2"].encoding.update(encoding_params)
        data["var2"].encoding.update(
            {
                "chunksizes": (20, 40),
                "original_shape": data.var2.shape,
                "blosc_shuffle": 1,
                "fletcher32": False,
            }
        )
        with self.roundtrip(data) as actual:
            expected_encoding = data["var2"].encoding.copy()
            # compression does not appear in the retrieved encoding, that differs
            # from the input encoding. shuffle also chantges. Here we modify the
            # expected encoding to account for this
            compression = expected_encoding.pop("compression")
            blosc_shuffle = expected_encoding.pop("blosc_shuffle")
            if compression is not None:
                if "blosc" in compression and blosc_shuffle:
                    expected_encoding["blosc"] = {
                        "compressor": compression,
                        "shuffle": blosc_shuffle,
                    }
                    expected_encoding["shuffle"] = False
                elif compression == "szip":
                    expected_encoding["szip"] = {
                        "coding": "nn",
                        "pixels_per_block": 8,
                    }
                    expected_encoding["shuffle"] = False
                else:
                    # This will set a key like zlib=true which is what appears in
                    # the encoding when we read it.
                    expected_encoding[compression] = True
                    if compression == "zstd":
                        expected_encoding["shuffle"] = False
            else:
                expected_encoding["shuffle"] = False

            actual_encoding = actual["var2"].encoding
            assert expected_encoding.items() <= actual_encoding.items()
        if (
            encoding_params["compression"] is not None
            and "blosc" not in encoding_params["compression"]
        ):
            # regression test for #156
            expected = data.isel(dim1=0)
            with self.roundtrip(expected) as actual:
                assert_equal(expected, actual)

    @pytest.mark.skip(reason="https://github.com/Unidata/netcdf4-python/issues/1195")
    def test_refresh_from_disk(self) -> None:
        super().test_refresh_from_disk()

    @requires_netCDF4_1_7_0_or_above
    def test_roundtrip_complex(self):
        expected = Dataset({"x": ("y", np.ones(5) + 1j * np.ones(5))})
        skwargs = dict(auto_complex=True)
        okwargs = dict(auto_complex=True)
        with self.roundtrip(
            expected, save_kwargs=skwargs, open_kwargs=okwargs
        ) as actual:
            assert_equal(expected, actual)


@requires_netCDF4
class TestNetCDF4AlreadyOpen:
    def test_base_case(self) -> None:
        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, mode="w") as nc:
                v = nc.createVariable("x", "int")
                v[...] = 42

            nc = nc4.Dataset(tmp_file, mode="r")
            store = backends.NetCDF4DataStore(nc)
            with open_dataset(store) as ds:
                expected = Dataset({"x": ((), 42)})
                assert_identical(expected, ds)

    def test_group(self) -> None:
        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, mode="w") as nc:
                group = nc.createGroup("g")
                v = group.createVariable("x", "int")
                v[...] = 42

            nc = nc4.Dataset(tmp_file, mode="r")
            store = backends.NetCDF4DataStore(nc.groups["g"])
            with open_dataset(store) as ds:
                expected = Dataset({"x": ((), 42)})
                assert_identical(expected, ds)

            nc = nc4.Dataset(tmp_file, mode="r")
            store = backends.NetCDF4DataStore(nc, group="g")
            with open_dataset(store) as ds:
                expected = Dataset({"x": ((), 42)})
                assert_identical(expected, ds)

            with nc4.Dataset(tmp_file, mode="r") as nc:
                with pytest.raises(ValueError, match="must supply a root"):
                    backends.NetCDF4DataStore(nc.groups["g"], group="g")

    def test_deepcopy(self) -> None:
        # regression test for https://github.com/pydata/xarray/issues/4425
        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, mode="w") as nc:
                nc.createDimension("x", 10)
                v = nc.createVariable("y", np.int32, ("x",))
                v[:] = np.arange(10)

            h5 = nc4.Dataset(tmp_file, mode="r")
            store = backends.NetCDF4DataStore(h5)
            with open_dataset(store) as ds:
                copied = ds.copy(deep=True)
                expected = Dataset({"y": ("x", np.arange(10))})
                assert_identical(expected, copied)


@requires_h5netcdf_or_netCDF4
class TestGenericNetCDF4InMemory(InMemoryNetCDFWithGroups):
    engine = None


@requires_netCDF4
class TestNetCDF4InMemory(InMemoryNetCDFWithGroups):
    engine: T_NetcdfEngine = "netcdf4"


@requires_netCDF4
@requires_dask
@pytest.mark.filterwarnings("ignore:deallocating CachingFileManager")
class TestNetCDF4ViaDaskData(TestNetCDF4Data):
    @contextlib.contextmanager
    def roundtrip(
        self, data, save_kwargs=None, open_kwargs=None, allow_cleanup_failure=False
    ):
        if open_kwargs is None:
            open_kwargs = {}
        if save_kwargs is None:
            save_kwargs = {}
        open_kwargs.setdefault("chunks", -1)
        with TestNetCDF4Data.roundtrip(
            self, data, save_kwargs, open_kwargs, allow_cleanup_failure
        ) as ds:
            yield ds

    def test_unsorted_index_raises(self) -> None:
        # Skip when using dask because dask rewrites indexers to getitem,
        # dask first pulls items by block.
        pass

    @pytest.mark.skip(reason="caching behavior differs for dask")
    def test_dataset_caching(self) -> None:
        pass

    def test_write_inconsistent_chunks(self) -> None:
        # Construct two variables with the same dimensions, but different
        # chunk sizes.
        da = dask_array_api
        x = da.zeros((100, 100), dtype="f4", chunks=(50, 100))
        x = DataArray(data=x, dims=("lat", "lon"), name="x")
        x.encoding["chunksizes"] = (50, 100)
        x.encoding["original_shape"] = (100, 100)
        y = da.ones((100, 100), dtype="f4", chunks=(100, 50))
        y = DataArray(data=y, dims=("lat", "lon"), name="y")
        y.encoding["chunksizes"] = (100, 50)
        y.encoding["original_shape"] = (100, 100)
        # Put them both into the same dataset
        ds = Dataset({"x": x, "y": y})
        with self.roundtrip(ds) as actual:
            assert actual["x"].encoding["chunksizes"] == (50, 100)
            assert actual["y"].encoding["chunksizes"] == (100, 50)

    # Flaky test. Very open to contributions on fixing this
    @pytest.mark.flaky
    def test_roundtrip_coordinates(self) -> None:
        super().test_roundtrip_coordinates()

    @requires_cftime
    def test_roundtrip_cftime_bnds(self):
        # Regression test for issue #7794
        import cftime

        original = xr.Dataset(
            {
                "foo": ("time", [0.0]),
                "time_bnds": (
                    ("time", "bnds"),
                    [
                        [
                            cftime.Datetime360Day(2005, 12, 1, 0, 0, 0, 0),
                            cftime.Datetime360Day(2005, 12, 2, 0, 0, 0, 0),
                        ]
                    ],
                ),
            },
            {"time": [cftime.Datetime360Day(2005, 12, 1, 12, 0, 0, 0)]},
        )

        with create_tmp_file() as tmp_file:
            original.to_netcdf(tmp_file)
            with open_dataset(tmp_file) as actual:
                # Operation to load actual time_bnds into memory
                assert_array_equal(actual.time_bnds.values, original.time_bnds.values)
                chunked = actual.chunk(time=1)
                with create_tmp_file() as tmp_file_chunked:
                    chunked.to_netcdf(tmp_file_chunked)


@requires_netCDF4
class TestNetCDF3ViaNetCDF4Data(NetCDF3Only, CFEncodedBase):
    engine: T_NetcdfEngine = "netcdf4"
    file_format: T_NetcdfTypes = "NETCDF3_CLASSIC"

    @contextlib.contextmanager
    def create_store(self):
        with create_tmp_file() as tmp_file:
            with backends.NetCDF4DataStore.open(
                tmp_file, mode="w", format="NETCDF3_CLASSIC"
            ) as store:
                yield store

    def test_encoding_kwarg_vlen_string(self) -> None:
        original = Dataset({"x": ["foo", "bar", "baz"]})
        kwargs = dict(encoding={"x": {"dtype": str}})
        with pytest.raises(ValueError, match=r"encoding dtype=str for vlen"):
            with self.roundtrip(original, save_kwargs=kwargs):
                pass


@requires_netCDF4
class TestNetCDF4ClassicViaNetCDF4Data(NetCDF3Only, CFEncodedBase):
    engine: T_NetcdfEngine = "netcdf4"
    file_format: T_NetcdfTypes = "NETCDF4_CLASSIC"

    @contextlib.contextmanager
    def create_store(self):
        with create_tmp_file() as tmp_file:
            with backends.NetCDF4DataStore.open(
                tmp_file, mode="w", format="NETCDF4_CLASSIC"
            ) as store:
                yield store

    @requires_h5netcdf
    def test_string_attributes_stored_as_char(self, tmp_path):
        import h5netcdf

        original = Dataset(attrs={"foo": "bar"})
        store_path = tmp_path / "tmp.nc"
        original.to_netcdf(store_path, engine=self.engine, format=self.file_format)
        with h5netcdf.File(store_path, "r") as ds:
            # Check that the attribute is stored as a char array
            assert ds._h5file.attrs["foo"].dtype == np.dtype("S3")


class TestEncodingInvalid:
    def test_extract_nc4_variable_encoding(self) -> None:
        var = xr.Variable(("x",), [1, 2, 3], {}, {"foo": "bar"})
        with pytest.raises(ValueError, match=r"unexpected encoding"):
            _extract_nc4_variable_encoding(var, raise_on_invalid=True)

        var = xr.Variable(("x",), [1, 2, 3], {}, {"chunking": (2, 1)})
        encoding = _extract_nc4_variable_encoding(var)
        assert {} == encoding

        # regression test
        var = xr.Variable(("x",), [1, 2, 3], {}, {"shuffle": True})
        encoding = _extract_nc4_variable_encoding(var, raise_on_invalid=True)
        assert {"shuffle": True} == encoding

        # Variables with unlim dims must be chunked on output.
        var = xr.Variable(("x",), [1, 2, 3], {}, {"contiguous": True})
        encoding = _extract_nc4_variable_encoding(var, unlimited_dims=("x",))
        assert {} == encoding

    @requires_netCDF4
    def test_extract_nc4_variable_encoding_netcdf4(self):
        # New netCDF4 1.6.0 compression argument.
        var = xr.Variable(("x",), [1, 2, 3], {}, {"compression": "szlib"})
        _extract_nc4_variable_encoding(var, backend="netCDF4", raise_on_invalid=True)

    @pytest.mark.xfail
    def test_extract_h5nc_encoding(self) -> None:
        # not supported with h5netcdf (yet)
        var = xr.Variable(("x",), [1, 2, 3], {}, {"least_significant_digit": 2})
        with pytest.raises(ValueError, match=r"unexpected encoding"):
            _extract_nc4_variable_encoding(var, raise_on_invalid=True)


@requires_netCDF4
def test_netcdf4_entrypoint(tmp_path: Path) -> None:
    entrypoint = NetCDF4BackendEntrypoint()
    ds = create_test_data()

    path = tmp_path / "foo"
    ds.to_netcdf(path, format="NETCDF3_CLASSIC")
    _check_guess_can_open_and_open(entrypoint, path, engine="netcdf4", expected=ds)
    _check_guess_can_open_and_open(entrypoint, str(path), engine="netcdf4", expected=ds)

    path = tmp_path / "bar"
    ds.to_netcdf(path, format="NETCDF4_CLASSIC")
    _check_guess_can_open_and_open(entrypoint, path, engine="netcdf4", expected=ds)
    _check_guess_can_open_and_open(entrypoint, str(path), engine="netcdf4", expected=ds)

    # Remote URLs without extensions return True (backward compatibility)
    assert entrypoint.guess_can_open("http://something/remote")
    # Remote URLs with netCDF extensions are also claimed
    assert entrypoint.guess_can_open("http://something/remote.nc")
    assert entrypoint.guess_can_open("something-local.nc")
    assert entrypoint.guess_can_open("something-local.nc4")
    assert entrypoint.guess_can_open("something-local.cdf")
    assert not entrypoint.guess_can_open("not-found-and-no-extension")

    contents = ds.to_netcdf(engine="netcdf4")
    _check_guess_can_open_and_open(entrypoint, contents, engine="netcdf4", expected=ds)

    path = tmp_path / "baz"
    with open(path, "wb") as f:
        f.write(b"not-a-netcdf-file")
    assert not entrypoint.guess_can_open(path)


@pytest.mark.parametrize(
    "engine",
    [
        pytest.param("netcdf4", marks=requires_netCDF4),
        pytest.param("h5netcdf", marks=requires_h5netcdf),
    ],
)
@pytest.mark.parametrize("dtype", ["f8", "i4"])
def test_roundtrip_non_native_endian_attrs(tmp_path: Path, engine, dtype) -> None:
    """Test that non-native-endian numeric attribute values round-trip."""
    values = np.array([1, 2], dtype=np.dtype(dtype).newbyteorder("S"))
    assert not values.dtype.isnative
    ds = xr.Dataset(
        {
            "x": xr.DataArray(
                [1.0],
                dims=("d",),
                attrs={"valid_range": values},
            )
        },
        attrs={"levels": values},
    )
    ds.to_netcdf(tmp_path / "test.nc", engine=engine)
    with xr.open_dataset(tmp_path / "test.nc", engine=engine) as ds2:
        assert_array_equal(ds2["x"].attrs["valid_range"], [1, 2])
        assert_array_equal(ds2.attrs["levels"], [1, 2])


def _dataset_with_metadata() -> Dataset:
    return Dataset(
        {
            f"v{i}": (("x", "y"), np.full((4, 3), float(i)), {"units": "m"})
            for i in range(10)
        },
        coords={"x": np.arange(4), "y": np.arange(3)},
        attrs={f"attr{i}": i for i in range(5)},
    )


@requires_netCDF4
def test_netcdf4_concurrent_writes(tmp_path: Path) -> None:
    # GH9779: netCDF-C is not thread-safe, so writing metadata must hold its lock
    original = _dataset_with_metadata()
    paths = [tmp_path / f"{i}.nc" for i in range(16)]
    with ThreadPoolExecutor(8) as executor:
        list(executor.map(lambda p: original.to_netcdf(p, engine="netcdf4"), paths))

    for path in paths:
        with open_dataset(path, engine="netcdf4") as actual:
            assert_identical(actual, original)


@requires_netCDF4
def test_netcdf4_concurrent_opens(tmp_path: Path) -> None:
    # GH9779: netCDF-C is not thread-safe, so reading metadata must hold its lock
    original = _dataset_with_metadata()
    paths = [tmp_path / f"{i}.nc" for i in range(16)]
    for path in paths:
        original.to_netcdf(path, engine="netcdf4")

    def load(path):
        with open_dataset(path, engine="netcdf4") as ds:
            return ds.load()

    with ThreadPoolExecutor(8) as executor:
        for actual in executor.map(load, paths * 4):
            assert_identical(actual, original)


@requires_netCDF4
@requires_dask
def test_open_mfdataset_netcdf4_parallel(tmp_path: Path) -> None:
    # GH11088: opening netCDF4 files in parallel threads crashed with segfaults
    paths, expected = _write_mfdataset_files(tmp_path, nfiles=8)
    for _ in range(10):
        with open_mfdataset(paths, engine="netcdf4", parallel=True) as actual:
            assert_identical(actual.load(), expected)
