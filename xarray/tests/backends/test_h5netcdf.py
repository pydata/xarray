from __future__ import annotations

import contextlib
import pickle
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import numpy as np
import pytest

import xarray as xr
from xarray import DataArray, Dataset, backends, load_dataset, open_dataset
from xarray.backends.common import _open_remote_file
from xarray.backends.h5netcdf_ import H5netcdfBackendEntrypoint
from xarray.tests import (
    assert_equal,
    assert_identical,
    dask_array_api,
    network,
    requires_dask,
    requires_fsspec,
    requires_h5netcdf,
    requires_h5netcdf_ros3,
    requires_netCDF4,
    requires_scipy,
)
from xarray.tests.backends import test_common, test_netCDF4
from xarray.tests.backends.base import (
    ON_WINDOWS,
    FileObjectNetCDF,
    InMemoryNetCDFWithGroups,
    NetCDF4Base,
    _check_guess_can_open_and_open,
    create_tmp_file,
    create_tmp_files,
)
from xarray.tests.test_dataset import create_test_data

with contextlib.suppress(ImportError):
    import netCDF4 as nc4

with contextlib.suppress(ImportError):
    import fsspec


if TYPE_CHECKING:
    from xarray.backends.api import T_NetcdfEngine, T_NetcdfTypes


@requires_h5netcdf
class TestNetCDF4ClassicViaH5NetCDFData(test_netCDF4.TestNetCDF4ClassicViaNetCDF4Data):
    engine: T_NetcdfEngine = "h5netcdf"
    file_format: T_NetcdfTypes = "NETCDF4_CLASSIC"

    @contextlib.contextmanager
    def create_store(self):
        with create_tmp_file() as tmp_file:
            with backends.H5NetCDFStore.open(
                tmp_file, mode="w", format="NETCDF4_CLASSIC"
            ) as store:
                yield store

    @requires_netCDF4
    def test_cross_engine_read_write_netcdf4(self) -> None:
        # Drop dim3, because its labels include strings. These appear to be
        # not properly read with python-netCDF4, which converts them into
        # unicode instead of leaving them as bytes.
        data = create_test_data().drop_vars("dim3")
        data.attrs["foo"] = "bar"
        valid_engines: list[T_NetcdfEngine] = ["netcdf4", "h5netcdf"]
        for write_engine in valid_engines:
            with create_tmp_file() as tmp_file:
                data.to_netcdf(tmp_file, engine=write_engine, format=self.file_format)
                for read_engine in valid_engines:
                    with open_dataset(tmp_file, engine=read_engine) as actual:
                        assert_identical(data, actual)

    def test_group_fails(self):
        # Check writing group data fails with CLASSIC format
        original = create_test_data()
        with pytest.raises(
            ValueError, match=r"Cannot create sub-groups in `NETCDF4_CLASSIC` format."
        ):
            original.to_netcdf(group="sub", format=self.file_format, engine=self.engine)


@requires_h5netcdf
@requires_netCDF4
@pytest.mark.filterwarnings("ignore:use make_scale(name) instead")
class TestH5NetCDFData(NetCDF4Base):
    engine: T_NetcdfEngine = "h5netcdf"

    @contextlib.contextmanager
    def create_store(self):
        with create_tmp_file() as tmp_file:
            yield backends.H5NetCDFStore.open(tmp_file, "w")

    def test_numpy_bool_(self) -> None:
        # h5netcdf loads booleans as numpy.bool_, this type needs to be supported
        # when writing invalid_netcdf datasets in order to support a roundtrip
        expected = Dataset({"x": ("y", np.ones(5), {"numpy_bool": np.bool_(True)})})
        save_kwargs = {"invalid_netcdf": True}
        with pytest.warns(UserWarning, match="You are writing invalid netcdf features"):
            with self.roundtrip(expected, save_kwargs=save_kwargs) as actual:
                assert_identical(expected, actual)

    def test_cross_engine_read_write_netcdf4(self) -> None:
        # Drop dim3, because its labels include strings. These appear to be
        # not properly read with python-netCDF4, which converts them into
        # unicode instead of leaving them as bytes.
        data = create_test_data().drop_vars("dim3")
        data.attrs["foo"] = "bar"
        valid_engines: list[T_NetcdfEngine] = ["netcdf4", "h5netcdf"]
        for write_engine in valid_engines:
            with create_tmp_file() as tmp_file:
                data.to_netcdf(tmp_file, engine=write_engine)
                for read_engine in valid_engines:
                    with open_dataset(tmp_file, engine=read_engine) as actual:
                        assert_identical(data, actual)

    def test_read_byte_attrs_as_unicode(self) -> None:
        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, "w") as nc:
                nc.foo = b"bar"
            with open_dataset(tmp_file) as actual:
                expected = Dataset(attrs={"foo": "bar"})
                assert_identical(expected, actual)

    def test_compression_encoding_h5py(self) -> None:
        ENCODINGS: tuple[tuple[dict[str, Any], dict[str, Any]], ...] = (
            # h5py style compression with gzip codec will be converted to
            # NetCDF4-Python style on round-trip
            (
                {"compression": "gzip", "compression_opts": 9},
                {"zlib": True, "complevel": 9},
            ),
            # What can't be expressed in NetCDF4-Python style is
            # round-tripped unaltered
            (
                {"compression": "lzf", "compression_opts": None},
                {"compression": "lzf", "compression_opts": None},
            ),
            # If both styles are used together, h5py format takes precedence
            (
                {
                    "compression": "lzf",
                    "compression_opts": None,
                    "zlib": True,
                    "complevel": 9,
                },
                {"compression": "lzf", "compression_opts": None},
            ),
        )

        for compr_in, compr_out in ENCODINGS:
            data = create_test_data()
            compr_common = {
                "chunksizes": (5, 5),
                "fletcher32": True,
                "shuffle": True,
                "original_shape": data.var2.shape,
            }
            data["var2"].encoding.update(compr_in)
            data["var2"].encoding.update(compr_common)
            compr_out.update(compr_common)
            data["scalar"] = ("scalar_dim", np.array([2.0]))
            data["scalar"] = data["scalar"][0]
            with self.roundtrip(data) as actual:
                for k, v in compr_out.items():
                    assert v == actual["var2"].encoding[k]

    def test_compression_check_encoding_h5py(self) -> None:
        """When mismatched h5py and NetCDF4-Python encodings are expressed
        in to_netcdf(encoding=...), must raise ValueError
        """
        data = Dataset({"x": ("y", np.arange(10.0))})
        # Compatible encodings are graciously supported
        with create_tmp_file() as tmp_file:
            data.to_netcdf(
                tmp_file,
                engine="h5netcdf",
                encoding={
                    "x": {
                        "compression": "gzip",
                        "zlib": True,
                        "compression_opts": 6,
                        "complevel": 6,
                    }
                },
            )
            with open_dataset(tmp_file, engine="h5netcdf") as actual:
                assert actual.x.encoding["zlib"] is True
                assert actual.x.encoding["complevel"] == 6

        with create_tmp_file() as tmp_file:
            with pytest.raises(
                ValueError,
                match=r"'complevel' and 'compression_opts' encodings mismatch",
            ):
                data.to_netcdf(
                    tmp_file,
                    engine="h5netcdf",
                    encoding={
                        "x": {
                            "compression": "gzip",
                            "compression_opts": 5,
                            "complevel": 6,
                        }
                    },
                )

    def test_dump_encodings_h5py(self) -> None:
        # regression test for #709
        ds = Dataset({"x": ("y", np.arange(10.0))})

        kwargs = {"encoding": {"x": {"compression": "gzip", "compression_opts": 9}}}
        with self.roundtrip(ds, save_kwargs=kwargs) as actual:
            assert actual.x.encoding["zlib"]
            assert actual.x.encoding["complevel"] == 9

        kwargs = {"encoding": {"x": {"compression": "lzf", "compression_opts": None}}}
        with self.roundtrip(ds, save_kwargs=kwargs) as actual:
            assert actual.x.encoding["compression"] == "lzf"
            assert actual.x.encoding["compression_opts"] is None

    def test_decode_utf8_warning(self) -> None:
        title = b"\xc3"
        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, "w") as f:
                f.title = title
            with pytest.warns(UnicodeWarning, match="returning bytes undecoded") as w:
                ds = xr.load_dataset(tmp_file, engine="h5netcdf")
            assert ds.title == title
            assert "attribute 'title' of h5netcdf object '/'" in str(w[0].message)

    def test_byte_attrs(self, byte_attrs_dataset: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match=byte_attrs_dataset["h5netcdf_error"]):
            super().test_byte_attrs(byte_attrs_dataset)

    def test_roundtrip_complex(self):
        expected = Dataset({"x": ("y", np.ones(5) + 1j * np.ones(5))})
        with self.roundtrip(expected) as actual:
            assert_equal(expected, actual)

    def test_phony_dims_warning(self) -> None:
        import h5py

        foo_data = np.arange(125).reshape(5, 5, 5)
        bar_data = np.arange(625).reshape(25, 5, 5)
        var = {"foo1": foo_data, "foo2": bar_data, "foo3": foo_data, "foo4": bar_data}
        with create_tmp_file() as tmp_file:
            with h5py.File(tmp_file, "w") as f:
                grps = ["bar", "baz"]
                for grp in grps:
                    fx = f.create_group(grp)
                    for k, v in var.items():
                        fx.create_dataset(k, data=v)
            with pytest.warns(UserWarning, match="The 'phony_dims' kwarg"):
                with xr.open_dataset(tmp_file, engine="h5netcdf", group="bar") as ds:
                    assert ds.sizes == {
                        "phony_dim_0": 5,
                        "phony_dim_1": 5,
                        "phony_dim_2": 5,
                        "phony_dim_3": 25,
                    }


@requires_h5netcdf
@requires_netCDF4
class TestH5NetCDFAlreadyOpen:
    def test_open_dataset_group(self) -> None:
        import h5netcdf

        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, mode="w") as nc:
                group = nc.createGroup("g")
                v = group.createVariable("x", "int")
                v[...] = 42

            kwargs = {"decode_vlen_strings": True}

            h5 = h5netcdf.File(tmp_file, mode="r", **kwargs)
            store = backends.H5NetCDFStore(h5["g"])
            with open_dataset(store) as ds:
                expected = Dataset({"x": ((), 42)})
                assert_identical(expected, ds)

            h5 = h5netcdf.File(tmp_file, mode="r", **kwargs)
            store = backends.H5NetCDFStore(h5, group="g")
            with open_dataset(store) as ds:
                expected = Dataset({"x": ((), 42)})
                assert_identical(expected, ds)

    def test_deepcopy(self) -> None:
        import h5netcdf

        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, mode="w") as nc:
                nc.createDimension("x", 10)
                v = nc.createVariable("y", np.int32, ("x",))
                v[:] = np.arange(10)

            kwargs = {"decode_vlen_strings": True}

            h5 = h5netcdf.File(tmp_file, mode="r", **kwargs)
            store = backends.H5NetCDFStore(h5)
            with open_dataset(store) as ds:
                copied = ds.copy(deep=True)
                expected = Dataset({"y": ("x", np.arange(10))})
                assert_identical(expected, copied)


@requires_h5netcdf
class TestH5NetCDFFileObject(TestH5NetCDFData, FileObjectNetCDF):
    engine: T_NetcdfEngine = "h5netcdf"

    def test_open_badbytes(self) -> None:
        with pytest.raises(
            ValueError, match=r"match in any of xarray's currently installed IO"
        ):
            with open_dataset(b"garbage"):
                pass
        with pytest.raises(
            ValueError, match=r"not the signature of a valid netCDF4 file"
        ):
            with open_dataset(b"garbage", engine="h5netcdf"):
                pass
        with pytest.raises(
            ValueError, match=r"not the signature of a valid netCDF4 file"
        ):
            with open_dataset(BytesIO(b"garbage"), engine="h5netcdf"):
                pass

    def test_open_twice(self) -> None:
        expected = create_test_data()
        with create_tmp_file() as tmp_file:
            expected.to_netcdf(tmp_file, engine=self.engine)
            with open(tmp_file, "rb") as f:
                with open_dataset(f, engine=self.engine):
                    with open_dataset(f, engine=self.engine):
                        pass  # should not crash

    @requires_scipy
    def test_open_fileobj(self) -> None:
        # open in-memory datasets instead of local file paths
        expected = create_test_data().drop_vars("dim3")
        expected.attrs["foo"] = "bar"
        with create_tmp_file() as tmp_file:
            expected.to_netcdf(tmp_file, engine="h5netcdf")

            with open(tmp_file, "rb") as f:
                with open_dataset(f, engine="h5netcdf") as actual:
                    assert_identical(expected, actual)

                f.seek(0)
                with open_dataset(f) as actual:
                    assert_identical(expected, actual)

                f.seek(0)
                with BytesIO(f.read()) as bio:
                    with open_dataset(bio, engine="h5netcdf") as actual:
                        assert_identical(expected, actual)

                f.seek(0)
                with pytest.raises(TypeError, match="not a valid NetCDF 3"):
                    open_dataset(f, engine="scipy")

            # TODO: this additional open is required since scipy seems to close the file
            # when it fails on the TypeError (though didn't when we used
            # `raises_regex`?). Ref https://github.com/pydata/xarray/pull/5191
            with open(tmp_file, "rb") as f:
                f.seek(8)
                with open_dataset(f):  # ensure file gets closed
                    pass

    @requires_fsspec
    def test_fsspec(self) -> None:
        expected = create_test_data()
        with create_tmp_file() as tmp_file:
            expected.to_netcdf(tmp_file, engine="h5netcdf")

            with fsspec.open(tmp_file, "rb") as f:
                with open_dataset(f, engine="h5netcdf") as actual:
                    assert_identical(actual, expected)

                    # fsspec.open() creates a pickleable file, unlike open()
                    with pickle.loads(pickle.dumps(actual)) as unpickled:
                        assert_identical(unpickled, expected)


@requires_h5netcdf
class TestH5NetCDFInMemoryData(InMemoryNetCDFWithGroups):
    engine: T_NetcdfEngine = "h5netcdf"


@requires_h5netcdf
@requires_dask
@pytest.mark.filterwarnings("ignore:deallocating CachingFileManager")
class TestH5NetCDFViaDaskData(TestH5NetCDFData):
    @contextlib.contextmanager
    def roundtrip(
        self, data, save_kwargs=None, open_kwargs=None, allow_cleanup_failure=False
    ):
        if save_kwargs is None:
            save_kwargs = {}
        if open_kwargs is None:
            open_kwargs = {}
        open_kwargs.setdefault("chunks", -1)
        with TestH5NetCDFData.roundtrip(
            self, data, save_kwargs, open_kwargs, allow_cleanup_failure
        ) as ds:
            yield ds

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


@requires_h5netcdf
@requires_fsspec
@pytest.mark.parametrize(
    "open_kwargs, expected_cache_type, expected_block_size",
    [
        # Default: blockcache with 4MB block size
        (None, "blockcache", 4 * 1024 * 1024),
        ({}, "blockcache", 4 * 1024 * 1024),
        # Custom block_size still uses blockcache
        ({"block_size": 8 * 1024 * 1024}, "blockcache", 8 * 1024 * 1024),
        # Explicit blockcache with default block_size
        ({"cache_type": "blockcache"}, "blockcache", 4 * 1024 * 1024),
        # Custom cache_type: no block_size default injected
        ({"cache_type": "readahead"}, "readahead", None),
    ],
    ids=["default", "empty-dict", "8mb", "blockcache", "readahead"],
)
def test_h5netcdf_open_kwargs(
    open_kwargs, expected_cache_type, expected_block_size
) -> None:
    """Test that open_kwargs are forwarded to the remote file opener."""
    expected = create_test_data()
    with create_tmp_file() as tmp_file:
        expected.to_netcdf(tmp_file, engine="h5netcdf")

        captured = {}

        def capturing_open_remote_file(
            file, mode, storage_options=None, open_kwargs=None
        ):
            captured["open_kwargs"] = open_kwargs
            return _open_remote_file(
                file,
                mode=mode,
                storage_options=storage_options,
                open_kwargs=open_kwargs,
            )

        with patch(
            "xarray.backends.h5netcdf_._open_remote_file",
            side_effect=capturing_open_remote_file,
        ):
            # Use a file:// URI so is_remote_uri returns True and _open_remote_file is called
            file_uri = f"file://{tmp_file}"
            with open_dataset(
                file_uri, engine="h5netcdf", open_kwargs=open_kwargs
            ) as actual:
                assert_identical(actual, expected)

        assert captured["open_kwargs"]["cache_type"] == expected_cache_type
        if expected_block_size is None:
            assert "block_size" not in captured["open_kwargs"]
        else:
            assert captured["open_kwargs"]["block_size"] == expected_block_size


@requires_netCDF4
@requires_h5netcdf
def test_memoryview_write_h5netcdf_read_netcdf4() -> None:
    original = create_test_data()
    result = original.to_netcdf(engine="h5netcdf")
    roundtrip = load_dataset(result, engine="netcdf4")
    assert_identical(roundtrip, original)


@requires_netCDF4
@requires_h5netcdf
def test_memoryview_write_netcdf4_read_h5netcdf() -> None:
    original = create_test_data()
    result = original.to_netcdf(engine="netcdf4")
    roundtrip = load_dataset(result, engine="h5netcdf")
    assert_identical(roundtrip, original)


@network
@requires_h5netcdf_ros3
class TestH5NetCDFDataRos3Driver(test_common.TestCommon):
    engine: T_NetcdfEngine = "h5netcdf"
    test_remote_dataset: str = "https://dandiarchive.s3.amazonaws.com/ros3test.hdf5"

    @property
    def ros3_kwargs(self) -> dict:
        from h5py import version as h5ver

        return (
            {} if h5ver.hdf5_version_tuple < (2, 0, 0) else {"aws_region": b"us-east-2"}
        )

    @pytest.mark.filterwarnings("ignore:Duplicate dimension names")
    def test_get_variable_list(self) -> None:
        with open_dataset(
            self.test_remote_dataset,
            engine="h5netcdf",
            backend_kwargs={
                "driver": "ros3",
                "driver_kwds": self.ros3_kwargs,
                "phony_dims": "access",
            },
        ) as actual:
            assert "mydataset" in list(actual)

    @pytest.mark.filterwarnings("ignore:Duplicate dimension names")
    def test_get_variable_list_empty_driver_kwds(self) -> None:
        driver_kwds = {
            "secret_id": b"",
            "secret_key": b"",
        }
        driver_kwds.update(self.ros3_kwargs)
        backend_kwargs = {
            "driver": "ros3",
            "driver_kwds": driver_kwds,
            "phony_dims": "access",
        }

        with open_dataset(
            self.test_remote_dataset, engine="h5netcdf", backend_kwargs=backend_kwargs
        ) as actual:
            assert "mydataset" in list(actual)


@requires_h5netcdf
@requires_netCDF4
def test_load_single_value_h5netcdf(tmp_path: Path) -> None:
    """Test that numeric single-element vector attributes are handled fine.

    At present (h5netcdf v0.8.1), the h5netcdf exposes single-valued numeric variable
    attributes as arrays of length 1, as opposed to scalars for the NetCDF4
    backend.  This was leading to a ValueError upon loading a single value from
    a file, see #4471.  Test that loading causes no failure.
    """
    ds = xr.Dataset(
        {
            "test": xr.DataArray(
                np.array([0]), dims=("x",), attrs={"scale_factor": 1, "add_offset": 0}
            )
        }
    )
    ds.to_netcdf(tmp_path / "test.nc")
    with xr.open_dataset(tmp_path / "test.nc", engine="h5netcdf") as ds2:
        ds2["test"][0].load()


@requires_h5netcdf
def test_h5netcdf_entrypoint(tmp_path: Path) -> None:
    entrypoint = H5netcdfBackendEntrypoint()
    ds = create_test_data()

    path = tmp_path / "foo"
    ds.to_netcdf(path, engine="h5netcdf")
    _check_guess_can_open_and_open(entrypoint, path, engine="h5netcdf", expected=ds)
    _check_guess_can_open_and_open(
        entrypoint, str(path), engine="h5netcdf", expected=ds
    )
    with open(path, "rb") as f:
        _check_guess_can_open_and_open(entrypoint, f, engine="h5netcdf", expected=ds)

    contents = ds.to_netcdf(engine="h5netcdf")
    _check_guess_can_open_and_open(entrypoint, contents, engine="h5netcdf", expected=ds)

    assert entrypoint.guess_can_open("something-local.nc")
    assert entrypoint.guess_can_open("something-local.nc4")
    assert entrypoint.guess_can_open("something-local.cdf")
    assert not entrypoint.guess_can_open("not-found-and-no-extension")


@requires_h5netcdf
@requires_fsspec
@requires_dask  # TODO remove after https://github.com/pydata/xarray/issues/9038
def test_h5netcdf_storage_options() -> None:
    with create_tmp_files(2, allow_cleanup_failure=ON_WINDOWS) as (f1, f2):
        ds1 = create_test_data()
        ds1.to_netcdf(f1, engine="h5netcdf")

        ds2 = create_test_data()
        ds2.to_netcdf(f2, engine="h5netcdf")

        files = [f"file://{f}" for f in [f1, f2]]
        with xr.open_mfdataset(
            files,
            engine="h5netcdf",
            concat_dim="time",
            data_vars="all",
            combine="nested",
            storage_options={"skip_instance_cache": False},
        ) as ds:
            assert_identical(xr.concat([ds1, ds2], dim="time", data_vars="all"), ds)
