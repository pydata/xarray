from __future__ import annotations

import contextlib
import io
import os.path
import pickle
import re
import sys
import warnings
from io import BytesIO
from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd
import pytest
from pandas.errors import OutOfBoundsDatetime

import xarray as xr
from xarray import (
    DataArray,
    Dataset,
    DataTree,
    load_dataarray,
    load_dataset,
    load_datatree,
    open_dataarray,
    open_dataset,
    open_mfdataset,
    save_mfdataset,
)
from xarray.backends.writers import get_default_netcdf_write_engine
from xarray.coders import CFDatetimeCoder
from xarray.coding.variables import SerializationWarning
from xarray.core.common import _contains_cftime_datetimes
from xarray.core.options import set_options
from xarray.structure.alignment import AlignmentError
from xarray.tests import (
    assert_allclose,
    assert_array_equal,
    assert_equal,
    assert_identical,
    assert_no_warnings,
    dask_array_type,
    has_netCDF4,
    has_numpy_2,
    has_scipy,
    requires_aiobotocore,
    requires_cftime,
    requires_dask,
    requires_fsspec,
    requires_h5netcdf,
    requires_netCDF4,
    requires_scipy,
    requires_scipy_or_netCDF4,
    requires_zarr,
)
from xarray.tests.backends.base import (
    DATA_DIR,
    ON_WINDOWS,
    CFEncodedBase,
    DatasetIOBase,
    NetCDF3Only,
    create_tmp_file,
    create_tmp_files,
    open_example_mfdataset,
)
from xarray.tests.test_coding_times import (
    _ALL_CALENDARS,
    _NON_STANDARD_CALENDARS,
    _STANDARD_CALENDARS,
)
from xarray.tests.test_dataset import create_test_data

with contextlib.suppress(ImportError):
    import dask

with contextlib.suppress(ImportError):
    import netCDF4 as nc4

if TYPE_CHECKING:
    from xarray.backends.api import T_NetcdfEngine, T_NetcdfTypes


@requires_netCDF4
@requires_scipy
@requires_h5netcdf
def test_get_default_netcdf_write_engine() -> None:
    assert xr.get_options()["netcdf_engine_order"] == ("netcdf4", "h5netcdf", "scipy")

    engine = get_default_netcdf_write_engine("", format=None)
    assert engine == "netcdf4"

    engine = get_default_netcdf_write_engine("", format="NETCDF4")
    assert engine == "netcdf4"

    engine = get_default_netcdf_write_engine("", format="NETCDF4_CLASSIC")
    assert engine == "netcdf4"

    engine = get_default_netcdf_write_engine("", format="NETCDF3_CLASSIC")
    assert engine == "netcdf4"

    engine = get_default_netcdf_write_engine(io.BytesIO(), format=None)
    assert engine == "h5netcdf"

    engine = get_default_netcdf_write_engine(io.BytesIO(), format="NETCDF4")
    assert engine == "h5netcdf"

    engine = get_default_netcdf_write_engine(io.BytesIO(), format="NETCDF3_CLASSIC")
    assert engine == "scipy"

    engine = get_default_netcdf_write_engine("path.zarr#mode=nczarr", format=None)
    assert engine == "netcdf4"

    with xr.set_options(netcdf_engine_order=["netcdf4", "scipy", "h5netcdf"]):
        engine = get_default_netcdf_write_engine(io.BytesIO(), format=None)
        assert engine == "scipy"

        engine = get_default_netcdf_write_engine(io.BytesIO(), format="NETCDF4")
        assert engine == "h5netcdf"

        engine = get_default_netcdf_write_engine(io.BytesIO(), format="NETCDF3_CLASSIC")
        assert engine == "scipy"

    with xr.set_options(netcdf_engine_order=["h5netcdf", "scipy", "netcdf4"]):
        engine = get_default_netcdf_write_engine("", format=None)
        assert engine == "h5netcdf"

        engine = get_default_netcdf_write_engine("", format="NETCDF4")
        assert engine == "h5netcdf"

        engine = get_default_netcdf_write_engine("", format="NETCDF4_CLASSIC")
        assert engine == "netcdf4"

        engine = get_default_netcdf_write_engine(io.BytesIO(), format="NETCDF4")
        assert engine == "h5netcdf"

        engine = get_default_netcdf_write_engine("", format="NETCDF3_CLASSIC")
        assert engine == "scipy"

        engine = get_default_netcdf_write_engine(io.BytesIO(), format="NETCDF3_CLASSIC")
        assert engine == "scipy"


@requires_h5netcdf
def test_default_engine_h5netcdf(monkeypatch):
    """Test the default netcdf engine when h5netcdf is the only importable module."""

    monkeypatch.delitem(sys.modules, "netCDF4", raising=False)
    monkeypatch.delitem(sys.modules, "scipy", raising=False)
    monkeypatch.setattr(sys, "meta_path", [])

    engine = get_default_netcdf_write_engine("", format=None)
    assert engine == "h5netcdf"

    with pytest.raises(
        ValueError,
        match=re.escape(
            "cannot write NetCDF files with format='NETCDF3_CLASSIC' because "
            "none of the suitable backend libraries (SUITABLE_BACKENDS) are installed"
        ).replace("SUITABLE_BACKENDS", r"(scipy, netCDF4)|(netCDF4, scipy)"),
    ):
        get_default_netcdf_write_engine("", format="NETCDF3_CLASSIC")


def test_default_engine_nczarr_no_netcdf4_python(monkeypatch):
    monkeypatch.delitem(sys.modules, "netCDF4", raising=False)
    monkeypatch.setattr(sys, "meta_path", [])

    with pytest.raises(
        ValueError,
        match=re.escape(
            "cannot write NetCDF files in NCZarr format because "
            "none of the suitable backend libraries (netCDF4) are installed"
        ),
    ):
        get_default_netcdf_write_engine("#mode=nczarr", format=None)


def test_custom_engine() -> None:
    expected = xr.Dataset(
        dict(a=2 * np.arange(5)), coords=dict(x=("x", np.arange(5), dict(units="s")))
    )

    class CustomBackend(xr.backends.BackendEntrypoint):
        def open_dataset(
            self,
            filename_or_obj,
            drop_variables=None,
            **kwargs,
        ) -> xr.Dataset:
            return expected.copy(deep=True)

    actual = xr.open_dataset("fake_filename", engine=CustomBackend)
    assert_identical(expected, actual)


def test_multiindex() -> None:
    # GH7139
    # Check that we properly handle backends that change index variables
    dataset = xr.Dataset(coords={"coord1": ["A", "B"], "coord2": [1, 2]})
    dataset = dataset.stack(z=["coord1", "coord2"])

    class MultiindexBackend(xr.backends.BackendEntrypoint):
        def open_dataset(
            self,
            filename_or_obj,
            drop_variables=None,
            **kwargs,
        ) -> xr.Dataset:
            return dataset.copy(deep=True)

    loaded = xr.open_dataset("fake_filename", engine=MultiindexBackend)
    assert_identical(dataset, loaded)


class PassThroughBackendEntrypoint(xr.backends.BackendEntrypoint):
    """Access an object passed to the `open_dataset` method."""

    def open_dataset(self, dataset, *, drop_variables=None):
        """Return the first argument."""
        return dataset


def explicit_chunks(chunks, shape):
    """Return explicit chunks, expanding any integer member to a tuple of integers."""
    # Emulate `dask.array.core.normalize_chunks` but for simpler inputs.
    return tuple(
        (
            (
                (size // chunk) * (chunk,)
                + ((size % chunk,) if size % chunk or size == 0 else ())
            )
            if isinstance(chunk, Number)
            else chunk
        )
        for chunk, size in zip(chunks, shape, strict=True)
    )


@requires_dask
class TestPreferredChunks:
    """Test behaviors related to the backend's preferred chunks."""

    var_name = "data"

    def create_dataset(self, shape, pref_chunks):
        """Return a dataset with a variable with the given shape and preferred chunks."""
        dims = tuple(f"dim_{idx}" for idx in range(len(shape)))
        return xr.Dataset(
            {
                self.var_name: xr.Variable(
                    dims,
                    np.empty(shape, dtype=np.dtype("V1")),
                    encoding={
                        "preferred_chunks": dict(zip(dims, pref_chunks, strict=True))
                    },
                )
            }
        )

    def check_dataset(self, initial, final, expected_chunks):
        assert_identical(initial, final)
        assert final[self.var_name].chunks == expected_chunks

    @pytest.mark.parametrize(
        "shape,pref_chunks",
        [
            # Represent preferred chunking with int.
            ((5,), (2,)),
            # Represent preferred chunking with tuple.
            ((5,), ((2, 2, 1),)),
            # Represent preferred chunking with int in two dims.
            ((5, 6), (4, 2)),
            # Represent preferred chunking with tuple in second dim.
            ((5, 6), (4, (2, 2, 2))),
        ],
    )
    @pytest.mark.parametrize("request_with_empty_map", [False, True])
    def test_honor_chunks(self, shape, pref_chunks, request_with_empty_map):
        """Honor the backend's preferred chunks when opening a dataset."""
        initial = self.create_dataset(shape, pref_chunks)
        # To keep the backend's preferred chunks, the `chunks` argument must be an
        # empty mapping or map dimensions to `None`.
        chunks = (
            {}
            if request_with_empty_map
            else dict.fromkeys(initial[self.var_name].dims, None)
        )
        final = xr.open_dataset(
            initial, engine=PassThroughBackendEntrypoint, chunks=chunks
        )
        self.check_dataset(initial, final, explicit_chunks(pref_chunks, shape))

    @pytest.mark.parametrize(
        "shape,pref_chunks,req_chunks",
        [
            # Preferred chunking is int; requested chunking is int.
            ((5,), (2,), (3,)),
            # Preferred chunking is int; requested chunking is tuple.
            ((5,), (2,), ((2, 1, 1, 1),)),
            # Preferred chunking is tuple; requested chunking is int.
            ((5,), ((2, 2, 1),), (3,)),
            # Preferred chunking is tuple; requested chunking is tuple.
            ((5,), ((2, 2, 1),), ((2, 1, 1, 1),)),
            # Split chunks along a dimension other than the first.
            ((1, 5), (1, 2), (1, 3)),
        ],
    )
    def test_split_chunks(self, shape, pref_chunks, req_chunks):
        """Warn when the requested chunks separate the backend's preferred chunks."""
        initial = self.create_dataset(shape, pref_chunks)
        with pytest.warns(UserWarning):
            final = xr.open_dataset(
                initial,
                engine=PassThroughBackendEntrypoint,
                chunks=dict(zip(initial[self.var_name].dims, req_chunks, strict=True)),
            )
        self.check_dataset(initial, final, explicit_chunks(req_chunks, shape))

    @pytest.mark.parametrize(
        "shape,pref_chunks,req_chunks",
        [
            # Keep preferred chunks using int representation.
            ((5,), (2,), (2,)),
            # Keep preferred chunks using tuple representation.
            ((5,), (2,), ((2, 2, 1),)),
            # Join chunks, leaving a final short chunk.
            ((5,), (2,), (4,)),
            # Join all chunks with an int larger than the dimension size.
            ((5,), (2,), (6,)),
            # Join one chunk using tuple representation.
            ((5,), (1,), ((1, 1, 2, 1),)),
            # Join one chunk using int representation.
            ((5,), ((1, 1, 2, 1),), (2,)),
            # Join multiple chunks using tuple representation.
            ((5,), ((1, 1, 2, 1),), ((2, 3),)),
            # Join chunks in multiple dimensions.
            ((5, 5), (2, (1, 1, 2, 1)), (4, (2, 3))),
        ],
    )
    def test_join_chunks(self, shape, pref_chunks, req_chunks):
        """Don't warn when the requested chunks join or keep the preferred chunks."""
        initial = self.create_dataset(shape, pref_chunks)
        with assert_no_warnings():
            final = xr.open_dataset(
                initial,
                engine=PassThroughBackendEntrypoint,
                chunks=dict(zip(initial[self.var_name].dims, req_chunks, strict=True)),
            )
        self.check_dataset(initial, final, explicit_chunks(req_chunks, shape))

    @pytest.mark.parametrize("create_default_indexes", [True, False])
    def test_default_indexes(self, create_default_indexes):
        """Create default indexes if the backend does not create them."""
        coords = xr.Coordinates({"x": ("x", [0, 1]), "y": list("abc")}, indexes={})
        initial = xr.Dataset({"a": ("x", [1, 2])}, coords=coords)

        with assert_no_warnings():
            final = xr.open_dataset(
                initial,
                engine=PassThroughBackendEntrypoint,
                create_default_indexes=create_default_indexes,
            )

        if create_default_indexes:
            assert all(name in final.xindexes for name in ["x", "y"])
        else:
            assert len(final.xindexes) == 0

    @pytest.mark.parametrize("create_default_indexes", [True, False])
    def test_default_indexes_passthrough(self, create_default_indexes):
        """Allow creating indexes in the backend."""

        initial = xr.Dataset(
            {"a": (["x", "y"], [[1, 2, 3], [4, 5, 6]])},
            coords={"x": ("x", [0, 1]), "y": ("y", list("abc"))},
        ).stack(z=["x", "y"])

        with assert_no_warnings():
            final = xr.open_dataset(
                initial,
                engine=PassThroughBackendEntrypoint,
                create_default_indexes=create_default_indexes,
            )

        assert initial.coords.equals(final.coords)


@requires_scipy_or_netCDF4
class TestGenericNetCDFData(NetCDF3Only, CFEncodedBase):
    # verify that we can read and write netCDF3 files as long as we have scipy
    # or netCDF4-python installed
    file_format: T_NetcdfTypes = "NETCDF3_64BIT"

    def test_write_store(self) -> None:
        # there's no specific store to test here
        pass

    @requires_scipy
    @requires_netCDF4
    def test_engine(self) -> None:
        data = create_test_data()

        with pytest.raises(ValueError, match=r"unrecognized engine"):
            data.to_netcdf("foo.nc", engine="foobar")  # type: ignore[call-overload]

        with create_tmp_file() as tmp_file:
            data.to_netcdf(tmp_file)
            with pytest.raises(ValueError, match=r"unrecognized engine"):
                open_dataset(tmp_file, engine="foobar")

        with pytest.raises(
            TypeError,
            match=re.escape("file objects are not supported by the netCDF4 backend"),
        ):
            data.to_netcdf(BytesIO(), engine="netcdf4")

        with pytest.raises(
            TypeError,
            match=re.escape("file objects are not supported by the netCDF4 backend"),
        ):
            open_dataset(BytesIO(), engine="netcdf4")

        bytes_io = BytesIO()
        data.to_netcdf(bytes_io, engine="scipy")
        with pytest.raises(ValueError, match=r"unrecognized engine"):
            open_dataset(bytes_io, engine="foobar")

    def test_cross_engine_read_write_netcdf3(self) -> None:
        data = create_test_data()
        valid_engines: set[T_NetcdfEngine] = set()
        if has_netCDF4:
            valid_engines.add("netcdf4")
        if has_scipy:
            valid_engines.add("scipy")

        for write_engine in valid_engines:
            for format in self.netcdf3_formats:
                with create_tmp_file() as tmp_file:
                    data.to_netcdf(tmp_file, format=format, engine=write_engine)
                    for read_engine in valid_engines:
                        with open_dataset(tmp_file, engine=read_engine) as actual:
                            # hack to allow test to work:
                            # coord comes back as DataArray rather than coord,
                            # and so need to loop through here rather than in
                            # the test function (or we get recursion)
                            [
                                assert_allclose(data[k].variable, actual[k].variable)
                                for k in data.variables
                            ]

    def test_encoding_unlimited_dims(self) -> None:
        ds = Dataset({"x": ("y", np.arange(10.0))})
        with self.roundtrip(ds, save_kwargs=dict(unlimited_dims=["y"])) as actual:
            assert actual.encoding["unlimited_dims"] == set("y")
            assert_equal(ds, actual)

        # Regression test for https://github.com/pydata/xarray/issues/2134
        with self.roundtrip(ds, save_kwargs=dict(unlimited_dims="y")) as actual:
            assert actual.encoding["unlimited_dims"] == set("y")
            assert_equal(ds, actual)

        ds.encoding = {"unlimited_dims": ["y"]}
        with self.roundtrip(ds) as actual:
            assert actual.encoding["unlimited_dims"] == set("y")
            assert_equal(ds, actual)

        # Regression test for https://github.com/pydata/xarray/issues/2134
        ds.encoding = {"unlimited_dims": "y"}
        with self.roundtrip(ds) as actual:
            assert actual.encoding["unlimited_dims"] == set("y")
            assert_equal(ds, actual)

    @requires_scipy
    def test_roundtrip_via_bytes(self) -> None:
        original = create_test_data()
        netcdf_bytes = original.to_netcdf()
        roundtrip = load_dataset(netcdf_bytes)
        assert_identical(roundtrip, original)

    @pytest.mark.xfail(
        reason="scipy.io.netcdf_file closes files upon garbage collection"
    )
    @requires_scipy
    def test_roundtrip_via_file_object(self) -> None:
        original = create_test_data()
        f = BytesIO()
        original.to_netcdf(f)
        assert not f.closed
        restored = open_dataset(f)
        assert not f.closed
        assert_identical(restored, original)
        restored.close()
        assert not f.closed


@pytest.fixture(
    params=[
        pytest.param("scipy", marks=requires_scipy),
        pytest.param("netcdf4", marks=requires_netCDF4),
        pytest.param("h5netcdf", marks=requires_h5netcdf),
        pytest.param("zarr", marks=requires_zarr),
    ]
)
def readengine(request):
    return request.param


@pytest.fixture(params=[1, 20])
def nfiles(request):
    return request.param


@pytest.fixture(params=[5, None])
def file_cache_maxsize(request):
    maxsize = request.param
    if maxsize is not None:
        with set_options(file_cache_maxsize=maxsize):
            yield maxsize
    else:
        yield maxsize


@pytest.fixture(params=[True, False])
def parallel(request):
    return request.param


@pytest.fixture(params=[None, 5])
def chunks(request):
    return request.param


@requires_dask
@pytest.mark.filterwarnings("ignore:use make_scale(name) instead")
def test_open_mfdataset_manyfiles(
    readengine, nfiles, parallel, chunks, file_cache_maxsize
):
    if readengine == "netcdf4" and parallel:
        pytest.skip(
            "netCDF4 reads metadata without holding the netCDF-C lock and can "
            "crash the worker, see GH9779"
        )
    randdata = np.random.randn(nfiles)
    original = Dataset({"foo": ("x", randdata)})
    # test standard open_mfdataset approach with too many files
    with create_tmp_files(nfiles) as tmpfiles:
        # split into multiple sets of temp files
        for ii in original.x.values:
            subds = original.isel(x=slice(ii, ii + 1))
            if readengine != "zarr":
                subds.to_netcdf(tmpfiles[ii], engine=readengine)
            else:  # if writeengine == "zarr":
                subds.to_zarr(store=tmpfiles[ii])

        # check that calculation on opened datasets works properly
        with open_mfdataset(
            tmpfiles,
            combine="nested",
            concat_dim="x",
            engine=readengine,
            parallel=parallel,
            chunks=chunks if (not chunks and readengine != "zarr") else "auto",
        ) as actual:
            # check that using open_mfdataset returns dask arrays for variables
            assert isinstance(actual["foo"].data, dask_array_type)

            assert_identical(original, actual)


@requires_netCDF4
@requires_dask
def test_open_mfdataset_can_open_path_objects() -> None:
    dataset = os.path.join(DATA_DIR, "example_1.nc")
    with open_mfdataset(Path(dataset)) as actual:
        assert isinstance(actual, Dataset)


@requires_netCDF4
@requires_dask
def test_open_mfdataset_list_attr() -> None:
    """
    Case when an attribute of type list differs across the multiple files
    """
    with create_tmp_files(2) as nfiles:
        for i in range(2):
            with nc4.Dataset(nfiles[i], "w") as f:
                f.createDimension("x", 3)
                vlvar = f.createVariable("test_var", np.int32, ("x"))
                # here create an attribute as a list
                vlvar.test_attr = [f"string a {i}", f"string b {i}"]
                vlvar[:] = np.arange(3)

        with open_dataset(nfiles[0]) as ds1:
            with open_dataset(nfiles[1]) as ds2:
                original = xr.concat([ds1, ds2], dim="x")
                with xr.open_mfdataset(
                    [nfiles[0], nfiles[1]], combine="nested", concat_dim="x"
                ) as actual:
                    assert_identical(actual, original)


@requires_cftime
@requires_netCDF4
@requires_dask
@pytest.mark.parametrize(
    "parallel",
    (
        pytest.param(
            True,
            marks=pytest.mark.skip(
                reason="Flaky in CI. Would be a welcome contribution to make a similar test reliable."
            ),
        ),
        False,
    ),
)
def test_open_mfdataset_multiple_files_parallel(parallel, tmp_path):
    lon = np.arange(100)
    time = xr.date_range("20010101", periods=100, calendar="360_day", use_cftime=True)
    data = np.random.random((time.size, lon.size))
    da = xr.DataArray(data, coords={"time": time, "lon": lon}, name="test")

    fnames = []
    for i in range(0, 100, 10):
        fname = tmp_path / f"test_{i}.nc"
        da.isel(time=slice(i, i + 10)).to_netcdf(fname)
        fnames.append(fname)

    for get in [dask.threaded.get, dask.multiprocessing.get, dask.local.get_sync, None]:
        with dask.config.set(scheduler=get):
            with xr.open_mfdataset(
                fnames, parallel=parallel, concat_dim="time", combine="nested"
            ) as tf:
                assert_identical(tf["test"], da)


@requires_scipy_or_netCDF4
@requires_dask
class TestOpenMFDatasetWithDataVarsAndCoordsKw:
    coord_name = "lon"
    var_name = "v1"

    @contextlib.contextmanager
    def setup_files_and_datasets(self, *, fuzz=0, new_combine_kwargs: bool = False):
        ds1, ds2 = self.gen_datasets_with_common_coord_and_time()

        # to test join='exact'
        ds1["x"] = ds1.x + fuzz

        with create_tmp_file() as tmpfile1:
            with create_tmp_file() as tmpfile2:
                # save data to the temporary files
                ds1.to_netcdf(tmpfile1)
                ds2.to_netcdf(tmpfile2)

                with set_options(use_new_combine_kwarg_defaults=new_combine_kwargs):
                    yield [tmpfile1, tmpfile2], [ds1, ds2]

    def gen_datasets_with_common_coord_and_time(self):
        # create coordinate data
        nx = 10
        nt = 10
        x = np.arange(nx)
        t1 = np.arange(nt)
        t2 = np.arange(nt, 2 * nt, 1)

        v1 = np.random.randn(nt, nx)
        v2 = np.random.randn(nt, nx)

        ds1 = Dataset(
            data_vars={self.var_name: (["t", "x"], v1), self.coord_name: ("x", 2 * x)},
            coords={"t": (["t"], t1), "x": (["x"], x)},
        )

        ds2 = Dataset(
            data_vars={self.var_name: (["t", "x"], v2), self.coord_name: ("x", 2 * x)},
            coords={"t": (["t"], t2), "x": (["x"], x)},
        )

        return ds1, ds2

    @pytest.mark.parametrize(
        "combine, concat_dim", [("nested", "t"), ("by_coords", None)]
    )
    @pytest.mark.parametrize("opt", ["all", "minimal", "different"])
    @pytest.mark.parametrize("join", ["outer", "inner", "left", "right"])
    def test_open_mfdataset_does_same_as_concat(
        self, combine, concat_dim, opt, join
    ) -> None:
        with self.setup_files_and_datasets() as (files, [ds1, ds2]):
            if combine == "by_coords":
                files.reverse()
            with open_mfdataset(
                files,
                data_vars=opt,
                combine=combine,
                concat_dim=concat_dim,
                join=join,
                compat="equals",
            ) as ds:
                ds_expect = xr.concat(
                    [ds1, ds2], data_vars=opt, dim="t", join=join, compat="equals"
                )
                assert_identical(ds, ds_expect)

    @pytest.mark.parametrize("use_new_combine_kwarg_defaults", [True, False])
    @pytest.mark.parametrize(
        ["combine_attrs", "attrs", "expected", "expect_error"],
        (
            pytest.param("drop", [{"a": 1}, {"a": 2}], {}, False, id="drop"),
            pytest.param(
                "override", [{"a": 1}, {"a": 2}], {"a": 1}, False, id="override"
            ),
            pytest.param(
                "no_conflicts", [{"a": 1}, {"a": 2}], None, True, id="no_conflicts"
            ),
            pytest.param(
                "identical",
                [{"a": 1, "b": 2}, {"a": 1, "c": 3}],
                None,
                True,
                id="identical",
            ),
            pytest.param(
                "drop_conflicts",
                [{"a": 1, "b": 2}, {"b": -1, "c": 3}],
                {"a": 1, "c": 3},
                False,
                id="drop_conflicts",
            ),
        ),
    )
    def test_open_mfdataset_dataset_combine_attrs(
        self,
        use_new_combine_kwarg_defaults,
        combine_attrs,
        attrs,
        expected,
        expect_error,
    ):
        with self.setup_files_and_datasets() as (files, [_ds1, _ds2]):
            # Give the files an inconsistent attribute
            for i, f in enumerate(files):
                ds = open_dataset(f).load()
                ds.attrs = attrs[i]
                ds.close()
                ds.to_netcdf(f)

            with set_options(
                use_new_combine_kwarg_defaults=use_new_combine_kwarg_defaults
            ):
                warning: contextlib.AbstractContextManager = (
                    pytest.warns(FutureWarning)
                    if not use_new_combine_kwarg_defaults
                    else contextlib.nullcontext()
                )
                error: contextlib.AbstractContextManager = (
                    pytest.raises(xr.MergeError)
                    if expect_error
                    else contextlib.nullcontext()
                )
                with warning:
                    with error:
                        with xr.open_mfdataset(
                            files,
                            combine="nested",
                            concat_dim="t",
                            combine_attrs=combine_attrs,
                        ) as ds:
                            assert ds.attrs == expected

    def test_open_mfdataset_dataset_attr_by_coords(self) -> None:
        """
        Case when an attribute differs across the multiple files
        """
        with self.setup_files_and_datasets() as (files, [_ds1, _ds2]):
            # Give the files an inconsistent attribute
            for i, f in enumerate(files):
                ds = open_dataset(f).load()
                ds.attrs["test_dataset_attr"] = 10 + i
                ds.close()
                ds.to_netcdf(f)

            with set_options(use_new_combine_kwarg_defaults=True):
                with xr.open_mfdataset(files, combine="nested", concat_dim="t") as ds:
                    assert ds.test_dataset_attr == 10

    def test_open_mfdataset_dataarray_attr_by_coords(self) -> None:
        """
        Case when an attribute of a member DataArray differs across the multiple files
        """
        with self.setup_files_and_datasets(new_combine_kwargs=True) as (
            files,
            [_ds1, _ds2],
        ):
            # Give the files an inconsistent attribute
            for i, f in enumerate(files):
                ds = open_dataset(f).load()
                ds["v1"].attrs["test_dataarray_attr"] = i
                ds.close()
                ds.to_netcdf(f)

                with xr.open_mfdataset(
                    files, data_vars=None, combine="nested", concat_dim="t"
                ) as ds:
                    assert ds["v1"].test_dataarray_attr == 0

    @pytest.mark.parametrize(
        "combine, concat_dim", [("nested", "t"), ("by_coords", None)]
    )
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"data_vars": "all"},
            {"data_vars": "minimal"},
            {
                "data_vars": "all",
                "coords": "different",
                "compat": "no_conflicts",
            },  # old defaults
            {
                "data_vars": None,
                "coords": "minimal",
                "compat": "override",
            },  # new defaults
            {"data_vars": "different", "compat": "no_conflicts"},
            {},
        ],
    )
    def test_open_mfdataset_exact_join_raises_error(
        self, combine, concat_dim, kwargs
    ) -> None:
        with self.setup_files_and_datasets(fuzz=0.1, new_combine_kwargs=True) as (
            files,
            _,
        ):
            if combine == "by_coords":
                files.reverse()

            with pytest.raises(
                ValueError, match="cannot align objects with join='exact'"
            ):
                open_mfdataset(
                    files,
                    **kwargs,
                    combine=combine,
                    concat_dim=concat_dim,
                    join="exact",
                )

    def test_open_mfdataset_defaults_with_exact_join_warns_as_well_as_raising(
        self,
    ) -> None:
        with self.setup_files_and_datasets(fuzz=0.1, new_combine_kwargs=True) as (
            files,
            _,
        ):
            files.reverse()
            with pytest.raises(
                ValueError, match="cannot align objects with join='exact'"
            ):
                open_mfdataset(files, combine="by_coords")

    def test_common_coord_when_datavars_all(self) -> None:
        opt: Final = "all"

        with self.setup_files_and_datasets() as (files, [ds1, ds2]):
            # open the files with the data_var option
            with open_mfdataset(
                files, data_vars=opt, combine="nested", concat_dim="t"
            ) as ds:
                coord_shape = ds[self.coord_name].shape
                coord_shape1 = ds1[self.coord_name].shape
                coord_shape2 = ds2[self.coord_name].shape

                var_shape = ds[self.var_name].shape

                assert var_shape == coord_shape
                assert coord_shape1 != coord_shape
                assert coord_shape2 != coord_shape

    def test_common_coord_when_datavars_minimal(self) -> None:
        opt: Final = "minimal"

        with self.setup_files_and_datasets(new_combine_kwargs=True) as (
            files,
            [ds1, ds2],
        ):
            # open the files using data_vars option
            with open_mfdataset(
                files, data_vars=opt, combine="nested", concat_dim="t"
            ) as ds:
                coord_shape = ds[self.coord_name].shape
                coord_shape1 = ds1[self.coord_name].shape
                coord_shape2 = ds2[self.coord_name].shape

                var_shape = ds[self.var_name].shape

                assert var_shape != coord_shape
                assert coord_shape1 == coord_shape
                assert coord_shape2 == coord_shape

    def test_invalid_data_vars_value_should_fail(self) -> None:
        with self.setup_files_and_datasets() as (files, _):
            with pytest.raises(ValueError):
                with open_mfdataset(files, data_vars="minimum", combine="by_coords"):  # type: ignore[arg-type]
                    pass

            # test invalid coord parameter
            with pytest.raises(ValueError):
                with open_mfdataset(files, coords="minimum", combine="by_coords"):
                    pass

    @pytest.mark.parametrize(
        "combine, concat_dim", [("nested", "t"), ("by_coords", None)]
    )
    @pytest.mark.parametrize(
        "kwargs", [{"data_vars": "different"}, {"coords": "different"}]
    )
    def test_open_mfdataset_warns_when_kwargs_set_to_different(
        self, combine, concat_dim, kwargs
    ) -> None:
        with self.setup_files_and_datasets(new_combine_kwargs=True) as (
            files,
            [ds1, ds2],
        ):
            if combine == "by_coords":
                files.reverse()
            with pytest.raises(
                ValueError, match="Previously the default was `compat='no_conflicts'`"
            ):
                open_mfdataset(files, combine=combine, concat_dim=concat_dim, **kwargs)
            with pytest.raises(
                ValueError, match="Previously the default was `compat='equals'`"
            ):
                xr.concat([ds1, ds2], dim="t", **kwargs)

            with set_options(use_new_combine_kwarg_defaults=False):
                expectation: contextlib.AbstractContextManager = (
                    pytest.warns(
                        FutureWarning,
                        match="will change from data_vars='all'",
                    )
                    if "data_vars" not in kwargs
                    else contextlib.nullcontext()
                )

                with pytest.warns(
                    FutureWarning,
                    match="will change from compat='equals'",
                ):
                    with expectation:
                        ds_expect = xr.concat([ds1, ds2], dim="t", **kwargs)
                with pytest.warns(
                    FutureWarning, match="will change from compat='no_conflicts'"
                ):
                    with expectation:
                        ds = open_mfdataset(
                            files, combine=combine, concat_dim=concat_dim, **kwargs
                        )
                with ds:
                    assert_identical(ds, ds_expect)


@requires_dask
@requires_scipy
@requires_netCDF4
class TestDask(DatasetIOBase):
    @contextlib.contextmanager
    def create_store(self):
        yield Dataset()

    @contextlib.contextmanager
    def roundtrip(
        self, data, save_kwargs=None, open_kwargs=None, allow_cleanup_failure=False
    ):
        yield data.chunk()

    # Override methods in DatasetIOBase - not applicable to dask
    def test_roundtrip_string_encoded_characters(self) -> None:
        pass

    def test_roundtrip_coordinates_with_space(self) -> None:
        pass

    @pytest.mark.skip(
        reason="nulls are only replaced when encoding, which does not happen here"
    )
    def test_roundtrip_stringdtype_nulls(self) -> None:
        pass

    def test_roundtrip_numpy_datetime_data(self) -> None:
        # Override method in DatasetIOBase - remove not applicable
        # save_kwargs
        times = pd.to_datetime(["2000-01-01", "2000-01-02", "NaT"], unit="ns")
        expected = Dataset({"t": ("t", times), "t0": times[0]})
        with self.roundtrip(expected) as actual:
            assert_identical(expected, actual)

    def test_roundtrip_cftime_datetime_data(self) -> None:
        # Override method in DatasetIOBase - remove not applicable
        # save_kwargs
        from xarray.tests.test_coding_times import _all_cftime_date_types

        date_types = _all_cftime_date_types()
        for date_type in date_types.values():
            times = [date_type(1, 1, 1), date_type(1, 1, 2)]
            expected = Dataset({"t": ("t", times), "t0": times[0]})
            expected_decoded_t = np.array(times)
            expected_decoded_t0 = np.array([date_type(1, 1, 1)])

            with self.roundtrip(expected) as actual:
                assert_array_equal(actual.t.values, expected_decoded_t)
                assert_array_equal(actual.t0.values, expected_decoded_t0)

    def test_write_store(self) -> None:
        # Override method in DatasetIOBase - not applicable to dask
        pass

    def test_dataset_caching(self) -> None:
        expected = Dataset({"foo": ("x", [5, 6, 7])})
        with self.roundtrip(expected) as actual:
            assert not actual.foo.variable._in_memory
            _ = actual.foo.values  # no caching
            assert not actual.foo.variable._in_memory

    def test_open_mfdataset(self) -> None:
        original = Dataset({"foo": ("x", np.random.randn(10))})
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                original.isel(x=slice(5)).to_netcdf(tmp1)
                original.isel(x=slice(5, 10)).to_netcdf(tmp2)
                with open_mfdataset(
                    [tmp1, tmp2], concat_dim="x", combine="nested"
                ) as actual:
                    assert isinstance(actual.foo.variable.data, dask_array_type)
                    assert actual.foo.variable.data.chunks == ((5, 5),)
                    assert_identical(original, actual)
                with open_mfdataset(
                    [tmp1, tmp2], concat_dim="x", combine="nested", chunks={"x": 3}
                ) as actual:
                    assert actual.foo.variable.data.chunks == ((3, 2, 3, 2),)

        with pytest.raises(OSError, match=r"no files to open"):
            open_mfdataset("foo-bar-baz-*.nc")
        with pytest.raises(ValueError, match=r"wild-card"):
            open_mfdataset("http://some/remote/uri")

    @requires_fsspec
    @requires_aiobotocore
    def test_open_mfdataset_no_files(self) -> None:
        # glob is attempted as of #4823, but finds no files
        with pytest.raises(OSError, match=r"no files"):
            open_mfdataset("http://some/remote/uri", engine="zarr")

    def test_open_mfdataset_2d(self) -> None:
        original = Dataset({"foo": (["x", "y"], np.random.randn(10, 8))})
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                with create_tmp_file() as tmp3:
                    with create_tmp_file() as tmp4:
                        original.isel(x=slice(5), y=slice(4)).to_netcdf(tmp1)
                        original.isel(x=slice(5, 10), y=slice(4)).to_netcdf(tmp2)
                        original.isel(x=slice(5), y=slice(4, 8)).to_netcdf(tmp3)
                        original.isel(x=slice(5, 10), y=slice(4, 8)).to_netcdf(tmp4)
                        with open_mfdataset(
                            [[tmp1, tmp2], [tmp3, tmp4]],
                            combine="nested",
                            concat_dim=["y", "x"],
                        ) as actual:
                            assert isinstance(
                                actual.foo.variable.data,
                                dask_array_type,
                            )
                            assert actual.foo.variable.data.chunks == ((5, 5), (4, 4))
                            assert_identical(original, actual)
                        with open_mfdataset(
                            [[tmp1, tmp2], [tmp3, tmp4]],
                            combine="nested",
                            concat_dim=["y", "x"],
                            chunks={"x": 3, "y": 2},
                        ) as actual:
                            assert actual.foo.variable.data.chunks == (
                                (3, 2, 3, 2),
                                (2, 2, 2, 2),
                            )

    def test_open_mfdataset_pathlib(self) -> None:
        original = Dataset({"foo": ("x", np.random.randn(10))})
        with create_tmp_file() as tmps1:
            with create_tmp_file() as tmps2:
                tmp1 = Path(tmps1)
                tmp2 = Path(tmps2)
                original.isel(x=slice(5)).to_netcdf(tmp1)
                original.isel(x=slice(5, 10)).to_netcdf(tmp2)
                with open_mfdataset(
                    [tmp1, tmp2], concat_dim="x", combine="nested"
                ) as actual:
                    assert_identical(original, actual)

    def test_open_mfdataset_2d_pathlib(self) -> None:
        original = Dataset({"foo": (["x", "y"], np.random.randn(10, 8))})
        with create_tmp_file() as tmps1:
            with create_tmp_file() as tmps2:
                with create_tmp_file() as tmps3:
                    with create_tmp_file() as tmps4:
                        tmp1 = Path(tmps1)
                        tmp2 = Path(tmps2)
                        tmp3 = Path(tmps3)
                        tmp4 = Path(tmps4)
                        original.isel(x=slice(5), y=slice(4)).to_netcdf(tmp1)
                        original.isel(x=slice(5, 10), y=slice(4)).to_netcdf(tmp2)
                        original.isel(x=slice(5), y=slice(4, 8)).to_netcdf(tmp3)
                        original.isel(x=slice(5, 10), y=slice(4, 8)).to_netcdf(tmp4)
                        with open_mfdataset(
                            [[tmp1, tmp2], [tmp3, tmp4]],
                            combine="nested",
                            concat_dim=["y", "x"],
                        ) as actual:
                            assert_identical(original, actual)

    def test_open_mfdataset_2(self) -> None:
        original = Dataset({"foo": ("x", np.random.randn(10))})
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                original.isel(x=slice(5)).to_netcdf(tmp1)
                original.isel(x=slice(5, 10)).to_netcdf(tmp2)

                with open_mfdataset(
                    [tmp1, tmp2], concat_dim="x", combine="nested"
                ) as actual:
                    assert_identical(original, actual)

    def test_open_mfdataset_with_ignore(self) -> None:
        original = Dataset({"foo": ("x", np.random.randn(10))})
        with create_tmp_files(2) as (tmp1, tmp2):
            ds1 = original.isel(x=slice(5))
            ds2 = original.isel(x=slice(5, 10))
            ds1.to_netcdf(tmp1)
            ds2.to_netcdf(tmp2)
            with open_mfdataset(
                [tmp1, "non-existent-file.nc", tmp2],
                concat_dim="x",
                combine="nested",
                errors="ignore",
            ) as actual:
                assert_identical(original, actual)

    def test_open_mfdataset_with_warn(self) -> None:
        original = Dataset({"foo": ("x", np.random.randn(10))})
        with create_tmp_files(2) as (tmp1, tmp2):
            ds1 = original.isel(x=slice(5))
            ds2 = original.isel(x=slice(5, 10))
            ds1.to_netcdf(tmp1)
            ds2.to_netcdf(tmp2)
            with pytest.warns(UserWarning, match=r"Ignoring."):
                actual = open_mfdataset(
                    [tmp1, "non-existent-file.nc", tmp2],
                    concat_dim="x",
                    combine="nested",
                    errors="warn",
                )
            with actual:
                assert_identical(original, actual)

    def test_open_mfdataset_2d_with_ignore(self) -> None:
        original = Dataset({"foo": (["x", "y"], np.random.randn(10, 8))})
        with create_tmp_files(4) as (tmp1, tmp2, tmp3, tmp4):
            original.isel(x=slice(5), y=slice(4)).to_netcdf(tmp1)
            original.isel(x=slice(5, 10), y=slice(4)).to_netcdf(tmp2)
            original.isel(x=slice(5), y=slice(4, 8)).to_netcdf(tmp3)
            original.isel(x=slice(5, 10), y=slice(4, 8)).to_netcdf(tmp4)
            with open_mfdataset(
                [[tmp1, tmp2], ["non-existent-file.nc", tmp3, tmp4]],
                combine="nested",
                concat_dim=["y", "x"],
                errors="ignore",
            ) as actual:
                assert_identical(original, actual)

    def test_open_mfdataset_2d_with_warn(self) -> None:
        original = Dataset({"foo": (["x", "y"], np.random.randn(10, 8))})
        with create_tmp_files(4) as (tmp1, tmp2, tmp3, tmp4):
            original.isel(x=slice(5), y=slice(4)).to_netcdf(tmp1)
            original.isel(x=slice(5, 10), y=slice(4)).to_netcdf(tmp2)
            original.isel(x=slice(5), y=slice(4, 8)).to_netcdf(tmp3)
            original.isel(x=slice(5, 10), y=slice(4, 8)).to_netcdf(tmp4)
            with pytest.warns(UserWarning, match=r"Ignoring."):
                actual = open_mfdataset(
                    [[tmp1, tmp2, "non-existent-file.nc"], [tmp3, tmp4]],
                    combine="nested",
                    concat_dim=["y", "x"],
                    errors="warn",
                )
            with actual:
                assert_identical(original, actual)

    def test_attrs_mfdataset(self) -> None:
        original = Dataset({"foo": ("x", np.random.randn(10))})
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                ds1 = original.isel(x=slice(5))
                ds2 = original.isel(x=slice(5, 10))
                ds1.attrs["test1"] = "foo"
                ds2.attrs["test2"] = "bar"
                ds1.to_netcdf(tmp1)
                ds2.to_netcdf(tmp2)
                with open_mfdataset(
                    [tmp1, tmp2], concat_dim="x", combine="nested"
                ) as actual:
                    # presumes that attributes inherited from
                    # first dataset loaded
                    assert actual.test1 == ds1.test1
                    # attributes from ds2 are not retained, e.g.,
                    with pytest.raises(AttributeError, match=r"no attribute"):
                        _ = actual.test2

    def test_open_mfdataset_attrs_file(self) -> None:
        original = Dataset({"foo": ("x", np.random.randn(10))})
        with create_tmp_files(2) as (tmp1, tmp2):
            ds1 = original.isel(x=slice(5))
            ds2 = original.isel(x=slice(5, 10))
            ds1.attrs["test1"] = "foo"
            ds2.attrs["test2"] = "bar"
            ds1.to_netcdf(tmp1)
            ds2.to_netcdf(tmp2)
            with open_mfdataset(
                [tmp1, tmp2], concat_dim="x", combine="nested", attrs_file=tmp2
            ) as actual:
                # attributes are inherited from the master file
                assert actual.attrs["test2"] == ds2.attrs["test2"]
                # attributes from ds1 are not retained, e.g.,
                assert "test1" not in actual.attrs

    def test_open_mfdataset_attrs_file_path(self) -> None:
        original = Dataset({"foo": ("x", np.random.randn(10))})
        with create_tmp_files(2) as (tmps1, tmps2):
            tmp1 = Path(tmps1)
            tmp2 = Path(tmps2)
            ds1 = original.isel(x=slice(5))
            ds2 = original.isel(x=slice(5, 10))
            ds1.attrs["test1"] = "foo"
            ds2.attrs["test2"] = "bar"
            ds1.to_netcdf(tmp1)
            ds2.to_netcdf(tmp2)
            with open_mfdataset(
                [tmp1, tmp2], concat_dim="x", combine="nested", attrs_file=tmp2
            ) as actual:
                # attributes are inherited from the master file
                assert actual.attrs["test2"] == ds2.attrs["test2"]
                # attributes from ds1 are not retained, e.g.,
                assert "test1" not in actual.attrs

    def test_open_mfdataset_auto_combine(self) -> None:
        original = Dataset({"foo": ("x", np.random.randn(10)), "x": np.arange(10)})
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                original.isel(x=slice(5)).to_netcdf(tmp1)
                original.isel(x=slice(5, 10)).to_netcdf(tmp2)

                with open_mfdataset([tmp2, tmp1], combine="by_coords") as actual:
                    assert_identical(original, actual)

    def test_open_mfdataset_raise_on_bad_combine_args(self) -> None:
        # Regression test for unhelpful error shown in #5230
        original = Dataset({"foo": ("x", np.random.randn(10)), "x": np.arange(10)})
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                original.isel(x=slice(5)).to_netcdf(tmp1)
                original.isel(x=slice(5, 10)).to_netcdf(tmp2)
                with pytest.raises(ValueError, match="`concat_dim` has no effect"):
                    open_mfdataset([tmp1, tmp2], concat_dim="x")

    def test_encoding_mfdataset(self) -> None:
        original = Dataset(
            {
                "foo": ("t", np.random.randn(10)),
                "t": ("t", pd.date_range(start="2010-01-01", periods=10, freq="1D")),
            }
        )
        original.t.encoding["units"] = "days since 2010-01-01"

        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                ds1 = original.isel(t=slice(5))
                ds2 = original.isel(t=slice(5, 10))
                ds1.t.encoding["units"] = "days since 2010-01-01"
                ds2.t.encoding["units"] = "days since 2000-01-01"
                ds1.to_netcdf(tmp1)
                ds2.to_netcdf(tmp2)
                with open_mfdataset(
                    [tmp1, tmp2], combine="nested", concat_dim="t"
                ) as actual:
                    assert actual.t.encoding["units"] == original.t.encoding["units"]
                    assert actual.t.encoding["units"] == ds1.t.encoding["units"]
                    assert actual.t.encoding["units"] != ds2.t.encoding["units"]

    def test_encoding_mfdataset_new_defaults(self) -> None:
        original = Dataset(
            {
                "foo": ("t", np.random.randn(10)),
                "t": ("t", pd.date_range(start="2010-01-01", periods=10, freq="1D")),
            }
        )
        original.t.encoding["units"] = "days since 2010-01-01"

        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                ds1 = original.isel(t=slice(5))
                ds2 = original.isel(t=slice(5, 10))
                ds1.t.encoding["units"] = "days since 2010-01-01"
                ds2.t.encoding["units"] = "days since 2000-01-01"
                ds1.to_netcdf(tmp1)
                ds2.to_netcdf(tmp2)

                for setting in [True, False]:
                    with set_options(use_new_combine_kwarg_defaults=setting):
                        with open_mfdataset(
                            [tmp1, tmp2], combine="nested", concat_dim="t"
                        ) as old:
                            assert (
                                old.t.encoding["units"] == original.t.encoding["units"]
                            )
                            assert old.t.encoding["units"] == ds1.t.encoding["units"]
                            assert old.t.encoding["units"] != ds2.t.encoding["units"]

                with set_options(use_new_combine_kwarg_defaults=True):
                    with pytest.raises(
                        AlignmentError, match="If you are intending to concatenate"
                    ):
                        open_mfdataset([tmp1, tmp2], combine="nested")

    def test_preprocess_mfdataset(self) -> None:
        original = Dataset({"foo": ("x", np.random.randn(10))})
        with create_tmp_file() as tmp:
            original.to_netcdf(tmp)

            def preprocess(ds):
                return ds.assign_coords(z=0)

            expected = preprocess(original)
            with open_mfdataset(
                tmp, preprocess=preprocess, combine="by_coords"
            ) as actual:
                assert_identical(expected, actual)

    def test_save_mfdataset_roundtrip(self) -> None:
        original = Dataset({"foo": ("x", np.random.randn(10))})
        datasets = [original.isel(x=slice(5)), original.isel(x=slice(5, 10))]
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                save_mfdataset(datasets, [tmp1, tmp2])
                with open_mfdataset(
                    [tmp1, tmp2], concat_dim="x", combine="nested"
                ) as actual:
                    assert_identical(actual, original)

    def test_save_mfdataset_invalid(self) -> None:
        ds = Dataset()
        with pytest.raises(ValueError, match=r"cannot use mode"):
            save_mfdataset([ds, ds], ["same", "same"])
        with pytest.raises(ValueError, match=r"same length"):
            save_mfdataset([ds, ds], ["only one path"])

    def test_save_mfdataset_invalid_dataarray(self) -> None:
        # regression test for GH1555
        da = DataArray([1, 2])
        with pytest.raises(TypeError, match=r"supports writing Dataset"):
            save_mfdataset([da], ["dataarray"])

    def test_save_mfdataset_pathlib_roundtrip(self) -> None:
        original = Dataset({"foo": ("x", np.random.randn(10))})
        datasets = [original.isel(x=slice(5)), original.isel(x=slice(5, 10))]
        with create_tmp_file() as tmps1:
            with create_tmp_file() as tmps2:
                tmp1 = Path(tmps1)
                tmp2 = Path(tmps2)
                save_mfdataset(datasets, [tmp1, tmp2])
                with open_mfdataset(
                    [tmp1, tmp2], concat_dim="x", combine="nested"
                ) as actual:
                    assert_identical(actual, original)

    def test_save_mfdataset_pass_kwargs(self) -> None:
        # create a timeseries to store in a netCDF file
        times = [0, 1]
        time = xr.DataArray(times, dims=("time",))

        # create a simple dataset to write using save_mfdataset
        test_ds = xr.Dataset()
        test_ds["time"] = time

        # make sure the times are written as double and
        # turn off fill values
        encoding = dict(time=dict(dtype="double"))
        unlimited_dims = ["time"]

        # set the output file name
        output_path = "test.nc"

        # attempt to write the dataset with the encoding and unlimited args
        # passed through
        xr.save_mfdataset(
            [test_ds], [output_path], encoding=encoding, unlimited_dims=unlimited_dims
        )

    def test_open_and_do_math(self) -> None:
        original = Dataset({"foo": ("x", np.random.randn(10))})
        with create_tmp_file() as tmp:
            original.to_netcdf(tmp)
            with open_mfdataset(tmp, combine="by_coords") as ds:
                actual = 1.0 * ds
                assert_allclose(original, actual, decode_bytes=False)

    @pytest.mark.parametrize(
        "kwargs",
        [pytest.param({"concat_dim": None}, id="none"), pytest.param({}, id="default")],
    )
    def test_open_mfdataset_concat_dim(self, kwargs) -> None:
        with set_options(use_new_combine_kwarg_defaults=True):
            with create_tmp_file() as tmp1:
                with create_tmp_file() as tmp2:
                    data = Dataset({"x": 0})
                    data.to_netcdf(tmp1)
                    Dataset({"x": np.nan}).to_netcdf(tmp2)
                    with open_mfdataset(
                        [tmp1, tmp2], **kwargs, combine="nested"
                    ) as actual:
                        assert_identical(data, actual)

    def test_open_dataset(self) -> None:
        original = Dataset({"foo": ("x", np.random.randn(10))})
        with create_tmp_file() as tmp:
            original.to_netcdf(tmp)
            with open_dataset(tmp, chunks={"x": 5}) as actual:
                assert isinstance(actual.foo.variable.data, dask_array_type)
                assert actual.foo.variable.data.chunks == ((5, 5),)
                assert_identical(original, actual)
            with open_dataset(tmp, chunks=5) as actual:
                assert_identical(original, actual)
            with open_dataset(tmp) as actual:
                assert isinstance(actual.foo.variable.data, np.ndarray)
                assert_identical(original, actual)

    def test_open_single_dataset(self) -> None:
        # Test for issue GH #1988. This makes sure that the
        # concat_dim is utilized when specified in open_mfdataset().
        rnddata = np.random.randn(10)
        original = Dataset({"foo": ("x", rnddata)})
        dim = DataArray([100], name="baz", dims="baz")
        expected = Dataset(
            {"foo": (("baz", "x"), rnddata[np.newaxis, :])}, {"baz": [100]}
        )
        with create_tmp_file() as tmp:
            original.to_netcdf(tmp)
            with open_mfdataset(
                [tmp], concat_dim=dim, data_vars="all", combine="nested"
            ) as actual:
                assert_identical(expected, actual)

    def test_open_multi_dataset(self) -> None:
        # Test for issue GH #1988 and #2647. This makes sure that the
        # concat_dim is utilized when specified in open_mfdataset().
        # The additional wrinkle is to ensure that a length greater
        # than one is tested as well due to numpy's implicit casting
        # of 1-length arrays to booleans in tests, which allowed
        # #2647 to still pass the test_open_single_dataset(),
        # which is itself still needed as-is because the original
        # bug caused one-length arrays to not be used correctly
        # in concatenation.
        rnddata = np.random.randn(10)
        original = Dataset({"foo": ("x", rnddata)})
        dim = DataArray([100, 150], name="baz", dims="baz")
        expected = Dataset(
            {"foo": (("baz", "x"), np.tile(rnddata[np.newaxis, :], (2, 1)))},
            {"baz": [100, 150]},
        )
        with create_tmp_file() as tmp1, create_tmp_file() as tmp2:
            original.to_netcdf(tmp1)
            original.to_netcdf(tmp2)
            with open_mfdataset(
                [tmp1, tmp2], concat_dim=dim, data_vars="all", combine="nested"
            ) as actual:
                assert_identical(expected, actual)

    @requires_cftime
    def test_open_dataset_cftime_autochunk(self) -> None:
        """Create a dataset with cftime datetime objects and
        ensure that auto-chunking works correctly."""
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
        with self.roundtrip(original, open_kwargs={"chunks": "auto"}) as actual:
            assert isinstance(actual.time_bnds.variable.data, dask_array_type)
            assert _contains_cftime_datetimes(actual.time)
            assert_identical(original, actual)

    # Flaky test. Very open to contributions on fixing this
    @pytest.mark.flaky
    def test_dask_roundtrip(self) -> None:
        with create_tmp_file() as tmp:
            data = create_test_data()
            data.to_netcdf(tmp)
            chunks = {"dim1": 4, "dim2": 4, "dim3": 4, "time": 10}
            with open_dataset(tmp, chunks=chunks) as dask_ds:
                assert_identical(data, dask_ds)
                with create_tmp_file() as tmp2:
                    dask_ds.to_netcdf(tmp2)
                    with open_dataset(tmp2) as on_disk:
                        assert_identical(data, on_disk)

    def test_deterministic_names(self) -> None:
        with create_tmp_file() as tmp:
            data = create_test_data()
            data.to_netcdf(tmp)
            with open_mfdataset(tmp, combine="by_coords") as ds:
                original_names = {k: v.data.name for k, v in ds.data_vars.items()}
            with open_mfdataset(tmp, combine="by_coords") as ds:
                repeat_names = {k: v.data.name for k, v in ds.data_vars.items()}
            for var_name, dask_name in original_names.items():
                assert var_name in dask_name
                assert dask_name[:13] == "open_dataset-"
            assert original_names == repeat_names

    def test_dataarray_compute(self) -> None:
        # Test DataArray.compute() on dask backend.
        # The test for Dataset.compute() is already in DatasetIOBase;
        # however dask is the only tested backend which supports DataArrays
        actual = DataArray([1, 2]).chunk()
        computed = actual.compute()
        assert not actual._in_memory
        assert computed._in_memory
        assert_allclose(actual, computed, decode_bytes=False)

    def test_save_mfdataset_compute_false_roundtrip(self) -> None:
        from dask.delayed import Delayed

        original = Dataset({"foo": ("x", np.random.randn(10))}).chunk()
        datasets = [original.isel(x=slice(5)), original.isel(x=slice(5, 10))]
        with create_tmp_file(allow_cleanup_failure=ON_WINDOWS) as tmp1:
            with create_tmp_file(allow_cleanup_failure=ON_WINDOWS) as tmp2:
                delayed_obj = save_mfdataset(
                    datasets, [tmp1, tmp2], engine=self.engine, compute=False
                )
                assert isinstance(delayed_obj, Delayed)
                delayed_obj.compute()
                with open_mfdataset(
                    [tmp1, tmp2], combine="nested", concat_dim="x"
                ) as actual:
                    assert_identical(actual, original)

    def test_load_dataset(self) -> None:
        with create_tmp_file() as tmp:
            original = Dataset({"foo": ("x", np.random.randn(10))})
            original.to_netcdf(tmp)
            ds = load_dataset(tmp)
            assert_identical(original, ds)
            # this would fail if we used open_dataset instead of load_dataset
            ds.to_netcdf(tmp)

    def test_load_dataarray(self) -> None:
        with create_tmp_file() as tmp:
            original = DataArray(np.random.randn(10), dims=["x"])
            original.to_netcdf(tmp)
            da = load_dataarray(tmp)
            assert_identical(original, da)
            # this would fail if we used open_dataarray instead of
            # load_dataarray
            da.to_netcdf(tmp)

    def test_load_datatree(self) -> None:
        with create_tmp_file() as tmp:
            original = DataTree(Dataset({"foo": ("x", np.random.randn(10))}))
            original.to_netcdf(tmp)
            dt = load_datatree(tmp)
            xr.testing.assert_identical(original, dt)
            # this would fail if we used open_datatree instead of
            # load_datatree
            dt.to_netcdf(tmp)

    @pytest.mark.skipif(
        ON_WINDOWS,
        reason="counting number of tasks in graph fails on windows for some reason",
    )
    @pytest.mark.skip_with_dask_array
    def test_inline_array(self) -> None:
        with create_tmp_file() as tmp:
            original = Dataset({"foo": ("x", np.random.randn(10))})
            original.to_netcdf(tmp)
            chunks = {"time": 10}

            def num_graph_nodes(obj):
                return len(obj.__dask_graph__())

            with (
                open_dataset(tmp, inline_array=False, chunks=chunks) as not_inlined_ds,
                open_dataset(tmp, inline_array=True, chunks=chunks) as inlined_ds,
            ):
                assert num_graph_nodes(inlined_ds) < num_graph_nodes(not_inlined_ds)

            with (
                open_dataarray(
                    tmp, inline_array=False, chunks=chunks
                ) as not_inlined_da,
                open_dataarray(tmp, inline_array=True, chunks=chunks) as inlined_da,
            ):
                assert num_graph_nodes(inlined_da) < num_graph_nodes(not_inlined_da)


class MiscObject:
    pass


@requires_netCDF4
class TestValidateAttrs:
    def test_validating_attrs(self) -> None:
        def new_dataset():
            return Dataset({"data": ("y", np.arange(10.0))}, {"y": np.arange(10)})

        def new_dataset_and_dataset_attrs():
            ds = new_dataset()
            return ds, ds.attrs

        def new_dataset_and_data_attrs():
            ds = new_dataset()
            return ds, ds.data.attrs

        def new_dataset_and_coord_attrs():
            ds = new_dataset()
            return ds, ds.coords["y"].attrs

        for new_dataset_and_attrs in [
            new_dataset_and_dataset_attrs,
            new_dataset_and_data_attrs,
            new_dataset_and_coord_attrs,
        ]:
            ds, attrs = new_dataset_and_attrs()

            attrs[123] = "test"
            with pytest.raises(TypeError, match=r"Invalid name for attr: 123"):
                ds.to_netcdf("test.nc")

            ds, attrs = new_dataset_and_attrs()
            attrs[MiscObject()] = "test"
            with pytest.raises(TypeError, match=r"Invalid name for attr: "):
                ds.to_netcdf("test.nc")

            ds, attrs = new_dataset_and_attrs()
            attrs[""] = "test"
            with pytest.raises(ValueError, match=r"Invalid name for attr '':"):
                ds.to_netcdf("test.nc")

            # This one should work
            ds, attrs = new_dataset_and_attrs()
            attrs["test"] = "test"
            with create_tmp_file() as tmp_file:
                ds.to_netcdf(tmp_file)

            ds, attrs = new_dataset_and_attrs()
            attrs["test"] = {"a": 5}
            with pytest.raises(TypeError, match=r"Invalid value for attr 'test'"):
                ds.to_netcdf("test.nc")

            ds, attrs = new_dataset_and_attrs()
            attrs["test"] = MiscObject()
            with pytest.raises(TypeError, match=r"Invalid value for attr 'test'"):
                ds.to_netcdf("test.nc")

            ds, attrs = new_dataset_and_attrs()
            attrs["test"] = 5
            with create_tmp_file() as tmp_file:
                ds.to_netcdf(tmp_file)

            ds, attrs = new_dataset_and_attrs()
            attrs["test"] = 3.14
            with create_tmp_file() as tmp_file:
                ds.to_netcdf(tmp_file)

            ds, attrs = new_dataset_and_attrs()
            attrs["test"] = [1, 2, 3, 4]
            with create_tmp_file() as tmp_file:
                ds.to_netcdf(tmp_file)

            ds, attrs = new_dataset_and_attrs()
            attrs["test"] = (1.9, 2.5)
            with create_tmp_file() as tmp_file:
                ds.to_netcdf(tmp_file)

            ds, attrs = new_dataset_and_attrs()
            attrs["test"] = np.arange(5)
            with create_tmp_file() as tmp_file:
                ds.to_netcdf(tmp_file)

            ds, attrs = new_dataset_and_attrs()
            attrs["test"] = "This is a string"
            with create_tmp_file() as tmp_file:
                ds.to_netcdf(tmp_file)

            ds, attrs = new_dataset_and_attrs()
            attrs["test"] = ""
            with create_tmp_file() as tmp_file:
                ds.to_netcdf(tmp_file)


@requires_scipy_or_netCDF4
class TestDataArrayToNetCDF:
    def test_dataarray_to_netcdf_no_name(self) -> None:
        original_da = DataArray(np.arange(12).reshape((3, 4)))

        with create_tmp_file() as tmp:
            original_da.to_netcdf(tmp)

            with open_dataarray(tmp) as loaded_da:
                assert_identical(original_da, loaded_da)

    def test_dataarray_to_netcdf_with_name(self) -> None:
        original_da = DataArray(np.arange(12).reshape((3, 4)), name="test")

        with create_tmp_file() as tmp:
            original_da.to_netcdf(tmp)

            with open_dataarray(tmp) as loaded_da:
                assert_identical(original_da, loaded_da)

    def test_dataarray_to_netcdf_coord_name_clash(self) -> None:
        original_da = DataArray(
            np.arange(12).reshape((3, 4)), dims=["x", "y"], name="x"
        )

        with create_tmp_file() as tmp:
            original_da.to_netcdf(tmp)

            with open_dataarray(tmp) as loaded_da:
                assert_identical(original_da, loaded_da)

    def test_open_dataarray_options(self) -> None:
        data = DataArray(np.arange(5), coords={"y": ("x", range(5))}, dims=["x"])

        with create_tmp_file() as tmp:
            data.to_netcdf(tmp)

            expected = data.drop_vars("y")
            with open_dataarray(tmp, drop_variables=["y"]) as loaded:
                assert_identical(expected, loaded)

    @requires_scipy
    def test_dataarray_to_netcdf_return_bytes(self) -> None:
        # regression test for GH1410
        data = xr.DataArray([1, 2, 3])
        output = data.to_netcdf(engine="scipy")
        assert isinstance(output, memoryview)

    def test_dataarray_to_netcdf_no_name_pathlib(self) -> None:
        original_da = DataArray(np.arange(12).reshape((3, 4)))

        with create_tmp_file() as tmps:
            tmp = Path(tmps)
            original_da.to_netcdf(tmp)

            with open_dataarray(tmp) as loaded_da:
                assert_identical(original_da, loaded_da)


@requires_scipy_or_netCDF4
def test_no_warning_from_dask_effective_get() -> None:
    with create_tmp_file() as tmpfile:
        with assert_no_warnings():
            ds = Dataset()
            ds.to_netcdf(tmpfile)


@requires_scipy_or_netCDF4
def test_source_encoding_always_present() -> None:
    # Test for GH issue #2550.
    rnddata = np.random.randn(10)
    original = Dataset({"foo": ("x", rnddata)})
    with create_tmp_file() as tmp:
        original.to_netcdf(tmp)
        with open_dataset(tmp) as ds:
            assert ds.encoding["source"] == tmp


@requires_scipy_or_netCDF4
def test_source_encoding_always_present_with_pathlib() -> None:
    # Test for GH issue #5888.
    rnddata = np.random.randn(10)
    original = Dataset({"foo": ("x", rnddata)})
    with create_tmp_file() as tmp:
        original.to_netcdf(tmp)
        with open_dataset(Path(tmp)) as ds:
            assert ds.encoding["source"] == tmp


@requires_h5netcdf
@requires_fsspec
@requires_dask  # TODO remove after https://github.com/pydata/xarray/issues/9038
def test_source_encoding_always_present_with_fsspec() -> None:
    import fsspec

    rnddata = np.random.randn(10)
    original = Dataset({"foo": ("x", rnddata)})
    with create_tmp_file() as tmp:
        original.to_netcdf(tmp)

        fs = fsspec.filesystem("file")
        with fs.open(tmp) as f, open_dataset(f) as ds:
            assert ds.encoding["source"] == tmp
        with fs.open(tmp) as f, open_mfdataset([f]) as ds:
            assert "foo" in ds


def _assert_no_dates_out_of_range_warning(record):
    undesired_message = "dates out of range"
    for warning in record:
        assert undesired_message not in str(warning.message)


@requires_scipy_or_netCDF4
@pytest.mark.filterwarnings("ignore:deallocating CachingFileManager")
@pytest.mark.parametrize("calendar", _STANDARD_CALENDARS)
def test_use_cftime_standard_calendar_default_in_range(calendar) -> None:
    x = [0, 1]
    time = [0, 720]
    units_date = "2000-01-01"
    units = "days since 2000-01-01"
    original = DataArray(x, [("time", time)], name="x").to_dataset()
    for v in ["x", "time"]:
        original[v].attrs["units"] = units
        original[v].attrs["calendar"] = calendar

    x_timedeltas = np.array(x).astype("timedelta64[D]")
    time_timedeltas = np.array(time).astype("timedelta64[D]")
    decoded_x = np.datetime64(units_date, "ns") + x_timedeltas
    decoded_time = np.datetime64(units_date, "ns") + time_timedeltas
    expected_x = DataArray(decoded_x, [("time", decoded_time)], name="x")
    expected_time = DataArray(decoded_time, [("time", decoded_time)], name="time")

    with create_tmp_file() as tmp_file:
        original.to_netcdf(tmp_file)
        with warnings.catch_warnings(record=True) as record:
            with open_dataset(tmp_file) as ds:
                assert_identical(expected_x, ds.x)
                assert_identical(expected_time, ds.time)
            _assert_no_dates_out_of_range_warning(record)


@requires_cftime
@requires_scipy_or_netCDF4
@pytest.mark.parametrize("calendar", ["standard", "gregorian"])
def test_use_cftime_standard_calendar_default_out_of_range(calendar) -> None:
    # todo: check, if we still need to test for two dates
    import cftime

    x = [0, 1]
    time = [0, 720]
    units = "days since 1582-01-01"
    original = DataArray(x, [("time", time)], name="x").to_dataset()
    for v in ["x", "time"]:
        original[v].attrs["units"] = units
        original[v].attrs["calendar"] = calendar

    decoded_x = cftime.num2date(x, units, calendar, only_use_cftime_datetimes=True)
    decoded_time = cftime.num2date(
        time, units, calendar, only_use_cftime_datetimes=True
    )
    expected_x = DataArray(decoded_x, [("time", decoded_time)], name="x")
    expected_time = DataArray(decoded_time, [("time", decoded_time)], name="time")

    with create_tmp_file() as tmp_file:
        original.to_netcdf(tmp_file)
        # load eagerly: the warning is raised again when lazy data is decoded
        with pytest.warns(SerializationWarning):
            ds = load_dataset(tmp_file)
        assert_identical(expected_x, ds.x)
        assert_identical(expected_time, ds.time)


@requires_cftime
@requires_scipy_or_netCDF4
@pytest.mark.parametrize("calendar", _ALL_CALENDARS)
@pytest.mark.parametrize("units_year", [1500, 2000, 2500])
def test_use_cftime_true(calendar, units_year) -> None:
    import cftime

    x = [0, 1]
    time = [0, 720]
    units = f"days since {units_year}-01-01"
    original = DataArray(x, [("time", time)], name="x").to_dataset()
    for v in ["x", "time"]:
        original[v].attrs["units"] = units
        original[v].attrs["calendar"] = calendar

    decoded_x = cftime.num2date(x, units, calendar, only_use_cftime_datetimes=True)
    decoded_time = cftime.num2date(
        time, units, calendar, only_use_cftime_datetimes=True
    )
    expected_x = DataArray(decoded_x, [("time", decoded_time)], name="x")
    expected_time = DataArray(decoded_time, [("time", decoded_time)], name="time")

    with create_tmp_file() as tmp_file:
        original.to_netcdf(tmp_file)
        with warnings.catch_warnings(record=True) as record:
            decoder = CFDatetimeCoder(use_cftime=True)
            with open_dataset(tmp_file, decode_times=decoder) as ds:
                assert_identical(expected_x, ds.x)
                assert_identical(expected_time, ds.time)
            _assert_no_dates_out_of_range_warning(record)


@requires_scipy_or_netCDF4
@pytest.mark.parametrize("calendar", _STANDARD_CALENDARS)
@pytest.mark.xfail(
    has_numpy_2, reason="https://github.com/pandas-dev/pandas/issues/56996"
)
def test_use_cftime_false_standard_calendar_in_range(calendar) -> None:
    x = [0, 1]
    time = [0, 720]
    units_date = "2000-01-01"
    units = "days since 2000-01-01"
    original = DataArray(x, [("time", time)], name="x").to_dataset()
    for v in ["x", "time"]:
        original[v].attrs["units"] = units
        original[v].attrs["calendar"] = calendar

    x_timedeltas = np.array(x).astype("timedelta64[D]")
    time_timedeltas = np.array(time).astype("timedelta64[D]")
    decoded_x = np.datetime64(units_date, "ns") + x_timedeltas
    decoded_time = np.datetime64(units_date, "ns") + time_timedeltas
    expected_x = DataArray(decoded_x, [("time", decoded_time)], name="x")
    expected_time = DataArray(decoded_time, [("time", decoded_time)], name="time")

    with create_tmp_file() as tmp_file:
        original.to_netcdf(tmp_file)
        with warnings.catch_warnings(record=True) as record:
            coder = xr.coders.CFDatetimeCoder(use_cftime=False)
            with open_dataset(tmp_file, decode_times=coder) as ds:
                assert_identical(expected_x, ds.x)
                assert_identical(expected_time, ds.time)
            _assert_no_dates_out_of_range_warning(record)


@requires_scipy_or_netCDF4
@pytest.mark.filterwarnings("ignore:deallocating CachingFileManager")
@pytest.mark.parametrize("calendar", ["standard", "gregorian"])
def test_use_cftime_false_standard_calendar_out_of_range(calendar) -> None:
    x = [0, 1]
    time = [0, 720]
    units = "days since 1582-01-01"
    original = DataArray(x, [("time", time)], name="x").to_dataset()
    for v in ["x", "time"]:
        original[v].attrs["units"] = units
        original[v].attrs["calendar"] = calendar

    with create_tmp_file() as tmp_file:
        original.to_netcdf(tmp_file)
        decoder = CFDatetimeCoder(use_cftime=False)
        with pytest.raises((OutOfBoundsDatetime, ValueError)):
            open_dataset(tmp_file, decode_times=decoder)


@requires_scipy_or_netCDF4
@pytest.mark.filterwarnings("ignore:deallocating CachingFileManager")
@pytest.mark.parametrize("calendar", _NON_STANDARD_CALENDARS)
@pytest.mark.parametrize("units_year", [1500, 2000, 2500])
def test_use_cftime_false_nonstandard_calendar(calendar, units_year) -> None:
    x = [0, 1]
    time = [0, 720]
    units = f"days since {units_year}"
    original = DataArray(x, [("time", time)], name="x").to_dataset()
    for v in ["x", "time"]:
        original[v].attrs["units"] = units
        original[v].attrs["calendar"] = calendar

    with create_tmp_file() as tmp_file:
        original.to_netcdf(tmp_file)
        decoder = CFDatetimeCoder(use_cftime=False)
        with pytest.raises((OutOfBoundsDatetime, ValueError)):
            open_dataset(tmp_file, decode_times=decoder)


@pytest.mark.parametrize("engine", ["netcdf4", "scipy"])
def test_invalid_netcdf_raises(engine) -> None:
    data = create_test_data()
    with pytest.raises(ValueError, match=r"unrecognized option 'invalid_netcdf'"):
        data.to_netcdf("foo.nc", engine=engine, invalid_netcdf=True)


@requires_netCDF4
@pytest.mark.parametrize("str_type", (str, np.str_))
def test_write_file_from_np_str(str_type: type[str | np.str_], tmpdir: str) -> None:
    # https://github.com/pydata/xarray/pull/5264
    scenarios = [str_type(v) for v in ["scenario_a", "scenario_b", "scenario_c"]]
    years = range(2015, 2100 + 1)
    tdf = pd.DataFrame(
        data=np.random.random((len(scenarios), len(years))),
        columns=years,
        index=scenarios,
    )
    tdf.index.name = "scenario"
    tdf.columns.name = "year"
    tdf_series = tdf.stack()
    tdf_series.name = "tas"  # type: ignore[union-attr]

    txr = tdf_series.to_xarray()

    txr.to_netcdf(tmpdir.join("test.nc"))


@requires_netCDF4
@requires_dask
@pytest.mark.usefixtures("default_zarr_format")
def test_pickle_open_mfdataset_dataset():
    with open_example_mfdataset(["bears.nc"]) as ds:
        assert_identical(ds, pickle.loads(pickle.dumps(ds)))


def test_get_mtime_non_file_paths() -> None:
    """_get_mtime should not raise on non-file paths (gh#11386)."""
    from xarray.backends.api import _get_mtime

    # Non-existent local path should return None, not crash
    assert _get_mtime("/nonexistent/path.nc") is None

    # GDAL virtual filesystem paths are not real files
    assert _get_mtime("/vsicurl/https://example.com/file.nc") is None
