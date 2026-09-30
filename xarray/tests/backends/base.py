from __future__ import annotations

import contextlib
import itertools
import math
import os.path
import pickle
import re
import shutil
import sys
import tempfile
import warnings
from collections.abc import Generator, Iterator
from contextlib import ExitStack
from io import BytesIO
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import pytest

import xarray as xr
from xarray import DataArray, Dataset, load_dataset, open_dataset, open_mfdataset
from xarray.backends.netcdf3 import _nc3_dtype_coercions
from xarray.coders import CFTimedeltaCoder
from xarray.coding.strings import check_vlen_dtype, create_vlen_dtype
from xarray.coding.variables import SerializationWarning
from xarray.compat.npcompat import HAS_STRING_DTYPE
from xarray.conventions import encode_dataset_coordinates
from xarray.core import indexing
from xarray.core.indexes import PandasIndex
from xarray.core.types import PDDatetimeUnitOptions
from xarray.core.utils import module_available
from xarray.tests import (
    assert_allclose,
    assert_array_equal,
    assert_equal,
    assert_identical,
    dask_array_api,
    dask_array_type,
    has_dask,
    raise_if_dask_computes,
    requires_cftime,
    requires_dask,
    requires_iris,
    requires_netcdf,
    requires_netCDF4,
    requires_scipy,
)
from xarray.tests.test_dataset import create_test_data

with contextlib.suppress(ImportError):
    import netCDF4 as nc4

with contextlib.suppress(ImportError):
    import dask


if TYPE_CHECKING:
    from xarray.backends.api import T_NetcdfEngine, T_NetcdfTypes


# These marks require the test to use the ``default_zarr_format`` fixture.
def skip_if_zarr_format_3(reason: str, condition: bool = True) -> pytest.MarkDecorator:
    return pytest.mark.skip_if_param(
        default_zarr_format=3,
        reason=f"Unsupported with zarr_format=3: {reason}",
        condition=condition,
    )


def skip_if_zarr_format_2(reason: str, condition: bool = True) -> pytest.MarkDecorator:
    return pytest.mark.skip_if_param(
        default_zarr_format=2,
        reason=f"Unsupported with zarr_format=2: {reason}",
        condition=condition,
    )


# Requires the test to use the ``tmp_store`` fixture.
skip_if_zip_store = pytest.mark.skip_if_param(
    tmp_store="ZipStore",
    reason="zarr-python 3.x doesn't support reopening ZipStore with a new mode.",
)


ON_WINDOWS = sys.platform == "win32"


default_value = object()


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def open_example_dataset(name, *args, **kwargs) -> Dataset:
    return open_dataset(os.path.join(DATA_DIR, name), *args, **kwargs)


def open_example_mfdataset(names, *args, **kwargs) -> Dataset:
    return open_mfdataset(
        [os.path.join(DATA_DIR, name) for name in names],
        *args,
        **kwargs,
    )


def create_masked_and_scaled_data(dtype: np.dtype) -> Dataset:
    x = np.array([np.nan, np.nan, 10, 10.1, 10.2], dtype=dtype)
    encoding = {
        "_FillValue": -1,
        "add_offset": dtype.type(10),
        "scale_factor": dtype.type(0.1),
        "dtype": "i2",
    }
    return Dataset({"x": ("t", x, {}, encoding)})


def create_encoded_masked_and_scaled_data(dtype: np.dtype) -> Dataset:
    attributes = {
        "_FillValue": -1,
        "add_offset": dtype.type(10),
        "scale_factor": dtype.type(0.1),
    }
    return Dataset(
        {"x": ("t", np.array([-1, -1, 0, 1, 2], dtype=np.int16), attributes)}
    )


def create_unsigned_masked_scaled_data(dtype: np.dtype) -> Dataset:
    encoding = {
        "_FillValue": -1,
        "_Unsigned": "true",
        "dtype": "i1",
        "add_offset": dtype.type(10),
        "scale_factor": dtype.type(0.1),
    }
    x = np.array([10.0, 10.1, 22.7, 22.8, np.nan], dtype=dtype)
    return Dataset({"x": ("t", x, {}, encoding)})


def create_encoded_unsigned_masked_scaled_data(dtype: np.dtype) -> Dataset:
    # These are values as written to the file: the _FillValue will
    # be represented in the signed form.
    attributes = {
        "_FillValue": -1,
        "_Unsigned": "true",
        "add_offset": dtype.type(10),
        "scale_factor": dtype.type(0.1),
    }
    # Create unsigned data corresponding to [0, 1, 127, 128, 255] unsigned
    sb = np.asarray([0, 1, 127, -128, -1], dtype="i1")
    return Dataset({"x": ("t", sb, attributes)})


def create_bad_unsigned_masked_scaled_data(dtype: np.dtype) -> Dataset:
    encoding = {
        "_FillValue": 255,
        "_Unsigned": True,
        "dtype": "i1",
        "add_offset": dtype.type(10),
        "scale_factor": dtype.type(0.1),
    }
    x = np.array([10.0, 10.1, 22.7, 22.8, np.nan], dtype=dtype)
    return Dataset({"x": ("t", x, {}, encoding)})


def create_bad_encoded_unsigned_masked_scaled_data(dtype: np.dtype) -> Dataset:
    # These are values as written to the file: the _FillValue will
    # be represented in the signed form.
    attributes = {
        "_FillValue": -1,
        "_Unsigned": True,
        "add_offset": dtype.type(10),
        "scale_factor": dtype.type(0.1),
    }
    # Create signed data corresponding to [0, 1, 127, 128, 255] unsigned
    sb = np.asarray([0, 1, 127, -128, -1], dtype="i1")
    return Dataset({"x": ("t", sb, attributes)})


def create_signed_masked_scaled_data(dtype: np.dtype) -> Dataset:
    encoding = {
        "_FillValue": -127,
        "_Unsigned": "false",
        "dtype": "i1",
        "add_offset": dtype.type(10),
        "scale_factor": dtype.type(0.1),
    }
    x = np.array([-1.0, 10.1, 22.7, np.nan], dtype=dtype)
    return Dataset({"x": ("t", x, {}, encoding)})


def create_encoded_signed_masked_scaled_data(dtype: np.dtype) -> Dataset:
    # These are values as written to the file: the _FillValue will
    # be represented in the signed form.
    attributes = {
        "_FillValue": -127,
        "_Unsigned": "false",
        "add_offset": dtype.type(10),
        "scale_factor": dtype.type(0.1),
    }
    # Create signed data corresponding to [0, 1, 127, 128, 255] unsigned
    sb = np.asarray([-110, 1, 127, -127], dtype="i1")
    return Dataset({"x": ("t", sb, attributes)})


def create_unsigned_false_masked_scaled_data(dtype: np.dtype) -> Dataset:
    encoding = {
        "_FillValue": 255,
        "_Unsigned": "false",
        "dtype": "u1",
        "add_offset": dtype.type(10),
        "scale_factor": dtype.type(0.1),
    }
    x = np.array([-1.0, 10.1, 22.7, np.nan], dtype=dtype)
    return Dataset({"x": ("t", x, {}, encoding)})


def create_encoded_unsigned_false_masked_scaled_data(dtype: np.dtype) -> Dataset:
    # These are values as written to the file: the _FillValue will
    # be represented in the unsigned form.
    attributes = {
        "_FillValue": 255,
        "_Unsigned": "false",
        "add_offset": dtype.type(10),
        "scale_factor": dtype.type(0.1),
    }
    # Create unsigned data corresponding to [-110, 1, 127, 255] signed
    sb = np.asarray([146, 1, 127, 255], dtype="u1")
    return Dataset({"x": ("t", sb, attributes)})


def create_boolean_data() -> Dataset:
    attributes = {"units": "-"}
    return Dataset(
        {
            "x": (
                ("t", "x"),
                [[False, True, False, True], [True, False, False, True]],
                attributes,
            )
        }
    )


MASK_AND_SCALE_DATA = [
    (
        create_unsigned_masked_scaled_data,
        create_encoded_unsigned_masked_scaled_data,
    ),
    pytest.param(
        create_bad_unsigned_masked_scaled_data,
        create_bad_encoded_unsigned_masked_scaled_data,
        marks=pytest.mark.xfail(reason="Bad _Unsigned attribute."),
    ),
    (
        create_signed_masked_scaled_data,
        create_encoded_signed_masked_scaled_data,
    ),
    (
        create_unsigned_false_masked_scaled_data,
        create_encoded_unsigned_false_masked_scaled_data,
    ),
    (create_masked_and_scaled_data, create_encoded_masked_and_scaled_data),
]


class NetCDF3Only:
    netcdf3_formats: tuple[T_NetcdfTypes, ...] = ("NETCDF3_CLASSIC", "NETCDF3_64BIT")

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="NetCDF backends don't support async loading")
    async def test_load_async(self) -> None:
        pass

    @requires_scipy
    def test_dtype_coercion_error(self) -> None:
        """Failing dtype coercion should lead to an error"""
        for dtype, format in itertools.product(
            _nc3_dtype_coercions, self.netcdf3_formats
        ):
            if dtype == "bool":
                # coerced upcast (bool to int8) ==> can never fail
                continue

            # Using the largest representable value, create some data that will
            # no longer compare equal after the coerced downcast
            maxval = np.iinfo(dtype).max
            x = np.array([0, 1, 2, maxval], dtype=dtype)
            ds = Dataset({"x": ("t", x, {})})

            with create_tmp_file(allow_cleanup_failure=False) as path:
                with pytest.raises(ValueError, match="could not safely cast"):
                    ds.to_netcdf(path, format=format)


class DatasetIOBase:
    engine: T_NetcdfEngine | None = None
    file_format: T_NetcdfTypes | None = None

    def create_store(self):
        raise NotImplementedError()

    @contextlib.contextmanager
    def roundtrip(
        self, data, save_kwargs=None, open_kwargs=None, allow_cleanup_failure=False
    ):
        if save_kwargs is None:
            save_kwargs = {}
        if open_kwargs is None:
            open_kwargs = {}

        with create_tmp_file(allow_cleanup_failure=allow_cleanup_failure) as path:
            self.save(data, path, **save_kwargs)
            with self.open(path, **open_kwargs) as ds:
                yield ds

    @contextlib.contextmanager
    def roundtrip_append(
        self, data, save_kwargs=None, open_kwargs=None, allow_cleanup_failure=False
    ):
        if save_kwargs is None:
            save_kwargs = {}
        if open_kwargs is None:
            open_kwargs = {}
        with create_tmp_file(allow_cleanup_failure=allow_cleanup_failure) as path:
            for i, key in enumerate(data.variables):
                mode = "a" if i > 0 else "w"
                self.save(data[[key]], path, mode=mode, **save_kwargs)
            with self.open(path, **open_kwargs) as ds:
                yield ds

    # The save/open methods may be overwritten below
    def save(self, dataset, path, **kwargs):
        return dataset.to_netcdf(
            path, engine=self.engine, format=self.file_format, **kwargs
        )

    @contextlib.contextmanager
    def open(self, path, **kwargs):
        with open_dataset(path, engine=self.engine, **kwargs) as ds:
            yield ds

    def test_zero_dimensional_variable(self) -> None:
        expected = create_test_data()
        expected["float_var"] = ([], 1.0e9, {"units": "units of awesome"})
        expected["bytes_var"] = ([], b"foobar")
        expected["string_var"] = ([], "foobar")
        with self.roundtrip(expected) as actual:
            assert_identical(expected, actual)

    def test_write_store(self) -> None:
        expected = create_test_data()
        with self.create_store() as store:
            expected.dump_to_store(store)
            # we need to cf decode the store because it has time and
            # non-dimension coordinates
            with xr.decode_cf(store) as actual:
                assert_allclose(expected, actual)

    def check_dtypes_roundtripped(self, expected, actual):
        for k in expected.variables:
            expected_dtype = expected.variables[k].dtype

            # For NetCDF3, the backend should perform dtype coercion
            if (
                isinstance(self, NetCDF3Only)
                and str(expected_dtype) in _nc3_dtype_coercions
            ):
                expected_dtype = np.dtype(_nc3_dtype_coercions[str(expected_dtype)])

            actual_dtype = actual.variables[k].dtype
            # TODO: check expected behavior for string dtypes more carefully
            string_kinds = {"O", "S", "U", "T"}
            assert expected_dtype == actual_dtype or (
                expected_dtype.kind in string_kinds
                and actual_dtype.kind in string_kinds
            )

    def test_roundtrip_test_data(self) -> None:
        expected = create_test_data()
        with self.roundtrip(expected) as actual:
            self.check_dtypes_roundtripped(expected, actual)
            assert_identical(expected, actual)

    def test_load(self) -> None:
        # Note: please keep this in sync with test_load_async below as much as possible!
        expected = create_test_data()

        @contextlib.contextmanager
        def assert_loads(vars=None):
            if vars is None:
                vars = expected
            with self.roundtrip(expected) as actual:
                for k, v in actual.variables.items():
                    # IndexVariables are eagerly loaded into memory
                    assert v._in_memory == (k in actual.dims)
                yield actual
                for k, v in actual.variables.items():
                    if k in vars:
                        assert v._in_memory
                assert_identical(expected, actual)

        with pytest.raises(AssertionError):
            # make sure the contextmanager works!
            with assert_loads() as ds:
                pass

        with assert_loads() as ds:
            ds.load()

        with assert_loads(["var1", "dim1", "dim2"]) as ds:
            ds["var1"].load()

        # verify we can read data even after closing the file
        with self.roundtrip(expected) as ds:
            actual = ds.load()
        assert_identical(expected, actual)

    @pytest.mark.asyncio
    async def test_load_async(self) -> None:
        # Note: please keep this in sync with test_load above as much as possible!

        # Copied from `test_load` on the base test class, but won't work for netcdf
        expected = create_test_data()

        @contextlib.contextmanager
        def assert_loads(vars=None):
            if vars is None:
                vars = expected
            with self.roundtrip(expected) as actual:
                for k, v in actual.variables.items():
                    # IndexVariables are eagerly loaded into memory
                    assert v._in_memory == (k in actual.dims)
                yield actual
                for k, v in actual.variables.items():
                    if k in vars:
                        assert v._in_memory
                assert_identical(expected, actual)

        with pytest.raises(AssertionError):
            # make sure the contextmanager works!
            with assert_loads() as ds:
                pass

        with assert_loads() as ds:
            await ds.load_async()

        with assert_loads(["var1", "dim1", "dim2"]) as ds:
            await ds["var1"].load_async()

        # verify we can read data even after closing the file
        with self.roundtrip(expected) as ds:
            actual = await ds.load_async()
        assert_identical(expected, actual)

    def test_dataset_compute(self) -> None:
        expected = create_test_data()

        with self.roundtrip(expected) as actual:
            # Test Dataset.compute()
            for k, v in actual.variables.items():
                # IndexVariables are eagerly cached
                assert v._in_memory == (k in actual.dims)

            computed = actual.compute()

            for k, v in actual.variables.items():
                assert v._in_memory == (k in actual.dims)
            for v in computed.variables.values():
                assert v._in_memory

            assert_identical(expected, actual)
            assert_identical(expected, computed)

    def test_pickle(self) -> None:
        expected = Dataset({"foo": ("x", [42])})
        with self.roundtrip(expected, allow_cleanup_failure=ON_WINDOWS) as roundtripped:
            # Windows doesn't like reopening an already open file
            raw_pickle = pickle.dumps(roundtripped)
            with pickle.loads(raw_pickle) as unpickled_ds:
                assert_identical(expected, unpickled_ds)

    def test_pickle_dataarray(self) -> None:
        expected = Dataset({"foo": ("x", [42])})
        with self.roundtrip(expected, allow_cleanup_failure=ON_WINDOWS) as roundtripped:
            raw_pickle = pickle.dumps(roundtripped["foo"])
            with pickle.loads(raw_pickle) as unpickled:
                assert_identical(expected["foo"], unpickled)

    def test_dataset_caching(self) -> None:
        expected = Dataset({"foo": ("x", [5, 6, 7])})
        with self.roundtrip(expected) as actual:
            assert isinstance(actual.foo.variable._data, indexing.MemoryCachedArray)
            assert not actual.foo.variable._in_memory
            _ = actual.foo.values  # cache
            assert actual.foo.variable._in_memory

        with self.roundtrip(expected, open_kwargs={"cache": False}) as actual:
            assert isinstance(actual.foo.variable._data, indexing.CopyOnWriteArray)
            assert not actual.foo.variable._in_memory
            _ = actual.foo.values  # no caching
            assert not actual.foo.variable._in_memory

    def test_roundtrip_None_variable(self) -> None:
        expected = Dataset({None: (("x", "y"), [[0, 1], [2, 3]])})
        with self.roundtrip(expected) as actual:
            assert_identical(expected, actual)

    def test_roundtrip_object_dtype(self) -> None:
        floats = np.array([0.0, 0.0, 1.0, 2.0, 3.0], dtype=object)
        floats_nans = np.array([np.nan, np.nan, 1.0, 2.0, 3.0], dtype=object)
        bytes_ = np.array([b"ab", b"cdef", b"g"], dtype=object)
        bytes_nans = np.array([b"ab", b"cdef", np.nan], dtype=object)
        strings = np.array(["ab", "cdef", "g"], dtype=object)
        strings_nans = np.array(["ab", "cdef", np.nan], dtype=object)
        all_nans = np.array([np.nan, np.nan], dtype=object)
        original = Dataset(
            {
                "floats": ("a", floats),
                "floats_nans": ("a", floats_nans),
                "bytes": ("b", bytes_),
                "bytes_nans": ("b", bytes_nans),
                "strings": ("b", strings),
                "strings_nans": ("b", strings_nans),
                "all_nans": ("c", all_nans),
                "nan": ([], np.nan),
            }
        )
        expected = original.copy(deep=True)
        with self.roundtrip(original) as actual:
            try:
                assert_identical(expected, actual)
            except AssertionError:
                # Most stores use '' for nans in strings, but some don't.
                # First try the ideal case (where the store returns exactly)
                # the original Dataset), then try a more realistic case.
                # This currently includes all netCDF files when encoding is not
                # explicitly set.
                # https://github.com/pydata/xarray/issues/1647
                # Also Zarr
                expected["bytes_nans"][-1] = b""
                expected["strings_nans"][-1] = ""
                assert_identical(expected, actual)

    def test_roundtrip_string_data(self) -> None:
        expected = Dataset({"x": ("t", ["ab", "cdef"])})
        with self.roundtrip(expected) as actual:
            assert_identical(expected, actual)

    @pytest.mark.skipif(not HAS_STRING_DTYPE, reason="requires StringDType")
    def test_roundtrip_stringdtype_data(self) -> None:
        # GH11199
        data = np.array(["ab", "cdef"], dtype=np.dtypes.StringDType())
        expected = Dataset({"x": ("t", data)})
        with self.roundtrip(expected) as actual:
            assert_identical(expected, actual)

    @pytest.mark.skipif(not HAS_STRING_DTYPE, reason="requires StringDType")
    def test_roundtrip_stringdtype_nulls(self) -> None:
        # GH11199 — null values in StringDType are written as empty strings
        data = np.array(["ab", None], dtype=np.dtypes.StringDType(na_object=None))
        ds = Dataset({"x": ("t", data)})
        with self.roundtrip(ds) as actual:
            expected = Dataset({"x": ("t", np.array(["ab", ""]))})
            assert_identical(expected, actual)

    @pytest.mark.skipif(not HAS_STRING_DTYPE, reason="requires StringDType")
    def test_roundtrip_stringdtype_with_na_object(self) -> None:
        # GH11199 — StringDType(na_object="") should roundtrip correctly
        data = np.array(["ab", "cdef"], dtype=np.dtypes.StringDType(na_object=""))
        expected = Dataset({"x": ("t", data)})
        with self.roundtrip(expected) as actual:
            assert_identical(expected, actual)

    def test_roundtrip_string_encoded_characters(self) -> None:
        expected = Dataset({"x": ("t", ["ab", "cdef"])})
        expected["x"].encoding["dtype"] = "S1"
        with self.roundtrip(expected) as actual:
            assert_identical(expected, actual)
            assert actual["x"].encoding["_Encoding"] == "utf-8"

        expected["x"].encoding["_Encoding"] = "ascii"
        with self.roundtrip(expected) as actual:
            assert_identical(expected, actual)
            assert actual["x"].encoding["_Encoding"] == "ascii"

    def test_roundtrip_numpy_datetime_data(self) -> None:
        times = pd.to_datetime(["2000-01-01", "2000-01-02", "NaT"], unit="ns")
        expected = Dataset({"t": ("t", times), "t0": times[0]})
        kwargs = {"encoding": {"t0": {"units": "days since 1950-01-01"}}}
        with self.roundtrip(expected, save_kwargs=kwargs) as actual:
            assert_identical(expected, actual)
            assert actual.t0.encoding["units"] == "days since 1950-01-01"

    @requires_cftime
    def test_roundtrip_cftime_datetime_data(self) -> None:
        from xarray.tests.test_coding_times import _all_cftime_date_types

        date_types = _all_cftime_date_types()
        for date_type in date_types.values():
            times = [date_type(1, 1, 1), date_type(1, 1, 2)]
            expected = Dataset({"t": ("t", times), "t0": times[0]})
            kwargs = {"encoding": {"t0": {"units": "days since 0001-01-01"}}}
            expected_decoded_t = np.array(times)
            expected_decoded_t0 = np.array([date_type(1, 1, 1)])
            expected_calendar = times[0].calendar

            with warnings.catch_warnings():
                if expected_calendar in {"proleptic_gregorian", "standard"}:
                    warnings.filterwarnings("ignore", "Unable to decode time axis")

                with self.roundtrip(expected, save_kwargs=kwargs) as actual:
                    # proleptic gregorian will be decoded into numpy datetime64
                    # fixing to expectations
                    if actual.t.dtype.kind == "M":
                        dtype = actual.t.dtype
                        expected_decoded_t = expected_decoded_t.astype(dtype)
                        expected_decoded_t0 = expected_decoded_t0.astype(dtype)
                    assert_array_equal(actual.t.values, expected_decoded_t)
                    assert (
                        actual.t.encoding["units"]
                        == "days since 0001-01-01 00:00:00.000000"
                    )
                    assert actual.t.encoding["calendar"] == expected_calendar
                    assert_array_equal(actual.t0.values, expected_decoded_t0)
                    assert actual.t0.encoding["units"] == "days since 0001-01-01"
                    assert actual.t.encoding["calendar"] == expected_calendar

    def test_roundtrip_timedelta_data(self) -> None:
        # todo: suggestion from review:
        #  roundtrip large microsecond or coarser resolution timedeltas,
        #  though we cannot test that until we fix the timedelta decoding
        #  to support large ranges
        time_deltas = pd.to_timedelta(["1h", "2h", "NaT"]).as_unit("s")  # type: ignore[arg-type, unused-ignore]
        encoding = {"units": "seconds"}
        expected = Dataset({"td": ("td", time_deltas), "td0": time_deltas[0]})
        expected["td"].encoding = encoding
        expected["td0"].encoding = encoding
        with self.roundtrip(
            expected, open_kwargs={"decode_timedelta": CFTimedeltaCoder(time_unit="s")}
        ) as actual:
            assert_identical(expected, actual)

    def test_roundtrip_timedelta_data_via_dtype(
        self, time_unit: PDDatetimeUnitOptions
    ) -> None:
        time_deltas = pd.to_timedelta(["1h", "2h", "NaT"]).as_unit(time_unit)  # type: ignore[arg-type, unused-ignore]
        expected = Dataset(
            {"td": ("td", time_deltas), "td0": time_deltas[0].to_numpy()}
        )
        with self.roundtrip(expected) as actual:
            assert_identical(expected, actual)

    def test_roundtrip_float64_data(self) -> None:
        expected = Dataset({"x": ("y", np.array([1.0, 2.0, np.pi], dtype="float64"))})
        with self.roundtrip(expected) as actual:
            assert_identical(expected, actual)

    @requires_netcdf
    def test_roundtrip_example_1_netcdf(self) -> None:
        with open_example_dataset("example_1.nc") as expected:
            with self.roundtrip(expected) as actual:
                # we allow the attributes to differ since that
                # will depend on the encoding used.  For example,
                # without CF encoding 'actual' will end up with
                # a dtype attribute.
                assert_equal(expected, actual)

    def test_roundtrip_coordinates(self) -> None:
        original = Dataset(
            {"foo": ("x", [0, 1])}, {"x": [2, 3], "y": ("a", [42]), "z": ("x", [4, 5])}
        )

        with self.roundtrip(original) as actual:
            assert_identical(original, actual)

        original["foo"].encoding["coordinates"] = "y"
        with self.roundtrip(original, open_kwargs={"decode_coords": False}) as expected:
            # check roundtripping when decode_coords=False
            with self.roundtrip(
                expected, open_kwargs={"decode_coords": False}
            ) as actual:
                assert_identical(expected, actual)

    def test_roundtrip_global_coordinates(self) -> None:
        original = Dataset(
            {"foo": ("x", [0, 1])}, {"x": [2, 3], "y": ("a", [42]), "z": ("x", [4, 5])}
        )
        with self.roundtrip(original) as actual:
            assert_identical(original, actual)

        # test that global "coordinates" is as expected
        _, attrs = encode_dataset_coordinates(original)
        assert attrs["coordinates"] == "y"

        # test warning when global "coordinates" is already set
        original.attrs["coordinates"] = "foo"
        with pytest.warns(SerializationWarning):
            _, attrs = encode_dataset_coordinates(original)
        assert attrs["coordinates"] == "foo"

    def test_roundtrip_coordinates_with_space(self) -> None:
        original = Dataset(coords={"x": 0, "y z": 1})
        expected = Dataset({"y z": 1}, {"x": 0})
        with pytest.warns(SerializationWarning):
            with self.roundtrip(original) as actual:
                assert_identical(expected, actual)

    def test_roundtrip_boolean_dtype(self) -> None:
        original = create_boolean_data()
        assert original["x"].dtype == "bool"
        with self.roundtrip(original) as actual:
            assert_identical(original, actual)
            assert actual["x"].dtype == "bool"
            # this checks for preserving dtype during second roundtrip
            # see https://github.com/pydata/xarray/issues/7652#issuecomment-1476956975
            with self.roundtrip(actual) as actual2:
                assert_identical(original, actual2)
                assert actual2["x"].dtype == "bool"
            with self.roundtrip(actual) as actual3:
                # GH10536
                assert_identical(original.transpose(), actual3.transpose())

    def test_orthogonal_indexing(self) -> None:
        in_memory = create_test_data()
        with self.roundtrip(in_memory) as on_disk:
            indexers = {"dim1": [1, 2, 0], "dim2": [3, 2, 0, 3], "dim3": np.arange(5)}
            expected = in_memory.isel(indexers)
            actual = on_disk.isel(**indexers)
            # make sure the array is not yet loaded into memory
            assert not actual["var1"].variable._in_memory
            assert_identical(expected, actual)
            # do it twice, to make sure we're switched from orthogonal -> numpy
            # when we cached the values
            actual = on_disk.isel(**indexers)
            assert_identical(expected, actual)

    def test_vectorized_indexing(self) -> None:
        in_memory = create_test_data()
        with self.roundtrip(in_memory) as on_disk:
            indexers = {
                "dim1": DataArray([0, 2, 0], dims="a"),
                "dim2": DataArray([0, 2, 3], dims="a"),
            }
            expected = in_memory.isel(indexers)
            actual = on_disk.isel(**indexers)
            # make sure the array is not yet loaded into memory
            assert not actual["var1"].variable._in_memory
            assert_identical(expected, actual.load())
            # do it twice, to make sure we're switched from
            # vectorized -> numpy when we cached the values
            actual = on_disk.isel(**indexers)
            assert_identical(expected, actual)

        def multiple_indexing(indexers):
            # make sure a sequence of lazy indexings certainly works.
            with self.roundtrip(in_memory) as on_disk:
                actual = on_disk["var3"]
                expected = in_memory["var3"]
                for ind in indexers:
                    actual = actual.isel(ind)
                    expected = expected.isel(ind)
                    # make sure the array is not yet loaded into memory
                    assert not actual.variable._in_memory
                assert_identical(expected, actual.load())

        # two-staged vectorized-indexing
        indexers2 = [
            {
                "dim1": DataArray([[0, 7], [2, 6], [3, 5]], dims=["a", "b"]),
                "dim3": DataArray([[0, 4], [1, 3], [2, 2]], dims=["a", "b"]),
            },
            {"a": DataArray([0, 1], dims=["c"]), "b": DataArray([0, 1], dims=["c"])},
        ]
        multiple_indexing(indexers2)

        # vectorized-slice mixed
        indexers3 = [
            {
                "dim1": DataArray([[0, 7], [2, 6], [3, 5]], dims=["a", "b"]),
                "dim3": slice(None, 10),
            }
        ]
        multiple_indexing(indexers3)

        # vectorized-integer mixed
        indexers4 = [
            {"dim3": 0},
            {"dim1": DataArray([[0, 7], [2, 6], [3, 5]], dims=["a", "b"])},
            {"a": slice(None, None, 2)},
        ]
        multiple_indexing(indexers4)

        # vectorized-integer mixed
        indexers5 = [
            {"dim3": 0},
            {"dim1": DataArray([[0, 7], [2, 6], [3, 5]], dims=["a", "b"])},
            {"a": 1, "b": 0},
        ]
        multiple_indexing(indexers5)

    def test_vectorized_indexing_negative_step(self, use_dask: bool) -> None:
        open_kwargs: dict[str, Any] | None = {"chunks": {}} if use_dask else None
        in_memory = create_test_data()

        def multiple_indexing(indexers):
            # make sure a sequence of lazy indexings certainly works.
            with self.roundtrip(in_memory, open_kwargs=open_kwargs) as on_disk:
                actual = on_disk["var3"]
                expected = in_memory["var3"]
                for ind in indexers:
                    actual = actual.isel(ind)
                    expected = expected.isel(ind)
                    # make sure the array is not yet loaded into memory
                    assert not actual.variable._in_memory
                assert_identical(expected, actual.load())

        # with negative step slice.
        indexers = [
            {
                "dim1": DataArray([[0, 7], [2, 6], [3, 5]], dims=["a", "b"]),
                "dim3": slice(-1, 1, -1),
            }
        ]
        multiple_indexing(indexers)

        # with negative step slice.
        indexers = [
            {
                "dim1": DataArray([[0, 7], [2, 6], [3, 5]], dims=["a", "b"]),
                "dim3": slice(-1, 1, -2),
            }
        ]
        multiple_indexing(indexers)

    def test_outer_indexing_reversed(self) -> None:
        # regression test for GH6560
        ds = xr.Dataset(
            {"z": (("t", "p", "y", "x"), np.ones((1, 1, 31, 40)))},
        )

        with self.roundtrip(ds) as on_disk:
            subset = on_disk.isel(t=[0], p=0).z[:, ::10, ::10][:, ::-1, :]
            assert subset.sizes == subset.load().sizes

    def test_isel_dataarray(self) -> None:
        # Make sure isel works lazily. GH:issue:1688
        in_memory = create_test_data()
        with self.roundtrip(in_memory) as on_disk:
            expected = in_memory.isel(dim2=in_memory["dim2"] < 3)
            actual = on_disk.isel(dim2=on_disk["dim2"] < 3)
            assert_identical(expected, actual)

    def test_empty_isel(self) -> None:
        # Make sure isel works lazily with empty indexer.
        # GH:issue:10867
        in_memory = xr.Dataset({"a": ("x", np.arange(4))}, coords={"x": np.arange(4)})
        with self.roundtrip(in_memory) as on_disk:
            expected = in_memory.isel(x=[])
            actual = on_disk.isel(x=[])
            assert_identical(expected, actual)

    def test_empty_isel_multidim(self) -> None:
        # Empty indexers must also work for multi-dimensional variables, where
        # the in-memory part of the decomposed indexer still has to line up with
        # the axes of the loaded array.
        # GH:issue:11625, GH:issue:9075
        in_memory = xr.Dataset(
            {"a": (("x", "y"), np.arange(12.0).reshape(4, 3))},
            coords={"x": np.arange(4), "y": np.arange(3)},
        )
        with self.roundtrip(in_memory) as on_disk:
            for x_indexer, y_indexer in (([], [0, 2]), ([1, 3], []), ([], [])):
                expected = in_memory.isel(x=x_indexer, y=y_indexer)
                actual = on_disk.isel(x=x_indexer, y=y_indexer)
                assert_identical(expected, actual)

    def validate_array_type(self, ds):

        # Make sure that only NumpyIndexingAdapter stores a bare np.ndarray.
        def find_and_validate_array(obj):
            # recursively called function. obj: array or array wrapper.
            if hasattr(obj, "array"):
                if isinstance(obj.array, indexing.ExplicitlyIndexed):
                    find_and_validate_array(obj.array)
                elif isinstance(obj.array, np.ndarray):
                    assert isinstance(obj, indexing.NumpyIndexingAdapter)
                elif has_dask and isinstance(obj.array, dask_array_type):
                    assert isinstance(obj, indexing.DaskIndexingAdapter)
                elif isinstance(obj.array, pd.Index):
                    assert isinstance(obj, indexing.PandasIndexingAdapter)
                else:
                    raise TypeError(f"{type(obj.array)} is wrapped by {type(obj)}")

        for v in ds.variables.values():
            find_and_validate_array(v._data)

    def test_array_type_after_indexing(self) -> None:
        in_memory = create_test_data()
        with self.roundtrip(in_memory) as on_disk:
            self.validate_array_type(on_disk)
            indexers = {"dim1": [1, 2, 0], "dim2": [3, 2, 0, 3], "dim3": np.arange(5)}
            expected = in_memory.isel(indexers)
            actual = on_disk.isel(**indexers)
            assert_identical(expected, actual)
            self.validate_array_type(actual)
            # do it twice, to make sure we're switched from orthogonal -> numpy
            # when we cached the values
            actual = on_disk.isel(**indexers)
            assert_identical(expected, actual)
            self.validate_array_type(actual)

    def test_dropna(self) -> None:
        # regression test for GH:issue:1694
        a = np.random.randn(4, 3)
        a[1, 1] = np.nan
        in_memory = xr.Dataset(
            {"a": (("y", "x"), a)}, coords={"y": np.arange(4), "x": np.arange(3)}
        )

        assert_identical(
            in_memory.dropna(dim="x"), in_memory.isel(x=slice(None, None, 2))
        )

        with self.roundtrip(in_memory) as on_disk:
            self.validate_array_type(on_disk)
            expected = in_memory.dropna(dim="x")
            actual = on_disk.dropna(dim="x")
            assert_identical(expected, actual)

    def test_ondisk_after_print(self) -> None:
        """Make sure print does not load file into memory"""
        in_memory = create_test_data()
        with self.roundtrip(in_memory) as on_disk:
            repr(on_disk)
            assert not on_disk["var1"]._in_memory


class CFEncodedBase(DatasetIOBase):
    def test_roundtrip_bytes_with_fill_value(self) -> None:
        values = np.array([b"ab", b"cdef", np.nan], dtype=object)
        encoding = {"_FillValue": b"X", "dtype": "S1"}
        original = Dataset({"x": ("t", values, {}, encoding)})
        expected = original.copy(deep=True)
        with self.roundtrip(original) as actual:
            assert_identical(expected, actual)

        original = Dataset({"x": ("t", values, {}, {"_FillValue": b""})})
        with self.roundtrip(original) as actual:
            assert_identical(expected, actual)

    def test_roundtrip_string_with_fill_value_nchar(self) -> None:
        values = np.array(["ab", "cdef", np.nan], dtype=object)
        expected = Dataset({"x": ("t", values)})

        encoding = {"dtype": "S1", "_FillValue": b"X"}
        original = Dataset({"x": ("t", values, {}, encoding)})
        # Not supported yet.
        with pytest.raises(NotImplementedError):
            with self.roundtrip(original) as actual:
                assert_identical(expected, actual)

    def test_roundtrip_empty_vlen_string_array(self) -> None:
        # checks preserving vlen dtype for empty arrays GH7862
        dtype = create_vlen_dtype(str)
        original = Dataset({"a": np.array([], dtype=dtype)})
        assert check_vlen_dtype(original["a"].dtype) is str
        with self.roundtrip(original) as actual:
            assert_identical(original, actual)
            if np.issubdtype(actual["a"].dtype, object):
                # only check metadata for capable backends
                # eg. NETCDF3 based backends do not roundtrip metadata
                if actual["a"].dtype.metadata is not None:
                    assert check_vlen_dtype(actual["a"].dtype) is str
            elif HAS_STRING_DTYPE:
                assert np.issubdtype(actual["a"].dtype, np.dtypes.StringDType())
            else:
                assert np.issubdtype(actual["a"].dtype, np.dtype("=U1"))

    @pytest.mark.parametrize("decoded_fn, encoded_fn", MASK_AND_SCALE_DATA)
    @pytest.mark.parametrize("dtype", [np.dtype("float64"), np.dtype("float32")])
    def test_roundtrip_mask_and_scale(self, decoded_fn, encoded_fn, dtype) -> None:
        decoded = decoded_fn(dtype)
        encoded = encoded_fn(dtype)
        if decoded["x"].encoding["dtype"] == "u1" and not (
            (self.engine == "netcdf4" and self.file_format is None)
            or self.file_format == "NETCDF4"
        ):
            pytest.skip("uint8 data can't be written to non-NetCDF4 data")

        with self.roundtrip(decoded) as actual:
            for k in decoded.variables:
                assert decoded.variables[k].dtype == actual.variables[k].dtype
                # CF _FillValue is always on-disk type
                assert (
                    decoded.variables[k].encoding["_FillValue"]
                    == actual.variables[k].encoding["_FillValue"]
                )
            assert_allclose(decoded, actual, decode_bytes=False)

        with self.roundtrip(decoded, open_kwargs=dict(decode_cf=False)) as actual:
            # TODO: this assumes that all roundtrips will first
            # encode.  Is that something we want to test for?
            for k in encoded.variables:
                assert encoded.variables[k].dtype == actual.variables[k].dtype
                # CF _FillValue is always on-disk type
                assert (
                    decoded.variables[k].encoding["_FillValue"]
                    == actual.variables[k].attrs["_FillValue"]
                )
            assert_allclose(encoded, actual, decode_bytes=False)

        with self.roundtrip(encoded, open_kwargs=dict(decode_cf=False)) as actual:
            for k in encoded.variables:
                assert encoded.variables[k].dtype == actual.variables[k].dtype
                # CF _FillValue is always on-disk type
                assert (
                    encoded.variables[k].attrs["_FillValue"]
                    == actual.variables[k].attrs["_FillValue"]
                )
            assert_allclose(encoded, actual, decode_bytes=False)

        # make sure roundtrip encoding didn't change the
        # original dataset.
        assert_allclose(encoded, encoded_fn(dtype), decode_bytes=False)

        with self.roundtrip(encoded) as actual:
            for k in decoded.variables:
                assert decoded.variables[k].dtype == actual.variables[k].dtype
            assert_allclose(decoded, actual, decode_bytes=False)

    @pytest.mark.parametrize(
        ("fill_value", "exp_fill_warning"),
        [
            (np.int8(-1), False),
            (np.uint8(255), True),
            (-1, False),
            (255, True),
        ],
    )
    def test_roundtrip_unsigned(self, fill_value, exp_fill_warning):
        @contextlib.contextmanager
        def _roundtrip_with_warnings(*args, **kwargs):
            is_np2 = module_available("numpy", minversion="2.0.0.dev0")
            if exp_fill_warning and is_np2:
                warn_checker: contextlib.AbstractContextManager = pytest.warns(
                    SerializationWarning,
                    match="_FillValue attribute can't be represented",
                )
            else:
                warn_checker = contextlib.nullcontext()
            with warn_checker:
                with self.roundtrip(*args, **kwargs) as actual:
                    yield actual

        # regression/numpy2 test for
        encoding = {
            "_FillValue": fill_value,
            "_Unsigned": "true",
            "dtype": "i1",
        }
        x = np.array([0, 1, 127, 128, 254, np.nan], dtype=np.float32)
        decoded = Dataset({"x": ("t", x, {}, encoding)})

        attributes = {
            "_FillValue": fill_value,
            "_Unsigned": "true",
        }
        # Create unsigned data corresponding to [0, 1, 127, 128, 255] unsigned
        sb = np.asarray([0, 1, 127, -128, -2, -1], dtype="i1")
        encoded = Dataset({"x": ("t", sb, attributes)})
        unsigned_dtype = np.dtype(f"u{sb.dtype.itemsize}")

        with _roundtrip_with_warnings(decoded) as actual:
            for k in decoded.variables:
                assert decoded.variables[k].dtype == actual.variables[k].dtype
                exp_fv = decoded.variables[k].encoding["_FillValue"]
                if exp_fill_warning:
                    exp_fv = np.array(exp_fv, dtype=unsigned_dtype).view(sb.dtype)
                assert exp_fv == actual.variables[k].encoding["_FillValue"]
            assert_allclose(decoded, actual, decode_bytes=False)

        with _roundtrip_with_warnings(
            decoded, open_kwargs=dict(decode_cf=False)
        ) as actual:
            for k in encoded.variables:
                assert encoded.variables[k].dtype == actual.variables[k].dtype
                exp_fv = encoded.variables[k].attrs["_FillValue"]
                if exp_fill_warning:
                    exp_fv = np.array(exp_fv, dtype=unsigned_dtype).view(sb.dtype)
                assert exp_fv == actual.variables[k].attrs["_FillValue"]
            assert_allclose(encoded, actual, decode_bytes=False)

    @staticmethod
    def _create_cf_dataset():
        original = Dataset(
            dict(
                variable=(
                    ("ln_p", "latitude", "longitude"),
                    np.arange(8, dtype="f4").reshape(2, 2, 2),
                    {"ancillary_variables": "std_devs det_lim"},
                ),
                std_devs=(
                    ("ln_p", "latitude", "longitude"),
                    np.arange(0.1, 0.9, 0.1).reshape(2, 2, 2),
                    {"standard_name": "standard_error"},
                ),
                det_lim=(
                    (),
                    0.1,
                    {"standard_name": "detection_minimum"},
                ),
            ),
            dict(
                latitude=("latitude", [0, 1], {"units": "degrees_north"}),
                longitude=("longitude", [0, 1], {"units": "degrees_east"}),
                latlon=((), -1, {"grid_mapping_name": "latitude_longitude"}),
                latitude_bnds=(("latitude", "bnds2"), [[0, 1], [1, 2]]),
                longitude_bnds=(("longitude", "bnds2"), [[0, 1], [1, 2]]),
                areas=(
                    ("latitude", "longitude"),
                    [[1, 1], [1, 1]],
                    {"units": "degree^2"},
                ),
                ln_p=(
                    "ln_p",
                    [1.0, 0.5],
                    {
                        "standard_name": "atmosphere_ln_pressure_coordinate",
                        "computed_standard_name": "air_pressure",
                    },
                ),
                P0=((), 1013.25, {"units": "hPa"}),
            ),
        )
        original["variable"].encoding.update(
            {"cell_measures": "area: areas", "grid_mapping": "latlon"},
        )
        original.coords["latitude"].encoding.update(
            dict(grid_mapping="latlon", bounds="latitude_bnds")
        )
        original.coords["longitude"].encoding.update(
            dict(grid_mapping="latlon", bounds="longitude_bnds")
        )
        original.coords["ln_p"].encoding.update({"formula_terms": "p0: P0 lev : ln_p"})
        return original

    def test_grid_mapping_and_bounds_are_not_coordinates_in_file(self) -> None:
        original = self._create_cf_dataset()
        with self.roundtrip(original, open_kwargs={"decode_coords": False}) as ds:
            assert ds.coords["latitude"].attrs["bounds"] == "latitude_bnds"
            assert ds.coords["longitude"].attrs["bounds"] == "longitude_bnds"
            assert "coordinates" not in ds["variable"].attrs
            assert "coordinates" not in ds.attrs

    def test_coordinate_variables_after_dataset_roundtrip(self) -> None:
        original = self._create_cf_dataset()
        with self.roundtrip(original, open_kwargs={"decode_coords": "all"}) as actual:
            assert_identical(actual, original)

        with self.roundtrip(original) as actual:
            expected = original.reset_coords(
                ["latitude_bnds", "longitude_bnds", "areas", "P0", "latlon"]
            )
            # equal checks that coords and data_vars are equal which
            # should be enough
            # identical would require resetting a number of attributes
            # skip that.
            assert_equal(actual, expected)

    def test_grid_mapping_and_bounds_are_coordinates_after_dataarray_roundtrip(
        self,
    ) -> None:
        original = self._create_cf_dataset()
        # The DataArray roundtrip should have the same warnings as the
        # Dataset, but we already tested for those, so just go for the
        # new warnings.  It would appear that there is no way to tell
        # pytest "This warning and also this warning should both be
        # present".
        # xarray/tests/test_conventions.py::TestCFEncodedDataStore
        # needs the to_dataset. The other backends should be fine
        # without it.
        with pytest.warns(
            UserWarning,
            match=(
                r"Variable\(s\) referenced in bounds not in variables: "
                r"\['l(at|ong)itude_bnds'\]"
            ),
        ):
            with self.roundtrip(
                original["variable"].to_dataset(), open_kwargs={"decode_coords": "all"}
            ) as actual:
                assert_identical(actual, original["variable"].to_dataset())

    @requires_iris
    @requires_netcdf
    def test_coordinate_variables_after_iris_roundtrip(self) -> None:
        original = self._create_cf_dataset()
        iris_cube = original["variable"].to_iris()
        actual = DataArray.from_iris(iris_cube)
        # Bounds will be missing (xfail)
        del original.coords["latitude_bnds"], original.coords["longitude_bnds"]
        # Ancillary vars will be missing
        # Those are data_vars, and will be dropped when grabbing the variable
        assert_identical(actual, original["variable"])

    def test_coordinates_encoding(self) -> None:
        def equals_latlon(obj):
            return obj in {"lat lon", "lon lat"}

        original = Dataset(
            {"temp": ("x", [0, 1]), "precip": ("x", [0, -1])},
            {"lat": ("x", [2, 3]), "lon": ("x", [4, 5])},
        )
        with self.roundtrip(original) as actual:
            assert_identical(actual, original)
        with self.roundtrip(original, open_kwargs=dict(decode_coords=False)) as ds:
            assert equals_latlon(ds["temp"].attrs["coordinates"])
            assert equals_latlon(ds["precip"].attrs["coordinates"])
            assert "coordinates" not in ds.attrs
            assert "coordinates" not in ds["lat"].attrs
            assert "coordinates" not in ds["lon"].attrs

        modified = original.drop_vars(["temp", "precip"])
        with self.roundtrip(modified) as actual:
            assert_identical(actual, modified)
        with self.roundtrip(modified, open_kwargs=dict(decode_coords=False)) as ds:
            assert equals_latlon(ds.attrs["coordinates"])
            assert "coordinates" not in ds["lat"].attrs
            assert "coordinates" not in ds["lon"].attrs

        original["temp"].encoding["coordinates"] = "lat"
        with self.roundtrip(original) as actual:
            assert_identical(actual, original)
        original["precip"].encoding["coordinates"] = "lat"
        with self.roundtrip(original, open_kwargs=dict(decode_coords=True)) as ds:
            assert "lon" not in ds["temp"].encoding["coordinates"]
            assert "lon" not in ds["precip"].encoding["coordinates"]
            assert "coordinates" not in ds["lat"].encoding
            assert "coordinates" not in ds["lon"].encoding

    def test_roundtrip_endian(self) -> None:
        ds = Dataset(
            {
                "x": np.arange(3, 10, dtype=">i2"),
                "y": np.arange(3, 20, dtype="<i4"),
                "z": np.arange(3, 30, dtype="=i8"),
                "w": ("x", np.arange(3, 10, dtype=float)),
            }
        )

        with self.roundtrip(ds) as actual:
            # technically these datasets are slightly different,
            # one hold mixed endian data (ds) the other should be
            # all big endian (actual).  assertDatasetIdentical
            # should still pass though.
            assert_identical(ds, actual)

        if self.engine == "netcdf4":
            ds["z"].encoding["endian"] = "big"
            with pytest.raises(NotImplementedError):
                with self.roundtrip(ds) as actual:
                    pass

    def test_invalid_dataarray_names_raise(self) -> None:
        te = (TypeError, "string or None")
        ve = (ValueError, "string must be length 1 or")
        data = np.random.random((2, 2))
        da = xr.DataArray(data)
        for name, (error, msg) in zip(
            [0, (4, 5), True, ""], [te, te, te, ve], strict=True
        ):
            ds = Dataset({name: da})
            with pytest.raises(error) as excinfo:
                with self.roundtrip(ds):
                    pass
            excinfo.match(msg)
            excinfo.match(repr(name))

    def test_encoding_kwarg(self) -> None:
        ds = Dataset({"x": ("y", np.arange(10.0))})

        kwargs: dict[str, Any] = dict(encoding={"x": {"dtype": "f4"}})
        with self.roundtrip(ds, save_kwargs=kwargs) as actual:
            encoded_dtype = actual.x.encoding["dtype"]
            # On OS X, dtype sometimes switches endianness for unclear reasons
            assert encoded_dtype.kind == "f" and encoded_dtype.itemsize == 4
        assert ds.x.encoding == {}

        kwargs = dict(encoding={"x": {"foo": "bar"}})
        with pytest.raises(ValueError, match=r"unexpected encoding"):
            with self.roundtrip(ds, save_kwargs=kwargs) as actual:
                pass

        kwargs = dict(encoding={"x": "foo"})
        with pytest.raises(ValueError, match=r"must be castable"):
            with self.roundtrip(ds, save_kwargs=kwargs) as actual:
                pass

        kwargs = dict(encoding={"invalid": {}})
        with pytest.raises(KeyError):
            with self.roundtrip(ds, save_kwargs=kwargs) as actual:
                pass

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

        # test unlimited_dims validation
        # https://github.com/pydata/xarray/issues/10549
        ds.encoding = {"unlimited_dims": "z"}
        with pytest.warns(
            UserWarning,
            match=r"Unlimited dimension\(s\) .* declared in 'dataset.encoding'",
        ):
            with self.roundtrip(ds) as _:
                pass
        ds.encoding = {}
        with pytest.raises(
            ValueError,
            match=r"Unlimited dimension\(s\) .* declared in 'unlimited_dims-kwarg'",
        ):
            with self.roundtrip(ds, save_kwargs=dict(unlimited_dims=["z"])) as _:
                pass

    def test_encoding_kwarg_dates(self) -> None:
        ds = Dataset({"t": pd.date_range("2000-01-01", periods=3, unit="ns")})
        units = "days since 1900-01-01"
        kwargs = dict(encoding={"t": {"units": units}})
        with self.roundtrip(ds, save_kwargs=kwargs) as actual:
            assert actual.t.encoding["units"] == units
            assert_identical(actual, ds)

    def test_encoding_kwarg_fixed_width_string(self) -> None:
        # regression test for GH2149
        for strings in [[b"foo", b"bar", b"baz"], ["foo", "bar", "baz"]]:
            ds = Dataset({"x": strings})
            kwargs = dict(encoding={"x": {"dtype": "S1"}})
            with self.roundtrip(ds, save_kwargs=kwargs) as actual:
                assert actual["x"].encoding["dtype"] == "S1"
                assert_identical(actual, ds)

    def test_default_fill_value(self) -> None:
        # Test default encoding for float:
        ds = Dataset({"x": ("y", np.arange(10.0))})
        kwargs = dict(encoding={"x": {"dtype": "f4"}})
        with self.roundtrip(ds, save_kwargs=kwargs) as actual:
            assert math.isnan(actual.x.encoding["_FillValue"])
        assert ds.x.encoding == {}

        # Test default encoding for int:
        ds = Dataset({"x": ("y", np.arange(10.0))})
        kwargs = dict(encoding={"x": {"dtype": "int16"}})
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*floating point data as an integer")
            with self.roundtrip(ds, save_kwargs=kwargs) as actual:
                assert "_FillValue" not in actual.x.encoding
        assert ds.x.encoding == {}

        # Test default encoding for implicit int:
        ds = Dataset({"x": ("y", np.arange(10, dtype="int16"))})
        with self.roundtrip(ds) as actual:
            assert "_FillValue" not in actual.x.encoding
        assert ds.x.encoding == {}

    def test_explicitly_omit_fill_value(self) -> None:
        ds = Dataset({"x": ("y", [np.pi, -np.pi])})
        ds.x.encoding["_FillValue"] = None
        with self.roundtrip(ds) as actual:
            assert "_FillValue" not in actual.x.encoding

    def test_explicitly_omit_fill_value_via_encoding_kwarg(self) -> None:
        ds = Dataset({"x": ("y", [np.pi, -np.pi])})
        kwargs = dict(encoding={"x": {"_FillValue": None}})
        # _FillValue is not a valid encoding for Zarr
        with self.roundtrip(ds, save_kwargs=kwargs) as actual:
            assert "_FillValue" not in actual.x.encoding
        assert ds.y.encoding == {}

    def test_explicitly_omit_fill_value_in_coord(self) -> None:
        ds = Dataset({"x": ("y", [np.pi, -np.pi])}, coords={"y": [0.0, 1.0]})
        ds.y.encoding["_FillValue"] = None
        with self.roundtrip(ds) as actual:
            assert "_FillValue" not in actual.y.encoding

    def test_explicitly_omit_fill_value_in_coord_via_encoding_kwarg(self) -> None:
        ds = Dataset({"x": ("y", [np.pi, -np.pi])}, coords={"y": [0.0, 1.0]})
        kwargs = dict(encoding={"y": {"_FillValue": None}})
        with self.roundtrip(ds, save_kwargs=kwargs) as actual:
            assert "_FillValue" not in actual.y.encoding
        assert ds.y.encoding == {}

    def test_encoding_same_dtype(self) -> None:
        ds = Dataset({"x": ("y", np.arange(10.0, dtype="f4"))})
        kwargs = dict(encoding={"x": {"dtype": "f4"}})
        with self.roundtrip(ds, save_kwargs=kwargs) as actual:
            encoded_dtype = actual.x.encoding["dtype"]
            # On OS X, dtype sometimes switches endianness for unclear reasons
            assert encoded_dtype.kind == "f" and encoded_dtype.itemsize == 4
        assert ds.x.encoding == {}

    def test_append_write(self) -> None:
        # regression for GH1215
        data = create_test_data()
        with self.roundtrip_append(data) as actual:
            assert_identical(data, actual)

    def test_append_overwrite_values(self) -> None:
        # regression for GH1215
        data = create_test_data()
        with create_tmp_file(allow_cleanup_failure=False) as tmp_file:
            self.save(data, tmp_file, mode="w")
            data["var2"][:] = -999
            data["var9"] = data["var2"] * 3
            self.save(data[["var2", "var9"]], tmp_file, mode="a")
            with self.open(tmp_file) as actual:
                assert_identical(data, actual)

    def test_append_with_invalid_dim_raises(self) -> None:
        data = create_test_data()
        with create_tmp_file(allow_cleanup_failure=False) as tmp_file:
            self.save(data, tmp_file, mode="w")
            data["var9"] = data["var2"] * 3
            data = data.isel(dim1=slice(2, 6))  # modify one dimension
            with pytest.raises(
                ValueError, match=r"Unable to update size for existing dimension"
            ):
                self.save(data, tmp_file, mode="a")

    def test_multiindex_not_implemented(self) -> None:
        ds = Dataset(coords={"y": ("x", [1, 2]), "z": ("x", ["a", "b"])}).set_index(
            x=["y", "z"]
        )
        with pytest.raises(NotImplementedError, match=r"MultiIndex"):
            with self.roundtrip(ds):
                pass

        # regression GH8628 (can serialize reset multi-index level coordinates)
        ds_reset = ds.reset_index("x")
        with self.roundtrip(ds_reset) as actual:
            assert_identical(actual, ds_reset)

    @requires_dask
    def test_string_object_warning(self) -> None:
        original = Dataset(
            {
                "x": (
                    [
                        "y",
                    ],
                    np.array(["foo", "bar"], dtype=object),
                )
            }
        ).chunk()
        with pytest.warns(SerializationWarning, match="dask array with dtype=object"):
            with self.roundtrip(original) as actual:
                assert_identical(original, actual)

    @pytest.mark.parametrize(
        "indexer",
        (
            {"y": [1]},
            {"y": slice(2)},
            {"y": 1},
            {"x": [1], "y": [1]},
            {"x": ("x0", [0, 1]), "y": ("x0", [0, 1])},
        ),
    )
    def test_indexing_roundtrip(self, indexer) -> None:
        # regression test for GH8909
        ds = xr.Dataset()
        ds["A"] = xr.DataArray([[1, "a"], [2, "b"]], dims=["x", "y"])
        with self.roundtrip(ds) as ds2:
            expected = ds2.sel(indexer)
            with self.roundtrip(expected) as actual:
                assert_identical(actual, expected)


class NetCDFBase(CFEncodedBase):
    """Tests for all netCDF3 and netCDF4 backends."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="NetCDF backends don't support async loading")
    async def test_load_async(self) -> None:
        await super().test_load_async()

    @pytest.mark.skipif(
        ON_WINDOWS, reason="Windows does not allow modifying open files"
    )
    def test_refresh_from_disk(self) -> None:
        # regression test for https://github.com/pydata/xarray/issues/4862

        with create_tmp_file() as example_1_path:
            with create_tmp_file() as example_1_modified_path:
                with open_example_dataset("example_1.nc") as example_1:
                    self.save(example_1, example_1_path)

                    example_1.rh.values += 100
                    self.save(example_1, example_1_modified_path)

                a = open_dataset(example_1_path, engine=self.engine).load()

                # Simulate external process modifying example_1.nc while this script is running
                shutil.copy(example_1_modified_path, example_1_path)

                # Reopen example_1.nc (modified) as `b`; note that `a` has NOT been closed
                b = open_dataset(example_1_path, engine=self.engine).load()

                try:
                    assert not np.array_equal(a.rh.values, b.rh.values)
                finally:
                    a.close()
                    b.close()

    @pytest.fixture
    def byte_attrs_dataset(self) -> dict[str, Any]:
        """For testing issue #9407"""
        null_byte = b"\x00"
        other_bytes = bytes(range(1, 256))
        ds = Dataset({"x": 1}, coords={"x_coord": [1]})
        ds["x"].attrs["null_byte"] = null_byte
        ds["x"].attrs["other_bytes"] = other_bytes

        expected = ds.copy()
        expected["x"].attrs["null_byte"] = ""
        expected["x"].attrs["other_bytes"] = other_bytes.decode(errors="replace")

        return {
            "input": ds,
            "expected": expected,
            "h5netcdf_error": r"Invalid value provided for attribute .*: .*\. Null characters .*",
        }

    def test_byte_attrs(self, byte_attrs_dataset: dict[str, Any]) -> None:
        # test for issue #9407
        input = byte_attrs_dataset["input"]
        expected = byte_attrs_dataset["expected"]
        with self.roundtrip(input) as actual:
            assert_identical(actual, expected)


_counter = itertools.count()


@contextlib.contextmanager
def create_tmp_file(
    suffix: str = ".nc", allow_cleanup_failure: bool = False
) -> Iterator[str]:
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, f"temp-{next(_counter)}{suffix}")
    try:
        yield path
    finally:
        try:
            shutil.rmtree(temp_dir)
        except OSError:
            if not allow_cleanup_failure:
                raise


@contextlib.contextmanager
def create_tmp_files(
    nfiles: int, suffix: str = ".nc", allow_cleanup_failure: bool = False
) -> Iterator[list[str]]:
    with ExitStack() as stack:
        files = [
            stack.enter_context(create_tmp_file(suffix, allow_cleanup_failure))
            for _ in range(nfiles)
        ]
        yield files


class NetCDF4Base(NetCDFBase):
    """Tests for both netCDF4-python and h5netcdf."""

    engine: T_NetcdfEngine = "netcdf4"

    def test_open_group(self) -> None:
        # Create a netCDF file with a dataset stored within a group
        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, "w") as rootgrp:
                foogrp = rootgrp.createGroup("foo")
                ds = foogrp
                ds.createDimension("time", size=10)
                x = np.arange(10)
                ds.createVariable("x", np.int32, dimensions=("time",))
                ds.variables["x"][:] = x

            expected = Dataset()
            expected["x"] = ("time", x)

            # check equivalent ways to specify group
            for group in "foo", "/foo", "foo/", "/foo/":
                with self.open(tmp_file, group=group) as actual:
                    assert_equal(actual["x"], expected["x"])

            # check that missing group raises appropriate exception
            with pytest.raises(OSError):
                open_dataset(tmp_file, group="bar")
            with pytest.raises(ValueError, match=r"must be a string"):
                open_dataset(tmp_file, group=(1, 2, 3))

    def test_open_subgroup(self) -> None:
        # Create a netCDF file with a dataset stored within a group within a
        # group
        with create_tmp_file() as tmp_file:
            rootgrp = nc4.Dataset(tmp_file, "w")
            foogrp = rootgrp.createGroup("foo")
            bargrp = foogrp.createGroup("bar")
            ds = bargrp
            ds.createDimension("time", size=10)
            x = np.arange(10)
            ds.createVariable("x", np.int32, dimensions=("time",))
            ds.variables["x"][:] = x
            rootgrp.close()

            expected = Dataset()
            expected["x"] = ("time", x)

            # check equivalent ways to specify group
            for group in "foo/bar", "/foo/bar", "foo/bar/", "/foo/bar/":
                with self.open(tmp_file, group=group) as actual:
                    assert_equal(actual["x"], expected["x"])

    def test_write_groups(self) -> None:
        data1 = create_test_data()
        data2 = data1 * 2
        with create_tmp_file() as tmp_file:
            self.save(data1, tmp_file, group="data/1")
            self.save(data2, tmp_file, group="data/2", mode="a")
            with self.open(tmp_file, group="data/1") as actual1:
                assert_identical(data1, actual1)
            with self.open(tmp_file, group="data/2") as actual2:
                assert_identical(data2, actual2)

    def test_child_group_with_inconsistent_dimensions(self) -> None:
        base = Dataset(coords={"x": [1, 2]})
        child = Dataset(coords={"x": [1, 2, 3]})
        with create_tmp_file() as tmp_file:
            self.save(base, tmp_file)
            self.save(child, tmp_file, group="child", mode="a")
            with self.open(tmp_file) as actual_base:
                assert_identical(base, actual_base)
            with self.open(tmp_file, group="child") as actual_child:
                assert_identical(child, actual_child)

    @pytest.mark.parametrize(
        "input_strings, is_bytes",
        [
            ([b"foo", b"bar", b"baz"], True),
            (["foo", "bar", "baz"], False),
            (["foó", "bár", "baź"], False),
        ],
    )
    def test_encoding_kwarg_vlen_string(
        self, input_strings: list[str], is_bytes: bool
    ) -> None:
        original = Dataset({"x": input_strings})

        expected_string = ["foo", "bar", "baz"] if is_bytes else input_strings
        expected = Dataset({"x": expected_string})
        kwargs = dict(encoding={"x": {"dtype": str}})
        with self.roundtrip(original, save_kwargs=kwargs) as actual:
            assert actual["x"].encoding["dtype"] == "=U3"
            assert actual["x"].dtype == "=U3"
            assert_identical(actual, expected)

    @pytest.mark.parametrize("fill_value", ["XXX", "", "bár"])
    def test_roundtrip_string_with_fill_value_vlen(self, fill_value: str) -> None:
        values = np.array(["ab", "cdef", np.nan], dtype=object)
        expected = Dataset({"x": ("t", values)})

        original = Dataset({"x": ("t", values, {}, {"_FillValue": fill_value})})
        with self.roundtrip(original) as actual:
            assert_identical(expected, actual)

        original = Dataset({"x": ("t", values, {}, {"_FillValue": ""})})
        with self.roundtrip(original) as actual:
            assert_identical(expected, actual)

    def test_roundtrip_character_array(self) -> None:
        with create_tmp_file() as tmp_file:
            values = np.array([["a", "b", "c"], ["d", "e", "f"]], dtype="S")

            with nc4.Dataset(tmp_file, mode="w") as nc:
                nc.createDimension("x", 2)
                nc.createDimension("string3", 3)
                v = nc.createVariable("x", np.dtype("S1"), ("x", "string3"))
                v[:] = values

            values = np.array(["abc", "def"], dtype="S")
            expected = Dataset({"x": ("x", values)})
            with open_dataset(tmp_file) as actual:
                assert_identical(expected, actual)
                # regression test for #157
                with self.roundtrip(actual) as roundtripped:
                    assert_identical(expected, roundtripped)

    def test_default_to_char_arrays(self) -> None:
        data = Dataset({"x": np.array(["foo", "zzzz"], dtype="S")})
        with self.roundtrip(data) as actual:
            assert_identical(data, actual)
            assert actual["x"].dtype == np.dtype("S4")

    def test_open_encodings(self) -> None:
        # Create a netCDF file with explicit time units
        # and make sure it makes it into the encodings
        # and survives a round trip
        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, "w") as ds:
                ds.createDimension("time", size=10)
                ds.createVariable("time", np.int32, dimensions=("time",))
                units = "days since 1999-01-01"
                ds.variables["time"].setncattr("units", units)
                ds.variables["time"][:] = np.arange(10) + 4

            expected = Dataset()
            time = pd.date_range("1999-01-05", periods=10, unit="ns")
            encoding = {"units": units, "dtype": np.dtype("int32")}
            expected["time"] = ("time", time, {}, encoding)

            with open_dataset(tmp_file) as actual:
                assert_equal(actual["time"], expected["time"])
                actual_encoding = {
                    k: v
                    for k, v in actual["time"].encoding.items()
                    if k in expected["time"].encoding
                }
                assert actual_encoding == expected["time"].encoding

    def test_dump_encodings(self) -> None:
        # regression test for #709
        ds = Dataset({"x": ("y", np.arange(10.0))})
        kwargs = dict(encoding={"x": {"zlib": True}})
        with self.roundtrip(ds, save_kwargs=kwargs) as actual:
            assert actual.x.encoding["zlib"]

    def test_dump_and_open_encodings(self) -> None:
        # Create a netCDF file with explicit time units
        # and make sure it makes it into the encodings
        # and survives a round trip
        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, "w") as ds:
                ds.createDimension("time", size=10)
                ds.createVariable("time", np.int32, dimensions=("time",))
                units = "days since 1999-01-01"
                ds.variables["time"].setncattr("units", units)
                ds.variables["time"][:] = np.arange(10) + 4

            with open_dataset(tmp_file) as xarray_dataset:
                with create_tmp_file() as tmp_file2:
                    xarray_dataset.to_netcdf(tmp_file2)
                    with nc4.Dataset(tmp_file2, "r") as ds:
                        assert ds.variables["time"].getncattr("units") == units
                        assert_array_equal(ds.variables["time"], np.arange(10) + 4)

    def test_compression_encoding_legacy(self) -> None:
        data = create_test_data()
        data["var2"].encoding.update(
            {
                "zlib": True,
                "chunksizes": (5, 5),
                "fletcher32": True,
                "shuffle": True,
                "original_shape": data.var2.shape,
            }
        )
        with self.roundtrip(data) as actual:
            for k, v in data["var2"].encoding.items():
                assert v == actual["var2"].encoding[k]

        # regression test for #156
        expected = data.isel(dim1=0)
        with self.roundtrip(expected) as actual:
            assert_equal(expected, actual)

    def test_encoding_kwarg_compression(self) -> None:
        ds = Dataset({"x": np.arange(10.0)})
        encoding = dict(
            dtype="f4",
            zlib=True,
            complevel=9,
            fletcher32=True,
            chunksizes=(5,),
            shuffle=True,
        )
        kwargs = dict(encoding=dict(x=encoding))

        with self.roundtrip(ds, save_kwargs=kwargs) as actual:
            assert_equal(actual, ds)
            assert actual.x.encoding["dtype"] == "f4"
            assert actual.x.encoding["zlib"]
            assert actual.x.encoding["complevel"] == 9
            assert actual.x.encoding["fletcher32"]
            assert actual.x.encoding["chunksizes"] == (5,)
            assert actual.x.encoding["shuffle"]

        assert ds.x.encoding == {}

    def test_keep_chunksizes_if_no_original_shape(self) -> None:
        ds = Dataset({"x": [1, 2, 3]})
        chunksizes = (2,)
        ds.variables["x"].encoding = {"chunksizes": chunksizes}

        with self.roundtrip(ds) as actual:
            assert_identical(ds, actual)
            assert_array_equal(
                ds["x"].encoding["chunksizes"], actual["x"].encoding["chunksizes"]
            )

    def test_preferred_chunks_is_present(self) -> None:
        ds = Dataset({"x": [1, 2, 3]})
        chunksizes = (2,)
        ds.variables["x"].encoding = {"chunksizes": chunksizes}

        with self.roundtrip(ds) as actual:
            assert actual["x"].encoding["preferred_chunks"] == {"x": 2}

    @requires_dask
    def test_auto_chunking_is_based_on_disk_chunk_sizes(self) -> None:
        x_size = y_size = 1000
        y_chunksize = y_size
        x_chunksize = 10

        with dask.config.set({"array.chunk-size": "100KiB"}):
            with self.chunked_roundtrip(
                (1, y_size, x_size),
                (1, y_chunksize, x_chunksize),
                open_kwargs={"chunks": "auto"},
            ) as ds:
                _t_chunks, y_chunks, x_chunks = ds["image"].data.chunks
                assert all(np.asanyarray(y_chunks) == y_chunksize)
                # Check that the chunk size is a multiple of the file chunk size
                assert all(np.asanyarray(x_chunks) % x_chunksize == 0)

    @requires_dask
    def test_base_chunking_uses_disk_chunk_sizes(self) -> None:
        x_size = y_size = 1000
        y_chunksize = y_size
        x_chunksize = 10

        with self.chunked_roundtrip(
            (1, y_size, x_size),
            (1, y_chunksize, x_chunksize),
            open_kwargs={"chunks": {}},
        ) as ds:
            for chunksizes, expected in zip(
                ds["image"].data.chunks, (1, y_chunksize, x_chunksize), strict=True
            ):
                assert all(np.asanyarray(chunksizes) == expected)

    @contextlib.contextmanager
    def chunked_roundtrip(
        self,
        array_shape: tuple[int, int, int],
        chunk_sizes: tuple[int, int, int],
        open_kwargs: dict[str, Any] | None = None,
    ) -> Generator[Dataset, None, None]:
        t_size, y_size, x_size = array_shape
        t_chunksize, y_chunksize, x_chunksize = chunk_sizes

        image = xr.DataArray(
            np.arange(t_size * x_size * y_size, dtype=np.int16).reshape(
                (t_size, y_size, x_size)
            ),
            dims=["t", "y", "x"],
        )
        image.encoding = {"chunksizes": (t_chunksize, y_chunksize, x_chunksize)}
        dataset = xr.Dataset(dict(image=image))

        with self.roundtrip(dataset, open_kwargs=open_kwargs) as ds:
            yield ds

    def test_preferred_chunks_are_disk_chunk_sizes(self) -> None:
        x_size = y_size = 1000
        y_chunksize = y_size
        x_chunksize = 10

        with self.chunked_roundtrip(
            (1, y_size, x_size), (1, y_chunksize, x_chunksize)
        ) as ds:
            assert ds["image"].encoding["preferred_chunks"] == {
                "t": 1,
                "y": y_chunksize,
                "x": x_chunksize,
            }

    def test_encoding_chunksizes_unlimited(self) -> None:
        # regression test for GH1225
        ds = Dataset({"x": [1, 2, 3], "y": ("x", [2, 3, 4])})
        ds.variables["x"].encoding = {
            "zlib": False,
            "shuffle": False,
            "complevel": 0,
            "fletcher32": False,
            "contiguous": False,
            "chunksizes": (2**20,),
            "original_shape": (3,),
        }
        with self.roundtrip(ds) as actual:
            assert_equal(ds, actual)

    def test_mask_and_scale(self) -> None:
        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, mode="w") as nc:
                nc.createDimension("t", 5)
                nc.createVariable("x", "int16", ("t",), fill_value=-1)
                v = nc.variables["x"]
                v.set_auto_maskandscale(False)
                v.add_offset = 10
                v.scale_factor = 0.1
                v[:] = np.array([-1, -1, 0, 1, 2])
                dtype = type(v.scale_factor)

            # first make sure netCDF4 reads the masked and scaled data
            # correctly
            with nc4.Dataset(tmp_file, mode="r") as nc:
                expected = np.ma.array(
                    [-1, -1, 10, 10.1, 10.2], mask=[True, True, False, False, False]
                )
                actual = nc.variables["x"][:]
                assert_array_equal(expected, actual)

            # now check xarray
            with open_dataset(tmp_file) as ds:
                expected_ds = create_masked_and_scaled_data(np.dtype(dtype))
                assert_identical(expected_ds, ds)

    def test_0dimensional_variable(self) -> None:
        # This fix verifies our work-around to this netCDF4-python bug:
        # https://github.com/Unidata/netcdf4-python/pull/220
        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, mode="w") as nc:
                v = nc.createVariable("x", "int16")
                v[...] = 123

            with open_dataset(tmp_file) as ds:
                expected = Dataset({"x": ((), 123)})
                assert_identical(expected, ds)

    def test_read_variable_len_strings(self) -> None:
        with create_tmp_file() as tmp_file:
            values = np.array(["foo", "bar", "baz"], dtype=object)

            with nc4.Dataset(tmp_file, mode="w") as nc:
                nc.createDimension("x", 3)
                v = nc.createVariable("x", str, ("x",))
                v[:] = values

            expected = Dataset({"x": ("x", values)})
            for kwargs in [{}, {"decode_cf": True}]:
                with open_dataset(tmp_file, **cast(dict, kwargs)) as actual:
                    assert_identical(expected, actual)

    def test_raise_on_forward_slashes_in_names(self) -> None:
        # test for forward slash in variable names and dimensions
        # see GH 7943
        data_vars: list[dict[str, Any]] = [
            {"PASS/FAIL": (["PASSFAIL"], np.array([0]))},
            {"PASS/FAIL": np.array([0])},
            {"PASSFAIL": (["PASS/FAIL"], np.array([0]))},
        ]
        for dv in data_vars:
            ds = Dataset(data_vars=dv)
            with pytest.raises(ValueError, match="Forward slashes '/' are not allowed"):
                with self.roundtrip(ds):
                    pass

    @requires_netCDF4
    def test_encoding_enum__no_fill_value(self, recwarn):
        with create_tmp_file() as tmp_file:
            cloud_type_dict = {"clear": 0, "cloudy": 1}
            with nc4.Dataset(tmp_file, mode="w") as nc:
                nc.createDimension("time", size=2)
                cloud_type = nc.createEnumType(np.uint8, "cloud_type", cloud_type_dict)
                v = nc.createVariable(
                    "clouds",
                    cloud_type,
                    "time",
                    fill_value=None,
                )
                v[:] = 1
            with open_dataset(tmp_file, engine="netcdf4") as original:
                # We don't expect any errors.
                # This is effectively a void context manager
                expected_warnings = 0
                if self.engine == "h5netcdf":
                    expected_warnings = 1
                    expected_msg = "Creating variable with default fill_value 0 which IS defined in enum type"

                with self.roundtrip(original) as actual:
                    assert len(recwarn) == expected_warnings
                    if expected_warnings:
                        assert issubclass(recwarn[0].category, UserWarning)
                        assert str(recwarn[0].message).startswith(expected_msg)
                    assert_equal(original, actual)
                    assert (
                        actual.clouds.encoding["dtype"].metadata["enum"]
                        == cloud_type_dict
                    )

    @requires_netCDF4
    def test_encoding_enum__multiple_variable_with_enum(self):
        with create_tmp_file() as tmp_file:
            cloud_type_dict = {"clear": 0, "cloudy": 1, "missing": 255}
            with nc4.Dataset(tmp_file, mode="w") as nc:
                nc.createDimension("time", size=2)
                cloud_type = nc.createEnumType(np.uint8, "cloud_type", cloud_type_dict)
                nc.createVariable(
                    "clouds",
                    cloud_type,
                    "time",
                    fill_value=255,
                )
                nc.createVariable(
                    "tifa",
                    cloud_type,
                    "time",
                    fill_value=255,
                )
            with open_dataset(tmp_file, engine="netcdf4") as original:
                with self.roundtrip(original) as actual:
                    assert_equal(original, actual)
                    assert (
                        actual.clouds.encoding["dtype"] == actual.tifa.encoding["dtype"]
                    )
                    assert (
                        actual.clouds.encoding["dtype"].metadata
                        == actual.tifa.encoding["dtype"].metadata
                    )
                    assert (
                        actual.clouds.encoding["dtype"].metadata["enum"]
                        == cloud_type_dict
                    )

    @requires_netCDF4
    def test_encoding_enum__error_multiple_variable_with_changing_enum(self):
        """
        Given 2 variables, if they share the same enum type,
        the 2 enum definition should be identical.
        """
        with create_tmp_file() as tmp_file:
            cloud_type_dict = {"clear": 0, "cloudy": 1, "missing": 255}
            with nc4.Dataset(tmp_file, mode="w") as nc:
                nc.createDimension("time", size=2)
                cloud_type = nc.createEnumType(np.uint8, "cloud_type", cloud_type_dict)
                nc.createVariable(
                    "clouds",
                    cloud_type,
                    "time",
                    fill_value=255,
                )
                nc.createVariable(
                    "tifa",
                    cloud_type,
                    "time",
                    fill_value=255,
                )
            with open_dataset(tmp_file, engine="netcdf4") as original:
                assert (
                    original.clouds.encoding["dtype"].metadata
                    == original.tifa.encoding["dtype"].metadata
                )
                modified_enum = original.clouds.encoding["dtype"].metadata["enum"]
                modified_enum.update({"neblig": 2})
                original.clouds.encoding["dtype"] = np.dtype(
                    "u1",
                    metadata={"enum": modified_enum, "enum_name": "cloud_type"},
                )

    @pytest.mark.parametrize("create_default_indexes", [True, False])
    def test_create_default_indexes(self, tmp_path, create_default_indexes) -> None:
        store_path = tmp_path / "tmp.nc"
        original_ds = xr.Dataset(
            {"data": ("x", np.arange(3))}, coords={"x": [-1, 0, 1]}
        )
        original_ds.to_netcdf(store_path, engine=self.engine, mode="w")

        with open_dataset(
            store_path,
            engine=self.engine,
            create_default_indexes=create_default_indexes,
        ) as loaded_ds:
            if create_default_indexes:
                assert list(loaded_ds.xindexes) == ["x"] and isinstance(
                    loaded_ds.xindexes["x"], PandasIndex
                )
            else:
                assert len(loaded_ds.xindexes) == 0

    @requires_dask
    @pytest.mark.skip_with_dask_array
    def test_encoding_masked_arrays(self, tmp_path) -> None:
        store_path = tmp_path / "tmp.nc"

        with raise_if_dask_computes():
            da = dask_array_api
            ds = xr.DataArray(
                da.from_array(
                    np.ma.masked_array(
                        np.array([[np.nan, np.nan], [np.nan, 2]]),
                        np.array([[True, True], [True, False]]),
                    ),
                    chunks=(2, 2),
                ).astype("float32"),
                dims=("x", "y"),
            ).to_dataset(name="mydata")

        expected = ds.mean("x")
        expected.to_netcdf(
            store_path, encoding=dict(mydata=dict(_FillValue=np.float32(1e20)))
        )
        with open_dataset(store_path, engine=self.engine) as actual:
            assert_identical(expected.compute(), actual.compute())


class InMemoryNetCDF:
    engine: T_NetcdfEngine | None

    def test_roundtrip_via_memoryview(self) -> None:
        original = create_test_data()
        result = original.to_netcdf(engine=self.engine)
        roundtrip = load_dataset(result, engine=self.engine)
        assert_identical(roundtrip, original)

    def test_roundtrip_via_bytes(self) -> None:
        original = create_test_data()
        result = bytes(original.to_netcdf(engine=self.engine))
        roundtrip = load_dataset(result, engine=self.engine)
        assert_identical(roundtrip, original)

    def test_pickle_open_dataset_from_bytes(self) -> None:
        original = Dataset({"foo": ("x", [1, 2, 3])})
        netcdf_bytes = bytes(original.to_netcdf(engine=self.engine))
        with open_dataset(netcdf_bytes, engine=self.engine) as roundtrip:
            with pickle.loads(pickle.dumps(roundtrip)) as unpickled:
                assert_identical(unpickled, original)

    def test_compute_false(self) -> None:
        original = create_test_data()
        with pytest.raises(
            NotImplementedError,
            match=re.escape("to_netcdf() with compute=False is not yet implemented"),
        ):
            original.to_netcdf(engine=self.engine, compute=False)


class InMemoryNetCDFWithGroups(InMemoryNetCDF):
    def test_roundtrip_group_via_memoryview(self) -> None:
        original = create_test_data()
        netcdf_bytes = original.to_netcdf(group="sub", engine=self.engine)
        roundtrip = load_dataset(netcdf_bytes, group="sub", engine=self.engine)
        assert_identical(roundtrip, original)


class FileObjectNetCDF:
    engine: T_NetcdfEngine

    def test_file_remains_open(self) -> None:
        data = Dataset({"foo": ("x", [1, 2, 3])})
        f = BytesIO()
        data.to_netcdf(f, engine=self.engine)
        assert not f.closed
        restored = open_dataset(f, engine=self.engine)
        assert not f.closed
        assert_identical(restored, data)
        restored.close()
        assert not f.closed


def _check_guess_can_open_and_open(entrypoint, obj, engine, expected):
    assert entrypoint.guess_can_open(obj)
    with open_dataset(obj, engine=engine) as actual:
        assert_identical(expected, actual)
