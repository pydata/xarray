# -*- coding: utf-8 -*-
import contextlib
import warnings

import numpy as np
import pandas as pd
import pytest

from xarray import (
    Dataset,
    SerializationWarning,
    Variable,
    coding,
    conventions,
    open_dataset,
)
from xarray.backends.common import WritableCFDataStore
from xarray.backends.memory import InMemoryDataStore
from xarray.conventions import decode_cf
from xarray.testing import assert_identical

from . import (
    assert_array_equal,
    raises_regex,
    requires_cftime_or_netCDF4,
    requires_dask,
    requires_netCDF4,
)
from .test_backends import CFEncodedBase


class TestBoolTypeArray:
    def test_booltype_array(self):
        x = np.array([1, 0, 1, 1, 0], dtype="i1")
        bx = conventions.BoolTypeArray(x)
        assert bx.dtype == np.bool
        assert_array_equal(
            bx, np.array([True, False, True, True, False], dtype=np.bool)
        )


class TestNativeEndiannessArray:
    def test(self):
        x = np.arange(5, dtype=">i8")
        expected = np.arange(5, dtype="int64")
        a = conventions.NativeEndiannessArray(x)
        assert a.dtype == expected.dtype
        assert a.dtype == expected[:].dtype
        assert_array_equal(a, expected)


def test_decode_cf_with_conflicting_fill_missing_value():
    expected = Variable(["t"], [np.nan, np.nan, 2], {"units": "foobar"})
    var = Variable(
        ["t"], np.arange(3), {"units": "foobar", "missing_value": 0, "_FillValue": 1}
    )
    with warnings.catch_warnings(record=True) as w:
        actual = conventions.decode_cf_variable("t", var)
        assert_identical(actual, expected)
        assert "has multiple fill" in str(w[0].message)

    expected = Variable(["t"], np.arange(10), {"units": "foobar"})

    var = Variable(
        ["t"],
        np.arange(10),
        {"units": "foobar", "missing_value": np.nan, "_FillValue": np.nan},
    )
    actual = conventions.decode_cf_variable("t", var)
    assert_identical(actual, expected)

    var = Variable(
        ["t"],
        np.arange(10),
        {
            "units": "foobar",
            "missing_value": np.float32(np.nan),
            "_FillValue": np.float32(np.nan),
        },
    )
    actual = conventions.decode_cf_variable("t", var)
    assert_identical(actual, expected)


@requires_cftime_or_netCDF4
class TestEncodeCFVariable:
    def test_incompatible_attributes(self):
        invalid_vars = [
            Variable(
                ["t"], pd.date_range("2000-01-01", periods=3), {"units": "foobar"}
            ),
            Variable(["t"], pd.to_timedelta(["1 day"]), {"units": "foobar"}),
            Variable(["t"], [0, 1, 2], {"add_offset": 0}, {"add_offset": 2}),
            Variable(["t"], [0, 1, 2], {"_FillValue": 0}, {"_FillValue": 2}),
        ]
        for var in invalid_vars:
            with pytest.raises(ValueError):
                conventions.encode_cf_variable(var)

    def test_missing_fillvalue(self):
        v = Variable(["x"], np.array([np.nan, 1, 2, 3]))
        v.encoding = {"dtype": "int16"}
        with pytest.warns(Warning, match="floating point data as an integer"):
            conventions.encode_cf_variable(v)

    def test_multidimensional_coordinates(self):
        # regression test for GH1763
        # Set up test case with coordinates that have overlapping (but not
        # identical) dimensions.
        zeros1 = np.zeros((1, 5, 3))
        zeros2 = np.zeros((1, 6, 3))
        zeros3 = np.zeros((1, 5, 4))
        orig = Dataset(
            {
                "lon1": (["x1", "y1"], zeros1.squeeze(0), {}),
                "lon2": (["x2", "y1"], zeros2.squeeze(0), {}),
                "lon3": (["x1", "y2"], zeros3.squeeze(0), {}),
                "lat1": (["x1", "y1"], zeros1.squeeze(0), {}),
                "lat2": (["x2", "y1"], zeros2.squeeze(0), {}),
                "lat3": (["x1", "y2"], zeros3.squeeze(0), {}),
                "foo1": (["time", "x1", "y1"], zeros1, {"coordinates": "lon1 lat1"}),
                "foo2": (["time", "x2", "y1"], zeros2, {"coordinates": "lon2 lat2"}),
                "foo3": (["time", "x1", "y2"], zeros3, {"coordinates": "lon3 lat3"}),
                "time": ("time", [0.0], {"units": "hours since 2017-01-01"}),
            }
        )
        orig = conventions.decode_cf(orig)
        # Encode the coordinates, as they would be in a netCDF output file.
        enc, attrs = conventions.encode_dataset_coordinates(orig)
        # Make sure we have the right coordinates for each variable.
        foo1_coords = enc["foo1"].attrs.get("coordinates", "")
        foo2_coords = enc["foo2"].attrs.get("coordinates", "")
        foo3_coords = enc["foo3"].attrs.get("coordinates", "")
        assert set(foo1_coords.split()) == {"lat1", "lon1"}
        assert set(foo2_coords.split()) == {"lat2", "lon2"}
        assert set(foo3_coords.split()) == {"lat3", "lon3"}
        # Should not have any global coordinates.
        assert "coordinates" not in attrs

    @requires_dask
    def test_string_object_warning(self):
        original = Variable(("x",), np.array(["foo", "bar"], dtype=object)).chunk()
        with pytest.warns(SerializationWarning, match="dask array with dtype=object"):
            encoded = conventions.encode_cf_variable(original)
        assert_identical(original, encoded)


@requires_cftime_or_netCDF4
class TestDecodeCF:
    def test_dataset(self):
        original = Dataset(
            {
                "t": ("t", [0, 1, 2], {"units": "days since 2000-01-01"}),
                "foo": ("t", [0, 0, 0], {"coordinates": "y", "units": "bar"}),
                "y": ("t", [5, 10, -999], {"_FillValue": -999}),
            }
        )
        expected = Dataset(
            {"foo": ("t", [0, 0, 0], {"units": "bar"})},
            {
                "t": pd.date_range("2000-01-01", periods=3),
                "y": ("t", [5.0, 10.0, np.nan]),
            },
        )
        actual = conventions.decode_cf(original)
        assert_identical(expected, actual)

    def test_invalid_coordinates(self):
        # regression test for GH308
        original = Dataset({"foo": ("t", [1, 2], {"coordinates": "invalid"})})
        actual = conventions.decode_cf(original)
        assert_identical(original, actual)

    def test_decode_coordinates(self):
        # regression test for GH610
        original = Dataset(
            {"foo": ("t", [1, 2], {"coordinates": "x"}), "x": ("t", [4, 5])}
        )
        actual = conventions.decode_cf(original)
        assert actual.foo.encoding["coordinates"] == "x"

    def test_0d_int32_encoding(self):
        original = Variable((), np.int32(0), encoding={"dtype": "int64"})
        expected = Variable((), np.int64(0))
        actual = conventions.maybe_encode_nonstring_dtype(original)
        assert_identical(expected, actual)

    def test_decode_cf_with_multiple_missing_values(self):
        original = Variable(["t"], [0, 1, 2], {"missing_value": np.array([0, 1])})
        expected = Variable(["t"], [np.nan, np.nan, 2], {})
        with warnings.catch_warnings(record=True) as w:
            actual = conventions.decode_cf_variable("t", original)
            assert_identical(expected, actual)
            assert "has multiple fill" in str(w[0].message)

    def test_decode_cf_with_drop_variables(self):
        original = Dataset(
            {
                "t": ("t", [0, 1, 2], {"units": "days since 2000-01-01"}),
                "x": ("x", [9, 8, 7], {"units": "km"}),
                "foo": (
                    ("t", "x"),
                    [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                    {"units": "bar"},
                ),
                "y": ("t", [5, 10, -999], {"_FillValue": -999}),
            }
        )
        expected = Dataset(
            {
                "t": pd.date_range("2000-01-01", periods=3),
                "foo": (
                    ("t", "x"),
                    [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                    {"units": "bar"},
                ),
                "y": ("t", [5, 10, np.nan]),
            }
        )
        actual = conventions.decode_cf(original, drop_variables=("x",))
        actual2 = conventions.decode_cf(original, drop_variables="x")
        assert_identical(expected, actual)
        assert_identical(expected, actual2)

    def test_invalid_time_units_raises_eagerly(self):
        ds = Dataset({"time": ("time", [0, 1], {"units": "foobar since 123"})})
        with raises_regex(ValueError, "unable to decode time"):
            decode_cf(ds)

    @requires_cftime_or_netCDF4
    def test_dataset_repr_with_netcdf4_datetimes(self):
        # regression test for #347
        attrs = {"units": "days since 0001-01-01", "calendar": "noleap"}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "unable to decode time")
            ds = decode_cf(Dataset({"time": ("time", [0, 1], attrs)}))
            assert "(time) object" in repr(ds)

        attrs = {"units": "days since 1900-01-01"}
        ds = decode_cf(Dataset({"time": ("time", [0, 1], attrs)}))
        assert "(time) datetime64[ns]" in repr(ds)

    @requires_cftime_or_netCDF4
    def test_decode_cf_datetime_transition_to_invalid(self):
        # manually create dataset with not-decoded date
        from datetime import datetime

        ds = Dataset(coords={"time": [0, 266 * 365]})
        units = "days since 2000-01-01 00:00:00"
        ds.time.attrs = dict(units=units)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "unable to decode time")
            ds_decoded = conventions.decode_cf(ds)

        expected = [datetime(2000, 1, 1, 0, 0), datetime(2265, 10, 28, 0, 0)]

        assert_array_equal(ds_decoded.time.values, expected)

    @requires_dask
    def test_decode_cf_with_dask(self):
        import dask.array as da

        original = Dataset(
            {
                "t": ("t", [0, 1, 2], {"units": "days since 2000-01-01"}),
                "foo": ("t", [0, 0, 0], {"coordinates": "y", "units": "bar"}),
                "bar": ("string2", [b"a", b"b"]),
                "baz": (("x"), [b"abc"], {"_Encoding": "utf-8"}),
                "y": ("t", [5, 10, -999], {"_FillValue": -999}),
            }
        ).chunk()
        decoded = conventions.decode_cf(original)
        print(decoded)
        assert all(
            isinstance(var.data, da.Array)
            for name, var in decoded.variables.items()
            if name not in decoded.indexes
        )
        assert_identical(decoded, conventions.decode_cf(original).compute())

    @requires_dask
    def test_decode_dask_times(self):
        original = Dataset.from_dict(
            {
                "coords": {},
                "dims": {"time": 5},
                "data_vars": {
                    "average_T1": {
                        "dims": ("time",),
                        "attrs": {"units": "days since 1958-01-01 00:00:00"},
                        "data": [87659.0, 88024.0, 88389.0, 88754.0, 89119.0],
                    }
                },
            }
        )
        assert_identical(
            conventions.decode_cf(original.chunk()),
            conventions.decode_cf(original).chunk(),
        )


class CFEncodedInMemoryStore(WritableCFDataStore, InMemoryDataStore):
    def encode_variable(self, var):
        """encode one variable"""
        coder = coding.strings.EncodedStringCoder(allows_unicode=True)
        var = coder.encode(var)
        return var


@requires_netCDF4
class TestCFEncodedDataStore(CFEncodedBase):
    @contextlib.contextmanager
    def create_store(self):
        yield CFEncodedInMemoryStore()

    @contextlib.contextmanager
    def roundtrip(
        self, data, save_kwargs={}, open_kwargs={}, allow_cleanup_failure=False
    ):
        store = CFEncodedInMemoryStore()
        data.dump_to_store(store, **save_kwargs)
        yield open_dataset(store, **open_kwargs)

    @pytest.mark.skip("cannot roundtrip coordinates yet for " "CFEncodedInMemoryStore")
    def test_roundtrip_coordinates(self):
        pass

    def test_invalid_dataarray_names_raise(self):
        # only relevant for on-disk file formats
        pass

    def test_encoding_kwarg(self):
        # we haven't bothered to raise errors yet for unexpected encodings in
        # this test dummy
        pass

    def test_encoding_kwarg_fixed_width_string(self):
        # CFEncodedInMemoryStore doesn't support explicit string encodings.
        pass
