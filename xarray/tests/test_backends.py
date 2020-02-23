import contextlib
import itertools
import math
import os.path
import pickle
import shutil
import sys
import tempfile
import warnings
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest
from pandas.errors import OutOfBoundsDatetime

import xarray as xr
from xarray import (
    DataArray,
    Dataset,
    backends,
    load_dataarray,
    load_dataset,
    open_dataarray,
    open_dataset,
    open_mfdataset,
    save_mfdataset,
)
from xarray.backends.common import robust_getitem
from xarray.backends.netCDF4_ import _extract_nc4_variable_encoding
from xarray.backends.pydap_ import PydapDataStore
from xarray.coding.variables import SerializationWarning
from xarray.conventions import encode_dataset_coordinates
from xarray.core import indexing
from xarray.core.options import set_options
from xarray.core.pycompat import dask_array_type
from xarray.tests import LooseVersion, mock

from . import (
    arm_xfail,
    assert_allclose,
    assert_array_equal,
    assert_equal,
    assert_identical,
    has_dask,
    has_netCDF4,
    has_scipy,
    network,
    raises_regex,
    requires_cfgrib,
    requires_cftime,
    requires_dask,
    requires_h5netcdf,
    requires_netCDF4,
    requires_pseudonetcdf,
    requires_pydap,
    requires_pynio,
    requires_rasterio,
    requires_scipy,
    requires_scipy_or_netCDF4,
    requires_zarr,
)
from .test_coding_times import (
    _ALL_CALENDARS,
    _NON_STANDARD_CALENDARS,
    _STANDARD_CALENDARS,
)
from .test_dataset import create_append_test_data, create_test_data

try:
    import netCDF4 as nc4
except ImportError:
    pass

try:
    import dask
    import dask.array as da

    dask_version = dask.__version__
except ImportError:
    # needed for xfailed tests when dask < 2.4.0
    # remove when min dask > 2.4.0
    dask_version = "10.0"

ON_WINDOWS = sys.platform == "win32"


def open_example_dataset(name, *args, **kwargs):
    return open_dataset(
        os.path.join(os.path.dirname(__file__), "data", name), *args, **kwargs
    )


def open_example_mfdataset(names, *args, **kwargs):
    return open_mfdataset(
        [os.path.join(os.path.dirname(__file__), "data", name) for name in names],
        *args,
        **kwargs,
    )


def create_masked_and_scaled_data():
    x = np.array([np.nan, np.nan, 10, 10.1, 10.2], dtype=np.float32)
    encoding = {
        "_FillValue": -1,
        "add_offset": 10,
        "scale_factor": np.float32(0.1),
        "dtype": "i2",
    }
    return Dataset({"x": ("t", x, {}, encoding)})


def create_encoded_masked_and_scaled_data():
    attributes = {"_FillValue": -1, "add_offset": 10, "scale_factor": np.float32(0.1)}
    return Dataset({"x": ("t", np.int16([-1, -1, 0, 1, 2]), attributes)})


def create_unsigned_masked_scaled_data():
    encoding = {
        "_FillValue": 255,
        "_Unsigned": "true",
        "dtype": "i1",
        "add_offset": 10,
        "scale_factor": np.float32(0.1),
    }
    x = np.array([10.0, 10.1, 22.7, 22.8, np.nan], dtype=np.float32)
    return Dataset({"x": ("t", x, {}, encoding)})


def create_encoded_unsigned_masked_scaled_data():
    # These are values as written to the file: the _FillValue will
    # be represented in the signed form.
    attributes = {
        "_FillValue": -1,
        "_Unsigned": "true",
        "add_offset": 10,
        "scale_factor": np.float32(0.1),
    }
    # Create unsigned data corresponding to [0, 1, 127, 128, 255] unsigned
    sb = np.asarray([0, 1, 127, -128, -1], dtype="i1")
    return Dataset({"x": ("t", sb, attributes)})


def create_bad_unsigned_masked_scaled_data():
    encoding = {
        "_FillValue": 255,
        "_Unsigned": True,
        "dtype": "i1",
        "add_offset": 10,
        "scale_factor": np.float32(0.1),
    }
    x = np.array([10.0, 10.1, 22.7, 22.8, np.nan], dtype=np.float32)
    return Dataset({"x": ("t", x, {}, encoding)})


def create_bad_encoded_unsigned_masked_scaled_data():
    # These are values as written to the file: the _FillValue will
    # be represented in the signed form.
    attributes = {
        "_FillValue": -1,
        "_Unsigned": True,
        "add_offset": 10,
        "scale_factor": np.float32(0.1),
    }
    # Create signed data corresponding to [0, 1, 127, 128, 255] unsigned
    sb = np.asarray([0, 1, 127, -128, -1], dtype="i1")
    return Dataset({"x": ("t", sb, attributes)})


def create_signed_masked_scaled_data():
    encoding = {
        "_FillValue": -127,
        "_Unsigned": "false",
        "dtype": "i1",
        "add_offset": 10,
        "scale_factor": np.float32(0.1),
    }
    x = np.array([-1.0, 10.1, 22.7, np.nan], dtype=np.float32)
    return Dataset({"x": ("t", x, {}, encoding)})


def create_encoded_signed_masked_scaled_data():
    # These are values as written to the file: the _FillValue will
    # be represented in the signed form.
    attributes = {
        "_FillValue": -127,
        "_Unsigned": "false",
        "add_offset": 10,
        "scale_factor": np.float32(0.1),
    }
    # Create signed data corresponding to [0, 1, 127, 128, 255] unsigned
    sb = np.asarray([-110, 1, 127, -127], dtype="i1")
    return Dataset({"x": ("t", sb, attributes)})


def create_boolean_data():
    attributes = {"units": "-"}
    return Dataset({"x": ("t", [True, False, False, True], attributes)})


class TestCommon:
    def test_robust_getitem(self):
        class UnreliableArrayFailure(Exception):
            pass

        class UnreliableArray:
            def __init__(self, array, failures=1):
                self.array = array
                self.failures = failures

            def __getitem__(self, key):
                if self.failures > 0:
                    self.failures -= 1
                    raise UnreliableArrayFailure
                return self.array[key]

        array = UnreliableArray([0])
        with pytest.raises(UnreliableArrayFailure):
            array[0]
        assert array[0] == 0

        actual = robust_getitem(array, 0, catch=UnreliableArrayFailure, initial_delay=0)
        assert actual == 0


class NetCDF3Only:
    pass


class DatasetIOBase:
    engine: Optional[str] = None
    file_format: Optional[str] = None

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

    def test_zero_dimensional_variable(self):
        expected = create_test_data()
        expected["float_var"] = ([], 1.0e9, {"units": "units of awesome"})
        expected["bytes_var"] = ([], b"foobar")
        expected["string_var"] = ([], "foobar")
        with self.roundtrip(expected) as actual:
            assert_identical(expected, actual)

    def test_write_store(self):
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
            if isinstance(self, NetCDF3Only) and expected_dtype == "int64":
                # downcast
                expected_dtype = np.dtype("int32")
            actual_dtype = actual.variables[k].dtype
            # TODO: check expected behavior for string dtypes more carefully
            string_kinds = {"O", "S", "U"}
            assert expected_dtype == actual_dtype or (
                expected_dtype.kind in string_kinds
                and actual_dtype.kind in string_kinds
            )

    def test_roundtrip_test_data(self):
        expected = create_test_data()
        with self.roundtrip(expected) as actual:
            self.check_dtypes_roundtripped(expected, actual)
            assert_identical(expected, actual)

    def test_load(self):
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

    def test_dataset_compute(self):
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

    def test_pickle(self):
        if not has_dask:
            pytest.xfail("pickling requires dask for SerializableLock")
        expected = Dataset({"foo": ("x", [42])})
        with self.roundtrip(expected, allow_cleanup_failure=ON_WINDOWS) as roundtripped:
            with roundtripped:
                # Windows doesn't like reopening an already open file
                raw_pickle = pickle.dumps(roundtripped)
            with pickle.loads(raw_pickle) as unpickled_ds:
                assert_identical(expected, unpickled_ds)

    @pytest.mark.filterwarnings("ignore:deallocating CachingFileManager")
    def test_pickle_dataarray(self):
        if not has_dask:
            pytest.xfail("pickling requires dask for SerializableLock")
        expected = Dataset({"foo": ("x", [42])})
        with self.roundtrip(expected, allow_cleanup_failure=ON_WINDOWS) as roundtripped:
            with roundtripped:
                raw_pickle = pickle.dumps(roundtripped["foo"])
            # TODO: figure out how to explicitly close the file for the
            # unpickled DataArray?
            unpickled = pickle.loads(raw_pickle)
            assert_identical(expected["foo"], unpickled)

    def test_dataset_caching(self):
        expected = Dataset({"foo": ("x", [5, 6, 7])})
        with self.roundtrip(expected) as actual:
            assert isinstance(actual.foo.variable._data, indexing.MemoryCachedArray)
            assert not actual.foo.variable._in_memory
            actual.foo.values  # cache
            assert actual.foo.variable._in_memory

        with self.roundtrip(expected, open_kwargs={"cache": False}) as actual:
            assert isinstance(actual.foo.variable._data, indexing.CopyOnWriteArray)
            assert not actual.foo.variable._in_memory
            actual.foo.values  # no caching
            assert not actual.foo.variable._in_memory

    def test_roundtrip_None_variable(self):
        expected = Dataset({None: (("x", "y"), [[0, 1], [2, 3]])})
        with self.roundtrip(expected) as actual:
            assert_identical(expected, actual)

    def test_roundtrip_object_dtype(self):
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
                expected["bytes_nans"][-1] = b""
                expected["strings_nans"][-1] = ""
                assert_identical(expected, actual)

    def test_roundtrip_string_data(self):
        expected = Dataset({"x": ("t", ["ab", "cdef"])})
        with self.roundtrip(expected) as actual:
            assert_identical(expected, actual)

    def test_roundtrip_string_encoded_characters(self):
        expected = Dataset({"x": ("t", ["ab", "cdef"])})
        expected["x"].encoding["dtype"] = "S1"
        with self.roundtrip(expected) as actual:
            assert_identical(expected, actual)
            assert actual["x"].encoding["_Encoding"] == "utf-8"

        expected["x"].encoding["_Encoding"] = "ascii"
        with self.roundtrip(expected) as actual:
            assert_identical(expected, actual)
            assert actual["x"].encoding["_Encoding"] == "ascii"

    @arm_xfail
    def test_roundtrip_numpy_datetime_data(self):
        times = pd.to_datetime(["2000-01-01", "2000-01-02", "NaT"])
        expected = Dataset({"t": ("t", times), "t0": times[0]})
        kwargs = {"encoding": {"t0": {"units": "days since 1950-01-01"}}}
        with self.roundtrip(expected, save_kwargs=kwargs) as actual:
            assert_identical(expected, actual)
            assert actual.t0.encoding["units"] == "days since 1950-01-01"

    @requires_cftime
    def test_roundtrip_cftime_datetime_data(self):
        from .test_coding_times import _all_cftime_date_types

        date_types = _all_cftime_date_types()
        for date_type in date_types.values():
            times = [date_type(1, 1, 1), date_type(1, 1, 2)]
            expected = Dataset({"t": ("t", times), "t0": times[0]})
            kwargs = {"encoding": {"t0": {"units": "days since 0001-01-01"}}}
            expected_decoded_t = np.array(times)
            expected_decoded_t0 = np.array([date_type(1, 1, 1)])
            expected_calendar = times[0].calendar

            with warnings.catch_warnings():
                if expected_calendar in {"proleptic_gregorian", "gregorian"}:
                    warnings.filterwarnings("ignore", "Unable to decode time axis")

                with self.roundtrip(expected, save_kwargs=kwargs) as actual:
                    abs_diff = abs(actual.t.values - expected_decoded_t)
                    assert (abs_diff <= np.timedelta64(1, "s")).all()
                    assert (
                        actual.t.encoding["units"]
                        == "days since 0001-01-01 00:00:00.000000"
                    )
                    assert actual.t.encoding["calendar"] == expected_calendar

                    abs_diff = abs(actual.t0.values - expected_decoded_t0)
                    assert (abs_diff <= np.timedelta64(1, "s")).all()
                    assert actual.t0.encoding["units"] == "days since 0001-01-01"
                    assert actual.t.encoding["calendar"] == expected_calendar

    def test_roundtrip_timedelta_data(self):
        time_deltas = pd.to_timedelta(["1h", "2h", "NaT"])
        expected = Dataset({"td": ("td", time_deltas), "td0": time_deltas[0]})
        with self.roundtrip(expected) as actual:
            assert_identical(expected, actual)

    def test_roundtrip_float64_data(self):
        expected = Dataset({"x": ("y", np.array([1.0, 2.0, np.pi], dtype="float64"))})
        with self.roundtrip(expected) as actual:
            assert_identical(expected, actual)

    def test_roundtrip_example_1_netcdf(self):
        with open_example_dataset("example_1.nc") as expected:
            with self.roundtrip(expected) as actual:
                # we allow the attributes to differ since that
                # will depend on the encoding used.  For example,
                # without CF encoding 'actual' will end up with
                # a dtype attribute.
                assert_equal(expected, actual)

    def test_roundtrip_coordinates(self):
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

    def test_roundtrip_global_coordinates(self):
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

    def test_roundtrip_coordinates_with_space(self):
        original = Dataset(coords={"x": 0, "y z": 1})
        expected = Dataset({"y z": 1}, {"x": 0})
        with pytest.warns(SerializationWarning):
            with self.roundtrip(original) as actual:
                assert_identical(expected, actual)

    def test_roundtrip_boolean_dtype(self):
        original = create_boolean_data()
        assert original["x"].dtype == "bool"
        with self.roundtrip(original) as actual:
            assert_identical(original, actual)
            assert actual["x"].dtype == "bool"

    def test_orthogonal_indexing(self):
        in_memory = create_test_data()
        with self.roundtrip(in_memory) as on_disk:
            indexers = {"dim1": [1, 2, 0], "dim2": [3, 2, 0, 3], "dim3": np.arange(5)}
            expected = in_memory.isel(**indexers)
            actual = on_disk.isel(**indexers)
            # make sure the array is not yet loaded into memory
            assert not actual["var1"].variable._in_memory
            assert_identical(expected, actual)
            # do it twice, to make sure we're switched from orthogonal -> numpy
            # when we cached the values
            actual = on_disk.isel(**indexers)
            assert_identical(expected, actual)

    def test_vectorized_indexing(self):
        in_memory = create_test_data()
        with self.roundtrip(in_memory) as on_disk:
            indexers = {
                "dim1": DataArray([0, 2, 0], dims="a"),
                "dim2": DataArray([0, 2, 3], dims="a"),
            }
            expected = in_memory.isel(**indexers)
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
                    actual = actual.isel(**ind)
                    expected = expected.isel(**ind)
                    # make sure the array is not yet loaded into memory
                    assert not actual.variable._in_memory
                assert_identical(expected, actual.load())

        # two-staged vectorized-indexing
        indexers = [
            {
                "dim1": DataArray([[0, 7], [2, 6], [3, 5]], dims=["a", "b"]),
                "dim3": DataArray([[0, 4], [1, 3], [2, 2]], dims=["a", "b"]),
            },
            {"a": DataArray([0, 1], dims=["c"]), "b": DataArray([0, 1], dims=["c"])},
        ]
        multiple_indexing(indexers)

        # vectorized-slice mixed
        indexers = [
            {
                "dim1": DataArray([[0, 7], [2, 6], [3, 5]], dims=["a", "b"]),
                "dim3": slice(None, 10),
            }
        ]
        multiple_indexing(indexers)

        # vectorized-integer mixed
        indexers = [
            {"dim3": 0},
            {"dim1": DataArray([[0, 7], [2, 6], [3, 5]], dims=["a", "b"])},
            {"a": slice(None, None, 2)},
        ]
        multiple_indexing(indexers)

        # vectorized-integer mixed
        indexers = [
            {"dim3": 0},
            {"dim1": DataArray([[0, 7], [2, 6], [3, 5]], dims=["a", "b"])},
            {"a": 1, "b": 0},
        ]
        multiple_indexing(indexers)

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

    def test_isel_dataarray(self):
        # Make sure isel works lazily. GH:issue:1688
        in_memory = create_test_data()
        with self.roundtrip(in_memory) as on_disk:
            expected = in_memory.isel(dim2=in_memory["dim2"] < 3)
            actual = on_disk.isel(dim2=on_disk["dim2"] < 3)
            assert_identical(expected, actual)

    def validate_array_type(self, ds):
        # Make sure that only NumpyIndexingAdapter stores a bare np.ndarray.
        def find_and_validate_array(obj):
            # recursively called function. obj: array or array wrapper.
            if hasattr(obj, "array"):
                if isinstance(obj.array, indexing.ExplicitlyIndexed):
                    find_and_validate_array(obj.array)
                else:
                    if isinstance(obj.array, np.ndarray):
                        assert isinstance(obj, indexing.NumpyIndexingAdapter)
                    elif isinstance(obj.array, dask_array_type):
                        assert isinstance(obj, indexing.DaskIndexingAdapter)
                    elif isinstance(obj.array, pd.Index):
                        assert isinstance(obj, indexing.PandasIndexAdapter)
                    else:
                        raise TypeError(
                            "{} is wrapped by {}".format(type(obj.array), type(obj))
                        )

        for k, v in ds.variables.items():
            find_and_validate_array(v._data)

    def test_array_type_after_indexing(self):
        in_memory = create_test_data()
        with self.roundtrip(in_memory) as on_disk:
            self.validate_array_type(on_disk)
            indexers = {"dim1": [1, 2, 0], "dim2": [3, 2, 0, 3], "dim3": np.arange(5)}
            expected = in_memory.isel(**indexers)
            actual = on_disk.isel(**indexers)
            assert_identical(expected, actual)
            self.validate_array_type(actual)
            # do it twice, to make sure we're switched from orthogonal -> numpy
            # when we cached the values
            actual = on_disk.isel(**indexers)
            assert_identical(expected, actual)
            self.validate_array_type(actual)

    def test_dropna(self):
        # regression test for GH:issue:1694
        a = np.random.randn(4, 3)
        a[1, 1] = np.NaN
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

    def test_ondisk_after_print(self):
        """ Make sure print does not load file into memory """
        in_memory = create_test_data()
        with self.roundtrip(in_memory) as on_disk:
            repr(on_disk)
            assert not on_disk["var1"]._in_memory


class CFEncodedBase(DatasetIOBase):
    def test_roundtrip_bytes_with_fill_value(self):
        values = np.array([b"ab", b"cdef", np.nan], dtype=object)
        encoding = {"_FillValue": b"X", "dtype": "S1"}
        original = Dataset({"x": ("t", values, {}, encoding)})
        expected = original.copy(deep=True)
        with self.roundtrip(original) as actual:
            assert_identical(expected, actual)

        original = Dataset({"x": ("t", values, {}, {"_FillValue": b""})})
        with self.roundtrip(original) as actual:
            assert_identical(expected, actual)

    def test_roundtrip_string_with_fill_value_nchar(self):
        values = np.array(["ab", "cdef", np.nan], dtype=object)
        expected = Dataset({"x": ("t", values)})

        encoding = {"dtype": "S1", "_FillValue": b"X"}
        original = Dataset({"x": ("t", values, {}, encoding)})
        # Not supported yet.
        with pytest.raises(NotImplementedError):
            with self.roundtrip(original) as actual:
                assert_identical(expected, actual)

    @pytest.mark.parametrize(
        "decoded_fn, encoded_fn",
        [
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
            (create_masked_and_scaled_data, create_encoded_masked_and_scaled_data),
        ],
    )
    def test_roundtrip_mask_and_scale(self, decoded_fn, encoded_fn):
        decoded = decoded_fn()
        encoded = encoded_fn()

        with self.roundtrip(decoded) as actual:
            for k in decoded.variables:
                assert decoded.variables[k].dtype == actual.variables[k].dtype
            assert_allclose(decoded, actual, decode_bytes=False)

        with self.roundtrip(decoded, open_kwargs=dict(decode_cf=False)) as actual:
            # TODO: this assumes that all roundtrips will first
            # encode.  Is that something we want to test for?
            for k in encoded.variables:
                assert encoded.variables[k].dtype == actual.variables[k].dtype
            assert_allclose(encoded, actual, decode_bytes=False)

        with self.roundtrip(encoded, open_kwargs=dict(decode_cf=False)) as actual:
            for k in encoded.variables:
                assert encoded.variables[k].dtype == actual.variables[k].dtype
            assert_allclose(encoded, actual, decode_bytes=False)

        # make sure roundtrip encoding didn't change the
        # original dataset.
        assert_allclose(encoded, encoded_fn(), decode_bytes=False)

        with self.roundtrip(encoded) as actual:
            for k in decoded.variables:
                assert decoded.variables[k].dtype == actual.variables[k].dtype
            assert_allclose(decoded, actual, decode_bytes=False)

    def test_coordinates_encoding(self):
        def equals_latlon(obj):
            return obj == "lat lon" or obj == "lon lat"

        original = Dataset(
            {"temp": ("x", [0, 1]), "precip": ("x", [0, -1])},
            {"lat": ("x", [2, 3]), "lon": ("x", [4, 5])},
        )
        with self.roundtrip(original) as actual:
            assert_identical(actual, original)
        with create_tmp_file() as tmp_file:
            original.to_netcdf(tmp_file)
            with open_dataset(tmp_file, decode_coords=False) as ds:
                assert equals_latlon(ds["temp"].attrs["coordinates"])
                assert equals_latlon(ds["precip"].attrs["coordinates"])
                assert "coordinates" not in ds.attrs
                assert "coordinates" not in ds["lat"].attrs
                assert "coordinates" not in ds["lon"].attrs

        modified = original.drop_vars(["temp", "precip"])
        with self.roundtrip(modified) as actual:
            assert_identical(actual, modified)
        with create_tmp_file() as tmp_file:
            modified.to_netcdf(tmp_file)
            with open_dataset(tmp_file, decode_coords=False) as ds:
                assert equals_latlon(ds.attrs["coordinates"])
                assert "coordinates" not in ds["lat"].attrs
                assert "coordinates" not in ds["lon"].attrs

        original["temp"].encoding["coordinates"] = "lat"
        with self.roundtrip(original) as actual:
            assert_identical(actual, original)
        original["precip"].encoding["coordinates"] = "lat"
        with create_tmp_file() as tmp_file:
            original.to_netcdf(tmp_file)
            with open_dataset(tmp_file, decode_coords=True) as ds:
                assert "lon" not in ds["temp"].encoding["coordinates"]
                assert "lon" not in ds["precip"].encoding["coordinates"]
                assert "coordinates" not in ds["lat"].encoding
                assert "coordinates" not in ds["lon"].encoding

    def test_roundtrip_endian(self):
        ds = Dataset(
            {
                "x": np.arange(3, 10, dtype=">i2"),
                "y": np.arange(3, 20, dtype="<i4"),
                "z": np.arange(3, 30, dtype="=i8"),
                "w": ("x", np.arange(3, 10, dtype=np.float)),
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

    def test_invalid_dataarray_names_raise(self):
        te = (TypeError, "string or None")
        ve = (ValueError, "string must be length 1 or")
        data = np.random.random((2, 2))
        da = xr.DataArray(data)
        for name, e in zip([0, (4, 5), True, ""], [te, te, te, ve]):
            ds = Dataset({name: da})
            with raises_regex(*e):
                with self.roundtrip(ds):
                    pass

    def test_encoding_kwarg(self):
        ds = Dataset({"x": ("y", np.arange(10.0))})
        kwargs = dict(encoding={"x": {"dtype": "f4"}})
        with self.roundtrip(ds, save_kwargs=kwargs) as actual:
            encoded_dtype = actual.x.encoding["dtype"]
            # On OS X, dtype sometimes switches endianness for unclear reasons
            assert encoded_dtype.kind == "f" and encoded_dtype.itemsize == 4
        assert ds.x.encoding == {}

        kwargs = dict(encoding={"x": {"foo": "bar"}})
        with raises_regex(ValueError, "unexpected encoding"):
            with self.roundtrip(ds, save_kwargs=kwargs) as actual:
                pass

        kwargs = dict(encoding={"x": "foo"})
        with raises_regex(ValueError, "must be castable"):
            with self.roundtrip(ds, save_kwargs=kwargs) as actual:
                pass

        kwargs = dict(encoding={"invalid": {}})
        with pytest.raises(KeyError):
            with self.roundtrip(ds, save_kwargs=kwargs) as actual:
                pass

    def test_encoding_kwarg_dates(self):
        ds = Dataset({"t": pd.date_range("2000-01-01", periods=3)})
        units = "days since 1900-01-01"
        kwargs = dict(encoding={"t": {"units": units}})
        with self.roundtrip(ds, save_kwargs=kwargs) as actual:
            assert actual.t.encoding["units"] == units
            assert_identical(actual, ds)

    def test_encoding_kwarg_fixed_width_string(self):
        # regression test for GH2149
        for strings in [[b"foo", b"bar", b"baz"], ["foo", "bar", "baz"]]:
            ds = Dataset({"x": strings})
            kwargs = dict(encoding={"x": {"dtype": "S1"}})
            with self.roundtrip(ds, save_kwargs=kwargs) as actual:
                assert actual["x"].encoding["dtype"] == "S1"
                assert_identical(actual, ds)

    def test_default_fill_value(self):
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

    def test_explicitly_omit_fill_value(self):
        ds = Dataset({"x": ("y", [np.pi, -np.pi])})
        ds.x.encoding["_FillValue"] = None
        with self.roundtrip(ds) as actual:
            assert "_FillValue" not in actual.x.encoding

    def test_explicitly_omit_fill_value_via_encoding_kwarg(self):
        ds = Dataset({"x": ("y", [np.pi, -np.pi])})
        kwargs = dict(encoding={"x": {"_FillValue": None}})
        with self.roundtrip(ds, save_kwargs=kwargs) as actual:
            assert "_FillValue" not in actual.x.encoding
        assert ds.y.encoding == {}

    def test_explicitly_omit_fill_value_in_coord(self):
        ds = Dataset({"x": ("y", [np.pi, -np.pi])}, coords={"y": [0.0, 1.0]})
        ds.y.encoding["_FillValue"] = None
        with self.roundtrip(ds) as actual:
            assert "_FillValue" not in actual.y.encoding

    def test_explicitly_omit_fill_value_in_coord_via_encoding_kwarg(self):
        ds = Dataset({"x": ("y", [np.pi, -np.pi])}, coords={"y": [0.0, 1.0]})
        kwargs = dict(encoding={"y": {"_FillValue": None}})
        with self.roundtrip(ds, save_kwargs=kwargs) as actual:
            assert "_FillValue" not in actual.y.encoding
        assert ds.y.encoding == {}

    def test_encoding_same_dtype(self):
        ds = Dataset({"x": ("y", np.arange(10.0, dtype="f4"))})
        kwargs = dict(encoding={"x": {"dtype": "f4"}})
        with self.roundtrip(ds, save_kwargs=kwargs) as actual:
            encoded_dtype = actual.x.encoding["dtype"]
            # On OS X, dtype sometimes switches endianness for unclear reasons
            assert encoded_dtype.kind == "f" and encoded_dtype.itemsize == 4
        assert ds.x.encoding == {}

    def test_append_write(self):
        # regression for GH1215
        data = create_test_data()
        with self.roundtrip_append(data) as actual:
            assert_identical(data, actual)

    def test_append_overwrite_values(self):
        # regression for GH1215
        data = create_test_data()
        with create_tmp_file(allow_cleanup_failure=False) as tmp_file:
            self.save(data, tmp_file, mode="w")
            data["var2"][:] = -999
            data["var9"] = data["var2"] * 3
            self.save(data[["var2", "var9"]], tmp_file, mode="a")
            with self.open(tmp_file) as actual:
                assert_identical(data, actual)

    def test_append_with_invalid_dim_raises(self):
        data = create_test_data()
        with create_tmp_file(allow_cleanup_failure=False) as tmp_file:
            self.save(data, tmp_file, mode="w")
            data["var9"] = data["var2"] * 3
            data = data.isel(dim1=slice(2, 6))  # modify one dimension
            with raises_regex(
                ValueError, "Unable to update size for existing dimension"
            ):
                self.save(data, tmp_file, mode="a")

    def test_multiindex_not_implemented(self):
        ds = Dataset(coords={"y": ("x", [1, 2]), "z": ("x", ["a", "b"])}).set_index(
            x=["y", "z"]
        )
        with raises_regex(NotImplementedError, "MultiIndex"):
            with self.roundtrip(ds):
                pass


_counter = itertools.count()


@contextlib.contextmanager
def create_tmp_file(suffix=".nc", allow_cleanup_failure=False):
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, "temp-{}{}".format(next(_counter), suffix))
    try:
        yield path
    finally:
        try:
            shutil.rmtree(temp_dir)
        except OSError:
            if not allow_cleanup_failure:
                raise


@contextlib.contextmanager
def create_tmp_files(nfiles, suffix=".nc", allow_cleanup_failure=False):
    with ExitStack() as stack:
        files = [
            stack.enter_context(create_tmp_file(suffix, allow_cleanup_failure))
            for apath in np.arange(nfiles)
        ]
        yield files


class NetCDF4Base(CFEncodedBase):
    """Tests for both netCDF4-python and h5netcdf."""

    engine = "netcdf4"

    def test_open_group(self):
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
            with pytest.raises(IOError):
                open_dataset(tmp_file, group="bar")
            with raises_regex(ValueError, "must be a string"):
                open_dataset(tmp_file, group=(1, 2, 3))

    def test_open_subgroup(self):
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

    def test_write_groups(self):
        data1 = create_test_data()
        data2 = data1 * 2
        with create_tmp_file() as tmp_file:
            self.save(data1, tmp_file, group="data/1")
            self.save(data2, tmp_file, group="data/2", mode="a")
            with self.open(tmp_file, group="data/1") as actual1:
                assert_identical(data1, actual1)
            with self.open(tmp_file, group="data/2") as actual2:
                assert_identical(data2, actual2)

    def test_encoding_kwarg_vlen_string(self):
        for input_strings in [[b"foo", b"bar", b"baz"], ["foo", "bar", "baz"]]:
            original = Dataset({"x": input_strings})
            expected = Dataset({"x": ["foo", "bar", "baz"]})
            kwargs = dict(encoding={"x": {"dtype": str}})
            with self.roundtrip(original, save_kwargs=kwargs) as actual:
                assert actual["x"].encoding["dtype"] is str
                assert_identical(actual, expected)

    def test_roundtrip_string_with_fill_value_vlen(self):
        values = np.array(["ab", "cdef", np.nan], dtype=object)
        expected = Dataset({"x": ("t", values)})

        # netCDF4-based backends don't support an explicit fillvalue
        # for variable length strings yet.
        # https://github.com/Unidata/netcdf4-python/issues/730
        # https://github.com/shoyer/h5netcdf/issues/37
        original = Dataset({"x": ("t", values, {}, {"_FillValue": "XXX"})})
        with pytest.raises(NotImplementedError):
            with self.roundtrip(original) as actual:
                assert_identical(expected, actual)

        original = Dataset({"x": ("t", values, {}, {"_FillValue": ""})})
        with pytest.raises(NotImplementedError):
            with self.roundtrip(original) as actual:
                assert_identical(expected, actual)

    def test_roundtrip_character_array(self):
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

    def test_default_to_char_arrays(self):
        data = Dataset({"x": np.array(["foo", "zzzz"], dtype="S")})
        with self.roundtrip(data) as actual:
            assert_identical(data, actual)
            assert actual["x"].dtype == np.dtype("S4")

    def test_open_encodings(self):
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

            time = pd.date_range("1999-01-05", periods=10)
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

    def test_dump_encodings(self):
        # regression test for #709
        ds = Dataset({"x": ("y", np.arange(10.0))})
        kwargs = dict(encoding={"x": {"zlib": True}})
        with self.roundtrip(ds, save_kwargs=kwargs) as actual:
            assert actual.x.encoding["zlib"]

    def test_dump_and_open_encodings(self):
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

    def test_compression_encoding(self):
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

    def test_encoding_kwarg_compression(self):
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

    def test_keep_chunksizes_if_no_original_shape(self):
        ds = Dataset({"x": [1, 2, 3]})
        chunksizes = (2,)
        ds.variables["x"].encoding = {"chunksizes": chunksizes}

        with self.roundtrip(ds) as actual:
            assert_identical(ds, actual)
            assert_array_equal(
                ds["x"].encoding["chunksizes"], actual["x"].encoding["chunksizes"]
            )

    def test_encoding_chunksizes_unlimited(self):
        # regression test for GH1225
        ds = Dataset({"x": [1, 2, 3], "y": ("x", [2, 3, 4])})
        ds.variables["x"].encoding = {
            "zlib": False,
            "shuffle": False,
            "complevel": 0,
            "fletcher32": False,
            "contiguous": False,
            "chunksizes": (2 ** 20,),
            "original_shape": (3,),
        }
        with self.roundtrip(ds) as actual:
            assert_equal(ds, actual)

    def test_mask_and_scale(self):
        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, mode="w") as nc:
                nc.createDimension("t", 5)
                nc.createVariable("x", "int16", ("t",), fill_value=-1)
                v = nc.variables["x"]
                v.set_auto_maskandscale(False)
                v.add_offset = 10
                v.scale_factor = 0.1
                v[:] = np.array([-1, -1, 0, 1, 2])

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
                expected = create_masked_and_scaled_data()
                assert_identical(expected, ds)

    def test_0dimensional_variable(self):
        # This fix verifies our work-around to this netCDF4-python bug:
        # https://github.com/Unidata/netcdf4-python/pull/220
        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, mode="w") as nc:
                v = nc.createVariable("x", "int16")
                v[...] = 123

            with open_dataset(tmp_file) as ds:
                expected = Dataset({"x": ((), 123)})
                assert_identical(expected, ds)

    def test_read_variable_len_strings(self):
        with create_tmp_file() as tmp_file:
            values = np.array(["foo", "bar", "baz"], dtype=object)

            with nc4.Dataset(tmp_file, mode="w") as nc:
                nc.createDimension("x", 3)
                v = nc.createVariable("x", str, ("x",))
                v[:] = values

            expected = Dataset({"x": ("x", values)})
            for kwargs in [{}, {"decode_cf": True}]:
                with open_dataset(tmp_file, **kwargs) as actual:
                    assert_identical(expected, actual)

    def test_encoding_unlimited_dims(self):
        ds = Dataset({"x": ("y", np.arange(10.0))})
        with self.roundtrip(ds, save_kwargs=dict(unlimited_dims=["y"])) as actual:
            assert actual.encoding["unlimited_dims"] == set("y")
            assert_equal(ds, actual)
        ds.encoding = {"unlimited_dims": ["y"]}
        with self.roundtrip(ds) as actual:
            assert actual.encoding["unlimited_dims"] == set("y")
            assert_equal(ds, actual)


@requires_netCDF4
class TestNetCDF4Data(NetCDF4Base):
    @contextlib.contextmanager
    def create_store(self):
        with create_tmp_file() as tmp_file:
            with backends.NetCDF4DataStore.open(tmp_file, mode="w") as store:
                yield store

    def test_variable_order(self):
        # doesn't work with scipy or h5py :(
        ds = Dataset()
        ds["a"] = 1
        ds["z"] = 2
        ds["b"] = 3
        ds.coords["c"] = 4

        with self.roundtrip(ds) as actual:
            assert list(ds.variables) == list(actual.variables)

    def test_unsorted_index_raises(self):
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
                ds2.randovar.values
            except IndexError as err:
                assert "first by calling .load" in str(err)

    def test_setncattr_string(self):
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

    def test_autoclose_future_warning(self):
        data = create_test_data()
        with create_tmp_file() as tmp_file:
            self.save(data, tmp_file)
            with pytest.warns(FutureWarning):
                with self.open(tmp_file, autoclose=True) as actual:
                    assert_identical(data, actual)

    def test_already_open_dataset(self):
        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, mode="w") as nc:
                v = nc.createVariable("x", "int")
                v[...] = 42

            nc = nc4.Dataset(tmp_file, mode="r")
            store = backends.NetCDF4DataStore(nc)
            with open_dataset(store) as ds:
                expected = Dataset({"x": ((), 42)})
                assert_identical(expected, ds)

    def test_already_open_dataset_group(self):
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

    def test_unsorted_index_raises(self):
        # Skip when using dask because dask rewrites indexers to getitem,
        # dask first pulls items by block.
        pass

    def test_dataset_caching(self):
        # caching behavior differs for dask
        pass

    def test_write_inconsistent_chunks(self):
        # Construct two variables with the same dimensions, but different
        # chunk sizes.
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


@requires_zarr
class ZarrBase(CFEncodedBase):

    DIMENSION_KEY = "_ARRAY_DIMENSIONS"

    @contextlib.contextmanager
    def create_store(self):
        with self.create_zarr_target() as store_target:
            yield backends.ZarrStore.open_group(store_target, mode="w")

    def save(self, dataset, store_target, **kwargs):
        return dataset.to_zarr(store=store_target, **kwargs)

    @contextlib.contextmanager
    def open(self, store_target, **kwargs):
        with xr.open_zarr(store_target, **kwargs) as ds:
            yield ds

    @contextlib.contextmanager
    def roundtrip(
        self, data, save_kwargs=None, open_kwargs=None, allow_cleanup_failure=False
    ):
        if save_kwargs is None:
            save_kwargs = {}
        if open_kwargs is None:
            open_kwargs = {}
        with self.create_zarr_target() as store_target:
            self.save(data, store_target, **save_kwargs)
            with self.open(store_target, **open_kwargs) as ds:
                yield ds

    @contextlib.contextmanager
    def roundtrip_append(
        self, data, save_kwargs=None, open_kwargs=None, allow_cleanup_failure=False
    ):
        pytest.skip("zarr backend does not support appending")

    def test_roundtrip_consolidated(self):
        pytest.importorskip("zarr", minversion="2.2.1.dev2")
        expected = create_test_data()
        with self.roundtrip(
            expected,
            save_kwargs={"consolidated": True},
            open_kwargs={"consolidated": True},
        ) as actual:
            self.check_dtypes_roundtripped(expected, actual)
            assert_identical(expected, actual)

    def test_auto_chunk(self):
        original = create_test_data().chunk()

        with self.roundtrip(original, open_kwargs={"chunks": None}) as actual:
            for k, v in actual.variables.items():
                # only index variables should be in memory
                assert v._in_memory == (k in actual.dims)
                # there should be no chunks
                assert v.chunks is None

        with self.roundtrip(original, open_kwargs={"chunks": "auto"}) as actual:
            for k, v in actual.variables.items():
                # only index variables should be in memory
                assert v._in_memory == (k in actual.dims)
                # chunk size should be the same as original
                assert v.chunks == original[k].chunks

    @pytest.mark.filterwarnings("ignore:Specified Dask chunks")
    def test_manual_chunk(self):
        original = create_test_data().chunk({"dim1": 3, "dim2": 4, "dim3": 3})

        # All of these should return non-chunked arrays
        NO_CHUNKS = (None, 0, {})
        for no_chunk in NO_CHUNKS:
            open_kwargs = {"chunks": no_chunk}
            with self.roundtrip(original, open_kwargs=open_kwargs) as actual:
                for k, v in actual.variables.items():
                    # only index variables should be in memory
                    assert v._in_memory == (k in actual.dims)
                    # there should be no chunks
                    assert v.chunks is None

        # uniform arrays
        for i in range(2, 6):
            rechunked = original.chunk(chunks=i)
            open_kwargs = {"chunks": i}
            with self.roundtrip(original, open_kwargs=open_kwargs) as actual:
                for k, v in actual.variables.items():
                    # only index variables should be in memory
                    assert v._in_memory == (k in actual.dims)
                    # chunk size should be the same as rechunked
                    assert v.chunks == rechunked[k].chunks

        chunks = {"dim1": 2, "dim2": 3, "dim3": 5}
        rechunked = original.chunk(chunks=chunks)

        open_kwargs = {"chunks": chunks, "overwrite_encoded_chunks": True}
        with self.roundtrip(original, open_kwargs=open_kwargs) as actual:
            for k, v in actual.variables.items():
                assert v.chunks == rechunked[k].chunks

            with self.roundtrip(actual) as auto:
                # encoding should have changed
                for k, v in actual.variables.items():
                    assert v.chunks == rechunked[k].chunks

                assert_identical(actual, auto)
                assert_identical(actual.load(), auto.load())

    def test_warning_on_bad_chunks(self):
        original = create_test_data().chunk({"dim1": 4, "dim2": 3, "dim3": 5})

        bad_chunks = (2, {"dim2": (3, 3, 2, 1)})
        for chunks in bad_chunks:
            kwargs = {"chunks": chunks}
            with pytest.warns(UserWarning):
                with self.roundtrip(original, open_kwargs=kwargs) as actual:
                    for k, v in actual.variables.items():
                        # only index variables should be in memory
                        assert v._in_memory == (k in actual.dims)

        good_chunks = ({"dim2": 3}, {"dim3": 10})
        for chunks in good_chunks:
            kwargs = {"chunks": chunks}
            with pytest.warns(None) as record:
                with self.roundtrip(original, open_kwargs=kwargs) as actual:
                    for k, v in actual.variables.items():
                        # only index variables should be in memory
                        assert v._in_memory == (k in actual.dims)
            assert len(record) == 0

    def test_deprecate_auto_chunk(self):
        original = create_test_data().chunk()
        with pytest.warns(FutureWarning):
            with self.roundtrip(original, open_kwargs={"auto_chunk": True}) as actual:
                for k, v in actual.variables.items():
                    # only index variables should be in memory
                    assert v._in_memory == (k in actual.dims)
                    # chunk size should be the same as original
                    assert v.chunks == original[k].chunks

        with pytest.warns(FutureWarning):
            with self.roundtrip(original, open_kwargs={"auto_chunk": False}) as actual:
                for k, v in actual.variables.items():
                    # only index variables should be in memory
                    assert v._in_memory == (k in actual.dims)
                    # there should be no chunks
                    assert v.chunks is None

    def test_write_uneven_dask_chunks(self):
        # regression for GH#2225
        original = create_test_data().chunk({"dim1": 3, "dim2": 4, "dim3": 3})
        with self.roundtrip(original, open_kwargs={"chunks": "auto"}) as actual:
            for k, v in actual.data_vars.items():
                print(k)
                assert v.chunks == actual[k].chunks

    def test_chunk_encoding(self):
        # These datasets have no dask chunks. All chunking specified in
        # encoding
        data = create_test_data()
        chunks = (5, 5)
        data["var2"].encoding.update({"chunks": chunks})

        with self.roundtrip(data) as actual:
            assert chunks == actual["var2"].encoding["chunks"]

        # expect an error with non-integer chunks
        data["var2"].encoding.update({"chunks": (5, 4.5)})
        with pytest.raises(TypeError):
            with self.roundtrip(data) as actual:
                pass

    def test_chunk_encoding_with_dask(self):
        # These datasets DO have dask chunks. Need to check for various
        # interactions between dask and zarr chunks
        ds = xr.DataArray((np.arange(12)), dims="x", name="var1").to_dataset()

        # - no encoding specified -
        # zarr automatically gets chunk information from dask chunks
        ds_chunk4 = ds.chunk({"x": 4})
        with self.roundtrip(ds_chunk4) as actual:
            assert (4,) == actual["var1"].encoding["chunks"]

        # should fail if dask_chunks are irregular...
        ds_chunk_irreg = ds.chunk({"x": (5, 4, 3)})
        with pytest.raises(ValueError) as e_info:
            with self.roundtrip(ds_chunk_irreg) as actual:
                pass
        # make sure this error message is correct and not some other error
        assert e_info.match("chunks")

        # ... except if the last chunk is smaller than the first
        ds_chunk_irreg = ds.chunk({"x": (5, 5, 2)})
        with self.roundtrip(ds_chunk_irreg) as actual:
            assert (5,) == actual["var1"].encoding["chunks"]
        # re-save Zarr arrays
        with self.roundtrip(ds_chunk_irreg) as original:
            with self.roundtrip(original) as actual:
                assert_identical(original, actual)

        # - encoding specified  -
        # specify compatible encodings
        for chunk_enc in 4, (4,):
            ds_chunk4["var1"].encoding.update({"chunks": chunk_enc})
            with self.roundtrip(ds_chunk4) as actual:
                assert (4,) == actual["var1"].encoding["chunks"]

        # TODO: remove this failure once syncronized overlapping writes are
        # supported by xarray
        ds_chunk4["var1"].encoding.update({"chunks": 5})
        with pytest.raises(NotImplementedError):
            with self.roundtrip(ds_chunk4) as actual:
                pass

    def test_hidden_zarr_keys(self):
        expected = create_test_data()
        with self.create_store() as store:
            expected.dump_to_store(store)
            zarr_group = store.ds

            # check that a variable hidden attribute is present and correct
            # JSON only has a single array type, which maps to list in Python.
            # In contrast, dims in xarray is always a tuple.
            for var in expected.variables.keys():
                dims = zarr_group[var].attrs[self.DIMENSION_KEY]
                assert dims == list(expected[var].dims)

            with xr.decode_cf(store):
                # make sure it is hidden
                for var in expected.variables.keys():
                    assert self.DIMENSION_KEY not in expected[var].attrs

            # put it back and try removing from a variable
            del zarr_group.var2.attrs[self.DIMENSION_KEY]
            with pytest.raises(KeyError):
                with xr.decode_cf(store):
                    pass

    @pytest.mark.skipif(LooseVersion(dask_version) < "2.4", reason="dask GH5334")
    def test_write_persistence_modes(self):
        original = create_test_data()

        # overwrite mode
        with self.roundtrip(original, save_kwargs={"mode": "w"}) as actual:
            assert_identical(original, actual)

        # don't overwrite mode
        with self.roundtrip(original, save_kwargs={"mode": "w-"}) as actual:
            assert_identical(original, actual)

        # make sure overwriting works as expected
        with self.create_zarr_target() as store:
            self.save(original, store)
            # should overwrite with no error
            self.save(original, store, mode="w")
            with self.open(store) as actual:
                assert_identical(original, actual)
                with pytest.raises(ValueError):
                    self.save(original, store, mode="w-")

        # check append mode for normal write
        with self.roundtrip(original, save_kwargs={"mode": "a"}) as actual:
            assert_identical(original, actual)

        ds, ds_to_append, _ = create_append_test_data()

        # check append mode for append write
        with self.create_zarr_target() as store_target:
            ds.to_zarr(store_target, mode="w")
            ds_to_append.to_zarr(store_target, append_dim="time")
            original = xr.concat([ds, ds_to_append], dim="time")
            assert_identical(original, xr.open_zarr(store_target))

    def test_compressor_encoding(self):
        original = create_test_data()
        # specify a custom compressor
        import zarr

        blosc_comp = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
        save_kwargs = dict(encoding={"var1": {"compressor": blosc_comp}})
        with self.roundtrip(original, save_kwargs=save_kwargs) as ds:
            actual = ds["var1"].encoding["compressor"]
            # get_config returns a dictionary of compressor attributes
            assert actual.get_config() == blosc_comp.get_config()

    def test_group(self):
        original = create_test_data()
        group = "some/random/path"
        with self.roundtrip(
            original, save_kwargs={"group": group}, open_kwargs={"group": group}
        ) as actual:
            assert_identical(original, actual)

    def test_encoding_kwarg_fixed_width_string(self):
        # not relevant for zarr, since we don't use EncodedStringCoder
        pass

    # TODO: someone who understand caching figure out whether chaching
    # makes sense for Zarr backend
    @pytest.mark.xfail(reason="Zarr caching not implemented")
    def test_dataset_caching(self):
        super().test_dataset_caching()

    @pytest.mark.skipif(LooseVersion(dask_version) < "2.4", reason="dask GH5334")
    def test_append_write(self):
        ds, ds_to_append, _ = create_append_test_data()
        with self.create_zarr_target() as store_target:
            ds.to_zarr(store_target, mode="w")
            ds_to_append.to_zarr(store_target, append_dim="time")
            original = xr.concat([ds, ds_to_append], dim="time")
            assert_identical(original, xr.open_zarr(store_target))

    @pytest.mark.xfail(reason="Zarr stores can not be appended to")
    def test_append_overwrite_values(self):
        super().test_append_overwrite_values()

    def test_append_with_invalid_dim_raises(self):

        ds, ds_to_append, _ = create_append_test_data()

        # check failure when append_dim not valid
        with pytest.raises(ValueError):
            with self.create_zarr_target() as store_target:
                ds.to_zarr(store_target, mode="w")
                ds_to_append.to_zarr(store_target, append_dim="notvalid")

    def test_append_with_append_dim_not_set_raises(self):

        ds, ds_to_append, _ = create_append_test_data()

        # check failure when append_dim not set
        with pytest.raises(ValueError):
            with self.create_zarr_target() as store_target:
                ds.to_zarr(store_target, mode="w")
                ds_to_append.to_zarr(store_target, mode="a")

    def test_append_with_mode_not_a_raises(self):

        ds, ds_to_append, _ = create_append_test_data()

        # check failure when append_dim is set and mode != 'a'
        with pytest.raises(ValueError):
            with self.create_zarr_target() as store_target:
                ds.to_zarr(store_target, mode="w")
                ds_to_append.to_zarr(store_target, mode="w", append_dim="time")

    def test_append_with_existing_encoding_raises(self):

        ds, ds_to_append, _ = create_append_test_data()

        # check failure when providing encoding to existing variable
        with pytest.raises(ValueError):
            with self.create_zarr_target() as store_target:
                ds.to_zarr(store_target, mode="w")
                ds_to_append.to_zarr(
                    store_target,
                    append_dim="time",
                    encoding={"da": {"compressor": None}},
                )

    def test_check_encoding_is_consistent_after_append(self):

        ds, ds_to_append, _ = create_append_test_data()

        # check encoding consistency
        with self.create_zarr_target() as store_target:
            import zarr

            compressor = zarr.Blosc()
            encoding = {"da": {"compressor": compressor}}
            ds.to_zarr(store_target, mode="w", encoding=encoding)
            ds_to_append.to_zarr(store_target, append_dim="time")
            actual_ds = xr.open_zarr(store_target)
            actual_encoding = actual_ds["da"].encoding["compressor"]
            assert actual_encoding.get_config() == compressor.get_config()
            assert_identical(
                xr.open_zarr(store_target).compute(),
                xr.concat([ds, ds_to_append], dim="time"),
            )

    @pytest.mark.skipif(LooseVersion(dask_version) < "2.4", reason="dask GH5334")
    def test_append_with_new_variable(self):

        ds, ds_to_append, ds_with_new_var = create_append_test_data()

        # check append mode for new variable
        with self.create_zarr_target() as store_target:
            xr.concat([ds, ds_to_append], dim="time").to_zarr(store_target, mode="w")
            ds_with_new_var.to_zarr(store_target, mode="a")
            combined = xr.concat([ds, ds_to_append], dim="time")
            combined["new_var"] = ds_with_new_var["new_var"]
            assert_identical(combined, xr.open_zarr(store_target))

    def test_to_zarr_compute_false_roundtrip(self):
        from dask.delayed import Delayed

        original = create_test_data().chunk()

        with self.create_zarr_target() as store:
            delayed_obj = self.save(original, store, compute=False)
            assert isinstance(delayed_obj, Delayed)

            # make sure target store has not been written to yet
            with pytest.raises(AssertionError):
                with self.open(store) as actual:
                    assert_identical(original, actual)

            delayed_obj.compute()

            with self.open(store) as actual:
                assert_identical(original, actual)

    def test_to_zarr_append_compute_false_roundtrip(self):
        from dask.delayed import Delayed

        ds, ds_to_append, _ = create_append_test_data()
        ds, ds_to_append = ds.chunk(), ds_to_append.chunk()

        with self.create_zarr_target() as store:
            delayed_obj = self.save(ds, store, compute=False, mode="w")
            assert isinstance(delayed_obj, Delayed)

            with pytest.raises(AssertionError):
                with self.open(store) as actual:
                    assert_identical(ds, actual)

            delayed_obj.compute()

            with self.open(store) as actual:
                assert_identical(ds, actual)

            delayed_obj = self.save(
                ds_to_append, store, compute=False, append_dim="time"
            )
            assert isinstance(delayed_obj, Delayed)

            with pytest.raises(AssertionError):
                with self.open(store) as actual:
                    assert_identical(xr.concat([ds, ds_to_append], dim="time"), actual)

            delayed_obj.compute()

            with self.open(store) as actual:
                assert_identical(xr.concat([ds, ds_to_append], dim="time"), actual)

    def test_encoding_chunksizes(self):
        # regression test for GH2278
        # see also test_encoding_chunksizes_unlimited
        nx, ny, nt = 4, 4, 5
        original = xr.Dataset(
            {}, coords={"x": np.arange(nx), "y": np.arange(ny), "t": np.arange(nt)}
        )
        original["v"] = xr.Variable(("x", "y", "t"), np.zeros((nx, ny, nt)))
        original = original.chunk({"t": 1, "x": 2, "y": 2})

        with self.roundtrip(original) as ds1:
            assert_equal(ds1, original)
            with self.roundtrip(ds1.isel(t=0)) as ds2:
                assert_equal(ds2, original.isel(t=0))


@requires_zarr
class TestZarrDictStore(ZarrBase):
    @contextlib.contextmanager
    def create_zarr_target(self):
        yield {}


@requires_zarr
class TestZarrDirectoryStore(ZarrBase):
    @contextlib.contextmanager
    def create_zarr_target(self):
        with create_tmp_file(suffix=".zarr") as tmp:
            yield tmp


class ScipyWriteBase(CFEncodedBase, NetCDF3Only):
    def test_append_write(self):
        import scipy

        if scipy.__version__ == "1.0.1":
            pytest.xfail("https://github.com/scipy/scipy/issues/8625")
        super().test_append_write()

    def test_append_overwrite_values(self):
        import scipy

        if scipy.__version__ == "1.0.1":
            pytest.xfail("https://github.com/scipy/scipy/issues/8625")
        super().test_append_overwrite_values()


@requires_scipy
class TestScipyInMemoryData(ScipyWriteBase):
    engine = "scipy"

    @contextlib.contextmanager
    def create_store(self):
        fobj = BytesIO()
        yield backends.ScipyDataStore(fobj, "w")

    def test_to_netcdf_explicit_engine(self):
        # regression test for GH1321
        Dataset({"foo": 42}).to_netcdf(engine="scipy")

    def test_bytes_pickle(self):
        data = Dataset({"foo": ("x", [1, 2, 3])})
        fobj = data.to_netcdf()
        with self.open(fobj) as ds:
            unpickled = pickle.loads(pickle.dumps(ds))
            assert_identical(unpickled, data)


@requires_scipy
class TestScipyFileObject(ScipyWriteBase):
    engine = "scipy"

    @contextlib.contextmanager
    def create_store(self):
        fobj = BytesIO()
        yield backends.ScipyDataStore(fobj, "w")

    @contextlib.contextmanager
    def roundtrip(
        self, data, save_kwargs=None, open_kwargs=None, allow_cleanup_failure=False
    ):
        if save_kwargs is None:
            save_kwargs = {}
        if open_kwargs is None:
            open_kwargs = {}
        with create_tmp_file() as tmp_file:
            with open(tmp_file, "wb") as f:
                self.save(data, f, **save_kwargs)
            with open(tmp_file, "rb") as f:
                with self.open(f, **open_kwargs) as ds:
                    yield ds

    @pytest.mark.skip(reason="cannot pickle file objects")
    def test_pickle(self):
        pass

    @pytest.mark.skip(reason="cannot pickle file objects")
    def test_pickle_dataarray(self):
        pass


@requires_scipy
class TestScipyFilePath(ScipyWriteBase):
    engine = "scipy"

    @contextlib.contextmanager
    def create_store(self):
        with create_tmp_file() as tmp_file:
            with backends.ScipyDataStore(tmp_file, mode="w") as store:
                yield store

    def test_array_attrs(self):
        ds = Dataset(attrs={"foo": [[1, 2], [3, 4]]})
        with raises_regex(ValueError, "must be 1-dimensional"):
            with self.roundtrip(ds):
                pass

    def test_roundtrip_example_1_netcdf_gz(self):
        with open_example_dataset("example_1.nc.gz") as expected:
            with open_example_dataset("example_1.nc") as actual:
                assert_identical(expected, actual)

    def test_netcdf3_endianness(self):
        # regression test for GH416
        with open_example_dataset("bears.nc", engine="scipy") as expected:
            for var in expected.variables.values():
                assert var.dtype.isnative

    @requires_netCDF4
    def test_nc4_scipy(self):
        with create_tmp_file(allow_cleanup_failure=True) as tmp_file:
            with nc4.Dataset(tmp_file, "w", format="NETCDF4") as rootgrp:
                rootgrp.createGroup("foo")

            with raises_regex(TypeError, "pip install netcdf4"):
                open_dataset(tmp_file, engine="scipy")


@requires_netCDF4
class TestNetCDF3ViaNetCDF4Data(CFEncodedBase, NetCDF3Only):
    engine = "netcdf4"
    file_format = "NETCDF3_CLASSIC"

    @contextlib.contextmanager
    def create_store(self):
        with create_tmp_file() as tmp_file:
            with backends.NetCDF4DataStore.open(
                tmp_file, mode="w", format="NETCDF3_CLASSIC"
            ) as store:
                yield store

    def test_encoding_kwarg_vlen_string(self):
        original = Dataset({"x": ["foo", "bar", "baz"]})
        kwargs = dict(encoding={"x": {"dtype": str}})
        with raises_regex(ValueError, "encoding dtype=str for vlen"):
            with self.roundtrip(original, save_kwargs=kwargs):
                pass


@requires_netCDF4
class TestNetCDF4ClassicViaNetCDF4Data(CFEncodedBase, NetCDF3Only):
    engine = "netcdf4"
    file_format = "NETCDF4_CLASSIC"

    @contextlib.contextmanager
    def create_store(self):
        with create_tmp_file() as tmp_file:
            with backends.NetCDF4DataStore.open(
                tmp_file, mode="w", format="NETCDF4_CLASSIC"
            ) as store:
                yield store


@requires_scipy_or_netCDF4
class TestGenericNetCDFData(CFEncodedBase, NetCDF3Only):
    # verify that we can read and write netCDF3 files as long as we have scipy
    # or netCDF4-python installed
    file_format = "netcdf3_64bit"

    def test_write_store(self):
        # there's no specific store to test here
        pass

    @requires_scipy
    def test_engine(self):
        data = create_test_data()
        with raises_regex(ValueError, "unrecognized engine"):
            data.to_netcdf("foo.nc", engine="foobar")
        with raises_regex(ValueError, "invalid engine"):
            data.to_netcdf(engine="netcdf4")

        with create_tmp_file() as tmp_file:
            data.to_netcdf(tmp_file)
            with raises_regex(ValueError, "unrecognized engine"):
                open_dataset(tmp_file, engine="foobar")

        netcdf_bytes = data.to_netcdf()
        with raises_regex(ValueError, "unrecognized engine"):
            open_dataset(BytesIO(netcdf_bytes), engine="foobar")

    def test_cross_engine_read_write_netcdf3(self):
        data = create_test_data()
        valid_engines = set()
        if has_netCDF4:
            valid_engines.add("netcdf4")
        if has_scipy:
            valid_engines.add("scipy")

        for write_engine in valid_engines:
            for format in ["NETCDF3_CLASSIC", "NETCDF3_64BIT"]:
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

    def test_encoding_unlimited_dims(self):
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


@requires_h5netcdf
@requires_netCDF4
@pytest.mark.filterwarnings("ignore:use make_scale(name) instead")
class TestH5NetCDFData(NetCDF4Base):
    engine = "h5netcdf"

    @contextlib.contextmanager
    def create_store(self):
        with create_tmp_file() as tmp_file:
            yield backends.H5NetCDFStore.open(tmp_file, "w")

    @pytest.mark.filterwarnings("ignore:complex dtypes are supported by h5py")
    @pytest.mark.parametrize(
        "invalid_netcdf, warntype, num_warns",
        [(None, FutureWarning, 1), (False, FutureWarning, 1), (True, None, 0)],
    )
    def test_complex(self, invalid_netcdf, warntype, num_warns):
        expected = Dataset({"x": ("y", np.ones(5) + 1j * np.ones(5))})
        save_kwargs = {"invalid_netcdf": invalid_netcdf}
        with pytest.warns(warntype) as record:
            with self.roundtrip(expected, save_kwargs=save_kwargs) as actual:
                assert_equal(expected, actual)

        recorded_num_warns = 0
        if warntype:
            for warning in record:
                if issubclass(warning.category, warntype) and (
                    "complex dtypes" in str(warning.message)
                ):
                    recorded_num_warns += 1

        assert recorded_num_warns == num_warns

    def test_cross_engine_read_write_netcdf4(self):
        # Drop dim3, because its labels include strings. These appear to be
        # not properly read with python-netCDF4, which converts them into
        # unicode instead of leaving them as bytes.
        data = create_test_data().drop_vars("dim3")
        data.attrs["foo"] = "bar"
        valid_engines = ["netcdf4", "h5netcdf"]
        for write_engine in valid_engines:
            with create_tmp_file() as tmp_file:
                data.to_netcdf(tmp_file, engine=write_engine)
                for read_engine in valid_engines:
                    with open_dataset(tmp_file, engine=read_engine) as actual:
                        assert_identical(data, actual)

    def test_read_byte_attrs_as_unicode(self):
        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, "w") as nc:
                nc.foo = b"bar"
            with open_dataset(tmp_file) as actual:
                expected = Dataset(attrs={"foo": "bar"})
                assert_identical(expected, actual)

    def test_encoding_unlimited_dims(self):
        ds = Dataset({"x": ("y", np.arange(10.0))})
        with self.roundtrip(ds, save_kwargs=dict(unlimited_dims=["y"])) as actual:
            assert actual.encoding["unlimited_dims"] == set("y")
            assert_equal(ds, actual)
        ds.encoding = {"unlimited_dims": ["y"]}
        with self.roundtrip(ds) as actual:
            assert actual.encoding["unlimited_dims"] == set("y")
            assert_equal(ds, actual)

    def test_compression_encoding_h5py(self):
        ENCODINGS = (
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

    def test_compression_check_encoding_h5py(self):
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

        # Incompatible encodings cause a crash
        with create_tmp_file() as tmp_file:
            with raises_regex(
                ValueError, "'zlib' and 'compression' encodings mismatch"
            ):
                data.to_netcdf(
                    tmp_file,
                    engine="h5netcdf",
                    encoding={"x": {"compression": "lzf", "zlib": True}},
                )

        with create_tmp_file() as tmp_file:
            with raises_regex(
                ValueError, "'complevel' and 'compression_opts' encodings mismatch"
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

    def test_dump_encodings_h5py(self):
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

    def test_already_open_dataset_group(self):
        import h5netcdf

        with create_tmp_file() as tmp_file:
            with nc4.Dataset(tmp_file, mode="w") as nc:
                group = nc.createGroup("g")
                v = group.createVariable("x", "int")
                v[...] = 42

            h5 = h5netcdf.File(tmp_file, mode="r")
            store = backends.H5NetCDFStore(h5["g"])
            with open_dataset(store) as ds:
                expected = Dataset({"x": ((), 42)})
                assert_identical(expected, ds)

            h5 = h5netcdf.File(tmp_file, mode="r")
            store = backends.H5NetCDFStore(h5, group="g")
            with open_dataset(store) as ds:
                expected = Dataset({"x": ((), 42)})
                assert_identical(expected, ds)


@requires_h5netcdf
class TestH5NetCDFFileObject(TestH5NetCDFData):
    engine = "h5netcdf"

    def test_open_badbytes(self):
        with raises_regex(ValueError, "HDF5 as bytes"):
            with open_dataset(b"\211HDF\r\n\032\n", engine="h5netcdf"):
                pass
        with raises_regex(ValueError, "not a valid netCDF"):
            with open_dataset(b"garbage"):
                pass
        with raises_regex(ValueError, "can only read bytes"):
            with open_dataset(b"garbage", engine="netcdf4"):
                pass
        with raises_regex(ValueError, "not a valid netCDF"):
            with open_dataset(BytesIO(b"garbage"), engine="h5netcdf"):
                pass

    def test_open_twice(self):
        expected = create_test_data()
        expected.attrs["foo"] = "bar"
        with raises_regex(ValueError, "read/write pointer not at zero"):
            with create_tmp_file() as tmp_file:
                expected.to_netcdf(tmp_file, engine="h5netcdf")
                with open(tmp_file, "rb") as f:
                    with open_dataset(f, engine="h5netcdf"):
                        with open_dataset(f, engine="h5netcdf"):
                            pass

    def test_open_fileobj(self):
        # open in-memory datasets instead of local file paths
        expected = create_test_data().drop_vars("dim3")
        expected.attrs["foo"] = "bar"
        with create_tmp_file() as tmp_file:
            expected.to_netcdf(tmp_file, engine="h5netcdf")

            with open(tmp_file, "rb") as f:
                with open_dataset(f, engine="h5netcdf") as actual:
                    assert_identical(expected, actual)

                f.seek(0)
                with BytesIO(f.read()) as bio:
                    with open_dataset(bio, engine="h5netcdf") as actual:
                        assert_identical(expected, actual)


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

    def test_dataset_caching(self):
        # caching behavior differs for dask
        pass

    def test_write_inconsistent_chunks(self):
        # Construct two variables with the same dimensions, but different
        # chunk sizes.
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


@pytest.fixture(params=["scipy", "netcdf4", "h5netcdf", "pynio"])
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


# using pytest.mark.skipif does not work so this a work around
def skip_if_not_engine(engine):
    if engine == "netcdf4":
        pytest.importorskip("netCDF4")
    elif engine == "pynio":
        pytest.importorskip("Nio")
    else:
        pytest.importorskip(engine)


@requires_dask
@pytest.mark.filterwarnings("ignore:use make_scale(name) instead")
def test_open_mfdataset_manyfiles(
    readengine, nfiles, parallel, chunks, file_cache_maxsize
):

    # skip certain combinations
    skip_if_not_engine(readengine)

    if ON_WINDOWS:
        pytest.skip("Skipping on Windows")

    randdata = np.random.randn(nfiles)
    original = Dataset({"foo": ("x", randdata)})
    # test standard open_mfdataset approach with too many files
    with create_tmp_files(nfiles) as tmpfiles:
        writeengine = readengine if readengine != "pynio" else "netcdf4"
        # split into multiple sets of temp files
        for ii in original.x.values:
            subds = original.isel(x=slice(ii, ii + 1))
            subds.to_netcdf(tmpfiles[ii], engine=writeengine)

        # check that calculation on opened datasets works properly
        with open_mfdataset(
            tmpfiles,
            combine="nested",
            concat_dim="x",
            engine=readengine,
            parallel=parallel,
            chunks=chunks,
        ) as actual:

            # check that using open_mfdataset returns dask arrays for variables
            assert isinstance(actual["foo"].data, dask_array_type)

            assert_identical(original, actual)


@requires_netCDF4
@requires_dask
def test_open_mfdataset_list_attr():
    """
    Case when an attribute of type list differs across the multiple files
    """
    from netCDF4 import Dataset

    with create_tmp_files(2) as nfiles:
        for i in range(2):
            f = Dataset(nfiles[i], "w")
            f.createDimension("x", 3)
            vlvar = f.createVariable("test_var", np.int32, ("x"))
            # here create an attribute as a list
            vlvar.test_attr = [f"string a {i}", f"string b {i}"]
            vlvar[:] = np.arange(3)
            f.close()
        ds1 = open_dataset(nfiles[0])
        ds2 = open_dataset(nfiles[1])
        original = xr.concat([ds1, ds2], dim="x")
        with xr.open_mfdataset(
            [nfiles[0], nfiles[1]], combine="nested", concat_dim="x"
        ) as actual:
            assert_identical(actual, original)


@requires_scipy_or_netCDF4
@requires_dask
class TestOpenMFDatasetWithDataVarsAndCoordsKw:
    coord_name = "lon"
    var_name = "v1"

    @contextlib.contextmanager
    def setup_files_and_datasets(self, fuzz=0):
        ds1, ds2 = self.gen_datasets_with_common_coord_and_time()

        # to test join='exact'
        ds1["x"] = ds1.x + fuzz

        with create_tmp_file() as tmpfile1:
            with create_tmp_file() as tmpfile2:

                # save data to the temporary files
                ds1.to_netcdf(tmpfile1)
                ds2.to_netcdf(tmpfile2)

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

    @pytest.mark.parametrize("combine", ["nested", "by_coords"])
    @pytest.mark.parametrize("opt", ["all", "minimal", "different"])
    @pytest.mark.parametrize("join", ["outer", "inner", "left", "right"])
    def test_open_mfdataset_does_same_as_concat(self, combine, opt, join):
        with self.setup_files_and_datasets() as (files, [ds1, ds2]):
            if combine == "by_coords":
                files.reverse()
            with open_mfdataset(
                files, data_vars=opt, combine=combine, concat_dim="t", join=join
            ) as ds:
                ds_expect = xr.concat([ds1, ds2], data_vars=opt, dim="t", join=join)
                assert_identical(ds, ds_expect)

    @pytest.mark.parametrize("combine", ["nested", "by_coords"])
    @pytest.mark.parametrize("opt", ["all", "minimal", "different"])
    def test_open_mfdataset_exact_join_raises_error(self, combine, opt):
        with self.setup_files_and_datasets(fuzz=0.1) as (files, [ds1, ds2]):
            if combine == "by_coords":
                files.reverse()
            with raises_regex(ValueError, "indexes along dimension"):
                open_mfdataset(
                    files, data_vars=opt, combine=combine, concat_dim="t", join="exact"
                )

    def test_common_coord_when_datavars_all(self):
        opt = "all"

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

    def test_common_coord_when_datavars_minimal(self):
        opt = "minimal"

        with self.setup_files_and_datasets() as (files, [ds1, ds2]):
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

    def test_invalid_data_vars_value_should_fail(self):

        with self.setup_files_and_datasets() as (files, _):
            with pytest.raises(ValueError):
                with open_mfdataset(files, data_vars="minimum", combine="by_coords"):
                    pass

            # test invalid coord parameter
            with pytest.raises(ValueError):
                with open_mfdataset(files, coords="minimum", combine="by_coords"):
                    pass


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
    def test_roundtrip_string_encoded_characters(self):
        pass

    def test_roundtrip_coordinates_with_space(self):
        pass

    def test_roundtrip_numpy_datetime_data(self):
        # Override method in DatasetIOBase - remove not applicable
        # save_kwargs
        times = pd.to_datetime(["2000-01-01", "2000-01-02", "NaT"])
        expected = Dataset({"t": ("t", times), "t0": times[0]})
        with self.roundtrip(expected) as actual:
            assert_identical(expected, actual)

    def test_roundtrip_cftime_datetime_data(self):
        # Override method in DatasetIOBase - remove not applicable
        # save_kwargs
        from .test_coding_times import _all_cftime_date_types

        date_types = _all_cftime_date_types()
        for date_type in date_types.values():
            times = [date_type(1, 1, 1), date_type(1, 1, 2)]
            expected = Dataset({"t": ("t", times), "t0": times[0]})
            expected_decoded_t = np.array(times)
            expected_decoded_t0 = np.array([date_type(1, 1, 1)])

            with self.roundtrip(expected) as actual:
                abs_diff = abs(actual.t.values - expected_decoded_t)
                assert (abs_diff <= np.timedelta64(1, "s")).all()

                abs_diff = abs(actual.t0.values - expected_decoded_t0)
                assert (abs_diff <= np.timedelta64(1, "s")).all()

    def test_write_store(self):
        # Override method in DatasetIOBase - not applicable to dask
        pass

    def test_dataset_caching(self):
        expected = Dataset({"foo": ("x", [5, 6, 7])})
        with self.roundtrip(expected) as actual:
            assert not actual.foo.variable._in_memory
            actual.foo.values  # no caching
            assert not actual.foo.variable._in_memory

    def test_open_mfdataset(self):
        original = Dataset({"foo": ("x", np.random.randn(10))})
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                original.isel(x=slice(5)).to_netcdf(tmp1)
                original.isel(x=slice(5, 10)).to_netcdf(tmp2)
                with open_mfdataset(
                    [tmp1, tmp2], concat_dim="x", combine="nested"
                ) as actual:
                    assert isinstance(actual.foo.variable.data, da.Array)
                    assert actual.foo.variable.data.chunks == ((5, 5),)
                    assert_identical(original, actual)
                with open_mfdataset(
                    [tmp1, tmp2], concat_dim="x", combine="nested", chunks={"x": 3}
                ) as actual:
                    assert actual.foo.variable.data.chunks == ((3, 2, 3, 2),)

        with raises_regex(IOError, "no files to open"):
            open_mfdataset("foo-bar-baz-*.nc")

        with raises_regex(ValueError, "wild-card"):
            open_mfdataset("http://some/remote/uri")

    def test_open_mfdataset_2d(self):
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
                            assert isinstance(actual.foo.variable.data, da.Array)
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

    def test_open_mfdataset_pathlib(self):
        original = Dataset({"foo": ("x", np.random.randn(10))})
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                tmp1 = Path(tmp1)
                tmp2 = Path(tmp2)
                original.isel(x=slice(5)).to_netcdf(tmp1)
                original.isel(x=slice(5, 10)).to_netcdf(tmp2)
                with open_mfdataset(
                    [tmp1, tmp2], concat_dim="x", combine="nested"
                ) as actual:
                    assert_identical(original, actual)

    def test_open_mfdataset_2d_pathlib(self):
        original = Dataset({"foo": (["x", "y"], np.random.randn(10, 8))})
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                with create_tmp_file() as tmp3:
                    with create_tmp_file() as tmp4:
                        tmp1 = Path(tmp1)
                        tmp2 = Path(tmp2)
                        tmp3 = Path(tmp3)
                        tmp4 = Path(tmp4)
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

    def test_open_mfdataset_2(self):
        original = Dataset({"foo": ("x", np.random.randn(10))})
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                original.isel(x=slice(5)).to_netcdf(tmp1)
                original.isel(x=slice(5, 10)).to_netcdf(tmp2)

                with open_mfdataset(
                    [tmp1, tmp2], concat_dim="x", combine="nested"
                ) as actual:
                    assert_identical(original, actual)

    def test_attrs_mfdataset(self):
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
                    with raises_regex(AttributeError, "no attribute"):
                        actual.test2

    def test_open_mfdataset_attrs_file(self):
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

    def test_open_mfdataset_attrs_file_path(self):
        original = Dataset({"foo": ("x", np.random.randn(10))})
        with create_tmp_files(2) as (tmp1, tmp2):
            tmp1 = Path(tmp1)
            tmp2 = Path(tmp2)
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

    def test_open_mfdataset_auto_combine(self):
        original = Dataset({"foo": ("x", np.random.randn(10)), "x": np.arange(10)})
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                original.isel(x=slice(5)).to_netcdf(tmp1)
                original.isel(x=slice(5, 10)).to_netcdf(tmp2)

                with open_mfdataset([tmp2, tmp1], combine="by_coords") as actual:
                    assert_identical(original, actual)

    def test_open_mfdataset_combine_nested_no_concat_dim(self):
        original = Dataset({"foo": ("x", np.random.randn(10)), "x": np.arange(10)})
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                original.isel(x=slice(5)).to_netcdf(tmp1)
                original.isel(x=slice(5, 10)).to_netcdf(tmp2)

                with raises_regex(ValueError, "Must supply concat_dim"):
                    open_mfdataset([tmp2, tmp1], combine="nested")

    @pytest.mark.xfail(reason="mfdataset loses encoding currently.")
    def test_encoding_mfdataset(self):
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
                with open_mfdataset([tmp1, tmp2], combine="nested") as actual:
                    assert actual.t.encoding["units"] == original.t.encoding["units"]
                    assert actual.t.encoding["units"] == ds1.t.encoding["units"]
                    assert actual.t.encoding["units"] != ds2.t.encoding["units"]

    def test_preprocess_mfdataset(self):
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

    def test_save_mfdataset_roundtrip(self):
        original = Dataset({"foo": ("x", np.random.randn(10))})
        datasets = [original.isel(x=slice(5)), original.isel(x=slice(5, 10))]
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                save_mfdataset(datasets, [tmp1, tmp2])
                with open_mfdataset(
                    [tmp1, tmp2], concat_dim="x", combine="nested"
                ) as actual:
                    assert_identical(actual, original)

    def test_save_mfdataset_invalid(self):
        ds = Dataset()
        with raises_regex(ValueError, "cannot use mode"):
            save_mfdataset([ds, ds], ["same", "same"])
        with raises_regex(ValueError, "same length"):
            save_mfdataset([ds, ds], ["only one path"])

    def test_save_mfdataset_invalid_dataarray(self):
        # regression test for GH1555
        da = DataArray([1, 2])
        with raises_regex(TypeError, "supports writing Dataset"):
            save_mfdataset([da], ["dataarray"])

    def test_save_mfdataset_pathlib_roundtrip(self):
        original = Dataset({"foo": ("x", np.random.randn(10))})
        datasets = [original.isel(x=slice(5)), original.isel(x=slice(5, 10))]
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                tmp1 = Path(tmp1)
                tmp2 = Path(tmp2)
                save_mfdataset(datasets, [tmp1, tmp2])
                with open_mfdataset(
                    [tmp1, tmp2], concat_dim="x", combine="nested"
                ) as actual:
                    assert_identical(actual, original)

    def test_open_and_do_math(self):
        original = Dataset({"foo": ("x", np.random.randn(10))})
        with create_tmp_file() as tmp:
            original.to_netcdf(tmp)
            with open_mfdataset(tmp, combine="by_coords") as ds:
                actual = 1.0 * ds
                assert_allclose(original, actual, decode_bytes=False)

    def test_open_mfdataset_concat_dim_none(self):
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                data = Dataset({"x": 0})
                data.to_netcdf(tmp1)
                Dataset({"x": np.nan}).to_netcdf(tmp2)
                with open_mfdataset(
                    [tmp1, tmp2], concat_dim=None, combine="nested"
                ) as actual:
                    assert_identical(data, actual)

    def test_open_dataset(self):
        original = Dataset({"foo": ("x", np.random.randn(10))})
        with create_tmp_file() as tmp:
            original.to_netcdf(tmp)
            with open_dataset(tmp, chunks={"x": 5}) as actual:
                assert isinstance(actual.foo.variable.data, da.Array)
                assert actual.foo.variable.data.chunks == ((5, 5),)
                assert_identical(original, actual)
            with open_dataset(tmp, chunks=5) as actual:
                assert_identical(original, actual)
            with open_dataset(tmp) as actual:
                assert isinstance(actual.foo.variable.data, np.ndarray)
                assert_identical(original, actual)

    def test_open_single_dataset(self):
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
            with open_mfdataset([tmp], concat_dim=dim, combine="nested") as actual:
                assert_identical(expected, actual)

    def test_open_multi_dataset(self):
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
                [tmp1, tmp2], concat_dim=dim, combine="nested"
            ) as actual:
                assert_identical(expected, actual)

    def test_dask_roundtrip(self):
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

    def test_deterministic_names(self):
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

    def test_dataarray_compute(self):
        # Test DataArray.compute() on dask backend.
        # The test for Dataset.compute() is already in DatasetIOBase;
        # however dask is the only tested backend which supports DataArrays
        actual = DataArray([1, 2]).chunk()
        computed = actual.compute()
        assert not actual._in_memory
        assert computed._in_memory
        assert_allclose(actual, computed, decode_bytes=False)

    def test_save_mfdataset_compute_false_roundtrip(self):
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

    def test_load_dataset(self):
        with create_tmp_file() as tmp:
            original = Dataset({"foo": ("x", np.random.randn(10))})
            original.to_netcdf(tmp)
            ds = load_dataset(tmp)
            # this would fail if we used open_dataset instead of load_dataset
            ds.to_netcdf(tmp)

    def test_load_dataarray(self):
        with create_tmp_file() as tmp:
            original = Dataset({"foo": ("x", np.random.randn(10))})
            original.to_netcdf(tmp)
            ds = load_dataarray(tmp)
            # this would fail if we used open_dataarray instead of
            # load_dataarray
            ds.to_netcdf(tmp)


@requires_scipy_or_netCDF4
@requires_dask
class TestOpenMFDataSetDeprecation:
    """
    Set of tests to check that FutureWarnings are correctly raised until the
    deprecation cycle is complete. #2616
    """

    def test_open_mfdataset_default(self):
        ds1, ds2 = Dataset({"x": [0]}), Dataset({"x": [1]})
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                ds1.to_netcdf(tmp1)
                ds2.to_netcdf(tmp2)

                with pytest.warns(
                    FutureWarning, match="default behaviour of" " `open_mfdataset`"
                ):
                    open_mfdataset([tmp1, tmp2])

    def test_open_mfdataset_with_concat_dim(self):
        ds1, ds2 = Dataset({"x": [0]}), Dataset({"x": [1]})
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                ds1.to_netcdf(tmp1)
                ds2.to_netcdf(tmp2)

                with pytest.warns(FutureWarning, match="`concat_dim`"):
                    open_mfdataset([tmp1, tmp2], concat_dim="x")

    def test_auto_combine_with_merge_and_concat(self):
        ds1, ds2 = Dataset({"x": [0]}), Dataset({"x": [1]})
        ds3 = Dataset({"z": ((), 99)})
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                with create_tmp_file() as tmp3:
                    ds1.to_netcdf(tmp1)
                    ds2.to_netcdf(tmp2)
                    ds3.to_netcdf(tmp3)

                    with pytest.warns(
                        FutureWarning, match="require both concatenation"
                    ):
                        open_mfdataset([tmp1, tmp2, tmp3])

    def test_auto_combine_with_coords(self):
        ds1 = Dataset({"foo": ("x", [0])}, coords={"x": ("x", [0])})
        ds2 = Dataset({"foo": ("x", [1])}, coords={"x": ("x", [1])})
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                ds1.to_netcdf(tmp1)
                ds2.to_netcdf(tmp2)

                with pytest.warns(FutureWarning, match="supplied have global"):
                    open_mfdataset([tmp1, tmp2])

    def test_auto_combine_without_coords(self):
        ds1, ds2 = Dataset({"foo": ("x", [0])}), Dataset({"foo": ("x", [1])})
        with create_tmp_file() as tmp1:
            with create_tmp_file() as tmp2:
                ds1.to_netcdf(tmp1)
                ds2.to_netcdf(tmp2)

                with pytest.warns(FutureWarning, match="supplied do not have global"):
                    open_mfdataset([tmp1, tmp2])


@requires_scipy_or_netCDF4
@requires_pydap
@pytest.mark.filterwarnings("ignore:The binary mode of fromstring is deprecated")
class TestPydap:
    def convert_to_pydap_dataset(self, original):
        from pydap.model import GridType, BaseType, DatasetType

        ds = DatasetType("bears", **original.attrs)
        for key, var in original.data_vars.items():
            v = GridType(key)
            v[key] = BaseType(key, var.values, dimensions=var.dims, **var.attrs)
            for d in var.dims:
                v[d] = BaseType(d, var[d].values)
            ds[key] = v
        # check all dims are stored in ds
        for d in original.coords:
            ds[d] = BaseType(
                d, original[d].values, dimensions=(d,), **original[d].attrs
            )
        return ds

    @contextlib.contextmanager
    def create_datasets(self, **kwargs):
        with open_example_dataset("bears.nc") as expected:
            pydap_ds = self.convert_to_pydap_dataset(expected)
            actual = open_dataset(PydapDataStore(pydap_ds))
            # TODO solve this workaround:
            # netcdf converts string to byte not unicode
            expected["bears"] = expected["bears"].astype(str)
            yield actual, expected

    def test_cmp_local_file(self):
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
            assert_equal(actual.isel(i=0, j=-1), expected.isel(i=0, j=-1))

        with self.create_datasets() as (actual, expected):
            assert_equal(actual.isel(j=slice(1, 2)), expected.isel(j=slice(1, 2)))

        with self.create_datasets() as (actual, expected):
            indexers = {"i": [1, 0, 0], "j": [1, 2, 0, 1]}
            assert_equal(actual.isel(**indexers), expected.isel(**indexers))

        with self.create_datasets() as (actual, expected):
            indexers = {
                "i": DataArray([0, 1, 0], dims="a"),
                "j": DataArray([0, 2, 1], dims="a"),
            }
            assert_equal(actual.isel(**indexers), expected.isel(**indexers))

    def test_compatible_to_netcdf(self):
        # make sure it can be saved as a netcdf
        with self.create_datasets() as (actual, expected):
            with create_tmp_file() as tmp_file:
                actual.to_netcdf(tmp_file)
                with open_dataset(tmp_file) as actual2:
                    actual2["bears"] = actual2["bears"].astype(str)
                    assert_equal(actual2, expected)

    @requires_dask
    def test_dask(self):
        with self.create_datasets(chunks={"j": 2}) as (actual, expected):
            assert_equal(actual, expected)


@network
@requires_scipy_or_netCDF4
@requires_pydap
class TestPydapOnline(TestPydap):
    @contextlib.contextmanager
    def create_datasets(self, **kwargs):
        url = "http://test.opendap.org/opendap/hyrax/data/nc/bears.nc"
        actual = open_dataset(url, engine="pydap", **kwargs)
        with open_example_dataset("bears.nc") as expected:
            # workaround to restore string which is converted to byte
            expected["bears"] = expected["bears"].astype(str)
            yield actual, expected

    def test_session(self):
        from pydap.cas.urs import setup_session

        session = setup_session("XarrayTestUser", "Xarray2017")
        with mock.patch("pydap.client.open_url") as mock_func:
            xr.backends.PydapDataStore.open("http://test.url", session=session)
        mock_func.assert_called_with("http://test.url", session=session)


@requires_scipy
@requires_pynio
class TestPyNio(ScipyWriteBase):
    def test_write_store(self):
        # pynio is read-only for now
        pass

    @contextlib.contextmanager
    def open(self, path, **kwargs):
        with open_dataset(path, engine="pynio", **kwargs) as ds:
            yield ds

    def test_kwargs(self):
        kwargs = {"format": "grib"}
        path = os.path.join(os.path.dirname(__file__), "data", "example")
        with backends.NioDataStore(path, **kwargs) as store:
            assert store._manager._kwargs["format"] == "grib"

    def save(self, dataset, path, **kwargs):
        return dataset.to_netcdf(path, engine="scipy", **kwargs)

    def test_weakrefs(self):
        example = Dataset({"foo": ("x", np.arange(5.0))})
        expected = example.rename({"foo": "bar", "x": "y"})

        with create_tmp_file() as tmp_file:
            example.to_netcdf(tmp_file, engine="scipy")
            on_disk = open_dataset(tmp_file, engine="pynio")
            actual = on_disk.rename({"foo": "bar", "x": "y"})
            del on_disk  # trigger garbage collection
            assert_identical(actual, expected)


@requires_cfgrib
class TestCfGrib:
    def test_read(self):
        expected = {
            "number": 2,
            "time": 3,
            "isobaricInhPa": 2,
            "latitude": 3,
            "longitude": 4,
        }
        with open_example_dataset("example.grib", engine="cfgrib") as ds:
            assert ds.dims == expected
            assert list(ds.data_vars) == ["z", "t"]
            assert ds["z"].min() == 12660.0

    def test_read_filter_by_keys(self):
        kwargs = {"filter_by_keys": {"shortName": "t"}}
        expected = {
            "number": 2,
            "time": 3,
            "isobaricInhPa": 2,
            "latitude": 3,
            "longitude": 4,
        }
        with open_example_dataset(
            "example.grib", engine="cfgrib", backend_kwargs=kwargs
        ) as ds:
            assert ds.dims == expected
            assert list(ds.data_vars) == ["t"]
            assert ds["t"].min() == 231.0


@requires_pseudonetcdf
@pytest.mark.filterwarnings("ignore:IOAPI_ISPH is assumed to be 6370000")
class TestPseudoNetCDFFormat:
    def open(self, path, **kwargs):
        return open_dataset(path, engine="pseudonetcdf", **kwargs)

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

    def test_ict_format(self):
        """
        Open a CAMx file and test data variables
        """
        ictfile = open_example_dataset(
            "example.ict", engine="pseudonetcdf", backend_kwargs={"format": "ffi1001"}
        )
        stdattr = {
            "fill_value": -9999.0,
            "missing_value": -9999,
            "scale": 1,
            "llod_flag": -8888,
            "llod_value": "N/A",
            "ulod_flag": -7777,
            "ulod_value": "N/A",
        }

        def myatts(**attrs):
            outattr = stdattr.copy()
            outattr.update(attrs)
            return outattr

        input = {
            "coords": {},
            "attrs": {
                "fmt": "1001",
                "n_header_lines": 27,
                "PI_NAME": "Henderson, Barron",
                "ORGANIZATION_NAME": "U.S. EPA",
                "SOURCE_DESCRIPTION": "Example file with artificial data",
                "MISSION_NAME": "JUST_A_TEST",
                "VOLUME_INFO": "1, 1",
                "SDATE": "2018, 04, 27",
                "WDATE": "2018, 04, 27",
                "TIME_INTERVAL": "0",
                "INDEPENDENT_VARIABLE": "Start_UTC",
                "ULOD_FLAG": "-7777",
                "ULOD_VALUE": "N/A",
                "LLOD_FLAG": "-8888",
                "LLOD_VALUE": ("N/A, N/A, N/A, N/A, 0.025"),
                "OTHER_COMMENTS": (
                    "www-air.larc.nasa.gov/missions/etc/" + "IcarttDataFormat.htm"
                ),
                "REVISION": "R0",
                "R0": "No comments for this revision.",
                "TFLAG": "Start_UTC",
            },
            "dims": {"POINTS": 4},
            "data_vars": {
                "Start_UTC": {
                    "data": [43200.0, 46800.0, 50400.0, 50400.0],
                    "dims": ("POINTS",),
                    "attrs": myatts(units="Start_UTC", standard_name="Start_UTC"),
                },
                "lat": {
                    "data": [41.0, 42.0, 42.0, 42.0],
                    "dims": ("POINTS",),
                    "attrs": myatts(units="degrees_north", standard_name="lat"),
                },
                "lon": {
                    "data": [-71.0, -72.0, -73.0, -74.0],
                    "dims": ("POINTS",),
                    "attrs": myatts(units="degrees_east", standard_name="lon"),
                },
                "elev": {
                    "data": [5.0, 15.0, 20.0, 25.0],
                    "dims": ("POINTS",),
                    "attrs": myatts(units="meters", standard_name="elev"),
                },
                "TEST_ppbv": {
                    "data": [1.2345, 2.3456, 3.4567, 4.5678],
                    "dims": ("POINTS",),
                    "attrs": myatts(units="ppbv", standard_name="TEST_ppbv"),
                },
                "TESTM_ppbv": {
                    "data": [2.22, -9999.0, -7777.0, -8888.0],
                    "dims": ("POINTS",),
                    "attrs": myatts(
                        units="ppbv", standard_name="TESTM_ppbv", llod_value=0.025
                    ),
                },
            },
        }
        chkfile = Dataset.from_dict(input)
        assert_identical(ictfile, chkfile)

    def test_ict_format_write(self):
        fmtkw = {"format": "ffi1001"}
        expected = open_example_dataset(
            "example.ict", engine="pseudonetcdf", backend_kwargs=fmtkw
        )
        with self.roundtrip(
            expected, save_kwargs=fmtkw, open_kwargs={"backend_kwargs": fmtkw}
        ) as actual:
            assert_identical(expected, actual)

    def test_uamiv_format_read(self):
        """
        Open a CAMx file and test data variables
        """

        camxfile = open_example_dataset(
            "example.uamiv", engine="pseudonetcdf", backend_kwargs={"format": "uamiv"}
        )
        data = np.arange(20, dtype="f").reshape(1, 1, 4, 5)
        expected = xr.Variable(
            ("TSTEP", "LAY", "ROW", "COL"),
            data,
            dict(units="ppm", long_name="O3".ljust(16), var_desc="O3".ljust(80)),
        )
        actual = camxfile.variables["O3"]
        assert_allclose(expected, actual)

        data = np.array([[[2002154, 0]]], dtype="i")
        expected = xr.Variable(
            ("TSTEP", "VAR", "DATE-TIME"),
            data,
            dict(
                long_name="TFLAG".ljust(16),
                var_desc="TFLAG".ljust(80),
                units="DATE-TIME".ljust(16),
            ),
        )
        actual = camxfile.variables["TFLAG"]
        assert_allclose(expected, actual)
        camxfile.close()

    def test_uamiv_format_mfread(self):
        """
        Open a CAMx file and test data variables
        """

        camxfile = open_example_mfdataset(
            ["example.uamiv", "example.uamiv"],
            engine="pseudonetcdf",
            concat_dim="TSTEP",
            backend_kwargs={"format": "uamiv"},
        )

        data1 = np.arange(20, dtype="f").reshape(1, 1, 4, 5)
        data = np.concatenate([data1] * 2, axis=0)
        expected = xr.Variable(
            ("TSTEP", "LAY", "ROW", "COL"),
            data,
            dict(units="ppm", long_name="O3".ljust(16), var_desc="O3".ljust(80)),
        )
        actual = camxfile.variables["O3"]
        assert_allclose(expected, actual)

        data = np.array([[[2002154, 0]]], dtype="i").repeat(2, 0)
        attrs = dict(
            long_name="TFLAG".ljust(16),
            var_desc="TFLAG".ljust(80),
            units="DATE-TIME".ljust(16),
        )
        dims = ("TSTEP", "VAR", "DATE-TIME")
        expected = xr.Variable(dims, data, attrs)
        actual = camxfile.variables["TFLAG"]
        assert_allclose(expected, actual)
        camxfile.close()

    def test_uamiv_format_write(self):
        fmtkw = {"format": "uamiv"}

        expected = open_example_dataset(
            "example.uamiv", engine="pseudonetcdf", backend_kwargs=fmtkw
        )
        with self.roundtrip(
            expected,
            save_kwargs=fmtkw,
            open_kwargs={"backend_kwargs": fmtkw},
            allow_cleanup_failure=True,
        ) as actual:
            assert_identical(expected, actual)

        expected.close()

    def save(self, dataset, path, **save_kwargs):
        import PseudoNetCDF as pnc

        pncf = pnc.PseudoNetCDFFile()
        pncf.dimensions = {
            k: pnc.PseudoNetCDFDimension(pncf, k, v) for k, v in dataset.dims.items()
        }
        pncf.variables = {
            k: pnc.PseudoNetCDFVariable(
                pncf, k, v.dtype.char, v.dims, values=v.data[...], **v.attrs
            )
            for k, v in dataset.variables.items()
        }
        for pk, pv in dataset.attrs.items():
            setattr(pncf, pk, pv)

        pnc.pncwrite(pncf, path, **save_kwargs)


@requires_rasterio
@contextlib.contextmanager
def create_tmp_geotiff(
    nx=4,
    ny=3,
    nz=3,
    transform=None,
    transform_args=[5000, 80000, 1000, 2000.0],
    crs={"units": "m", "no_defs": True, "ellps": "WGS84", "proj": "utm", "zone": 18},
    open_kwargs=None,
    additional_attrs=None,
):
    # yields a temporary geotiff file and a corresponding expected DataArray
    import rasterio
    from rasterio.transform import from_origin

    if open_kwargs is None:
        open_kwargs = {}

    with create_tmp_file(suffix=".tif", allow_cleanup_failure=ON_WINDOWS) as tmp_file:
        # allow 2d or 3d shapes
        if nz == 1:
            data_shape = ny, nx
            write_kwargs = {"indexes": 1}
        else:
            data_shape = nz, ny, nx
            write_kwargs = {}
        data = np.arange(nz * ny * nx, dtype=rasterio.float32).reshape(*data_shape)
        if transform is None:
            transform = from_origin(*transform_args)
        if additional_attrs is None:
            additional_attrs = {
                "descriptions": tuple("d{}".format(n + 1) for n in range(nz)),
                "units": tuple("u{}".format(n + 1) for n in range(nz)),
            }
        with rasterio.open(
            tmp_file,
            "w",
            driver="GTiff",
            height=ny,
            width=nx,
            count=nz,
            crs=crs,
            transform=transform,
            dtype=rasterio.float32,
            **open_kwargs,
        ) as s:
            for attr, val in additional_attrs.items():
                setattr(s, attr, val)
            s.write(data, **write_kwargs)
            dx, dy = s.res[0], -s.res[1]

        a, b, c, d = transform_args
        data = data[np.newaxis, ...] if nz == 1 else data
        expected = DataArray(
            data,
            dims=("band", "y", "x"),
            coords={
                "band": np.arange(nz) + 1,
                "y": -np.arange(ny) * d + b + dy / 2,
                "x": np.arange(nx) * c + a + dx / 2,
            },
        )
        yield tmp_file, expected


@requires_rasterio
class TestRasterio:
    @requires_scipy_or_netCDF4
    def test_serialization(self):
        with create_tmp_geotiff(additional_attrs={}) as (tmp_file, expected):
            # Write it to a netcdf and read again (roundtrip)
            with xr.open_rasterio(tmp_file) as rioda:
                with create_tmp_file(suffix=".nc") as tmp_nc_file:
                    rioda.to_netcdf(tmp_nc_file)
                    with xr.open_dataarray(tmp_nc_file) as ncds:
                        assert_identical(rioda, ncds)

    def test_utm(self):
        with create_tmp_geotiff() as (tmp_file, expected):
            with xr.open_rasterio(tmp_file) as rioda:
                assert_allclose(rioda, expected)
                assert rioda.attrs["scales"] == (1.0, 1.0, 1.0)
                assert rioda.attrs["offsets"] == (0.0, 0.0, 0.0)
                assert rioda.attrs["descriptions"] == ("d1", "d2", "d3")
                assert rioda.attrs["units"] == ("u1", "u2", "u3")
                assert isinstance(rioda.attrs["crs"], str)
                assert isinstance(rioda.attrs["res"], tuple)
                assert isinstance(rioda.attrs["is_tiled"], np.uint8)
                assert isinstance(rioda.attrs["transform"], tuple)
                assert len(rioda.attrs["transform"]) == 6
                np.testing.assert_array_equal(
                    rioda.attrs["nodatavals"], [np.NaN, np.NaN, np.NaN]
                )

            # Check no parse coords
            with xr.open_rasterio(tmp_file, parse_coordinates=False) as rioda:
                assert "x" not in rioda.coords
                assert "y" not in rioda.coords

    def test_non_rectilinear(self):
        from rasterio.transform import from_origin

        # Create a geotiff file with 2d coordinates
        with create_tmp_geotiff(
            transform=from_origin(0, 3, 1, 1).rotation(45), crs=None
        ) as (tmp_file, _):
            # Default is to not parse coords
            with xr.open_rasterio(tmp_file) as rioda:
                assert "x" not in rioda.coords
                assert "y" not in rioda.coords
                assert "crs" not in rioda.attrs
                assert rioda.attrs["scales"] == (1.0, 1.0, 1.0)
                assert rioda.attrs["offsets"] == (0.0, 0.0, 0.0)
                assert rioda.attrs["descriptions"] == ("d1", "d2", "d3")
                assert rioda.attrs["units"] == ("u1", "u2", "u3")
                assert isinstance(rioda.attrs["res"], tuple)
                assert isinstance(rioda.attrs["is_tiled"], np.uint8)
                assert isinstance(rioda.attrs["transform"], tuple)
                assert len(rioda.attrs["transform"]) == 6

            # See if a warning is raised if we force it
            with pytest.warns(Warning, match="transformation isn't rectilinear"):
                with xr.open_rasterio(tmp_file, parse_coordinates=True) as rioda:
                    assert "x" not in rioda.coords
                    assert "y" not in rioda.coords

    def test_platecarree(self):
        with create_tmp_geotiff(
            8,
            10,
            1,
            transform_args=[1, 2, 0.5, 2.0],
            crs="+proj=latlong",
            open_kwargs={"nodata": -9765},
        ) as (tmp_file, expected):
            with xr.open_rasterio(tmp_file) as rioda:
                assert_allclose(rioda, expected)
                assert rioda.attrs["scales"] == (1.0,)
                assert rioda.attrs["offsets"] == (0.0,)
                assert isinstance(rioda.attrs["descriptions"], tuple)
                assert isinstance(rioda.attrs["units"], tuple)
                assert isinstance(rioda.attrs["crs"], str)
                assert isinstance(rioda.attrs["res"], tuple)
                assert isinstance(rioda.attrs["is_tiled"], np.uint8)
                assert isinstance(rioda.attrs["transform"], tuple)
                assert len(rioda.attrs["transform"]) == 6
                np.testing.assert_array_equal(rioda.attrs["nodatavals"], [-9765.0])

    def test_notransform(self):
        # regression test for https://github.com/pydata/xarray/issues/1686
        import rasterio
        import warnings

        # Create a geotiff file
        with warnings.catch_warnings():
            # rasterio throws a NotGeoreferencedWarning here, which is
            # expected since we test rasterio's defaults in this case.
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="Dataset has no geotransform set",
            )
            with create_tmp_file(suffix=".tif") as tmp_file:
                # data
                nx, ny, nz = 4, 3, 3
                data = np.arange(nx * ny * nz, dtype=rasterio.float32).reshape(
                    nz, ny, nx
                )
                with rasterio.open(
                    tmp_file,
                    "w",
                    driver="GTiff",
                    height=ny,
                    width=nx,
                    count=nz,
                    dtype=rasterio.float32,
                ) as s:
                    s.descriptions = ("nx", "ny", "nz")
                    s.units = ("cm", "m", "km")
                    s.write(data)

                # Tests
                expected = DataArray(
                    data,
                    dims=("band", "y", "x"),
                    coords={
                        "band": [1, 2, 3],
                        "y": [0.5, 1.5, 2.5],
                        "x": [0.5, 1.5, 2.5, 3.5],
                    },
                )
                with xr.open_rasterio(tmp_file) as rioda:
                    assert_allclose(rioda, expected)
                    assert rioda.attrs["scales"] == (1.0, 1.0, 1.0)
                    assert rioda.attrs["offsets"] == (0.0, 0.0, 0.0)
                    assert rioda.attrs["descriptions"] == ("nx", "ny", "nz")
                    assert rioda.attrs["units"] == ("cm", "m", "km")
                    assert isinstance(rioda.attrs["res"], tuple)
                    assert isinstance(rioda.attrs["is_tiled"], np.uint8)
                    assert isinstance(rioda.attrs["transform"], tuple)
                    assert len(rioda.attrs["transform"]) == 6

    def test_indexing(self):
        with create_tmp_geotiff(
            8, 10, 3, transform_args=[1, 2, 0.5, 2.0], crs="+proj=latlong"
        ) as (tmp_file, expected):
            with xr.open_rasterio(tmp_file, cache=False) as actual:

                # tests
                # assert_allclose checks all data + coordinates
                assert_allclose(actual, expected)
                assert not actual.variable._in_memory

                # Basic indexer
                ind = {"x": slice(2, 5), "y": slice(5, 7)}
                assert_allclose(expected.isel(**ind), actual.isel(**ind))
                assert not actual.variable._in_memory

                ind = {"band": slice(1, 2), "x": slice(2, 5), "y": slice(5, 7)}
                assert_allclose(expected.isel(**ind), actual.isel(**ind))
                assert not actual.variable._in_memory

                ind = {"band": slice(1, 2), "x": slice(2, 5), "y": 0}
                assert_allclose(expected.isel(**ind), actual.isel(**ind))
                assert not actual.variable._in_memory

                # orthogonal indexer
                ind = {
                    "band": np.array([2, 1, 0]),
                    "x": np.array([1, 0]),
                    "y": np.array([0, 2]),
                }
                assert_allclose(expected.isel(**ind), actual.isel(**ind))
                assert not actual.variable._in_memory

                ind = {"band": np.array([2, 1, 0]), "x": np.array([1, 0]), "y": 0}
                assert_allclose(expected.isel(**ind), actual.isel(**ind))
                assert not actual.variable._in_memory

                ind = {"band": 0, "x": np.array([0, 0]), "y": np.array([1, 1, 1])}
                assert_allclose(expected.isel(**ind), actual.isel(**ind))
                assert not actual.variable._in_memory

                # minus-stepped slice
                ind = {"band": np.array([2, 1, 0]), "x": slice(-1, None, -1), "y": 0}
                assert_allclose(expected.isel(**ind), actual.isel(**ind))
                assert not actual.variable._in_memory

                ind = {"band": np.array([2, 1, 0]), "x": 1, "y": slice(-1, 1, -2)}
                assert_allclose(expected.isel(**ind), actual.isel(**ind))
                assert not actual.variable._in_memory

                # empty selection
                ind = {"band": np.array([2, 1, 0]), "x": 1, "y": slice(2, 2, 1)}
                assert_allclose(expected.isel(**ind), actual.isel(**ind))
                assert not actual.variable._in_memory

                ind = {"band": slice(0, 0), "x": 1, "y": 2}
                assert_allclose(expected.isel(**ind), actual.isel(**ind))
                assert not actual.variable._in_memory

                # vectorized indexer
                ind = {
                    "band": DataArray([2, 1, 0], dims="a"),
                    "x": DataArray([1, 0, 0], dims="a"),
                    "y": np.array([0, 2]),
                }
                assert_allclose(expected.isel(**ind), actual.isel(**ind))
                assert not actual.variable._in_memory

                ind = {
                    "band": DataArray([[2, 1, 0], [1, 0, 2]], dims=["a", "b"]),
                    "x": DataArray([[1, 0, 0], [0, 1, 0]], dims=["a", "b"]),
                    "y": 0,
                }
                assert_allclose(expected.isel(**ind), actual.isel(**ind))
                assert not actual.variable._in_memory

                # Selecting lists of bands is fine
                ex = expected.isel(band=[1, 2])
                ac = actual.isel(band=[1, 2])
                assert_allclose(ac, ex)
                ex = expected.isel(band=[0, 2])
                ac = actual.isel(band=[0, 2])
                assert_allclose(ac, ex)

                # Integer indexing
                ex = expected.isel(band=1)
                ac = actual.isel(band=1)
                assert_allclose(ac, ex)

                ex = expected.isel(x=1, y=2)
                ac = actual.isel(x=1, y=2)
                assert_allclose(ac, ex)

                ex = expected.isel(band=0, x=1, y=2)
                ac = actual.isel(band=0, x=1, y=2)
                assert_allclose(ac, ex)

                # Mixed
                ex = actual.isel(x=slice(2), y=slice(2))
                ac = actual.isel(x=[0, 1], y=[0, 1])
                assert_allclose(ac, ex)

                ex = expected.isel(band=0, x=1, y=slice(5, 7))
                ac = actual.isel(band=0, x=1, y=slice(5, 7))
                assert_allclose(ac, ex)

                ex = expected.isel(band=0, x=slice(2, 5), y=2)
                ac = actual.isel(band=0, x=slice(2, 5), y=2)
                assert_allclose(ac, ex)

                # One-element lists
                ex = expected.isel(band=[0], x=slice(2, 5), y=[2])
                ac = actual.isel(band=[0], x=slice(2, 5), y=[2])
                assert_allclose(ac, ex)

    def test_caching(self):
        with create_tmp_geotiff(
            8, 10, 3, transform_args=[1, 2, 0.5, 2.0], crs="+proj=latlong"
        ) as (tmp_file, expected):
            # Cache is the default
            with xr.open_rasterio(tmp_file) as actual:

                # This should cache everything
                assert_allclose(actual, expected)

                # once cached, non-windowed indexing should become possible
                ac = actual.isel(x=[2, 4])
                ex = expected.isel(x=[2, 4])
                assert_allclose(ac, ex)

    @requires_dask
    def test_chunks(self):
        with create_tmp_geotiff(
            8, 10, 3, transform_args=[1, 2, 0.5, 2.0], crs="+proj=latlong"
        ) as (tmp_file, expected):
            # Chunk at open time
            with xr.open_rasterio(tmp_file, chunks=(1, 2, 2)) as actual:

                import dask.array as da

                assert isinstance(actual.data, da.Array)
                assert "open_rasterio" in actual.data.name

                # do some arithmetic
                ac = actual.mean()
                ex = expected.mean()
                assert_allclose(ac, ex)

                ac = actual.sel(band=1).mean(dim="x")
                ex = expected.sel(band=1).mean(dim="x")
                assert_allclose(ac, ex)

    def test_pickle_rasterio(self):
        # regression test for https://github.com/pydata/xarray/issues/2121
        with create_tmp_geotiff() as (tmp_file, expected):
            with xr.open_rasterio(tmp_file) as rioda:
                temp = pickle.dumps(rioda)
                with pickle.loads(temp) as actual:
                    assert_equal(actual, rioda)

    def test_ENVI_tags(self):
        rasterio = pytest.importorskip("rasterio", minversion="1.0a")
        from rasterio.transform import from_origin

        # Create an ENVI file with some tags in the ENVI namespace
        # this test uses a custom driver, so we can't use create_tmp_geotiff
        with create_tmp_file(suffix=".dat") as tmp_file:
            # data
            nx, ny, nz = 4, 3, 3
            data = np.arange(nx * ny * nz, dtype=rasterio.float32).reshape(nz, ny, nx)
            transform = from_origin(5000, 80000, 1000, 2000.0)
            with rasterio.open(
                tmp_file,
                "w",
                driver="ENVI",
                height=ny,
                width=nx,
                count=nz,
                crs={
                    "units": "m",
                    "no_defs": True,
                    "ellps": "WGS84",
                    "proj": "utm",
                    "zone": 18,
                },
                transform=transform,
                dtype=rasterio.float32,
            ) as s:
                s.update_tags(
                    ns="ENVI",
                    description="{Tagged file}",
                    wavelength="{123.000000, 234.234000, 345.345678}",
                    fwhm="{1.000000, 0.234000, 0.000345}",
                )
                s.write(data)
                dx, dy = s.res[0], -s.res[1]

            # Tests
            coords = {
                "band": [1, 2, 3],
                "y": -np.arange(ny) * 2000 + 80000 + dy / 2,
                "x": np.arange(nx) * 1000 + 5000 + dx / 2,
                "wavelength": ("band", np.array([123, 234.234, 345.345678])),
                "fwhm": ("band", np.array([1, 0.234, 0.000345])),
            }
            expected = DataArray(data, dims=("band", "y", "x"), coords=coords)

            with xr.open_rasterio(tmp_file) as rioda:
                assert_allclose(rioda, expected)
                assert isinstance(rioda.attrs["crs"], str)
                assert isinstance(rioda.attrs["res"], tuple)
                assert isinstance(rioda.attrs["is_tiled"], np.uint8)
                assert isinstance(rioda.attrs["transform"], tuple)
                assert len(rioda.attrs["transform"]) == 6
                # from ENVI tags
                assert isinstance(rioda.attrs["description"], str)
                assert isinstance(rioda.attrs["map_info"], str)
                assert isinstance(rioda.attrs["samples"], str)

    def test_geotiff_tags(self):
        # Create a geotiff file with some tags
        with create_tmp_geotiff() as (tmp_file, _):
            with xr.open_rasterio(tmp_file) as rioda:
                assert isinstance(rioda.attrs["AREA_OR_POINT"], str)

    def test_no_mftime(self):
        # rasterio can accept "filename" urguments that are actually urls,
        # including paths to remote files.
        # In issue #1816, we found that these caused dask to break, because
        # the modification time was used to determine the dask token. This
        # tests ensure we can still chunk such files when reading with
        # rasterio.
        with create_tmp_geotiff(
            8, 10, 3, transform_args=[1, 2, 0.5, 2.0], crs="+proj=latlong"
        ) as (tmp_file, expected):
            with mock.patch("os.path.getmtime", side_effect=OSError):
                with xr.open_rasterio(tmp_file, chunks=(1, 2, 2)) as actual:
                    import dask.array as da

                    assert isinstance(actual.data, da.Array)
                    assert_allclose(actual, expected)

    @network
    def test_http_url(self):
        # more examples urls here
        # http://download.osgeo.org/geotiff/samples/
        url = "http://download.osgeo.org/geotiff/samples/made_up/ntf_nord.tif"
        with xr.open_rasterio(url) as actual:
            assert actual.shape == (1, 512, 512)
        # make sure chunking works
        with xr.open_rasterio(url, chunks=(1, 256, 256)) as actual:
            import dask.array as da

            assert isinstance(actual.data, da.Array)

    def test_rasterio_environment(self):
        import rasterio

        with create_tmp_geotiff() as (tmp_file, expected):
            # Should fail with error since suffix not allowed
            with pytest.raises(Exception):
                with rasterio.Env(GDAL_SKIP="GTiff"):
                    with xr.open_rasterio(tmp_file) as actual:
                        assert_allclose(actual, expected)

    @pytest.mark.xfail(reason="rasterio 1.1.1 is broken. GH3573")
    def test_rasterio_vrt(self):
        import rasterio

        # tmp_file default crs is UTM: CRS({'init': 'epsg:32618'}
        with create_tmp_geotiff() as (tmp_file, expected):
            with rasterio.open(tmp_file) as src:
                with rasterio.vrt.WarpedVRT(src, crs="epsg:4326") as vrt:
                    expected_shape = (vrt.width, vrt.height)
                    expected_crs = vrt.crs
                    expected_res = vrt.res
                    # Value of single pixel in center of image
                    lon, lat = vrt.xy(vrt.width // 2, vrt.height // 2)
                    expected_val = next(vrt.sample([(lon, lat)]))
                    with xr.open_rasterio(vrt) as da:
                        actual_shape = (da.sizes["x"], da.sizes["y"])
                        actual_crs = da.crs
                        actual_res = da.res
                        actual_val = da.sel(dict(x=lon, y=lat), method="nearest").data

                        assert actual_crs == expected_crs
                        assert actual_res == expected_res
                        assert actual_shape == expected_shape
                        assert expected_val.all() == actual_val.all()

    def test_rasterio_vrt_with_transform_and_size(self):
        # Test open_rasterio() support of WarpedVRT with transform, width and
        # height (issue #2864)
        import rasterio
        from rasterio.warp import calculate_default_transform
        from affine import Affine

        with create_tmp_geotiff() as (tmp_file, expected):
            with rasterio.open(tmp_file) as src:
                # Estimate the transform, width and height
                # for a change of resolution
                # tmp_file initial res is (1000,2000) (default values)
                trans, w, h = calculate_default_transform(
                    src.crs, src.crs, src.width, src.height, resolution=500, *src.bounds
                )
                with rasterio.vrt.WarpedVRT(
                    src, transform=trans, width=w, height=h
                ) as vrt:
                    expected_shape = (vrt.width, vrt.height)
                    expected_res = vrt.res
                    expected_transform = vrt.transform
                    with xr.open_rasterio(vrt) as da:
                        actual_shape = (da.sizes["x"], da.sizes["y"])
                        actual_res = da.res
                        actual_transform = Affine(*da.transform)
                        assert actual_res == expected_res
                        assert actual_shape == expected_shape
                        assert actual_transform == expected_transform

    @network
    def test_rasterio_vrt_network(self):
        # Make sure loading w/ rasterio give same results as xarray
        import rasterio

        # use same url that rasterio package uses in tests
        prefix = "https://landsat-pds.s3.amazonaws.com/L8/139/045/"
        image = "LC81390452014295LGN00/LC81390452014295LGN00_B1.TIF"
        httpstif = prefix + image
        with rasterio.Env(aws_unsigned=True):
            with rasterio.open(httpstif) as src:
                with rasterio.vrt.WarpedVRT(src, crs="epsg:4326") as vrt:
                    expected_shape = vrt.width, vrt.height
                    expected_res = vrt.res
                    # Value of single pixel in center of image
                    lon, lat = vrt.xy(vrt.width // 2, vrt.height // 2)
                    expected_val = next(vrt.sample([(lon, lat)]))
                    with xr.open_rasterio(vrt) as da:
                        actual_shape = da.sizes["x"], da.sizes["y"]
                        actual_res = da.res
                        actual_val = da.sel(dict(x=lon, y=lat), method="nearest").data

                        assert actual_shape == expected_shape
                        assert actual_res == expected_res
                        assert expected_val == actual_val


class TestEncodingInvalid:
    def test_extract_nc4_variable_encoding(self):
        var = xr.Variable(("x",), [1, 2, 3], {}, {"foo": "bar"})
        with raises_regex(ValueError, "unexpected encoding"):
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

    def test_extract_h5nc_encoding(self):
        # not supported with h5netcdf (yet)
        var = xr.Variable(("x",), [1, 2, 3], {}, {"least_sigificant_digit": 2})
        with raises_regex(ValueError, "unexpected encoding"):
            _extract_nc4_variable_encoding(var, raise_on_invalid=True)


class MiscObject:
    pass


@requires_netCDF4
class TestValidateAttrs:
    def test_validating_attrs(self):
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
            with raises_regex(TypeError, "Invalid name for attr"):
                ds.to_netcdf("test.nc")

            ds, attrs = new_dataset_and_attrs()
            attrs[MiscObject()] = "test"
            with raises_regex(TypeError, "Invalid name for attr"):
                ds.to_netcdf("test.nc")

            ds, attrs = new_dataset_and_attrs()
            attrs[""] = "test"
            with raises_regex(ValueError, "Invalid name for attr"):
                ds.to_netcdf("test.nc")

            # This one should work
            ds, attrs = new_dataset_and_attrs()
            attrs["test"] = "test"
            with create_tmp_file() as tmp_file:
                ds.to_netcdf(tmp_file)

            ds, attrs = new_dataset_and_attrs()
            attrs["test"] = {"a": 5}
            with raises_regex(TypeError, "Invalid value for attr"):
                ds.to_netcdf("test.nc")

            ds, attrs = new_dataset_and_attrs()
            attrs["test"] = MiscObject()
            with raises_regex(TypeError, "Invalid value for attr"):
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
    def test_dataarray_to_netcdf_no_name(self):
        original_da = DataArray(np.arange(12).reshape((3, 4)))

        with create_tmp_file() as tmp:
            original_da.to_netcdf(tmp)

            with open_dataarray(tmp) as loaded_da:
                assert_identical(original_da, loaded_da)

    def test_dataarray_to_netcdf_with_name(self):
        original_da = DataArray(np.arange(12).reshape((3, 4)), name="test")

        with create_tmp_file() as tmp:
            original_da.to_netcdf(tmp)

            with open_dataarray(tmp) as loaded_da:
                assert_identical(original_da, loaded_da)

    def test_dataarray_to_netcdf_coord_name_clash(self):
        original_da = DataArray(
            np.arange(12).reshape((3, 4)), dims=["x", "y"], name="x"
        )

        with create_tmp_file() as tmp:
            original_da.to_netcdf(tmp)

            with open_dataarray(tmp) as loaded_da:
                assert_identical(original_da, loaded_da)

    def test_open_dataarray_options(self):
        data = DataArray(np.arange(5), coords={"y": ("x", range(5))}, dims=["x"])

        with create_tmp_file() as tmp:
            data.to_netcdf(tmp)

            expected = data.drop_vars("y")
            with open_dataarray(tmp, drop_variables=["y"]) as loaded:
                assert_identical(expected, loaded)

    @requires_scipy
    def test_dataarray_to_netcdf_return_bytes(self):
        # regression test for GH1410
        data = xr.DataArray([1, 2, 3])
        output = data.to_netcdf()
        assert isinstance(output, bytes)

    def test_dataarray_to_netcdf_no_name_pathlib(self):
        original_da = DataArray(np.arange(12).reshape((3, 4)))

        with create_tmp_file() as tmp:
            tmp = Path(tmp)
            original_da.to_netcdf(tmp)

            with open_dataarray(tmp) as loaded_da:
                assert_identical(original_da, loaded_da)


@requires_scipy_or_netCDF4
def test_no_warning_from_dask_effective_get():
    with create_tmp_file() as tmpfile:
        with pytest.warns(None) as record:
            ds = Dataset()
            ds.to_netcdf(tmpfile)
        assert len(record) == 0


@requires_scipy_or_netCDF4
def test_source_encoding_always_present():
    # Test for GH issue #2550.
    rnddata = np.random.randn(10)
    original = Dataset({"foo": ("x", rnddata)})
    with create_tmp_file() as tmp:
        original.to_netcdf(tmp)
        with open_dataset(tmp) as ds:
            assert ds.encoding["source"] == tmp


@requires_scipy_or_netCDF4
@pytest.mark.parametrize("calendar", _STANDARD_CALENDARS)
def test_use_cftime_standard_calendar_default_in_range(calendar):
    x = [0, 1]
    time = [0, 720]
    units_date = "2000-01-01"
    units = "days since 2000-01-01"
    original = DataArray(x, [("time", time)], name="x")
    original = original.to_dataset()
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
        with pytest.warns(None) as record:
            with open_dataset(tmp_file) as ds:
                assert_identical(expected_x, ds.x)
                assert_identical(expected_time, ds.time)
            assert not record


@requires_cftime
@requires_scipy_or_netCDF4
@pytest.mark.parametrize("calendar", _STANDARD_CALENDARS)
@pytest.mark.parametrize("units_year", [1500, 2500])
def test_use_cftime_standard_calendar_default_out_of_range(calendar, units_year):
    import cftime

    x = [0, 1]
    time = [0, 720]
    units = f"days since {units_year}-01-01"
    original = DataArray(x, [("time", time)], name="x")
    original = original.to_dataset()
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
        with pytest.warns(SerializationWarning):
            with open_dataset(tmp_file) as ds:
                assert_identical(expected_x, ds.x)
                assert_identical(expected_time, ds.time)


@requires_cftime
@requires_scipy_or_netCDF4
@pytest.mark.parametrize("calendar", _ALL_CALENDARS)
@pytest.mark.parametrize("units_year", [1500, 2000, 2500])
def test_use_cftime_true(calendar, units_year):
    import cftime

    x = [0, 1]
    time = [0, 720]
    units = f"days since {units_year}-01-01"
    original = DataArray(x, [("time", time)], name="x")
    original = original.to_dataset()
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
        with pytest.warns(None) as record:
            with open_dataset(tmp_file, use_cftime=True) as ds:
                assert_identical(expected_x, ds.x)
                assert_identical(expected_time, ds.time)
            assert not record


@requires_scipy_or_netCDF4
@pytest.mark.parametrize("calendar", _STANDARD_CALENDARS)
def test_use_cftime_false_standard_calendar_in_range(calendar):
    x = [0, 1]
    time = [0, 720]
    units_date = "2000-01-01"
    units = "days since 2000-01-01"
    original = DataArray(x, [("time", time)], name="x")
    original = original.to_dataset()
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
        with pytest.warns(None) as record:
            with open_dataset(tmp_file, use_cftime=False) as ds:
                assert_identical(expected_x, ds.x)
                assert_identical(expected_time, ds.time)
            assert not record


@requires_scipy_or_netCDF4
@pytest.mark.parametrize("calendar", _STANDARD_CALENDARS)
@pytest.mark.parametrize("units_year", [1500, 2500])
def test_use_cftime_false_standard_calendar_out_of_range(calendar, units_year):
    x = [0, 1]
    time = [0, 720]
    units = f"days since {units_year}-01-01"
    original = DataArray(x, [("time", time)], name="x")
    original = original.to_dataset()
    for v in ["x", "time"]:
        original[v].attrs["units"] = units
        original[v].attrs["calendar"] = calendar

    with create_tmp_file() as tmp_file:
        original.to_netcdf(tmp_file)
        with pytest.raises((OutOfBoundsDatetime, ValueError)):
            open_dataset(tmp_file, use_cftime=False)


@requires_scipy_or_netCDF4
@pytest.mark.parametrize("calendar", _NON_STANDARD_CALENDARS)
@pytest.mark.parametrize("units_year", [1500, 2000, 2500])
def test_use_cftime_false_nonstandard_calendar(calendar, units_year):
    x = [0, 1]
    time = [0, 720]
    units = f"days since {units_year}"
    original = DataArray(x, [("time", time)], name="x")
    original = original.to_dataset()
    for v in ["x", "time"]:
        original[v].attrs["units"] = units
        original[v].attrs["calendar"] = calendar

    with create_tmp_file() as tmp_file:
        original.to_netcdf(tmp_file)
        with pytest.raises((OutOfBoundsDatetime, ValueError)):
            open_dataset(tmp_file, use_cftime=False)


@pytest.mark.parametrize("engine", ["netcdf4", "scipy"])
def test_invalid_netcdf_raises(engine):
    data = create_test_data()
    with raises_regex(ValueError, "unrecognized option 'invalid_netcdf'"):
        data.to_netcdf("foo.nc", engine=engine, invalid_netcdf=True)
