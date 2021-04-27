from distutils.version import LooseVersion

import numpy as np
import pandas as pd
import pytest

import xarray as xr

from . import (
    assert_array_equal,
    assert_chunks_equal,
    assert_equal,
    assert_identical,
    raise_if_dask_computes,
    requires_cftime,
    requires_dask,
)


class TestDatetimeAccessor:
    @pytest.fixture(autouse=True)
    def setup(self):
        nt = 100
        data = np.random.rand(10, 10, nt)
        lons = np.linspace(0, 11, 10)
        lats = np.linspace(0, 20, 10)
        self.times = pd.date_range(start="2000/01/01", freq="H", periods=nt)

        self.data = xr.DataArray(
            data,
            coords=[lons, lats, self.times],
            dims=["lon", "lat", "time"],
            name="data",
        )

        self.times_arr = np.random.choice(self.times, size=(10, 10, nt))
        self.times_data = xr.DataArray(
            self.times_arr,
            coords=[lons, lats, self.times],
            dims=["lon", "lat", "time"],
            name="data",
        )

    @pytest.mark.parametrize(
        "field",
        [
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "microsecond",
            "nanosecond",
            "week",
            "weekofyear",
            "dayofweek",
            "weekday",
            "dayofyear",
            "quarter",
            "date",
            "time",
            "is_month_start",
            "is_month_end",
            "is_quarter_start",
            "is_quarter_end",
            "is_year_start",
            "is_year_end",
            "is_leap_year",
        ],
    )
    def test_field_access(self, field):

        if LooseVersion(pd.__version__) >= "1.1.0" and field in ["week", "weekofyear"]:
            data = self.times.isocalendar()["week"]
        else:
            data = getattr(self.times, field)

        expected = xr.DataArray(data, name=field, coords=[self.times], dims=["time"])

        if field in ["week", "weekofyear"]:
            with pytest.warns(
                FutureWarning, match="dt.weekofyear and dt.week have been deprecated"
            ):
                actual = getattr(self.data.time.dt, field)
        else:
            actual = getattr(self.data.time.dt, field)

        assert_equal(expected, actual)

    @pytest.mark.parametrize(
        "field, pandas_field",
        [
            ("year", "year"),
            ("week", "week"),
            ("weekday", "day"),
        ],
    )
    def test_isocalendar(self, field, pandas_field):

        if LooseVersion(pd.__version__) < "1.1.0":
            with pytest.raises(
                AttributeError, match=r"'isocalendar' not available in pandas < 1.1.0"
            ):
                self.data.time.dt.isocalendar()[field]
            return

        # pandas isocalendar has dtypy UInt32Dtype, convert to Int64
        expected = pd.Int64Index(getattr(self.times.isocalendar(), pandas_field))
        expected = xr.DataArray(
            expected, name=field, coords=[self.times], dims=["time"]
        )

        actual = self.data.time.dt.isocalendar()[field]
        assert_equal(expected, actual)

    def test_strftime(self):
        assert (
            "2000-01-01 01:00:00" == self.data.time.dt.strftime("%Y-%m-%d %H:%M:%S")[1]
        )

    def test_not_datetime_type(self):
        nontime_data = self.data.copy()
        int_data = np.arange(len(self.data.time)).astype("int8")
        nontime_data = nontime_data.assign_coords(time=int_data)
        with pytest.raises(TypeError, match=r"dt"):
            nontime_data.time.dt

    @pytest.mark.filterwarnings("ignore:dt.weekofyear and dt.week have been deprecated")
    @requires_dask
    @pytest.mark.parametrize(
        "field",
        [
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "microsecond",
            "nanosecond",
            "week",
            "weekofyear",
            "dayofweek",
            "weekday",
            "dayofyear",
            "quarter",
            "date",
            "time",
            "is_month_start",
            "is_month_end",
            "is_quarter_start",
            "is_quarter_end",
            "is_year_start",
            "is_year_end",
            "is_leap_year",
        ],
    )
    def test_dask_field_access(self, field):
        import dask.array as da

        expected = getattr(self.times_data.dt, field)

        dask_times_arr = da.from_array(self.times_arr, chunks=(5, 5, 50))
        dask_times_2d = xr.DataArray(
            dask_times_arr, coords=self.data.coords, dims=self.data.dims, name="data"
        )

        with raise_if_dask_computes():
            actual = getattr(dask_times_2d.dt, field)

        assert isinstance(actual.data, da.Array)
        assert_chunks_equal(actual, dask_times_2d)
        assert_equal(actual.compute(), expected.compute())

    @requires_dask
    @pytest.mark.parametrize(
        "field",
        [
            "year",
            "week",
            "weekday",
        ],
    )
    def test_isocalendar_dask(self, field):
        import dask.array as da

        if LooseVersion(pd.__version__) < "1.1.0":
            with pytest.raises(
                AttributeError, match=r"'isocalendar' not available in pandas < 1.1.0"
            ):
                self.data.time.dt.isocalendar()[field]
            return

        expected = getattr(self.times_data.dt.isocalendar(), field)

        dask_times_arr = da.from_array(self.times_arr, chunks=(5, 5, 50))
        dask_times_2d = xr.DataArray(
            dask_times_arr, coords=self.data.coords, dims=self.data.dims, name="data"
        )

        with raise_if_dask_computes():
            actual = dask_times_2d.dt.isocalendar()[field]

        assert isinstance(actual.data, da.Array)
        assert_chunks_equal(actual, dask_times_2d)
        assert_equal(actual.compute(), expected.compute())

    @requires_dask
    @pytest.mark.parametrize(
        "method, parameters",
        [
            ("floor", "D"),
            ("ceil", "D"),
            ("round", "D"),
            ("strftime", "%Y-%m-%d %H:%M:%S"),
        ],
    )
    def test_dask_accessor_method(self, method, parameters):
        import dask.array as da

        expected = getattr(self.times_data.dt, method)(parameters)
        dask_times_arr = da.from_array(self.times_arr, chunks=(5, 5, 50))
        dask_times_2d = xr.DataArray(
            dask_times_arr, coords=self.data.coords, dims=self.data.dims, name="data"
        )

        with raise_if_dask_computes():
            actual = getattr(dask_times_2d.dt, method)(parameters)

        assert isinstance(actual.data, da.Array)
        assert_chunks_equal(actual, dask_times_2d)
        assert_equal(actual.compute(), expected.compute())

    def test_seasons(self):
        dates = pd.date_range(start="2000/01/01", freq="M", periods=12)
        dates = xr.DataArray(dates)
        seasons = [
            "DJF",
            "DJF",
            "MAM",
            "MAM",
            "MAM",
            "JJA",
            "JJA",
            "JJA",
            "SON",
            "SON",
            "SON",
            "DJF",
        ]
        seasons = xr.DataArray(seasons)

        assert_array_equal(seasons.values, dates.dt.season.values)

    @pytest.mark.parametrize(
        "method, parameters", [("floor", "D"), ("ceil", "D"), ("round", "D")]
    )
    def test_accessor_method(self, method, parameters):
        dates = pd.date_range("2014-01-01", "2014-05-01", freq="H")
        xdates = xr.DataArray(dates, dims=["time"])
        expected = getattr(dates, method)(parameters)
        actual = getattr(xdates.dt, method)(parameters)
        assert_array_equal(expected, actual)


class TestTimedeltaAccessor:
    @pytest.fixture(autouse=True)
    def setup(self):
        nt = 100
        data = np.random.rand(10, 10, nt)
        lons = np.linspace(0, 11, 10)
        lats = np.linspace(0, 20, 10)
        self.times = pd.timedelta_range(start="1 day", freq="6H", periods=nt)

        self.data = xr.DataArray(
            data,
            coords=[lons, lats, self.times],
            dims=["lon", "lat", "time"],
            name="data",
        )

        self.times_arr = np.random.choice(self.times, size=(10, 10, nt))
        self.times_data = xr.DataArray(
            self.times_arr,
            coords=[lons, lats, self.times],
            dims=["lon", "lat", "time"],
            name="data",
        )

    def test_not_datetime_type(self):
        nontime_data = self.data.copy()
        int_data = np.arange(len(self.data.time)).astype("int8")
        nontime_data = nontime_data.assign_coords(time=int_data)
        with pytest.raises(TypeError, match=r"dt"):
            nontime_data.time.dt

    @pytest.mark.parametrize(
        "field", ["days", "seconds", "microseconds", "nanoseconds"]
    )
    def test_field_access(self, field):
        expected = xr.DataArray(
            getattr(self.times, field), name=field, coords=[self.times], dims=["time"]
        )
        actual = getattr(self.data.time.dt, field)
        assert_equal(expected, actual)

    @pytest.mark.parametrize(
        "method, parameters", [("floor", "D"), ("ceil", "D"), ("round", "D")]
    )
    def test_accessor_methods(self, method, parameters):
        dates = pd.timedelta_range(start="1 day", end="30 days", freq="6H")
        xdates = xr.DataArray(dates, dims=["time"])
        expected = getattr(dates, method)(parameters)
        actual = getattr(xdates.dt, method)(parameters)
        assert_array_equal(expected, actual)

    @requires_dask
    @pytest.mark.parametrize(
        "field", ["days", "seconds", "microseconds", "nanoseconds"]
    )
    def test_dask_field_access(self, field):
        import dask.array as da

        expected = getattr(self.times_data.dt, field)

        dask_times_arr = da.from_array(self.times_arr, chunks=(5, 5, 50))
        dask_times_2d = xr.DataArray(
            dask_times_arr, coords=self.data.coords, dims=self.data.dims, name="data"
        )

        with raise_if_dask_computes():
            actual = getattr(dask_times_2d.dt, field)

        assert isinstance(actual.data, da.Array)
        assert_chunks_equal(actual, dask_times_2d)
        assert_equal(actual, expected)

    @requires_dask
    @pytest.mark.parametrize(
        "method, parameters", [("floor", "D"), ("ceil", "D"), ("round", "D")]
    )
    def test_dask_accessor_method(self, method, parameters):
        import dask.array as da

        expected = getattr(self.times_data.dt, method)(parameters)
        dask_times_arr = da.from_array(self.times_arr, chunks=(5, 5, 50))
        dask_times_2d = xr.DataArray(
            dask_times_arr, coords=self.data.coords, dims=self.data.dims, name="data"
        )

        with raise_if_dask_computes():
            actual = getattr(dask_times_2d.dt, method)(parameters)

        assert isinstance(actual.data, da.Array)
        assert_chunks_equal(actual, dask_times_2d)
        assert_equal(actual.compute(), expected.compute())


_CFTIME_CALENDARS = [
    "365_day",
    "360_day",
    "julian",
    "all_leap",
    "366_day",
    "gregorian",
    "proleptic_gregorian",
]
_NT = 100


@pytest.fixture(params=_CFTIME_CALENDARS)
def calendar(request):
    return request.param


@pytest.fixture()
def times(calendar):
    import cftime

    return cftime.num2date(
        np.arange(_NT),
        units="hours since 2000-01-01",
        calendar=calendar,
        only_use_cftime_datetimes=True,
    )


@pytest.fixture()
def data(times):
    data = np.random.rand(10, 10, _NT)
    lons = np.linspace(0, 11, 10)
    lats = np.linspace(0, 20, 10)
    return xr.DataArray(
        data, coords=[lons, lats, times], dims=["lon", "lat", "time"], name="data"
    )


@pytest.fixture()
def times_3d(times):
    lons = np.linspace(0, 11, 10)
    lats = np.linspace(0, 20, 10)
    times_arr = np.random.choice(times, size=(10, 10, _NT))
    return xr.DataArray(
        times_arr, coords=[lons, lats, times], dims=["lon", "lat", "time"], name="data"
    )


@requires_cftime
@pytest.mark.parametrize(
    "field", ["year", "month", "day", "hour", "dayofyear", "dayofweek"]
)
def test_field_access(data, field):
    if field == "dayofyear" or field == "dayofweek":
        pytest.importorskip("cftime", minversion="1.0.2.1")
    result = getattr(data.time.dt, field)
    expected = xr.DataArray(
        getattr(xr.coding.cftimeindex.CFTimeIndex(data.time.values), field),
        name=field,
        coords=data.time.coords,
        dims=data.time.dims,
    )

    assert_equal(result, expected)


@requires_cftime
def test_isocalendar_cftime(data):

    with pytest.raises(
        AttributeError, match=r"'CFTimeIndex' object has no attribute 'isocalendar'"
    ):
        data.time.dt.isocalendar()


@requires_cftime
def test_date_cftime(data):

    with pytest.raises(
        AttributeError,
        match=r"'CFTimeIndex' object has no attribute `date`. Consider using the floor method instead, for instance: `.time.dt.floor\('D'\)`.",
    ):
        data.time.dt.date()


@requires_cftime
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_cftime_strftime_access(data):
    """compare cftime formatting against datetime formatting"""
    date_format = "%Y%m%d%H"
    result = data.time.dt.strftime(date_format)
    datetime_array = xr.DataArray(
        xr.coding.cftimeindex.CFTimeIndex(data.time.values).to_datetimeindex(),
        name="stftime",
        coords=data.time.coords,
        dims=data.time.dims,
    )
    expected = datetime_array.dt.strftime(date_format)
    assert_equal(result, expected)


@requires_cftime
@requires_dask
@pytest.mark.parametrize(
    "field", ["year", "month", "day", "hour", "dayofyear", "dayofweek"]
)
def test_dask_field_access_1d(data, field):
    import dask.array as da

    if field == "dayofyear" or field == "dayofweek":
        pytest.importorskip("cftime", minversion="1.0.2.1")
    expected = xr.DataArray(
        getattr(xr.coding.cftimeindex.CFTimeIndex(data.time.values), field),
        name=field,
        dims=["time"],
    )
    times = xr.DataArray(data.time.values, dims=["time"]).chunk({"time": 50})
    result = getattr(times.dt, field)
    assert isinstance(result.data, da.Array)
    assert result.chunks == times.chunks
    assert_equal(result.compute(), expected)


@requires_cftime
@requires_dask
@pytest.mark.parametrize(
    "field", ["year", "month", "day", "hour", "dayofyear", "dayofweek"]
)
def test_dask_field_access(times_3d, data, field):
    import dask.array as da

    if field == "dayofyear" or field == "dayofweek":
        pytest.importorskip("cftime", minversion="1.0.2.1")
    expected = xr.DataArray(
        getattr(
            xr.coding.cftimeindex.CFTimeIndex(times_3d.values.ravel()), field
        ).reshape(times_3d.shape),
        name=field,
        coords=times_3d.coords,
        dims=times_3d.dims,
    )
    times_3d = times_3d.chunk({"lon": 5, "lat": 5, "time": 50})
    result = getattr(times_3d.dt, field)
    assert isinstance(result.data, da.Array)
    assert result.chunks == times_3d.chunks
    assert_equal(result.compute(), expected)


@pytest.fixture()
def cftime_date_type(calendar):
    from .test_coding_times import _all_cftime_date_types

    return _all_cftime_date_types()[calendar]


@requires_cftime
def test_seasons(cftime_date_type):
    dates = np.array([cftime_date_type(2000, month, 15) for month in range(1, 13)])
    dates = xr.DataArray(dates)
    seasons = [
        "DJF",
        "DJF",
        "MAM",
        "MAM",
        "MAM",
        "JJA",
        "JJA",
        "JJA",
        "SON",
        "SON",
        "SON",
        "DJF",
    ]
    seasons = xr.DataArray(seasons)

    assert_array_equal(seasons.values, dates.dt.season.values)


@pytest.fixture
def cftime_rounding_dataarray(cftime_date_type):
    return xr.DataArray(
        [
            [cftime_date_type(1, 1, 1, 1), cftime_date_type(1, 1, 1, 15)],
            [cftime_date_type(1, 1, 1, 23), cftime_date_type(1, 1, 2, 1)],
        ]
    )


@requires_cftime
@requires_dask
@pytest.mark.parametrize("use_dask", [False, True])
def test_cftime_floor_accessor(cftime_rounding_dataarray, cftime_date_type, use_dask):
    import dask.array as da

    freq = "D"
    expected = xr.DataArray(
        [
            [cftime_date_type(1, 1, 1, 0), cftime_date_type(1, 1, 1, 0)],
            [cftime_date_type(1, 1, 1, 0), cftime_date_type(1, 1, 2, 0)],
        ],
        name="floor",
    )

    if use_dask:
        chunks = {"dim_0": 1}
        # Currently a compute is done to inspect a single value of the array
        # if it is of object dtype to check if it is a cftime.datetime (if not
        # we raise an error when using the dt accessor).
        with raise_if_dask_computes(max_computes=1):
            result = cftime_rounding_dataarray.chunk(chunks).dt.floor(freq)
        expected = expected.chunk(chunks)
        assert isinstance(result.data, da.Array)
        assert result.chunks == expected.chunks
    else:
        result = cftime_rounding_dataarray.dt.floor(freq)

    assert_identical(result, expected)


@requires_cftime
@requires_dask
@pytest.mark.parametrize("use_dask", [False, True])
def test_cftime_ceil_accessor(cftime_rounding_dataarray, cftime_date_type, use_dask):
    import dask.array as da

    freq = "D"
    expected = xr.DataArray(
        [
            [cftime_date_type(1, 1, 2, 0), cftime_date_type(1, 1, 2, 0)],
            [cftime_date_type(1, 1, 2, 0), cftime_date_type(1, 1, 3, 0)],
        ],
        name="ceil",
    )

    if use_dask:
        chunks = {"dim_0": 1}
        # Currently a compute is done to inspect a single value of the array
        # if it is of object dtype to check if it is a cftime.datetime (if not
        # we raise an error when using the dt accessor).
        with raise_if_dask_computes(max_computes=1):
            result = cftime_rounding_dataarray.chunk(chunks).dt.ceil(freq)
        expected = expected.chunk(chunks)
        assert isinstance(result.data, da.Array)
        assert result.chunks == expected.chunks
    else:
        result = cftime_rounding_dataarray.dt.ceil(freq)

    assert_identical(result, expected)


@requires_cftime
@requires_dask
@pytest.mark.parametrize("use_dask", [False, True])
def test_cftime_round_accessor(cftime_rounding_dataarray, cftime_date_type, use_dask):
    import dask.array as da

    freq = "D"
    expected = xr.DataArray(
        [
            [cftime_date_type(1, 1, 1, 0), cftime_date_type(1, 1, 2, 0)],
            [cftime_date_type(1, 1, 2, 0), cftime_date_type(1, 1, 2, 0)],
        ],
        name="round",
    )

    if use_dask:
        chunks = {"dim_0": 1}
        # Currently a compute is done to inspect a single value of the array
        # if it is of object dtype to check if it is a cftime.datetime (if not
        # we raise an error when using the dt accessor).
        with raise_if_dask_computes(max_computes=1):
            result = cftime_rounding_dataarray.chunk(chunks).dt.round(freq)
        expected = expected.chunk(chunks)
        assert isinstance(result.data, da.Array)
        assert result.chunks == expected.chunks
    else:
        result = cftime_rounding_dataarray.dt.round(freq)

    assert_identical(result, expected)
