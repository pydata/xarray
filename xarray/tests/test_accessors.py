from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import pytest

import xarray as xr

from . import (
    TestCase, assert_array_equal, assert_equal, raises_regex, requires_dask,
    has_cftime, has_dask, has_cftime_or_netCDF4)


class TestDatetimeAccessor(TestCase):
    def setUp(self):
        nt = 100
        data = np.random.rand(10, 10, nt)
        lons = np.linspace(0, 11, 10)
        lats = np.linspace(0, 20, 10)
        self.times = pd.date_range(start="2000/01/01", freq='H', periods=nt)

        self.data = xr.DataArray(data, coords=[lons, lats, self.times],
                                 dims=['lon', 'lat', 'time'], name='data')

        self.times_arr = np.random.choice(self.times, size=(10, 10, nt))
        self.times_data = xr.DataArray(self.times_arr,
                                       coords=[lons, lats, self.times],
                                       dims=['lon', 'lat', 'time'],
                                       name='data')

    def test_field_access(self):
        years = xr.DataArray(self.times.year, name='year',
                             coords=[self.times, ], dims=['time', ])
        months = xr.DataArray(self.times.month, name='month',
                              coords=[self.times, ], dims=['time', ])
        days = xr.DataArray(self.times.day, name='day',
                            coords=[self.times, ], dims=['time', ])
        hours = xr.DataArray(self.times.hour, name='hour',
                             coords=[self.times, ], dims=['time', ])

        assert_equal(years, self.data.time.dt.year)
        assert_equal(months, self.data.time.dt.month)
        assert_equal(days, self.data.time.dt.day)
        assert_equal(hours, self.data.time.dt.hour)

    def test_not_datetime_type(self):
        nontime_data = self.data.copy()
        int_data = np.arange(len(self.data.time)).astype('int8')
        nontime_data['time'].values = int_data
        with raises_regex(TypeError, 'dt'):
            nontime_data.time.dt

    @requires_dask
    def test_dask_field_access(self):
        import dask.array as da

        years = self.times_data.dt.year
        months = self.times_data.dt.month
        hours = self.times_data.dt.hour
        days = self.times_data.dt.day
        floor = self.times_data.dt.floor('D')
        ceil = self.times_data.dt.ceil('D')
        round = self.times_data.dt.round('D')

        dask_times_arr = da.from_array(self.times_arr, chunks=(5, 5, 50))
        dask_times_2d = xr.DataArray(dask_times_arr,
                                     coords=self.data.coords,
                                     dims=self.data.dims,
                                     name='data')
        dask_year = dask_times_2d.dt.year
        dask_month = dask_times_2d.dt.month
        dask_day = dask_times_2d.dt.day
        dask_hour = dask_times_2d.dt.hour
        dask_floor = dask_times_2d.dt.floor('D')
        dask_ceil = dask_times_2d.dt.ceil('D')
        dask_round = dask_times_2d.dt.round('D')

        # Test that the data isn't eagerly evaluated
        assert isinstance(dask_year.data, da.Array)
        assert isinstance(dask_month.data, da.Array)
        assert isinstance(dask_day.data, da.Array)
        assert isinstance(dask_hour.data, da.Array)

        # Double check that outcome chunksize is unchanged
        dask_chunks = dask_times_2d.chunks
        assert dask_year.data.chunks == dask_chunks
        assert dask_month.data.chunks == dask_chunks
        assert dask_day.data.chunks == dask_chunks
        assert dask_hour.data.chunks == dask_chunks

        # Check the actual output from the accessors
        assert_equal(years, dask_year.compute())
        assert_equal(months, dask_month.compute())
        assert_equal(days, dask_day.compute())
        assert_equal(hours, dask_hour.compute())
        assert_equal(floor, dask_floor.compute())
        assert_equal(ceil, dask_ceil.compute())
        assert_equal(round, dask_round.compute())

    def test_seasons(self):
        dates = pd.date_range(start="2000/01/01", freq="M", periods=12)
        dates = xr.DataArray(dates)
        seasons = ["DJF", "DJF", "MAM", "MAM", "MAM", "JJA", "JJA", "JJA",
                   "SON", "SON", "SON", "DJF"]
        seasons = xr.DataArray(seasons)

        assert_array_equal(seasons.values, dates.dt.season.values)

    def test_rounders(self):
        dates = pd.date_range("2014-01-01", "2014-05-01", freq='H')
        xdates = xr.DataArray(np.arange(len(dates)),
                              dims=['time'], coords=[dates])
        assert_array_equal(dates.floor('D').values,
                           xdates.time.dt.floor('D').values)
        assert_array_equal(dates.ceil('D').values,
                           xdates.time.dt.ceil('D').values)
        assert_array_equal(dates.round('D').values,
                           xdates.time.dt.round('D').values)


_CFTIME_CALENDARS = ['365_day', '360_day', 'julian', 'all_leap',
                     '366_day', 'gregorian', 'proleptic_gregorian']
_NT = 100


@pytest.fixture(params=_CFTIME_CALENDARS)
def calendar(request):
    return request.param


@pytest.fixture()
def times(calendar):
    import cftime

    return cftime.num2date(
        np.arange(_NT), units='hours since 2000-01-01', calendar=calendar,
        only_use_cftime_datetimes=True)


@pytest.fixture()
def data(times):
    data = np.random.rand(10, 10, _NT)
    lons = np.linspace(0, 11, 10)
    lats = np.linspace(0, 20, 10)
    return xr.DataArray(data, coords=[lons, lats, times],
                        dims=['lon', 'lat', 'time'], name='data')


@pytest.fixture()
def times_3d(times):
    lons = np.linspace(0, 11, 10)
    lats = np.linspace(0, 20, 10)
    times_arr = np.random.choice(times, size=(10, 10, _NT))
    return xr.DataArray(times_arr, coords=[lons, lats, times],
                        dims=['lon', 'lat', 'time'],
                        name='data')


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize('field', ['year', 'month', 'day', 'hour'])
def test_field_access(data, field):
    result = getattr(data.time.dt, field)
    expected = xr.DataArray(
        getattr(xr.coding.cftimeindex.CFTimeIndex(data.time.values), field),
        name=field, coords=data.time.coords, dims=data.time.dims)

    assert_equal(result, expected)


@pytest.mark.skipif(not has_dask, reason='dask not installed')
@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize('field', ['year', 'month', 'day', 'hour'])
def test_dask_field_access_1d(data, field):
    import dask.array as da

    expected = xr.DataArray(
        getattr(xr.coding.cftimeindex.CFTimeIndex(data.time.values), field),
        name=field, dims=['time'])
    times = xr.DataArray(data.time.values, dims=['time']).chunk({'time': 50})
    result = getattr(times.dt, field)
    assert isinstance(result.data, da.Array)
    assert result.chunks == times.chunks
    assert_equal(result.compute(), expected)


@pytest.mark.skipif(not has_dask, reason='dask not installed')
@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize('field', ['year', 'month', 'day', 'hour'])
def test_dask_field_access(times_3d, data, field):
    import dask.array as da

    expected = xr.DataArray(
        getattr(xr.coding.cftimeindex.CFTimeIndex(times_3d.values.ravel()),
                field).reshape(times_3d.shape),
        name=field, coords=times_3d.coords, dims=times_3d.dims)
    times_3d = times_3d.chunk({'lon': 5, 'lat': 5, 'time': 50})
    result = getattr(times_3d.dt, field)
    assert isinstance(result.data, da.Array)
    assert result.chunks == times_3d.chunks
    assert_equal(result.compute(), expected)


@pytest.fixture()
def cftime_date_type(calendar):
    from .test_coding_times import _all_cftime_date_types

    return _all_cftime_date_types()[calendar]


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
def test_seasons(cftime_date_type):
    dates = np.array([cftime_date_type(2000, month, 15)
                      for month in range(1, 13)])
    dates = xr.DataArray(dates)
    seasons = ['DJF', 'DJF', 'MAM', 'MAM', 'MAM', 'JJA',
               'JJA', 'JJA', 'SON', 'SON', 'SON', 'DJF']
    seasons = xr.DataArray(seasons)

    assert_array_equal(seasons.values, dates.dt.season.values)


@pytest.mark.skipif(not has_cftime_or_netCDF4,
                    reason='cftime or netCDF4 not installed')
def test_dt_accessor_error_netCDF4(cftime_date_type):
    da = xr.DataArray(
        [cftime_date_type(1, 1, 1), cftime_date_type(2, 1, 1)],
        dims=['time'])
    if not has_cftime:
        with pytest.raises(TypeError):
            da.dt.month
    else:
        da.dt.month
