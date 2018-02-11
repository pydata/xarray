from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xarray as xr
import numpy as np
import pandas as pd

from . import (TestCase, requires_dask, raises_regex, assert_equal,
               assert_array_equal)


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
