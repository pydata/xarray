import unittest
import numpy as np
import pandas as pd
import netCDF4 as nc4

from xray import backends, xarray, dataset
import datetime


class TestDatetime(unittest.TestCase):

    def test_convert_to_cf_variable(self):

        # test with the default calendar
        dates = pd.date_range(datetime.datetime(2012, 2, 25),
                              datetime.datetime(2012, 3, 5))
        units = 'hours since 2012-1-1'
        time = xarray.XArray(('time'), dates,
                             attributes={'cf_units': units})
        cf_time = backends.convert_to_cf_variable(time)
        self.assertEqual(units, cf_time.attributes['units'])
        self.assertNotIn('calendar', cf_time.attributes)

        cf_dates = nc4.num2date(cf_time.data, units, calendar='standard')
        cf_dtindex = pd.DatetimeIndex(cf_dates)
        self.assertTrue(all(dates.values == cf_dtindex.values))

        # test with an assigned calendar
        calendar = 'proleptic_gregorian'
        time = xarray.XArray(('time'), dates,
                             attributes={'cf_units': units,
                                         'cf_calendar': calendar})
        cf_time = backends.convert_to_cf_variable(time)
        self.assertEqual(units, cf_time.attributes['units'])
        self.assertEqual(calendar, cf_time.attributes['calendar'])

        cf_dates = nc4.num2date(cf_time.data, units, calendar=calendar)
        cf_dtindex = pd.DatetimeIndex(cf_dates)
        self.assertTrue(all(dates.values == cf_dtindex.values))

        # test with irregularly spaced dates
        more_dates = pd.date_range(datetime.datetime(2013, 2, 25),
                                   datetime.datetime(2013, 3, 5))
        more_dates = np.concatenate([dates.to_pydatetime(),
                                     more_dates.to_pydatetime()])
        more_dates = pd.DatetimeIndex(more_dates)

        units = 'hours since 2012-1-1'
        time = xarray.XArray(('time'), more_dates,
                             attributes={'cf_units': units})
        cf_time = backends.convert_to_cf_variable(time)
        self.assertEqual(units, cf_time.attributes['units'])
        self.assertNotIn('calendar', cf_time.attributes)

        cf_dates = nc4.num2date(cf_time.data, units, calendar='standard')
        cf_dtindex = pd.DatetimeIndex(cf_dates)
        self.assertTrue(all(more_dates.values == cf_dtindex.values))
