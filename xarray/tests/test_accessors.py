from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
try:
    import cPickle as pickle
except ImportError:
    import pickle

import xarray as xr
import numpy as np
import pandas as pd

from . import TestCase


class TestDatetimeAccessor(TestCase):
    def setUp(self):
        nt = 10000
        data = np.random.rand(10, 10, nt)
        lons = np.linspace(0, 11, 10)
        lats = np.linspace(0, 20, 10)
        self.times = pd.date_range(start="2000/01/01", freq='H', periods=nt)

        self.data = xr.DataArray(data, coords=[lons, lats, self.times],
                                 dims=['lon', 'lat', 'time'], name='data')

    def test_field_access(self):
        years = self.times.year
        months = self.times.month
        days = self.times.day
        hours = self.times.hour

        self.assertArrayEqual(years, self.data.time.dt.year)
        self.assertArrayEqual(months, self.data.time.dt.month)
        self.assertArrayEqual(days, self.data.time.dt.day)
        self.assertArrayEqual(hours, self.data.time.dt.hour)

    def test_not_datetime_type(self):
        nontime_data = self.data.copy()
        nontime_data['time'].values = \
            np.arange(len(self.data.time)).astype('int8')
        with self.assertRaisesRegexp(TypeError, 'dt'):
            nontime_data.time.dt.year