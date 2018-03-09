from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import xarray as xr

from . import parameterized, randn, requires_dask

nx = 3000
ny = 2000
nt = 1000
window = 20


class Rolling(object):
    def setup(self, *args, **kwargs):
        self.ds = xr.Dataset(
            {'var1': (('x', 'y'), randn((nx, ny), frac_nan=0.1)),
             'var2': (('x', 't'), randn((nx, nt))),
             'var3': (('t', ), randn(nt))},
            coords={'x': np.arange(nx),
                    'y': np.linspace(0, 1, ny),
                    't': pd.date_range('1970-01-01', periods=nt, freq='D'),
                    'x_coords': ('x', np.linspace(1.1, 2.1, nx))})

    @parameterized(['func', 'center'],
                   (['mean', 'count'], [True, False]))
    def time_rolling(self, func, center):
        getattr(self.ds.rolling(x=window, center=center), func)()

    @parameterized(['window_', 'min_periods'],
                   ([20, 40], [5, None]))
    def time_rolling_np(self, window_, min_periods):
        self.ds.rolling(x=window_, center=False,
                        min_periods=min_periods).reduce(getattr(np, 'nanmean'))

    @parameterized(['center', 'stride'],
                   ([True, False], [1, 200]))
    def time_rolling_construct(self, center, stride):
        self.ds.rolling(x=window, center=center).construct(
            'window_dim', stride=stride).mean(dim='window_dim')


class RollingDask(Rolling):
    def setup(self, *args, **kwargs):
        requires_dask()
        super(RollingDask, self).setup(**kwargs)
        self.ds = self.ds.chunk({'x': 100, 'y': 50, 't': 50})
