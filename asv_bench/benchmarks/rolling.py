from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import xarray as xr

from . import randn


nx = 3000
ny = 2000
nt = 1000
window = 20
ds = xr.Dataset({'var1': (('x', 'y'), randn((nx, ny), frac_nan=0.1)),
                 'var2': (('x', 't'), randn((nx, nt))),
                 'var3': (('t', ), randn(nt))},
                coords={'x': np.arange(nx),
                        'y': np.linspace(0, 1, ny),
                        't': pd.date_range('1970-01-01', periods=nt, freq='D'),
                        'x_coords': ('x', np.linspace(1.1, 2.1, nx))})


def time_rolling(func, center):
    getattr(ds.rolling(x=window, center=center), func)()


time_rolling.param_names = ['func', 'center']
time_rolling.params = (['mean', 'count'], [True, False])


def time_rolling_np(window_, min_periods):
    ds.rolling(x=window_, center=False, min_periods=min_periods).reduce(
        getattr(np, 'nanmean'))


time_rolling_np.param_names = ['window_', 'min_periods']
time_rolling_np.params = ([20, 40], [5, None])


def time_rolling_to_dataset(center, stride):
    ds.rolling(x=window, center=center).to_dataset(
        'window_dim', stride=stride).mean(dim='window_dim')


time_rolling_to_dataset.param_names = ['center', 'stride']
time_rolling_to_dataset.params = ([True, False], [1, 200])
