from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

try:
    import dask
    import dask.multiprocessing
except ImportError:
    pass

import xarray as xr

from . import randn, randint, requires_dask


nx = 300
ny = 200
nt = 100
ds = xr.Dataset({'var1': (('x', 'y'), randn((nx, ny), frac_nan=0.1)),
                 'var2': (('x', 't'), randn((nx, nt))),
                 'var3': (('t', ), randn(nt))},
                coords={'x': np.arange(nx),
                        'y': np.linspace(0, 1, ny),
                        't': pd.date_range('1970-01-01', periods=nt, freq='D'),
                        'x_coords': ('x', np.linspace(1.1, 2.1, nx))})


vectorized_indexes = [
    {'x': xr.DataArray(randint(0, nx, 400), dims='a')},
    {'x': xr.DataArray(randint(0, nx, 400), dims='a'),
     'y': xr.DataArray(randint(0, ny, 400), dims='a')},
    {'x': xr.DataArray(randint(0, nx, 400).reshape(4, 100), dims=['a', 'b']),
     'y': xr.DataArray(randint(0, ny, 400).reshape(4, 100), dims=['a', 'b']),
     't': xr.DataArray(randint(0, nt, 400).reshape(4, 100), dims=['a', 'b'])},
]


def time_basic_indexing(index):
    ds.isel(index)


time_basic_indexing.param_names = ['index']
time_basic_indexing.params = [
    {'x': slice(0, 3)},
    {'x': 0, 'y': slice(0, None, 3)},
    {'x': slice(3, -3, 3), 'y': 1, 't': slice(None, -3, 3)},
]


def time_outer_indexing(index):
    ds.isel(index)


time_outer_indexing.param_names = ['index']
time_outer_indexing.params = [
    {'x': randint(0, nx, 400)},
    {'x': randint(0, nx, 500), 'y': randint(0, ny, 400)},
    {'x': randint(0, nx, 100), 'y': 1, 't': randint(0, nt, 400)},
]
