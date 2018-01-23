from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import xarray as xr

from . import randn, randint


nx = 3000
ny = 2000
nt = 1000
ds = xr.Dataset({'var1': (('x', 'y'), randn((nx, ny), frac_nan=0.1)),
                 'var2': (('x', 't'), randn((nx, nt))),
                 'var3': (('t', ), randn(nt))},
                coords={'x': np.arange(nx),
                        'y': np.linspace(0, 1, ny),
                        't': pd.date_range('1970-01-01', periods=nt, freq='D'),
                        'x_coords': ('x', np.linspace(1.1, 2.1, nx))})

basic_indexes = [
    {'x': slice(0, 3)},
    {'x': 0, 'y': slice(None, None, 3)},
    {'x': slice(3, -3, 3), 'y': 1, 't': slice(None, -3, 3)}
]

basic_assignment_values = [
    xr.DataArray(randn((3, ny), frac_nan=0.1), dims=['x', 'y']),
    xr.DataArray(randn(int(ny / 3) + 1, frac_nan=0.1), dims=['y']),
    xr.DataArray(randn(int((nx - 6) / 3), frac_nan=0.1), dims=['x']),
]


def time_indexing_basic():
    for ind in basic_indexes:
        ds.isel(**ind)


def time_assignment_basic():
    tmp = ds.copy(deep=True)
    for ind, val in zip(basic_indexes, basic_assignment_values):
        tmp['var1'][ind.get('x', slice(None)), ind.get('y', slice(None))] = val


outer_indexes = [
    {'x': randint(0, nx, 400)},
    {'x': randint(0, nx, 500), 'y': randint(0, ny, 400)},
    {'x': randint(0, nx, 100), 'y': 1, 't': randint(0, nt, 400)},
]

outer_assignment_values = [
    xr.DataArray(randn((400, ny), frac_nan=0.1), dims=['x', 'y']),
    xr.DataArray(randn((500, 400), frac_nan=0.1), dims=['x', 'y']),
    xr.DataArray(randn(100, frac_nan=0.1), dims=['x']),
]


def time_indexing_outer():
    for ind in outer_indexes:
        ds.isel(**ind)


def time_assignment_outer():
    tmp = ds.copy(deep=True)
    for ind, val in zip(outer_indexes, outer_assignment_values):
        tmp['var1'][ind.get('x', slice(None)), ind.get('y', slice(None))] = val


vectorized_indexes = [
    {'x': xr.DataArray(randint(0, nx, 400), dims='a')},
    {'x': xr.DataArray(randint(0, nx, 400), dims='a'),
     'y': xr.DataArray(randint(0, ny, 400), dims='a')},
    {'x': xr.DataArray(randint(0, nx, 400).reshape(4, 100), dims=['a', 'b']),
     'y': xr.DataArray(randint(0, ny, 400).reshape(4, 100), dims=['a', 'b']),
     't': xr.DataArray(randint(0, nt, 400).reshape(4, 100), dims=['a', 'b'])},
]

vectorized_assignment_values = [
    xr.DataArray(randn((400, 2000)), dims=['a', 'y'],
                 coords={'a': randn(400)}),
    xr.DataArray(randn(400), dims=['a', ], coords={'a': randn(400)}),
    xr.DataArray(randn((4, 100)), dims=['a', 'b'],
                 coords={'a': randn(4), 'b': randn(100)}),
]


def time_indexing_vectorized():
    for ind in vectorized_indexes:
        ds.isel(**ind)


def time_assignment_vectorized():
    tmp = ds.copy(deep=True)
    for ind, val in zip(vectorized_indexes, vectorized_assignment_values):
        tmp['var1'][ind.get('x', slice(None)), ind.get('y', slice(None))] = val


try:
    ds_dask = ds.chunk({'x': 100, 'y': 50, 't': 50})

    def time_indexing_basic_dask():
        for ind in basic_indexes:
            ds_dask.isel(**ind)

    def time_indexing_outer_dask():
        for ind in outer_indexes:
            ds_dask.isel(**ind)

    def time_indexing_vectorized_dask():
        for ind in vectorized_indexes:
            ds_dask.isel(**ind)

except ImportError:
    pass
