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


class Indexing(object):
    def setup(self):
        self.nx = 300
        self.ny = 200
        self.nt = 100
        var1 = randn((self.nx, self.ny), frac_nan=0.1)
        var2 = randn((self.nx, self.nt))
        var3 = randn(self.nt)
        self.x = np.arange(self.nx)
        self.y = np.linspace(0, 1, self.ny)
        self.t = pd.date_range('1970-01-01', periods=self.nt, freq='D')
        self.x_coords = np.linspace(0, 1, self.nx)
        self.ds = xr.Dataset({'var1': (('x', 'y'), var1),
                              'var2': (('x', 't'), var2),
                              'var3': (('t', ), var3)},
                             coords={'x': self.x, 'y': self.y, 't': self.t,
                                     'x_coords': ('x', self.x_coords)})

        self.outer_indexes = [
            (randint(0, self.nx, 400), ),
            (randint(0, self.nx, 500), randint(0, self.ny, 400))]

    def time_outer_indexing(self):
        for ind in self.outer_indexes:
            ind_x = xr.DataArray(ind[-1], dims='y',
                                 coords={'y': self.x[ind[0]]})
            self.ds['var1'][(ind_x,) + ind[1:]]

    def time_outer_assignment(self):
        inds = self.outer_indexes()
        for ind in inds:
            self.ds['var1'][ind] = xr.DataArray(np.ones(400), dims='y')

    def time_vectorized_indexing(self):
        inds = [(xr.DataArray(randint(0, self.nx, self.ny), dims=['y']), ),
                (xr.DataArray(randint(0, self.nx, 500), dims=['a']),
                 xr.DataArray(randint(0, self.ny, 500), dims=['a'])),
                (xr.DataArray(randint(0, self.ny, 500).reshape(25, 20)),
                 xr.DataArray(randint(0, self.ny, 500).reshape(25, 20)))]
        for ind in inds:
            self.ds['var1'][ind]

    def time_vectorized_indexing_coords(self):
        ind = randint(0, self.nx, self.ny)
        inds = [(xr.DataArray(ind, dims=['y'], coords={'y': self.y}), ),
                (xr.DataArray(randint(0, self.nx, 500), dims=['a'],
                              coords={'a': np.linspace(0, 1, 500)}),
                 xr.DataArray(randint(0, self.ny, 500), dims=['a'],
                              coords={'a': np.linspace(0, 1, 500)})),
                (xr.DataArray(randint(0, self.ny, 500).reshape(25, 20),
                              dims=['a', 'b'],
                              coords={'a': np.arange(25), 'b': np.arange(20)}),
                 xr.DataArray(randint(0, self.ny, 500).reshape(25, 20),
                              dims=['a', 'b'],
                              coords={'a': np.arange(25), 'b': np.arange(20)}))
                ]
        for ind in inds:
            self.ds['var1'][ind]
