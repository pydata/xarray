import numpy as np

import xarray as xr

from . import parameterized, randn, requires_dask

ndegs = (2, 5, 20)
nxs = (10, 100, 1000, 5000)

xs = {nx: xr.DataArray(randn((nx, nx)), dims=("x", "y")) for nx in nxs}
coeffs = {ndeg: xr.DataArray(randn(ndeg), dims="degree") for ndeg in ndegs}


class Polyval:
    def setup(self, *args, **kwargs):
        self.coeffs = coeffs
        self.xs = xs

    @parameterized(["nx", "ndeg"], [nxs, ndegs])
    def time_polyval(self, nx, ndeg):
        x = self.xs[nx]
        c = self.coeffs[ndeg]
        xr.polyval(x, c)

    @parameterized(["nx", "ndeg"], [nxs, ndegs])
    def peakmem_polyval(self, nx, ndeg):
        x = self.xs[nx]
        c = self.coeffs[ndeg]
        xr.polyval(x, c)


class PolyvalDask(Polyval):
    def setup(self, *args, **kwargs):
        requires_dask()
        super().setup(*args, **kwargs)
        self.xs = {nx: self.xs[nx].chunk({"x": 100, "y": 100}) for nx in nxs}
