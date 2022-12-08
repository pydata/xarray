import numpy as np

import xarray as xr

from . import parameterized, randn, requires_dask

NDEGS = (2, 5, 20)
NX = (10**2, 10**6)


class Polyval:
    def setup(self, *args, **kwargs):
        self.xs = {nx: xr.DataArray(randn((nx,)), dims="x", name="x") for nx in NX}
        self.coeffs = {
            ndeg: xr.DataArray(
                randn((ndeg,)), dims="degree", coords={"degree": np.arange(ndeg)}
            )
            for ndeg in NDEGS
        }

    @parameterized(["nx", "ndeg"], [NX, NDEGS])
    def time_polyval(self, nx, ndeg):
        x = self.xs[nx]
        c = self.coeffs[ndeg]
        xr.polyval(x, c).compute()

    @parameterized(["nx", "ndeg"], [NX, NDEGS])
    def peakmem_polyval(self, nx, ndeg):
        x = self.xs[nx]
        c = self.coeffs[ndeg]
        xr.polyval(x, c).compute()


class PolyvalDask(Polyval):
    def setup(self, *args, **kwargs):
        requires_dask()
        super().setup(*args, **kwargs)
        self.xs = {k: v.chunk({"x": 10000}) for k, v in self.xs.items()}
