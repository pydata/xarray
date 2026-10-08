import xarray as xr

from . import parameterized, randn, requires_dask

# 256 MB per array, chunked along the dimension we reduce over
shape = (200, 400, 400)
chunks = (10, 400, 400)


class CovCorrDask:
    def setup(self, *args, **kwargs):
        requires_dask()
        dims = ("time", "y", "x")
        self.da_a = xr.DataArray(randn(shape, chunks=chunks, seed=0), dims=dims)
        self.da_b = xr.DataArray(randn(shape, chunks=chunks, seed=1), dims=dims)

    @parameterized(["func"], (["cov", "corr"]))
    def time_cov_corr(self, func):
        getattr(xr, func)(self.da_a, self.da_b, dim="time").compute()

    @parameterized(["func"], (["cov", "corr"]))
    def peakmem_cov_corr(self, func):
        # the synchronous scheduler makes the peak memory independent of the
        # number of cores, otherwise the chunks in flight would dominate it
        result = getattr(xr, func)(self.da_a, self.da_b, dim="time")
        result.compute(scheduler="synchronous")
