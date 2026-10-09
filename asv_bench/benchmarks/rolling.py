import numpy as np
import pandas as pd

import xarray as xr

from . import parameterized, randn, requires_dask

nx = 3000
long_nx = 30000
ny = 200
nt = 1000
window = 20
# numbagg and bottleneck are only used for rolling along a single dimension
BACKENDS = ["numbagg", "bottleneck", "numpy"]

randn_xy = randn((nx, ny), frac_nan=0.1)
randn_xt = randn((nx, nt))
randn_t = randn((nt,))
randn_long = randn((long_nx,), frac_nan=0.1)


def _create_ds():
    return xr.Dataset(
        {
            "var1": (("x", "y"), randn_xy),
            "var2": (("x", "t"), randn_xt),
            "var3": (("t",), randn_t),
        },
        coords={
            "x": np.arange(nx),
            "y": np.linspace(0, 1, ny),
            "t": pd.date_range("1970-01-01", periods=nt, freq="D"),
            "x_coords": ("x", np.linspace(1.1, 2.1, nx)),
        },
    )


def _backend_options(backend):
    # numbagg is used before bottleneck if both are enabled and have the function
    return xr.set_options(
        use_numbagg=backend == "numbagg", use_bottleneck=backend == "bottleneck"
    )


def _requires_method(obj, name):
    # skip methods that don't exist in older versions
    if not hasattr(obj, name):
        raise NotImplementedError(f"{type(obj).__name__}.{name} doesn't exist")


class Rolling:
    def setup(self, *args, **kwargs):
        self.ds = _create_ds()
        self.da_long = xr.DataArray(
            randn_long, dims="x", coords={"x": np.arange(long_nx) * 0.1}
        )

    @parameterized(
        ["func", "center", "backend"], (["mean", "count"], [True, False], BACKENDS)
    )
    def time_rolling(self, func, center, backend):
        with _backend_options(backend):
            getattr(self.ds.rolling(x=window, center=center), func)().load()

    @parameterized(
        ["func", "center", "backend"],
        (
            ["mean", "count", "max", "argmax", "idxmax"],
            [True, False],
            [*BACKENDS, "pandas"],
        ),
    )
    def time_rolling_long(self, func, center, backend):
        if backend == "pandas":
            rolling = self.da_long.to_series().rolling(
                window=window, center=center, min_periods=1
            )
            _requires_method(rolling, func)
            getattr(rolling, func)()
        else:
            rolling = self.da_long.rolling(x=window, center=center, min_periods=1)
            _requires_method(rolling, func)
            with _backend_options(backend):
                getattr(rolling, func)().load()

    @parameterized(["window_"], ([20, 40],))
    def time_rolling_np(self, window_):
        self.ds.rolling(x=window_, min_periods=5).reduce(np.nansum).load()

    @parameterized(["center"], ([True, False],))
    def time_rolling_construct(self, center):
        self.ds.rolling(x=window, center=center).construct("window_dim").sum(
            dim="window_dim"
        ).load()


class RollingDask(Rolling):
    def setup(self, *args, **kwargs):
        requires_dask()
        super().setup(**kwargs)
        # small chunks make dask's automatic rechunking of the windows explode
        # the number of tasks
        self.ds = self.ds.chunk({"x": 500, "y": 200, "t": 500})
        self.da_long = self.da_long.chunk({"x": 10000})


class Cumulative:
    # cumulative windows span the whole dimension, so the reductions without
    # bottleneck scale quadratically with its size
    def setup(self, *args, **kwargs):
        self.da = xr.DataArray(
            randn((20, nt), frac_nan=0.1),
            dims=("x", "t"),
            coords={"t": pd.date_range("1970-01-01", periods=nt, freq="D")},
        )

    @parameterized(
        ["func", "backend"], (["sum", "mean", "max", "argmax", "idxmax"], BACKENDS)
    )
    def time_cumulative(self, func, backend):
        cumulative = self.da.cumulative("t")
        _requires_method(cumulative, func)
        with _backend_options(backend):
            getattr(cumulative, func)().load()

    def time_cumsum(self):
        # reference for time_cumulative with func="sum"
        self.da.cumsum("t").load()

    @parameterized(["func", "backend"], (["sum", "max", "argmax"], BACKENDS))
    def peakmem_cumulative(self, func, backend):
        with _backend_options(backend):
            getattr(self.da.cumulative("t"), func)().load()


class CumulativeDask(Cumulative):
    def setup(self, *args, **kwargs):
        requires_dask()
        super().setup(**kwargs)
        self.da = self.da.chunk({"t": 250})


class RollingMemory:
    def setup(self, *args, **kwargs):
        self.ds = _create_ds()


class DataArrayRollingMemory(RollingMemory):
    @parameterized(["func"], (["sum", "max", "mean"],))
    def peakmem_ndrolling_reduce(self, func):
        getattr(self.ds.var1.rolling(x=10, y=4), func)()

    @parameterized(["func", "backend"], (["sum", "max", "mean"], BACKENDS))
    def peakmem_1drolling_reduce(self, func, backend):
        with _backend_options(backend):
            getattr(self.ds.var3.rolling(t=100), func)()

    @parameterized(["stride"], ([None, 5, 50],))
    def peakmem_1drolling_construct(self, stride):
        self.ds.var2.rolling(t=100).construct("w", stride=stride)
        self.ds.var3.rolling(t=100).construct("w", stride=stride)


class DatasetRollingMemory(RollingMemory):
    @parameterized(["func"], (["sum", "max", "mean"],))
    def peakmem_ndrolling_reduce(self, func):
        getattr(self.ds.rolling(x=10, y=4), func)()

    @parameterized(["func", "backend"], (["sum", "max", "mean"], BACKENDS))
    def peakmem_1drolling_reduce(self, func, backend):
        with _backend_options(backend):
            getattr(self.ds.rolling(t=100), func)()

    @parameterized(["stride"], ([None, 5, 50],))
    def peakmem_1drolling_construct(self, stride):
        self.ds.rolling(t=100).construct("w", stride=stride)
