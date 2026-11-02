import inspect
import itertools
import os

import numpy as np
from asv_runner.benchmarks.mark import skip_benchmark

_counter = itertools.count()


def requires_dask():
    try:
        import dask  # noqa: F401
    except ImportError as err:
        raise NotImplementedError() from err


def requires_zarr():
    try:
        import zarr  # noqa: F401
    except ImportError as err:
        raise NotImplementedError() from err


def requires_sparse():
    try:
        import sparse  # noqa: F401
    except ImportError as err:
        raise NotImplementedError() from err


def randn(shape, frac_nan=None, chunks=None, seed=0):
    rng = np.random.default_rng(seed)
    if chunks is None:
        x = rng.standard_normal(shape)
    else:
        import dask.array as da

        rng = da.random.default_rng(seed)
        x = rng.standard_normal(shape, chunks=chunks)

    if frac_nan is not None:
        inds = rng.choice(range(x.size), int(x.size * frac_nan))
        x.flat[inds] = np.nan

    return x


def randint(low, high=None, size=None, frac_minus=None, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(low, high, size)
    if frac_minus is not None:
        inds = rng.choice(range(x.size), int(x.size * frac_minus))
        x.flat[inds] = -1

    return x


_BENCHMARK_PREFIXES = ("time_", "timeraw_", "mem_", "peakmem_", "track_")


def skip_slow[T](obj: T) -> T:
    """Skip a slow or highly demanding benchmark, or all benchmarks of a class.

    The benchmarks are skipped if the ``ASV_SKIP_SLOW`` environment variable is
    set to ``1`` (as in CI).

    asv only looks for ``skip_benchmark`` on the benchmark methods themselves, so
    for a class the (possibly inherited) benchmark methods are wrapped on the
    class itself, leaving any parent class untouched.

    Examples
    --------
    >>> from . import skip_slow
    >>> @skip_slow
    ... def time_something_slow():
    ...     pass
    ...
    """
    if os.environ.get("ASV_SKIP_SLOW", "0") != "1":
        return obj
    if not inspect.isclass(obj):
        return skip_benchmark(obj)
    for name, member in inspect.getmembers(obj, inspect.isfunction):
        if name.startswith(_BENCHMARK_PREFIXES):
            setattr(obj, name, skip_benchmark(member))
    return obj
