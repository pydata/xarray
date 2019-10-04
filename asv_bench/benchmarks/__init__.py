import itertools

import numpy as np

_counter = itertools.count()


def parameterized(names, params):
    def decorator(func):
        func.param_names = names
        func.params = params
        return func

    return decorator


def requires_dask():
    try:
        import dask  # noqa: F401
    except ImportError:
        raise NotImplementedError()


def randn(shape, frac_nan=None, chunks=None, seed=0):
    rng = np.random.RandomState(seed)
    if chunks is None:
        x = rng.standard_normal(shape)
    else:
        import dask.array as da

        rng = da.random.RandomState(seed)
        x = rng.standard_normal(shape, chunks=chunks)

    if frac_nan is not None:
        inds = rng.choice(range(x.size), int(x.size * frac_nan))
        x.flat[inds] = np.nan

    return x


def randint(low, high=None, size=None, frac_minus=None, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.randint(low, high, size)
    if frac_minus is not None:
        inds = rng.choice(range(x.size), int(x.size * frac_minus))
        x.flat[inds] = -1

    return x
