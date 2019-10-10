from distutils.version import LooseVersion

import dask.array as da
import numpy as np
from dask import __version__ as dask_version

if LooseVersion(dask_version) >= LooseVersion("2.0.0"):
    meta_from_array = da.utils.meta_from_array
else:
    # Copied from dask v2.4.0
    # Used under the terms of Dask's license, see licenses/DASK_LICENSE.
    import numbers

    def meta_from_array(x, ndim=None, dtype=None):
        """ Normalize an array to appropriate meta object

        Parameters
        ----------
        x: array-like, callable
        Either an object that looks sufficiently like a Numpy array,
        or a callable that accepts shape and dtype keywords
        ndim: int
        Number of dimensions of the array
        dtype: Numpy dtype
        A valid input for ``np.dtype``

        Returns
        -------
        array-like with zero elements of the correct dtype
        """
        # If using x._meta, x must be a Dask Array, some libraries (e.g. zarr)
        # implement a _meta attribute that are incompatible with Dask Array._meta
        if hasattr(x, "_meta") and isinstance(x, da.Array):
            x = x._meta

        if dtype is None and x is None:
            raise ValueError("You must specify the meta or dtype of the array")

        if np.isscalar(x):
            x = np.array(x)

        if x is None:
            x = np.ndarray

        if isinstance(x, type):
            x = x(shape=(0,) * (ndim or 0), dtype=dtype)

        if (
            not hasattr(x, "shape")
            or not hasattr(x, "dtype")
            or not isinstance(x.shape, tuple)
        ):
            return x

        if isinstance(x, list) or isinstance(x, tuple):
            ndims = [
                0
                if isinstance(a, numbers.Number)
                else a.ndim
                if hasattr(a, "ndim")
                else len(a)
                for a in x
            ]
            a = [a if nd == 0 else meta_from_array(a, nd) for a, nd in zip(x, ndims)]
            return a if isinstance(x, list) else tuple(x)

        if ndim is None:
            ndim = x.ndim

        try:
            meta = x[tuple(slice(0, 0, None) for _ in range(x.ndim))]
            if meta.ndim != ndim:
                if ndim > x.ndim:
                    meta = meta[
                        (Ellipsis,) + tuple(None for _ in range(ndim - meta.ndim))
                    ]
                    meta = meta[tuple(slice(0, 0, None) for _ in range(meta.ndim))]
                elif ndim == 0:
                    meta = meta.sum()
                else:
                    meta = meta.reshape((0,) * ndim)
        except Exception:
            meta = np.empty((0,) * ndim, dtype=dtype or x.dtype)

        if np.isscalar(meta):
            meta = np.array(meta)

        if dtype and meta.dtype != dtype:
            meta = meta.astype(dtype)

        return meta
