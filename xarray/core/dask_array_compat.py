import warnings
from distutils.version import LooseVersion
from typing import Iterable

import numpy as np

from .pycompat import dask_array_type

try:
    import dask.array as da
    from dask import __version__ as dask_version
except ImportError:
    dask_version = "0.0.0"
    da = None

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
        if hasattr(x, "_meta") and isinstance(x, dask_array_type):
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


def _validate_pad_output_shape(input_shape, pad_width, output_shape):
    """ Validates the output shape of dask.array.pad, raising a RuntimeError if they do not match.
    In the current versions of dask (2.2/2.4), dask.array.pad with mode='reflect' sometimes returns
    an invalid shape.
    """
    isint = lambda i: isinstance(i, int)

    if isint(pad_width):
        pass
    elif len(pad_width) == 2 and all(map(isint, pad_width)):
        pad_width = sum(pad_width)
    elif (
        len(pad_width) == len(input_shape)
        and all(map(lambda x: len(x) == 2, pad_width))
        and all((isint(i) for p in pad_width for i in p))
    ):
        pad_width = np.sum(pad_width, axis=1)
    else:
        # unreachable: dask.array.pad should already have thrown an error
        raise ValueError("Invalid value for `pad_width`")

    if not np.array_equal(np.array(input_shape) + pad_width, output_shape):
        raise RuntimeError(
            "There seems to be something wrong with the shape of the output of dask.array.pad, "
            "try upgrading Dask, use a different pad mode e.g. mode='constant' or first convert "
            "your DataArray/Dataset to one backed by a numpy array by calling the `compute()` method."
            "See: https://github.com/dask/dask/issues/5303"
        )


def pad(array, pad_width, mode="constant", **kwargs):
    padded = da.pad(array, pad_width, mode=mode, **kwargs)
    # workaround for inconsistency between numpy and dask: https://github.com/dask/dask/issues/5303
    if mode == "mean" and issubclass(array.dtype.type, np.integer):
        warnings.warn(
            'dask.array.pad(mode="mean") converts integers to floats. xarray converts '
            "these floats back to integers to keep the interface consistent. There is a chance that "
            "this introduces rounding errors. If you wish to keep the values as floats, first change "
            "the dtype to a float before calling pad.",
            UserWarning,
        )
        return da.round(padded).astype(array.dtype)
    _validate_pad_output_shape(array.shape, pad_width, padded.shape)
    return padded


if LooseVersion(dask_version) >= LooseVersion("2.8.1"):
    median = da.median
else:
    # Copied from dask v2.8.1
    # Used under the terms of Dask's license, see licenses/DASK_LICENSE.
    def median(a, axis=None, keepdims=False):
        """
        This works by automatically chunking the reduced axes to a single chunk
        and then calling ``numpy.median`` function across the remaining dimensions
        """

        if axis is None:
            raise NotImplementedError(
                "The da.median function only works along an axis.  "
                "The full algorithm is difficult to do in parallel"
            )

        if not isinstance(axis, Iterable):
            axis = (axis,)

        axis = [ax + a.ndim if ax < 0 else ax for ax in axis]

        a = a.rechunk({ax: -1 if ax in axis else "auto" for ax in range(a.ndim)})

        result = a.map_blocks(
            np.median,
            axis=axis,
            keepdims=keepdims,
            drop_axis=axis if not keepdims else None,
            chunks=[1 if ax in axis else c for ax, c in enumerate(a.chunks)]
            if keepdims
            else None,
        )

        return result


if LooseVersion(dask_version) > LooseVersion("2.9.0"):
    nanmedian = da.nanmedian
else:

    def nanmedian(a, axis=None, keepdims=False):
        """
        This works by automatically chunking the reduced axes to a single chunk
        and then calling ``numpy.nanmedian`` function across the remaining dimensions
        """

        if axis is None:
            raise NotImplementedError(
                "The da.nanmedian function only works along an axis.  "
                "The full algorithm is difficult to do in parallel"
            )

        if not isinstance(axis, Iterable):
            axis = (axis,)

        axis = [ax + a.ndim if ax < 0 else ax for ax in axis]

        a = a.rechunk({ax: -1 if ax in axis else "auto" for ax in range(a.ndim)})

        result = a.map_blocks(
            np.nanmedian,
            axis=axis,
            keepdims=keepdims,
            drop_axis=axis if not keepdims else None,
            chunks=[1 if ax in axis else c for ax, c in enumerate(a.chunks)]
            if keepdims
            else None,
        )

        return result
