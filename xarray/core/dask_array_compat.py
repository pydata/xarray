import warnings

import numpy as np

from .pycompat import dask_version

try:
    import dask.array as da
except ImportError:
    da = None


def _validate_pad_output_shape(input_shape, pad_width, output_shape):
    """Validates the output shape of dask.array.pad, raising a RuntimeError if they do not match.
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
        and all(isint(i) for p in pad_width for i in p)
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


if dask_version > "2.30.0":
    ensure_minimum_chunksize = da.overlap.ensure_minimum_chunksize
else:

    # copied from dask
    def ensure_minimum_chunksize(size, chunks):
        """Determine new chunks to ensure that every chunk >= size

        Parameters
        ----------
        size : int
            The maximum size of any chunk.
        chunks : tuple
            Chunks along one axis, e.g. ``(3, 3, 2)``

        Examples
        --------
        >>> ensure_minimum_chunksize(10, (20, 20, 1))
        (20, 11, 10)
        >>> ensure_minimum_chunksize(3, (1, 1, 3))
        (5,)

        See Also
        --------
        overlap
        """
        if size <= min(chunks):
            return chunks

        # add too-small chunks to chunks before them
        output = []
        new = 0
        for c in chunks:
            if c < size:
                if new > size + (size - c):
                    output.append(new - (size - c))
                    new = size
                else:
                    new += c
            if new >= size:
                output.append(new)
                new = 0
            if c >= size:
                new += c
        if new >= size:
            output.append(new)
        elif len(output) >= 1:
            output[-1] += new
        else:
            raise ValueError(
                f"The overlapping depth {size} is larger than your "
                f"array {sum(chunks)}."
            )

        return tuple(output)


if dask_version > "2021.03.0":
    sliding_window_view = da.lib.stride_tricks.sliding_window_view
else:

    def sliding_window_view(x, window_shape, axis=None):
        from dask.array.overlap import map_overlap
        from numpy.core.numeric import normalize_axis_tuple

        from .npcompat import sliding_window_view as _np_sliding_window_view

        window_shape = (
            tuple(window_shape) if np.iterable(window_shape) else (window_shape,)
        )

        window_shape_array = np.array(window_shape)
        if np.any(window_shape_array <= 0):
            raise ValueError("`window_shape` must contain positive values")

        if axis is None:
            axis = tuple(range(x.ndim))
            if len(window_shape) != len(axis):
                raise ValueError(
                    f"Since axis is `None`, must provide "
                    f"window_shape for all dimensions of `x`; "
                    f"got {len(window_shape)} window_shape elements "
                    f"and `x.ndim` is {x.ndim}."
                )
        else:
            axis = normalize_axis_tuple(axis, x.ndim, allow_duplicate=True)
            if len(window_shape) != len(axis):
                raise ValueError(
                    f"Must provide matching length window_shape and "
                    f"axis; got {len(window_shape)} window_shape "
                    f"elements and {len(axis)} axes elements."
                )

        depths = [0] * x.ndim
        for ax, window in zip(axis, window_shape):
            depths[ax] += window - 1

        # Ensure that each chunk is big enough to leave at least a size-1 chunk
        # after windowing (this is only really necessary for the last chunk).
        safe_chunks = tuple(
            ensure_minimum_chunksize(d + 1, c) for d, c in zip(depths, x.chunks)
        )
        x = x.rechunk(safe_chunks)

        # result.shape = x_shape_trimmed + window_shape,
        # where x_shape_trimmed is x.shape with every entry
        # reduced by one less than the corresponding window size.
        # trim chunks to match x_shape_trimmed
        newchunks = tuple(
            c[:-1] + (c[-1] - d,) for d, c in zip(depths, x.chunks)
        ) + tuple((window,) for window in window_shape)

        kwargs = dict(
            depth=tuple((0, d) for d in depths),  # Overlap on +ve side only
            boundary="none",
            meta=x._meta,
            new_axis=range(x.ndim, x.ndim + len(axis)),
            chunks=newchunks,
            trim=False,
            window_shape=window_shape,
            axis=axis,
        )
        # map_overlap's signature changed in https://github.com/dask/dask/pull/6165
        if dask_version > "2.18.0":
            return map_overlap(_np_sliding_window_view, x, align_arrays=False, **kwargs)
        else:
            return map_overlap(x, _np_sliding_window_view, **kwargs)
