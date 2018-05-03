from __future__ import absolute_import, division, print_function
from functools import partial

import numpy as np
from .computation import apply_ufunc
from .pycompat import (OrderedDict, dask_array_type)


def _localize(obj, index_coord):
    """ Speed up for linear and nearest neighbor method.
    Only consider a subspace that is needed for the interpolation
    """
    for dim, [x, new_x] in index_coord.items():
        try:
            imin = x.to_index().get_loc(np.min(new_x), method='ffill')
            imax = x.to_index().get_loc(np.max(new_x), method='bfill')

            idx = slice(np.maximum(imin-1, 0), imax+1)
            index_coord[dim] = (x[idx], new_x)
            obj = obj.isel(**{dim: idx})
        except:
            pass
    return obj, index_coord


def interpolate(obj, indexes_coords, method, fill_value, kwargs):
    if len(indexes_coords) == 0:
        return obj
    if len(indexes_coords) == 1:
        return interpolate_1d(obj, indexes_coords, method, fill_value, kwargs)


def interpolate_1d(obj, index_coord, method, fill_value, kwargs):
    """ Make an interpolation of Variable

    Parameters
    ----------
    obj: Variable
    index_coord:
        mapping from dimension name to a pair of original and new coordinates.
    method: string or callable similar to scipy.interpolate
    fill_value:
        fill value if extrapolate
    kwargs:
        keyword arguments that are passed to scipy.interpolate

    Returns
    -------
    Interpolated Variable
    """
    try:
        import scipy.interpolate
    except ImportError:
        raise ImportError(
            'Interpolation with method `%s` requires scipy' % method)

    # simple speed up
    if method in ['linear', 'nearest']:
        obj, index_coord = _localize(obj, index_coord)

    dim, [x, new_x] = list(index_coord.items())[0]

    if method in ['linear', 'nearest', 'zero', 'slinear', 'quadratic',
                  'cubic']:
        func = partial(scipy.interpolate.interp1d, kind=method, axis=-1,
                       bounds_error=False, fill_value=fill_value)
    else:
        raise NotImplementedError

    rslt = apply_ufunc(_interpolate1d_func, obj, input_core_dims=[[dim]],
                       output_core_dims=[new_x.dims],
                       output_dtypes=[obj.dtype], dask='allowed',
                       kwargs={'x':x, 'new_x': new_x, 'func': func},
                       keep_attrs=True)
    if x.dims == new_x.dims:
        return rslt.transpose(*obj.dims)
    return rslt


def _interpolate1d_func(obj, x, new_x, func):
    if isinstance(obj, dask_array_type):
        import dask.array as da

        _assert_single_chunks(obj, [-1])
        chunks = obj.chunks[:-1] + (len(new_x), )
        return da.map_blocks(_interpolate1d_func, obj, x, new_x, func,
                             dtype=obj.dtype, chunks=chunks)

    if len(new_x.dims) > 1:
        rslt = func(x, obj)(np.ravel(new_x))
        return rslt.reshape(obj.shape[:-1] + new_x.shape)
    return func(x, obj)(new_x)


def _assert_single_chunks(obj, axes):
    for axis in axes:
        if len(obj.chunks[axis]) > 1:
            raise ValueError('Chunk along the dimension to be interpolated '
                             '({}) is not allowed.'.format(axis))
