from __future__ import absolute_import, division, print_function
from functools import partial

import numpy as np
from .computation import apply_ufunc
from .pycompat import dask_array_type
from .variable import broadcast_variables


def _localize(obj, index_coord):
    """ Speed up for linear and nearest neighbor method.
    Only consider a subspace that is needed for the interpolation
    """
    for dim, [x, new_x] in index_coord.items():
        try:
            imin = x.to_index().get_loc(np.min(new_x), method='ffill')
            imax = x.to_index().get_loc(np.max(new_x), method='bfill')

            idx = slice(np.maximum(imin - 1, 0), imax + 1)
            index_coord[dim] = (x[idx], new_x)
            obj = obj.isel(**{dim: idx})
        except ValueError:  # if index is not sorted.
            pass
    return obj, index_coord


def interpolate(obj, indexes_coords, method, fill_value, kwargs):
    """ Make an interpolation of Variable

    Parameters
    ----------
    obj: Variable
    index_coord:
        mapping from dimension name to a pair of original and new coordinates.
    method: string
        One of {'linear', 'nearest', 'zero', 'slinear', 'quadratic',
        'cubic'}. For multidimensional interpolation, only
        {'linear', 'nearest'} can be used.
    fill_value:
        fill value for extrapolation
    kwargs:
        keyword arguments to be passed to scipy.interpolate

    Returns
    -------
    Interpolated Variable
    """
    try:
        import scipy.interpolate
    except ImportError:
        raise ImportError(
            'Interpolation with method `%s` requires scipy' % method)

    if len(indexes_coords) == 0:
        return obj

    # simple speed up for the local interpolation
    if method in ['linear', 'nearest']:
        obj, indexes_coords = _localize(obj, indexes_coords)

    # target dimensions
    dims = list(indexes_coords)
    x = [indexes_coords[d][0] for d in dims]
    new_x = [indexes_coords[d][1] for d in dims]
    destination = broadcast_variables(*new_x)

    if len(indexes_coords) == 1:
        if method in ['linear', 'nearest', 'zero', 'slinear', 'quadratic',
                      'cubic']:
            func = partial(scipy.interpolate.interp1d, kind=method, axis=-1,
                           bounds_error=False, fill_value=fill_value)
        else:
            raise NotImplementedError

        rslt = apply_ufunc(_interpolate_1d, obj,
                           input_core_dims=[dims],
                           output_core_dims=[destination[0].dims],
                           output_dtypes=[obj.dtype], dask='allowed',
                           kwargs={'x': x, 'new_x': destination, 'func': func},
                           keep_attrs=True)
    else:
        if method in ['linear', 'nearest']:
            func = partial(scipy.interpolate.RegularGridInterpolator,
                           method=method, bounds_error=False,
                           fill_value=fill_value)
        else:
            raise NotImplementedError

        rslt = apply_ufunc(_interpolate_nd, obj,
                           input_core_dims=[dims],
                           output_core_dims=[destination[0].dims],
                           output_dtypes=[obj.dtype], dask='allowed',
                           kwargs={'x': x, 'new_x': destination, 'func': func},
                           keep_attrs=True)
    if all(x1.dims == new_x1.dims for x1, new_x1 in zip(x, new_x)):
        return rslt.transpose(*obj.dims)
    return rslt


def _interpolate_1d(obj, x, new_x, func):
    if isinstance(obj, dask_array_type):
        import dask.array as da

        _assert_single_chunks(obj, [-1])
        chunks = obj.chunks[:-len(x)] + new_x[0].shape
        drop_axis = range(obj.ndim - len(x), obj.ndim)
        new_axis = range(obj.ndim - len(x), obj.ndim - len(x) + new_x[0].ndim)
        # call this function recursively
        return da.map_blocks(_interpolate_1d, obj, x, new_x, func,
                             dtype=obj.dtype, chunks=chunks,
                             new_axis=new_axis, drop_axis=drop_axis)

    # x, new_x are tuples of size 1.
    x, new_x = x[0], new_x[0]
    rslt = func(x, obj)(np.ravel(new_x))
    if new_x.ndim > 1:
        return rslt.reshape(obj.shape[:-1] + new_x.shape)
    if new_x.ndim == 0:
        return rslt[..., -1]
    return rslt


def _interpolate_nd(obj, x, new_x, func):
    """ dask compatible interpolation function.
    The last len(x) dimensions are used for the interpolation
    """
    if isinstance(obj, dask_array_type):
        import dask.array as da

        _assert_single_chunks(obj, range(-len(x), 0))
        chunks = obj.chunks[:-len(x)] + new_x[0].shape
        drop_axis = range(obj.ndim - len(x), obj.ndim)
        new_axis = range(obj.ndim - len(x), obj.ndim - len(x) + new_x[0].ndim)
        return da.map_blocks(_interpolate_nd, obj, x, new_x, func,
                             dtype=obj.dtype, chunks=chunks,
                             new_axis=new_axis, drop_axis=drop_axis)

    # move the interpolation axes to the start position
    obj = obj.transpose(range(-len(x), obj.ndim - len(x)))
    # stack new_x to 1 vector, with reshape
    xi = np.stack([x1.values.ravel() for x1 in new_x], axis=-1)
    rslt = func(x, obj)(xi)
    # move back the interpolation axes to the last position
    rslt = rslt.transpose(range(-rslt.ndim + 1, 1))
    return rslt.reshape(rslt.shape[:-1] + new_x[0].shape)


def _assert_single_chunks(obj, axes):
    for axis in axes:
        if len(obj.chunks[axis]) > 1:
            raise ValueError('Chunk along the dimension to be interpolated '
                             '({}) is not allowed.'.format(axis))
