from __future__ import absolute_import, division, print_function

import numpy as np
from collections import OrderedDict
from .computation import apply_ufunc


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
    #if method in ['linear', 'nearest']:
    #    return interpolate_1d_local(obj, method, fill_value, kwargs, **coords)
    try:
        import scipy
    except ImportError:
        raise ImportError(
            'Interpolation with method `%s` requires scipy' % method)

    dim, [x, new_x] = list(index_coord.items())[0]

    if method in ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']:
        def interpolator(arr):
            return scipy.interpolate.interp1d(
                x, arr, kind=method, bounds_error=False,
                fill_value=fill_value)(new_x)
    else:
        raise NotImplementedError

    return apply_ufunc(interpolator, obj, input_core_dims=[[dim]],
                       output_core_dims=[[dim]], output_dtypes=[obj.dtype],
                       keep_attrs=True).transpose(*obj.dims)


'''
def interpolate_local(obj, method, bounds_error, fill_value, **coords):
    """ Interpolator for linear and nearestneighbor """
    # Only use necessary region for the efficiency
    for dim, coord in coords.items():
        try:
            min_index = index.get_loc(np.min(coord))
            max_index = index.get_loc(np.max(coord))
            obj = obj.isel(**{dim: slice(np.maximum(min_index-1, 0),
                                         max_index+1)})
        except:  # TODO specify the exception
            pass

    raise NotImplementedError


def interpolate_nd(obj, func, bounds_error, fill_value, **coords):
    """
    Most of this function is stolen from
    https://gist.github.com/crusaderky/b0aa6b8fdf6e036cb364f6f40476cc67
    """
    for dim in coords:
        if len(getattr(obj.chunks, dim, [])) > 1:
            raise ValueError('Chunking along the interpolated dimension ({}) '
                             'is not supported. Given {}.'.format(
                                dim, obj.chunks[dim]))

    def interp_func(x, y, *xnew):
        return func(x, y)(xnew)

    rslt = apply_ufunc(interp_func, obj, *[obj[d] for d in dims]
                       input_core_dims=[dims] + [[d] for d in dims],
                       dask='parallelized')
    # TODO consider coords.dims
    return rslt.transpose(obj.dims)


def _assert_no_chunks_along(obj, dim):
    if len(obj.chunks[dim]) > 1:
        raise ValueError('Chunk along ** is not allowed')
'''
