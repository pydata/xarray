from __future__ import absolute_import, division, print_function

import numpy as np
from .computation import apply_ufunc


def interpolate_dataarray(obj, method='linear', bounds_error=True,
                          fill_value=None, **coords):
    """ Make an interpolation of dataarray
    Parameters
    ----------
    variable: xr.DataArray
    **coords: mapping from dimension name to the new coordinate.

    Returns
    -------
    Interpolated DataArray

    Note
    ----
    The method should
    """
    # a simple speedup for linear interpolator
    if method in ['linear', 'nearest']:
        indexers = {d: _get_used_range for d, c in coords.items()}
        obj = obj.isel(**indexers)

    if len(coords) == 1:
        return interpolate_1d(obj, method, bounds_error, fill_value, **coords)

    if len(set(getattr(c, 'dim', d) for d, c in coord.items)) == len(coords):
        # grid->grid interpolation
        return interpolate_grid(obj, method, bounds_error, fill_value, **coords)

    # sampling-like interpolation
    return interpolate_nd(obj, method, bounds_error, fill_value, **coords)


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
