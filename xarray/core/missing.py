from __future__ import absolute_import, division, print_function

from collections import Iterable
from functools import partial

import numpy as np
import pandas as pd

from . import rolling
from .computation import apply_ufunc
from .npcompat import flip
from .pycompat import iteritems
from .utils import is_scalar
from .variable import broadcast_variables
from .duck_array_ops import dask_array_type


class BaseInterpolator(object):
    '''gerneric interpolator class for normalizing interpolation methods'''
    cons_kwargs = {}
    call_kwargs = {}
    f = None
    method = None

    def __init__(self, xi, yi, method=None, **kwargs):
        self.method = method
        self.call_kwargs = kwargs

    def __call__(self, x):
        return self.f(x, **self.call_kwargs)

    def __repr__(self):
        return "{type}: method={method}".format(type=self.__class__.__name__,
                                                method=self.method)


class NumpyInterpolator(BaseInterpolator):
    '''One-dimensional linear interpolation.

    See Also
    --------
    numpy.interp
    '''

    def __init__(self, xi, yi, method='linear', fill_value=None, **kwargs):

        if method != 'linear':
            raise ValueError(
                'only method `linear` is valid for the NumpyInterpolator')

        self.method = method
        self.f = np.interp
        self.cons_kwargs = kwargs
        self.call_kwargs = {'period': self.cons_kwargs.pop('period', None)}

        self._xi = xi
        self._yi = yi

        if self.cons_kwargs:
            raise ValueError(
                'recieved invalid kwargs: %r' % self.cons_kwargs.keys())

        if fill_value is None:
            self._left = np.nan
            self._right = np.nan
        elif isinstance(fill_value, Iterable) and len(fill_value) == 2:
            self._left = fill_value[0]
            self._right = fill_value[1]
        elif is_scalar(fill_value):
            self._left = fill_value
            self._right = fill_value
        else:
            raise ValueError('%s is not a valid fill_value' % fill_value)

    def __call__(self, x):
        return self.f(x, self._xi, self._yi, left=self._left,
                      right=self._right, **self.call_kwargs)


class ScipyInterpolator(BaseInterpolator):
    '''Interpolate a 1-D function using Scipy interp1d

    See Also
    --------
    scipy.interpolate.interp1d
    '''

    def __init__(self, xi, yi, method=None, fill_value=None,
                 assume_sorted=True, copy=False, bounds_error=False, **kwargs):
        from scipy.interpolate import interp1d

        if method is None:
            raise ValueError('method is a required argument, please supply a '
                             'valid scipy.inter1d method (kind)')

        if method == 'polynomial':
            method = kwargs.pop('order', None)
            if method is None:
                raise ValueError('order is required when method=polynomial')

        self.method = method

        self.cons_kwargs = kwargs
        self.call_kwargs = {}

        if fill_value is None and method == 'linear':
            fill_value = kwargs.pop('fill_value', (np.nan, np.nan))
        elif fill_value is None:
            fill_value = np.nan

        self.f = interp1d(xi, yi, kind=self.method, fill_value=fill_value,
                          bounds_error=False, assume_sorted=assume_sorted,
                          copy=copy, **self.cons_kwargs)


class SplineInterpolator(BaseInterpolator):
    '''One-dimensional smoothing spline fit to a given set of data points.

    See Also
    --------
    scipy.interpolate.UnivariateSpline
    '''

    def __init__(self, xi, yi, method='spline', fill_value=None, order=3,
                 **kwargs):
        from scipy.interpolate import UnivariateSpline

        if method != 'spline':
            raise ValueError(
                'only method `spline` is valid for the SplineInterpolator')

        self.method = method
        self.cons_kwargs = kwargs
        self.call_kwargs['nu'] = kwargs.pop('nu', 0)
        self.call_kwargs['ext'] = kwargs.pop('ext', None)

        if fill_value is not None:
            raise ValueError('SplineInterpolator does not support fill_value')

        self.f = UnivariateSpline(xi, yi, k=order, **self.cons_kwargs)


def _apply_over_vars_with_dim(func, self, dim=None, **kwargs):
    '''wrapper for datasets'''

    ds = type(self)(coords=self.coords, attrs=self.attrs)

    for name, var in iteritems(self.data_vars):
        if dim in var.dims:
            ds[name] = func(var, dim=dim, **kwargs)
        else:
            ds[name] = var

    return ds


def get_clean_interp_index(arr, dim, use_coordinate=True, **kwargs):
    '''get index to use for x values in interpolation.

    If use_coordinate is True, the coordinate that shares the name of the
    dimension along which interpolation is being performed will be used as the
    x values.

    If use_coordinate is False, the x values are set as an equally spaced
    sequence.
    '''
    if use_coordinate:
        if use_coordinate is True:
            index = arr.get_index(dim)
        else:
            index = arr.coords[use_coordinate]
            if index.ndim != 1:
                raise ValueError(
                    'Coordinates used for interpolation must be 1D, '
                    '%s is %dD.' % (use_coordinate, index.ndim))

        # raise if index cannot be cast to a float (e.g. MultiIndex)
        try:
            index = index.values.astype(np.float64)
        except (TypeError, ValueError):
            # pandas raises a TypeError
            # xarray/nuppy raise a ValueError
            raise TypeError('Index must be castable to float64 to support'
                            'interpolation, got: %s' % type(index))
        # check index sorting now so we can skip it later
        if not (np.diff(index) > 0).all():
            raise ValueError("Index must be monotonicly increasing")
    else:
        axis = arr.get_axis_num(dim)
        index = np.arange(arr.shape[axis], dtype=np.float64)

    return index


def interp_na(self, dim=None, use_coordinate=True, method='linear', limit=None,
              **kwargs):
    '''Interpolate values according to different methods.'''

    if dim is None:
        raise NotImplementedError('dim is a required argument')

    if limit is not None:
        valids = _get_valid_fill_mask(self, dim, limit)

    # method
    index = get_clean_interp_index(self, dim, use_coordinate=use_coordinate,
                                   **kwargs)
    interp_class, kwargs = _get_interpolator(method, **kwargs)
    interpolator = partial(func_interpolate_na, interp_class, **kwargs)

    arr = apply_ufunc(interpolator, index, self,
                      input_core_dims=[[dim], [dim]],
                      output_core_dims=[[dim]],
                      output_dtypes=[self.dtype],
                      dask='parallelized',
                      vectorize=True,
                      keep_attrs=True).transpose(*self.dims)

    if limit is not None:
        arr = arr.where(valids)

    return arr


def func_interpolate_na(interpolator, x, y, **kwargs):
    '''helper function to apply interpolation along 1 dimension'''
    # it would be nice if this wasn't necessary, works around:
    # "ValueError: assignment destination is read-only" in assignment below
    out = y.copy()

    nans = pd.isnull(y)
    nonans = ~nans

    # fast track for no-nans and all-nans cases
    n_nans = nans.sum()
    if n_nans == 0 or n_nans == len(y):
        return y

    f = interpolator(x[nonans], y[nonans], **kwargs)
    out[nans] = f(x[nans])
    return out


def _bfill(arr, n=None, axis=-1):
    '''inverse of ffill'''
    import bottleneck as bn

    arr = flip(arr, axis=axis)

    # fill
    arr = bn.push(arr, axis=axis, n=n)

    # reverse back to original
    return flip(arr, axis=axis)


def ffill(arr, dim=None, limit=None):
    '''forward fill missing values'''
    import bottleneck as bn

    axis = arr.get_axis_num(dim)

    # work around for bottleneck 178
    _limit = limit if limit is not None else arr.shape[axis]

    return apply_ufunc(bn.push, arr,
                       dask='parallelized',
                       keep_attrs=True,
                       output_dtypes=[arr.dtype],
                       kwargs=dict(n=_limit, axis=axis)).transpose(*arr.dims)


def bfill(arr, dim=None, limit=None):
    '''backfill missing values'''
    axis = arr.get_axis_num(dim)

    # work around for bottleneck 178
    _limit = limit if limit is not None else arr.shape[axis]

    return apply_ufunc(_bfill, arr,
                       dask='parallelized',
                       keep_attrs=True,
                       output_dtypes=[arr.dtype],
                       kwargs=dict(n=_limit, axis=axis)).transpose(*arr.dims)


def _get_interpolator(method, **kwargs):
    '''helper function to select the appropriate interpolator class

    returns interpolator class and keyword arguments for the class
    '''
    interp1d_methods = ['linear', 'nearest', 'zero', 'slinear', 'quadratic',
                        'cubic', 'polynomial']
    valid_methods = interp1d_methods + ['barycentric', 'krog', 'pchip',
                                        'spline', 'akima']

    try:
        from scipy import interpolate
    except ImportError:
        # scipy.interpolate should prior
        if (method == 'linear' and not
                kwargs.get('fill_value', None) == 'extrapolate'):
            kwargs.update(method=method)
            return NumpyInterpolator, kwargs

        raise ImportError(
            'Interpolation with method `%s` requires scipy' % method)

    if method in valid_methods:
        if method in interp1d_methods:
            kwargs.update(method=method)
            interp_class = ScipyInterpolator
        elif method == 'barycentric':
            interp_class = interpolate.BarycentricInterpolator
        elif method == 'krog':
            interp_class = interpolate.KroghInterpolator
        elif method == 'pchip':
            interp_class = interpolate.PchipInterpolator
        elif method == 'spline':
            kwargs.update(method=method)
            if 'fill_value' in kwargs:
                del kwargs['fill_value']
            interp_class = SplineInterpolator
        elif method == 'akima':
            interp_class = interpolate.Akima1DInterpolator
        else:
            raise ValueError('%s is not a valid scipy interpolator' % method)
    else:
        raise ValueError('%s is not a valid interpolator' % method)

    return interp_class, kwargs


def _get_interpolator_nd(method, **kwargs):
    '''helper function to select the appropriate interpolator class

    returns interpolator class and keyword arguments for the class
    '''
    valid_methods = ['linear', 'nearest']

    try:
        from scipy import interpolate
    except ImportError:
        raise ImportError(
            'Interpolation with method `%s` requires scipy' % method)

    if method in valid_methods:
        kwargs.update(method=method)
        interp_class = interpolate.interpn
    else:
        raise ValueError('%s is not a valid interpolator' % method)

    return interp_class, kwargs


def _get_valid_fill_mask(arr, dim, limit):
    '''helper function to determine values that can be filled when limit is not
    None'''
    kw = {dim: limit + 1}
    # we explicitly use construct method to avoid copy.
    new_dim = rolling._get_new_dimname(arr.dims, '_window')
    return (arr.isnull().rolling(min_periods=1, **kw)
            .construct(new_dim, fill_value=False)
            .sum(new_dim, skipna=False)) <= limit


def _assert_single_chunk(obj, axes):
    for axis in axes:
        if len(obj.chunks[axis]) > 1 or obj.chunks[axis][0] < obj.shape[axis]:
            raise ValueError('Chunk along the dimension to be interpolated '
                             '({}) is not allowed.'.format(axis))


def _localize(obj, indexes_coords):
    """ Speed up for linear and nearest neighbor method.
    Only consider a subspace that is needed for the interpolation
    """
    for dim, [x, new_x] in indexes_coords.items():
        try:
            imin = x.to_index().get_loc(np.min(new_x.values), method='nearest')
            imax = x.to_index().get_loc(np.max(new_x.values), method='nearest')

            idx = slice(np.maximum(imin - 2, 0), imax + 2)
            indexes_coords[dim] = (x[idx], new_x)
            obj = obj.isel(**{dim: idx})
        except ValueError:  # if index is not sorted.
            pass
    return obj, indexes_coords


def interp(obj, indexes_coords, method, **kwargs):
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
    if len(indexes_coords) == 0:
        return obj

    # simple speed up for the local interpolation
    if method in ['linear', 'nearest']:
        obj, indexes_coords = _localize(obj, indexes_coords)

    if method not in ['linear', 'nearest', 'zero', 'slinear', 'quadratic',
                      'cubic', 'polynomial']:
        raise ValueError('{} is not a valid method.'.format(method))

    # default behavior
    kwargs['bounds_error'] = kwargs.get('bounds_error', False)

    # target dimensions
    dims = list(indexes_coords)
    x = [indexes_coords[d][0] for d in dims]
    new_x = [indexes_coords[d][1] for d in dims]
    destination = broadcast_variables(*new_x)

    rslt = apply_ufunc(interp_func, obj,
                       input_core_dims=[dims],
                       output_core_dims=[destination[0].dims],
                       output_dtypes=[obj.dtype], dask='allowed',
                       kwargs={'x': x, 'new_x': destination, 'method': method,
                               'kwargs': kwargs},
                       keep_attrs=True)

    if all(x1.dims == new_x1.dims for x1, new_x1 in zip(x, new_x)):
        return rslt.transpose(*obj.dims)
    return rslt


def interp_func(obj, x, new_x, method, kwargs):
    """
    multi-dimensional interpolation for array-like.

    Parameters
    ----------
    obj: np.ndarray or dask.array.Array
        Array to be interpolated. The final dimension is interpolated.
    x: a list of 1d array.
        Original coordinates. Should not contain NaN.
    new_x: 1d array
        Original coordinates. Should not contain NaN.
    method: string
        {'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'} for
        1-dimensional itnterpolation.
        {'linear', 'nearest'} for multidimensional interpolation
    kwargs:
        Optional keyword arguments to be passed to scipy.interpolator

    Returns
    -------
    interpolated: array
        Interpolated array

    Note
    ----
    This requiers scipy installed.

    See Also
    --------
    scipy.interpolate.interp1d
    """
    if len(x) == 0:
        return obj

    if isinstance(obj, dask_array_type):
        import dask.array as da

        _assert_single_chunk(obj, range(obj.ndim-len(x), obj.ndim))
        chunks = obj.chunks[:-len(x)] + new_x[0].shape
        drop_axis = range(obj.ndim - len(x), obj.ndim)
        new_axis = range(obj.ndim - len(x), obj.ndim - len(x) + new_x[0].ndim)
        # call this function recursively
        return da.map_blocks(interp_func, obj, x, new_x, method, kwargs,
                             dtype=obj.dtype, chunks=chunks,
                             new_axis=new_axis, drop_axis=drop_axis)
    if len(x) == 1:
        func, kwargs = _get_interpolator(method, **kwargs)
        return _interp1d(obj, x, new_x, func, kwargs)

    func, kwargs = _get_interpolator_nd(method, **kwargs)
    return _interpnd(obj, x, new_x, func, kwargs)


def _interp1d(obj, x, new_x, func, kwargs):
    # x, new_x are tuples of size 1.
    x, new_x = x[0], new_x[0]
    rslt = func(x, obj, **kwargs)(np.ravel(new_x))
    if new_x.ndim > 1:
        return rslt.reshape(obj.shape[:-1] + new_x.shape)
    if new_x.ndim == 0:
        return rslt[..., -1]
    return rslt


def _interpnd(obj, x, new_x, func, kwargs):
    # move the interpolation axes to the start position
    obj = obj.transpose(range(-len(x), obj.ndim - len(x)))
    # stack new_x to 1 vector, with reshape
    xi = np.stack([x1.values.ravel() for x1 in new_x], axis=-1)
    rslt = func(x, obj, xi, **kwargs)
    # move back the interpolation axes to the last position
    rslt = rslt.transpose(range(-rslt.ndim + 1, 1))
    return rslt.reshape(rslt.shape[:-1] + new_x[0].shape)
