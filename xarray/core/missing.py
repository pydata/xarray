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
    interpolator = _get_interpolator(method, **kwargs)

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


def wrap_interpolator(interpolator, x, y, **kwargs):
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

    returns a partial of wrap_interpolator
    '''
    interp1d_methods = ['linear', 'nearest', 'zero', 'slinear', 'quadratic',
                        'cubic', 'polynomial']
    valid_methods = interp1d_methods + ['barycentric', 'krog', 'pchip',
                                        'spline', 'akima']

    if (method == 'linear' and not
            kwargs.get('fill_value', None) == 'extrapolate'):
        kwargs.update(method=method)
        interp_class = NumpyInterpolator
    elif method in valid_methods:
        try:
            from scipy import interpolate
        except ImportError:
            raise ImportError(
                'Interpolation with method `%s` requires scipy' % method)
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
            interp_class = SplineInterpolator
        elif method == 'akima':
            interp_class = interpolate.Akima1DInterpolator
        else:
            raise ValueError('%s is not a valid scipy interpolator' % method)
    else:
        raise ValueError('%s is not a valid interpolator' % method)

    return partial(wrap_interpolator, interp_class, **kwargs)


def _get_valid_fill_mask(arr, dim, limit):
    '''helper function to determine values that can be filled when limit is not
    None'''
    kw = {dim: limit + 1}
    # we explicitly use construct method to avoid copy.
    new_dim = rolling._get_new_dimname(arr.dims, '_window')
    return (arr.isnull().rolling(min_periods=1, **kw)
            .construct(new_dim, fill_value=False)
            .sum(new_dim, skipna=False)) <= limit
