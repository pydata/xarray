from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Iterable
from functools import partial

import numpy as np
import pandas as pd


from .computation import apply_ufunc
from .utils import is_scalar


class BaseInterpolator(object):
    '''gerneric interpolator class for normalizing interpolation methods'''
    cons_kwargs = {}
    call_kwargs = {}
    f = None
    kind = None

    def __init__(self, xi, yi, kind=None, **kwargs):
        self.kind = kind
        self.call_kwargs = kwargs

    def __call__(self, x):
        return self.f(x, **self.call_kwargs)

    def __repr__(self):
        return "{type}: kind={kind}".format(type=self.__class__.__name__,
                                            kind=self.kind)


class NumpyInterpolator(BaseInterpolator):
    def __init__(self, xi, yi, kind='linear', fill_value=None, **kwargs):
        self.kind = kind
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
            self._right = yi[-1]
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
    def __init__(self, xi, yi, kind=None, fill_value=None, assume_sorted=True,
                 copy=False, bounds_error=False, **kwargs):
        from scipy.interpolate import interp1d

        if kind is None:
            raise ValueError('kind is a required argument')

        if kind == 'polynomial':
            kind = kwargs.pop('order', None)
            if kind is None:
                raise ValueError('order is required when kind=polynomial')

        self.kind = kind

        self.cons_kwargs = kwargs
        self.call_kwargs = {}

        if fill_value is None and kind == 'linear':
            fill_value = kwargs.pop('fill_value', (np.nan, yi[-1]))
        else:
            fill_value = np.nan

        self.f = interp1d(xi, yi, kind=self.kind, fill_value=fill_value,
                          bounds_error=False, **self.cons_kwargs)


class FromDerivativesInterpolator(BaseInterpolator):
    def __init__(self, xi, yi, kind=None, fill_value=None, **kwargs):
        from scipy.interpolate import BPoly

        if kind is None:
            raise ValueError('kind is a required argument')

        self.kind = kind
        self.cons_kwargs = kwargs

        if fill_value is not None:
            raise ValueError('from_derivatives does not support fill_value')

        self.f = BPoly.from_derivatives(xi, yi, **self.cons_kwargs)


class SplineInterpolator(BaseInterpolator):
    def __init__(self, xi, yi, kind=None, fill_value=None, order=3, **kwargs):
        from scipy.interpolate import UnivariateSpline

        if kind is None:
            raise ValueError('kind is a required argument')

        self.kind = kind
        self.cons_kwargs = kwargs
        self.call_kwargs['nu'] = kwargs.pop('nu', 0)
        self.call_kwargs['ext'] = kwargs.pop('ext', None)

        if fill_value is not None:
            raise ValueError('SplineInterpolator does not support fill_value')

        self.f = UnivariateSpline(xi, yi, k=order, **self.cons_kwargs)


def get_clean_interp_index(arr, dim, use_coordinate=True, **kwargs):
    '''get index to use for x values in interpolation
    '''

    if use_coordinate:
        index = arr.get_index(dim)
        if isinstance(index, pd.DatetimeIndex):
            index = index.values.astype(np.float64)

        # check index sorting now so we can skip it later
        if not (np.diff(index) > 0).all():
            raise ValueError("Index must be monotonicly increasing")

    else:
        axis = arr.get_axis_num(dim)
        index = np.arange(arr.shape[axis])

    return index


def interp_na(self, dim=None, use_coordinate=True, method='linear', limit=None,
              inplace=False, **kwargs):
    '''Interpolate values according to different methods.'''

    arr = self if inplace else self.copy()

    if dim is None:
        raise NotImplementedError('dim is a required argument')

    if limit is not None:
        valids = _get_valid_fill_mask(arr, dim, limit)

    # method
    index = get_clean_interp_index(arr, dim, use_coordinate=use_coordinate,
                                   **kwargs)
    kwargs.update(kind=method)

    interpolator = _get_interpolator(method, **kwargs)

    arr = apply_ufunc(interpolator, index, arr,
                      input_core_dims=[[dim], [dim]],
                      output_core_dims=[[dim]],
                      dask='parallelized',
                      vectorize=True,
                      keep_attrs=True).transpose(*arr.dims)

    if limit is not None:
        arr = arr.where(valids)

    return arr


def wrap_interpolator(interpolator, x, y, **kwargs):
    '''helper function to apply interpolation along 1 dimension'''
    if x.shape != y.shape:
        # this can probably be removed once I get apply_ufuncs to work
        raise AssertionError("x and y shapes do not match "
                             "%s != %s" % (x.shape, y.shape))

    nans = pd.isnull(y)
    nonans = ~nans

    # fast track for no-nans and all-nans cases
    n_nans = nans.sum()
    if n_nans == 0 or n_nans == len(y):
        return y

    f = interpolator(x[nonans], y[nonans], **kwargs)
    return f(x[nans])


def _bfill(arr, n=None, axis=-1):
    '''inverse of ffill'''
    import bottleneck as bn

    arr = np.flip(arr, axis=axis)

    # fill
    arr = bn.push(arr, axis=axis, n=n)

    # reverse back to original
    return np.flip(arr, axis=axis)


def ffill(arr, dim=None, limit=None):
    ''' '''
    import bottleneck as bn

    axis = arr.get_axis_num(dim)

    if limit is not None:
        valids = _get_valid_fill_mask(arr, dim, limit)

    # work around for bottleneck 178
    _limit = limit if limit is not None else arr.shape[axis]

    new = apply_ufunc(bn.push, arr,
                      dask='parallelized',
                      keep_attrs=True,
                      kwargs=dict(n=_limit, axis=axis)).transpose(*arr.dims)

    if limit is not None:
        new = new.where(valids)

    return new


def bfill(arr, dim=None, limit=None):
    ''' '''
    axis = arr.get_axis_num(dim)

    if limit is not None:
        valids = _get_valid_fill_mask(arr, dim, limit)

    # work around for bottleneck 178
    _limit = limit if limit is not None else arr.shape[axis]

    new = apply_ufunc(_bfill, arr,
                      dask='parallelized',
                      keep_attrs=True,
                      kwargs=dict(n=_limit, axis=axis)).transpose(*arr.dims)
    if limit is not None:
        new = new.where(valids)

    return new


def _get_interpolator(method, **kwargs):
    interp1d_methods = ['linear', 'nearest', 'zero', 'slinear', 'quadratic',
                        'cubic', 'polynomial']
    valid_methods = interp1d_methods + ['piecewise_polynomial', 'barycentric',
                                        'krog', 'pchip', 'spline']

    if (method == 'linear' and not
            kwargs.get('fill_value', None) == 'extrapolate'):
        interp_class = NumpyInterpolator
    elif method in valid_methods:
        try:
            from scipy import interpolate
        except ImportError:
            raise ImportError(
                'Interpolation with method `%s` requires scipy' % method)
        if method in interp1d_methods:
            interp_class = ScipyInterpolator
        elif method == 'piecewise_polynomial':
            interp_class = FromDerivativesInterpolator
        elif method == 'barycentric':
            interp_class = interpolate.BarycentricInterpolator
        elif method == 'krog':
            interp_class = interpolate.KroghInterpolator
        elif method == 'pchip':
            interp_class = interpolate.PchipInterpolator
        elif method == 'spline':
            interp_class = SplineInterpolator
        else:
            raise ValueError('%s is not a valid scipy interpolator' % method)
    else:
        raise ValueError('%s is not a valid interpolator' % method)

    return partial(wrap_interpolator, interp_class, **kwargs)


def _get_valid_fill_mask(arr, dim, limit):
    kw = {dim: limit + 1}
    return arr.isnull().rolling(min_periods=1, **kw).sum() <= limit
