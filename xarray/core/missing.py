import warnings
from collections.abc import Iterable
from functools import partial
from typing import Any, Dict

import numpy as np
import pandas as pd

from . import utils
from .common import _contains_datetime_like_objects
from .computation import apply_ufunc
from .duck_array_ops import dask_array_type, datetime_to_numeric
from .utils import OrderedSet, is_scalar
from .variable import Variable, broadcast_variables


class BaseInterpolator(object):
    '''gerneric interpolator class for normalizing interpolation methods'''
    cons_kwargs = {}  # type: Dict[str, Any]
    call_kwargs = {}  # type: Dict[str, Any]
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
                'received invalid kwargs: %r' % self.cons_kwargs.keys())

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

    for name, var in self.data_vars.items():
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

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'overflow', RuntimeWarning)
        warnings.filterwarnings('ignore', 'invalid value', RuntimeWarning)
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

    arr = np.flip(arr, axis=axis)

    # fill
    arr = bn.push(arr, axis=axis, n=n)

    # reverse back to original
    return np.flip(arr, axis=axis)


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


def _get_interpolator(method, vectorizeable_only=False, **kwargs):
    '''helper function to select the appropriate interpolator class

    returns interpolator class and keyword arguments for the class
    '''
    interp1d_methods = ['linear', 'nearest', 'zero', 'slinear', 'quadratic',
                        'cubic', 'polynomial']
    valid_methods = interp1d_methods + ['barycentric', 'krog', 'pchip',
                                        'spline', 'akima']

    has_scipy = True
    try:
        from scipy import interpolate
    except ImportError:
        has_scipy = False

    # prioritize scipy.interpolate
    if (method == 'linear' and not
            kwargs.get('fill_value', None) == 'extrapolate' and
            not vectorizeable_only):
        kwargs.update(method=method)
        interp_class = NumpyInterpolator

    elif method in valid_methods:
        if not has_scipy:
            raise ImportError(
                'Interpolation with method `%s` requires scipy' % method)

        if method in interp1d_methods:
            kwargs.update(method=method)
            interp_class = ScipyInterpolator
        elif vectorizeable_only:
            raise ValueError('{} is not a vectorizeable interpolator. '
                             'Available methods are {}'.format(
                                 method, interp1d_methods))
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
        raise ValueError('%s is not a valid interpolator for interpolating '
                         'over multiple dimensions.' % method)

    return interp_class, kwargs


def _get_valid_fill_mask(arr, dim, limit):
    '''helper function to determine values that can be filled when limit is not
    None'''
    kw = {dim: limit + 1}
    # we explicitly use construct method to avoid copy.
    new_dim = utils.get_temp_dimname(arr.dims, '_window')
    return (arr.isnull().rolling(min_periods=1, **kw)
            .construct(new_dim, fill_value=False)
            .sum(new_dim, skipna=False)) <= limit


def _assert_single_chunk(var, axes):
    for axis in axes:
        if len(var.chunks[axis]) > 1 or var.chunks[axis][0] < var.shape[axis]:
            raise NotImplementedError(
                'Chunking along the dimension to be interpolated '
                '({}) is not yet supported.'.format(axis))


def _localize(var, indexes_coords):
    """ Speed up for linear and nearest neighbor method.
    Only consider a subspace that is needed for the interpolation
    """
    indexes = {}
    for dim, [x, new_x] in indexes_coords.items():
        index = x.to_index()
        imin = index.get_loc(np.min(new_x.values), method='nearest')
        imax = index.get_loc(np.max(new_x.values), method='nearest')

        indexes[dim] = slice(max(imin - 2, 0), imax + 2)
        indexes_coords[dim] = (x[indexes[dim]], new_x)
    return var.isel(**indexes), indexes_coords


def _floatize_x(x, new_x):
    """ Make x and new_x float.
    This is particulary useful for datetime dtype.
    x, new_x: tuple of np.ndarray
    """
    x = list(x)
    new_x = list(new_x)
    for i in range(len(x)):
        if _contains_datetime_like_objects(x[i]):
            # Scipy casts coordinates to np.float64, which is not accurate
            # enough for datetime64 (uses 64bit integer).
            # We assume that the most of the bits are used to represent the
            # offset (min(x)) and the variation (x - min(x)) can be
            # represented by float.
            xmin = x[i].values.min()
            x[i] = x[i]._to_numeric(offset=xmin, dtype=np.float64)
            new_x[i] = new_x[i]._to_numeric(offset=xmin, dtype=np.float64)
    return x, new_x


def interp(var, indexes_coords, method, **kwargs):
    """ Make an interpolation of Variable

    Parameters
    ----------
    var: Variable
    index_coords:
        Mapping from dimension name to a pair of original and new coordinates.
        Original coordinates should be sorted in strictly ascending order.
        Note that all the coordinates should be Variable objects.
    method: string
        One of {'linear', 'nearest', 'zero', 'slinear', 'quadratic',
        'cubic'}. For multidimensional interpolation, only
        {'linear', 'nearest'} can be used.
    **kwargs:
        keyword arguments to be passed to scipy.interpolate

    Returns
    -------
    Interpolated Variable

    See Also
    --------
    DataArray.interp
    Dataset.interp
    """
    if not indexes_coords:
        return var.copy()

    # simple speed up for the local interpolation
    if method in ['linear', 'nearest']:
        var, indexes_coords = _localize(var, indexes_coords)

    # default behavior
    kwargs['bounds_error'] = kwargs.get('bounds_error', False)

    # target dimensions
    dims = list(indexes_coords)
    x, new_x = zip(*[indexes_coords[d] for d in dims])
    destination = broadcast_variables(*new_x)

    # transpose to make the interpolated axis to the last position
    broadcast_dims = [d for d in var.dims if d not in dims]
    original_dims = broadcast_dims + dims
    new_dims = broadcast_dims + list(destination[0].dims)
    interped = interp_func(var.transpose(*original_dims).data,
                           x, destination, method, kwargs)

    result = Variable(new_dims, interped, attrs=var.attrs)

    # dimension of the output array
    out_dims = OrderedSet()
    for d in var.dims:
        if d in dims:
            out_dims.update(indexes_coords[d][1].dims)
        else:
            out_dims.add(d)
    return result.transpose(*tuple(out_dims))


def interp_func(var, x, new_x, method, kwargs):
    """
    multi-dimensional interpolation for array-like. Interpolated axes should be
    located in the last position.

    Parameters
    ----------
    var: np.ndarray or dask.array.Array
        Array to be interpolated. The final dimension is interpolated.
    x: a list of 1d array.
        Original coordinates. Should not contain NaN.
    new_x: a list of 1d array
        New coordinates. Should not contain NaN.
    method: string
        {'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'} for
        1-dimensional itnterpolation.
        {'linear', 'nearest'} for multidimensional interpolation
    **kwargs:
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
    if not x:
        return var.copy()

    if len(x) == 1:
        func, kwargs = _get_interpolator(method, vectorizeable_only=True,
                                         **kwargs)
    else:
        func, kwargs = _get_interpolator_nd(method, **kwargs)

    if isinstance(var, dask_array_type):
        import dask.array as da

        _assert_single_chunk(var, range(var.ndim - len(x), var.ndim))
        chunks = var.chunks[:-len(x)] + new_x[0].shape
        drop_axis = range(var.ndim - len(x), var.ndim)
        new_axis = range(var.ndim - len(x), var.ndim - len(x) + new_x[0].ndim)
        return da.map_blocks(_interpnd, var, x, new_x, func, kwargs,
                             dtype=var.dtype, chunks=chunks,
                             new_axis=new_axis, drop_axis=drop_axis)

    return _interpnd(var, x, new_x, func, kwargs)


def _interp1d(var, x, new_x, func, kwargs):
    # x, new_x are tuples of size 1.
    x, new_x = x[0], new_x[0]
    rslt = func(x, var, assume_sorted=True, **kwargs)(np.ravel(new_x))
    if new_x.ndim > 1:
        return rslt.reshape(var.shape[:-1] + new_x.shape)
    if new_x.ndim == 0:
        return rslt[..., -1]
    return rslt


def _interpnd(var, x, new_x, func, kwargs):
    x, new_x = _floatize_x(x, new_x)

    if len(x) == 1:
        return _interp1d(var, x, new_x, func, kwargs)

    # move the interpolation axes to the start position
    var = var.transpose(range(-len(x), var.ndim - len(x)))
    # stack new_x to 1 vector, with reshape
    xi = np.stack([x1.values.ravel() for x1 in new_x], axis=-1)
    rslt = func(x, var, xi, **kwargs)
    # move back the interpolation axes to the last position
    rslt = rslt.transpose(range(-rslt.ndim + 1, 1))
    return rslt.reshape(rslt.shape[:-1] + new_x[0].shape)
