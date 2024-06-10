from __future__ import annotations

import datetime as dt
import warnings
from collections.abc import Hashable, Sequence
from functools import partial
from numbers import Number
from typing import TYPE_CHECKING, Any, Callable, get_args

import numpy as np
import pandas as pd

from xarray.core.common import _contains_datetime_like_objects
from xarray.core.computation import apply_ufunc
from xarray.core.duck_array_ops import (
    datetime_to_numeric,
    push,
    reshape,
    timedelta_to_numeric,
)
from xarray.core.options import _get_keep_attrs
from xarray.core.types import (
    Interp1dOptions,
    InterpOptions,
    LimitAreaOptions,
    LimitDirectionOptions,
)
from xarray.core.utils import OrderedSet, is_scalar
from xarray.core.variable import Variable, broadcast_variables
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array

if TYPE_CHECKING:
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset


def _get_gap_left_edge(
    obj: Dataset | DataArray | Variable, dim: Hashable, index: Variable, outside=False
):
    left = index.where(~obj.isnull()).ffill(dim).transpose(*obj.dims)
    if outside:
        return left.fillna(index[0])
    return left


def _get_gap_right_edge(
    obj: Dataset | DataArray | Variable, dim: Hashable, index: Variable, outside=False
):
    right = index.where(~obj.isnull()).bfill(dim).transpose(*obj.dims)
    if outside:
        return right.fillna(index[-1])
    return right


def _get_gap_dist_to_left_edge(
    obj: Dataset | DataArray | Variable, dim: Hashable, index: Variable
):
    return (index - _get_gap_left_edge(obj, dim, index)).transpose(*obj.dims)


def _get_gap_dist_to_right_edge(
    obj: Dataset | DataArray | Variable, dim: Hashable, index: Variable
):
    return (_get_gap_right_edge(obj, dim, index) - index).transpose(*obj.dims)


def _get_limit_fill_mask(
    obj: Dataset | DataArray | Variable,
    dim: Hashable,
    index: Variable,
    limit,
    limit_direction,
):
    if limit_direction == "forward":
        limit_mask = _get_gap_dist_to_left_edge(obj, dim, index) <= limit
    elif limit_direction == "backward":
        limit_mask = _get_gap_dist_to_right_edge(obj, dim, index) <= limit
    elif limit_direction == "both":
        limit_mask = (_get_gap_dist_to_left_edge(obj, dim, index) <= limit) | (
            _get_gap_dist_to_right_edge(obj, dim, index) <= limit
        )
    else:
        raise ValueError(
            f"limit_direction must be one of 'forward', 'backward', 'both'. Got {limit_direction}"
        )
    return limit_mask


def _get_limit_area_mask(
    obj: Dataset | DataArray | Variable, dim: Hashable, index: Variable, limit_area
):
    if limit_area == "inside":
        area_mask = (
            _get_gap_left_edge(obj, dim, index).notnull()
            & _get_gap_right_edge(obj, dim, index).notnull()
        )
        area_mask = area_mask | obj.notnull()
    elif limit_area == "outside":
        area_mask = (
            _get_gap_left_edge(obj, dim, index).isnull()
            | _get_gap_right_edge(obj, dim, index).isnull()
        )
        area_mask = area_mask | obj.notnull()
    else:
        raise ValueError(
            f"limit_area must be one of 'inside', 'outside' or None. Got {limit_area}"
        )
    return area_mask


def _get_nan_block_lengths(
    obj: Dataset | DataArray | Variable, dim: Hashable, index: Variable
):
    """
    Return an object where each NaN element in 'obj' is replaced by the
    length of the gap the element is in.
    """
    return _get_gap_right_edge(obj, dim, index, outside=True) - _get_gap_left_edge(
        obj, dim, index, outside=True
    )


def _get_max_gap_mask(
    obj: Dataset | DataArray | Variable,
    dim: Hashable,
    index: Variable,
    max_gap: int | float | str | pd.Timedelta | np.timedelta64 | dt.timedelta,
):
    nan_block_lengths = _get_nan_block_lengths(obj, dim, index)
    return nan_block_lengths <= max_gap


def _get_gap_masks(
    obj: Dataset | DataArray | Variable,
    dim: Hashable,
    limit=None,
    limit_direction="both",
    limit_area=None,
    limit_use_coordinate=False,
    max_gap=None,
    max_gap_use_coordinate=False,
):
    # Input checking
    ##Limit
    if not is_scalar(limit):
        raise ValueError("limit must be a scalar.")

    if limit is None:
        limit = np.inf
    else:
        if limit_use_coordinate is False:
            if not isinstance(limit, (Number, np.number)):
                raise TypeError(
                    f"Expected integer or floating point limit since limit_use_coordinate=False. Received {type(limit).__name__}."
                )
        if _is_time_index(_get_raw_interp_index(obj, dim, limit_use_coordinate)):
            limit = timedelta_to_numeric(limit)

    ## Max_gap
    if max_gap is not None:
        if not is_scalar(max_gap):
            raise ValueError("max_gap must be a scalar.")

        if _is_time_index(_get_raw_interp_index(obj, dim, max_gap_use_coordinate)):
            max_gap = timedelta_to_numeric(max_gap)

        if not max_gap_use_coordinate:
            if not isinstance(max_gap, (Number, np.number)):
                raise TypeError(
                    f"Expected integer or floating point max_gap since use_coordinate=False. Received {type(max_gap).__name__}."
                )
    # Which masks are really needed?
    need_limit_mask = limit != np.inf or limit_direction != "both"
    need_area_mask = limit_area is not None
    need_max_gap_mask = max_gap is not None
    # Calculate indexes
    if need_limit_mask or need_area_mask:
        index_limit = get_clean_interp_index(
            obj, dim, use_coordinate=limit_use_coordinate
        )
        # index_limit = ones_like(obj) * index_limit
    if need_max_gap_mask:
        index_max_gap = get_clean_interp_index(
            obj, dim, use_coordinate=max_gap_use_coordinate
        )
        # index_max_gap = ones_like(obj) * index_max_gap
    # Calculate fill masks
    limit_mask = None
    if need_limit_mask:
        limit_mask = _get_limit_fill_mask(obj, dim, index_limit, limit, limit_direction)

    limit_area_mask = None
    if need_area_mask:
        limit_area_mask = _get_limit_area_mask(obj, dim, index_limit, limit_area)

    max_gap_mask = None
    if need_max_gap_mask:
        max_gap_mask = _get_max_gap_mask(obj, dim, index_max_gap, max_gap)
    return limit_mask, limit_area_mask, max_gap_mask


class BaseInterpolator:
    """Generic interpolator class for normalizing interpolation methods"""

    cons_kwargs: dict[str, Any]
    call_kwargs: dict[str, Any]
    f: Callable
    method: str

    def __call__(self, x):
        return self.f(x, **self.call_kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}: method={self.method}"


class NumpyInterpolator(BaseInterpolator):
    """One-dimensional linear interpolation.

    See Also
    --------
    numpy.interp
    """

    def __init__(self, xi, yi, method="linear", fill_value=None, period=None):
        if method != "linear":
            raise ValueError("only method `linear` is valid for the NumpyInterpolator")

        self.method = method
        self.f = np.interp
        self.cons_kwargs = {}
        self.call_kwargs = {"period": period}

        self._xi = xi
        self._yi = yi

        nan = np.nan if yi.dtype.kind != "c" else np.nan + np.nan * 1j

        if fill_value is None:
            self._left = nan
            self._right = nan
        elif isinstance(fill_value, Sequence) and len(fill_value) == 2:
            self._left = fill_value[0]
            self._right = fill_value[1]
        elif is_scalar(fill_value):
            self._left = fill_value
            self._right = fill_value
        else:
            raise ValueError(f"{fill_value} is not a valid fill_value")

    def __call__(self, x):
        return self.f(
            x,
            self._xi,
            self._yi,
            left=self._left,
            right=self._right,
            **self.call_kwargs,
        )


class ScipyInterpolator(BaseInterpolator):
    """Interpolate a 1-D function using Scipy interp1d

    See Also
    --------
    scipy.interpolate.interp1d
    """

    def __init__(
        self,
        xi,
        yi,
        method=None,
        fill_value=None,
        assume_sorted=True,
        copy=False,
        bounds_error=False,
        order=None,
        **kwargs,
    ):
        from scipy.interpolate import interp1d

        if method is None:
            raise ValueError(
                "method is a required argument, please supply a "
                "valid scipy.inter1d method (kind)"
            )

        if method == "polynomial":
            if order is None:
                raise ValueError("order is required when method=polynomial")
            method = order

        self.method = method

        self.cons_kwargs = kwargs
        self.call_kwargs = {}

        nan = np.nan if yi.dtype.kind != "c" else np.nan + np.nan * 1j

        if fill_value is None and method == "linear":
            fill_value = nan, nan
        elif fill_value is None:
            fill_value = nan

        self.f = interp1d(
            xi,
            yi,
            kind=self.method,
            fill_value=fill_value,
            bounds_error=bounds_error,
            assume_sorted=assume_sorted,
            copy=copy,
            **self.cons_kwargs,
        )


class SplineInterpolator(BaseInterpolator):
    """One-dimensional smoothing spline fit to a given set of data points.

    See Also
    --------
    scipy.interpolate.UnivariateSpline
    """

    def __init__(
        self,
        xi,
        yi,
        method="spline",
        fill_value=None,
        order=3,
        nu=0,
        ext=None,
        **kwargs,
    ):
        from scipy.interpolate import UnivariateSpline

        if method != "spline":
            raise ValueError("only method `spline` is valid for the SplineInterpolator")

        self.method = method
        self.cons_kwargs = kwargs
        self.call_kwargs = {"nu": nu, "ext": ext}

        if fill_value is not None:
            raise ValueError("SplineInterpolator does not support fill_value")

        self.f = UnivariateSpline(xi, yi, k=order, **self.cons_kwargs)


def _apply_over_vars_with_dim(func, self, dim=None, **kwargs):
    """Wrapper for datasets"""
    ds = type(self)(coords=self.coords, attrs=self.attrs)

    for name, var in self.data_vars.items():
        if dim in var.dims:
            ds[name] = func(var, dim=dim, **kwargs)
        else:
            ds[name] = var

    return ds


def _get_raw_interp_index(arr, dim: Hashable, use_coordinate: bool | Hashable = True):
    """Return index to use for x values in interpolation or curve fitting.
    In comparison to get_clean_interp_index, this function does not convert
    to numeric values."""

    if dim not in arr.dims:
        raise ValueError(f"{dim} is not a valid dimension")

    if use_coordinate is False:
        return pd.RangeIndex(arr.sizes[dim], name=dim)

    elif use_coordinate is True:
        coordinate = arr.coords[
            dim
        ]  # this will default to a linear coordinate, if no index is present
    else:  # string/hashable
        coordinate = arr.coords[use_coordinate]
        if dim not in coordinate.dims:
            raise ValueError(
                f"Coordinate given by {use_coordinate} must have dimension {dim}."
            )

    if coordinate.ndim != 1:
        raise ValueError(
            f"Coordinates used for interpolation must be 1D, "
            f"{use_coordinate} is {coordinate.ndim}D."
        )
    index = coordinate.to_index()
    return index


def get_clean_interp_index(
    arr, dim: Hashable, use_coordinate: bool | Hashable = True, strict: bool = True
):
    """Return index to use for x values in interpolation or curve fitting.

    Parameters
    ----------
    arr : DataArray
        Array to interpolate or fit to a curve.
    dim : str
        Name of dimension along which to fit.
    use_coordinate : bool or hashable
        If use_coordinate is True, the coordinate that shares the name of the
        dimension along which interpolation is being performed will be used as the
        x values. If False, the x values are set as an equally spaced sequence.
    strict : bool
        Whether to raise errors if the index is either non-unique or non-monotonic (default).

    Returns
    -------
    Variable
        Numerical values for the x-coordinates.

    Notes
    -----
    If indexing is along the time dimension, datetime coordinates are converted
    to time deltas with respect to 1970-01-01.
    """

    from xarray.coding.cftimeindex import CFTimeIndex

    index = _get_raw_interp_index(arr, dim, use_coordinate)
    # index.name is None for multiindexes
    # set name for nice error messages below
    if isinstance(index, pd.MultiIndex):
        index.name = dim

    if strict:
        if not index.is_monotonic_increasing:
            raise ValueError(f"Index {index.name!r} must be monotonically increasing")

        if not index.is_unique:
            raise ValueError(f"Index {index.name!r} has duplicate values")

    # Special case for non-standard calendar indexes
    # Numerical datetime values are defined with respect to 1970-01-01T00:00:00 in units of nanoseconds
    if isinstance(index, (CFTimeIndex, pd.DatetimeIndex)):
        offset = type(index[0])(1970, 1, 1)
        if isinstance(index, CFTimeIndex):
            index = index.values
        index = Variable(
            data=datetime_to_numeric(index, offset=offset, datetime_unit="ns"),
            dims=(dim,),
        )

    # raise if index cannot be cast to a float (e.g. MultiIndex)
    try:
        index = index.values.astype(np.float64)
    except (TypeError, ValueError):
        # pandas raises a TypeError
        # xarray/numpy raise a ValueError
        raise TypeError(
            f"Index {index.name!r} must be castable to float64 to support "
            f"interpolation or curve fitting, got {type(index).__name__}."
        )
    index = Variable([dim], index)
    return index


def _is_time_index(index):
    from xarray.coding.cftimeindex import CFTimeIndex

    return isinstance(index, (pd.DatetimeIndex, CFTimeIndex))


def interp_na(
    self,
    dim: Hashable | None = None,
    method: InterpOptions = "linear",
    use_coordinate: bool | str = True,
    limit: (
        int | float | str | pd.Timedelta | np.timedelta64 | dt.timedelta | None
    ) = None,
    limit_direction: LimitDirectionOptions = "forward",
    limit_area: LimitAreaOptions | None = None,
    limit_use_coordinate: (
        bool | str
    ) = False,  # backward compatibility + pandas (2.1.4) compatibility
    max_gap: int | float | str | pd.Timedelta | np.timedelta64 | dt.timedelta = None,
    keep_attrs: bool | None = None,
    **kwargs,
):
    """Interpolate values according to different methods."""

    # Preprocess arguments and do consistency checks
    if dim is None:
        raise NotImplementedError("dim is a required argument")

    masks = _get_gap_masks(
        self,
        dim,
        limit,
        limit_direction,
        limit_area,
        limit_use_coordinate,
        max_gap,
        use_coordinate,
    )

    # method
    index = get_clean_interp_index(self, dim, use_coordinate=use_coordinate)
    interp_class, kwargs = _get_interpolator(method, **kwargs)
    interpolator = partial(func_interpolate_na, interp_class, **kwargs)

    if keep_attrs is None:
        keep_attrs = _get_keep_attrs(default=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "overflow", RuntimeWarning)
        warnings.filterwarnings("ignore", "invalid value", RuntimeWarning)
        arr = apply_ufunc(
            interpolator,
            self,
            index.values,
            input_core_dims=[[dim], [dim]],
            output_core_dims=[[dim]],
            output_dtypes=[self.dtype],
            dask="parallelized",
            vectorize=True,
            keep_attrs=keep_attrs,
        ).transpose(*self.dims)

    for m in masks:
        if m is not None:
            arr = arr.where(m)

    return arr


def func_interpolate_na(interpolator, y, x, **kwargs):
    """helper function to apply interpolation along 1 dimension"""
    # reversed arguments are so that attrs are preserved from da, not index
    # it would be nice if this wasn't necessary, works around:
    # "ValueError: assignment destination is read-only" in assignment below
    out = y.copy()

    nans = pd.isnull(y)
    nonans = ~nans

    # fast track for no-nans, all nan but one, and all-nans cases
    n_nans = nans.sum()
    if n_nans == 0 or n_nans >= len(y) - 1:
        return y

    f = interpolator(x[nonans], y[nonans], **kwargs)
    out[nans] = f(x[nans])
    return out


def _bfill(arr, n=None, axis=-1):
    """inverse of ffill"""
    arr = np.flip(arr, axis=axis)

    # fill
    arr = push(arr, axis=axis, n=n)

    # reverse back to original
    return np.flip(arr, axis=axis)


def ffill(arr, dim=None, limit=None):
    """forward fill missing values"""

    axis = arr.get_axis_num(dim)

    # work around for bottleneck 178
    _limit = limit if limit is not None else arr.shape[axis]

    return apply_ufunc(
        push,
        arr,
        dask="allowed",
        keep_attrs=True,
        output_dtypes=[arr.dtype],
        kwargs=dict(n=_limit, axis=axis),
    ).transpose(*arr.dims)


def bfill(arr, dim=None, limit=None):
    """backfill missing values"""

    axis = arr.get_axis_num(dim)

    # work around for bottleneck 178
    _limit = limit if limit is not None else arr.shape[axis]

    return apply_ufunc(
        _bfill,
        arr,
        dask="allowed",
        keep_attrs=True,
        output_dtypes=[arr.dtype],
        kwargs=dict(n=_limit, axis=axis),
    ).transpose(*arr.dims)


def _import_interpolant(interpolant, method):
    """Import interpolant from scipy.interpolate."""
    try:
        from scipy import interpolate

        return getattr(interpolate, interpolant)
    except ImportError as e:
        raise ImportError(f"Interpolation with method {method} requires scipy.") from e


def _get_interpolator(
    method: InterpOptions, vectorizeable_only: bool = False, **kwargs
):
    """helper function to select the appropriate interpolator class

    returns interpolator class and keyword arguments for the class
    """
    interp_class: (
        type[NumpyInterpolator] | type[ScipyInterpolator] | type[SplineInterpolator]
    )

    interp1d_methods = get_args(Interp1dOptions)
    valid_methods = tuple(vv for v in get_args(InterpOptions) for vv in get_args(v))

    # prioritize scipy.interpolate
    if (
        method == "linear"
        and not kwargs.get("fill_value", None) == "extrapolate"
        and not vectorizeable_only
    ):
        kwargs.update(method=method)
        interp_class = NumpyInterpolator

    elif method in valid_methods:
        if method in interp1d_methods:
            kwargs.update(method=method)
            interp_class = ScipyInterpolator
        elif vectorizeable_only:
            raise ValueError(
                f"{method} is not a vectorizeable interpolator. "
                f"Available methods are {interp1d_methods}"
            )
        elif method == "barycentric":
            interp_class = _import_interpolant("BarycentricInterpolator", method)
        elif method in ["krogh", "krog"]:
            interp_class = _import_interpolant("KroghInterpolator", method)
        elif method == "pchip":
            interp_class = _import_interpolant("PchipInterpolator", method)
        elif method == "spline":
            kwargs.update(method=method)
            interp_class = SplineInterpolator
        elif method == "akima":
            interp_class = _import_interpolant("Akima1DInterpolator", method)
        else:
            raise ValueError(f"{method} is not a valid scipy interpolator")
    else:
        raise ValueError(f"{method} is not a valid interpolator")

    return interp_class, kwargs


def _get_interpolator_nd(method, **kwargs):
    """helper function to select the appropriate interpolator class

    returns interpolator class and keyword arguments for the class
    """
    valid_methods = ["linear", "nearest"]

    if method in valid_methods:
        kwargs.update(method=method)
        interp_class = _import_interpolant("interpn", method)
    else:
        raise ValueError(
            f"{method} is not a valid interpolator for interpolating "
            "over multiple dimensions."
        )

    return interp_class, kwargs


def _localize(var, indexes_coords):
    """Speed up for linear and nearest neighbor method.
    Only consider a subspace that is needed for the interpolation
    """
    indexes = {}
    for dim, [x, new_x] in indexes_coords.items():
        new_x_loaded = new_x.values
        minval = np.nanmin(new_x_loaded)
        maxval = np.nanmax(new_x_loaded)
        index = x.to_index()
        imin, imax = index.get_indexer([minval, maxval], method="nearest")
        indexes[dim] = slice(max(imin - 2, 0), imax + 2)
        indexes_coords[dim] = (x[indexes[dim]], new_x)
    return var.isel(**indexes), indexes_coords


def _floatize_x(x, new_x):
    """Make x and new_x float.
    This is particularly useful for datetime dtype.
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


def interp(var, indexes_coords, method: InterpOptions, **kwargs):
    """Make an interpolation of Variable

    Parameters
    ----------
    var : Variable
    indexes_coords
        Mapping from dimension name to a pair of original and new coordinates.
        Original coordinates should be sorted in strictly ascending order.
        Note that all the coordinates should be Variable objects.
    method : string
        One of {'linear', 'nearest', 'zero', 'slinear', 'quadratic',
        'cubic'}. For multidimensional interpolation, only
        {'linear', 'nearest'} can be used.
    **kwargs
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

    # default behavior
    kwargs["bounds_error"] = kwargs.get("bounds_error", False)

    result = var
    # decompose the interpolation into a succession of independent interpolation
    for indexes_coords in decompose_interp(indexes_coords):
        var = result

        # target dimensions
        dims = list(indexes_coords)
        x, new_x = zip(*[indexes_coords[d] for d in dims])
        destination = broadcast_variables(*new_x)

        # transpose to make the interpolated axis to the last position
        broadcast_dims = [d for d in var.dims if d not in dims]
        original_dims = broadcast_dims + dims
        new_dims = broadcast_dims + list(destination[0].dims)
        interped = interp_func(
            var.transpose(*original_dims).data, x, destination, method, kwargs
        )

        result = Variable(new_dims, interped, attrs=var.attrs, fastpath=True)

        # dimension of the output array
        out_dims: OrderedSet = OrderedSet()
        for d in var.dims:
            if d in dims:
                out_dims.update(indexes_coords[d][1].dims)
            else:
                out_dims.add(d)
        if len(out_dims) > 1:
            result = result.transpose(*out_dims)
    return result


def interp_func(var, x, new_x, method: InterpOptions, kwargs):
    """
    multi-dimensional interpolation for array-like. Interpolated axes should be
    located in the last position.

    Parameters
    ----------
    var : np.ndarray or dask.array.Array
        Array to be interpolated. The final dimension is interpolated.
    x : a list of 1d array.
        Original coordinates. Should not contain NaN.
    new_x : a list of 1d array
        New coordinates. Should not contain NaN.
    method : string
        {'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'} for
        1-dimensional interpolation.
        {'linear', 'nearest'} for multidimensional interpolation
    **kwargs
        Optional keyword arguments to be passed to scipy.interpolator

    Returns
    -------
    interpolated: array
        Interpolated array

    Notes
    -----
    This requires scipy installed.

    See Also
    --------
    scipy.interpolate.interp1d
    """
    if not x:
        return var.copy()

    if len(x) == 1:
        func, kwargs = _get_interpolator(method, vectorizeable_only=True, **kwargs)
    else:
        func, kwargs = _get_interpolator_nd(method, **kwargs)

    if is_chunked_array(var):
        chunkmanager = get_chunked_array_type(var)

        ndim = var.ndim
        nconst = ndim - len(x)

        out_ind = list(range(nconst)) + list(range(ndim, ndim + new_x[0].ndim))

        # blockwise args format
        x_arginds = [[_x, (nconst + index,)] for index, _x in enumerate(x)]
        x_arginds = [item for pair in x_arginds for item in pair]
        new_x_arginds = [
            [_x, [ndim + index for index in range(_x.ndim)]] for _x in new_x
        ]
        new_x_arginds = [item for pair in new_x_arginds for item in pair]

        args = (var, range(ndim), *x_arginds, *new_x_arginds)

        _, rechunked = chunkmanager.unify_chunks(*args)

        args = tuple(elem for pair in zip(rechunked, args[1::2]) for elem in pair)

        new_x = rechunked[1 + (len(rechunked) - 1) // 2 :]

        new_x0_chunks = new_x[0].chunks
        new_x0_shape = new_x[0].shape
        new_x0_chunks_is_not_none = new_x0_chunks is not None
        new_axes = {
            ndim + i: new_x0_chunks[i] if new_x0_chunks_is_not_none else new_x0_shape[i]
            for i in range(new_x[0].ndim)
        }

        # if useful, reuse localize for each chunk of new_x
        localize = (method in ["linear", "nearest"]) and new_x0_chunks_is_not_none

        # scipy.interpolate.interp1d always forces to float.
        # Use the same check for blockwise as well:
        if not issubclass(var.dtype.type, np.inexact):
            dtype = float
        else:
            dtype = var.dtype

        meta = var._meta

        return chunkmanager.blockwise(
            _chunked_aware_interpnd,
            out_ind,
            *args,
            interp_func=func,
            interp_kwargs=kwargs,
            localize=localize,
            concatenate=True,
            dtype=dtype,
            new_axes=new_axes,
            meta=meta,
            align_arrays=False,
        )

    return _interpnd(var, x, new_x, func, kwargs)


def _interp1d(var, x, new_x, func, kwargs):
    # x, new_x are tuples of size 1.
    x, new_x = x[0], new_x[0]
    rslt = func(x, var, assume_sorted=True, **kwargs)(np.ravel(new_x))
    if new_x.ndim > 1:
        return reshape(rslt, (var.shape[:-1] + new_x.shape))
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
    return reshape(rslt, rslt.shape[:-1] + new_x[0].shape)


def _chunked_aware_interpnd(var, *coords, interp_func, interp_kwargs, localize=True):
    """Wrapper for `_interpnd` through `blockwise` for chunked arrays.

    The first half arrays in `coords` are original coordinates,
    the other half are destination coordinates
    """
    n_x = len(coords) // 2
    nconst = len(var.shape) - n_x

    # _interpnd expect coords to be Variables
    x = [Variable([f"dim_{nconst + dim}"], _x) for dim, _x in enumerate(coords[:n_x])]
    new_x = [
        Variable([f"dim_{len(var.shape) + dim}" for dim in range(len(_x.shape))], _x)
        for _x in coords[n_x:]
    ]

    if localize:
        # _localize expect var to be a Variable
        var = Variable([f"dim_{dim}" for dim in range(len(var.shape))], var)

        indexes_coords = {_x.dims[0]: (_x, _new_x) for _x, _new_x in zip(x, new_x)}

        # simple speed up for the local interpolation
        var, indexes_coords = _localize(var, indexes_coords)
        x, new_x = zip(*[indexes_coords[d] for d in indexes_coords])

        # put var back as a ndarray
        var = var.data

    return _interpnd(var, x, new_x, interp_func, interp_kwargs)


def decompose_interp(indexes_coords):
    """Decompose the interpolation into a succession of independent interpolation keeping the order"""

    dest_dims = [
        dest[1].dims if dest[1].ndim > 0 else [dim]
        for dim, dest in indexes_coords.items()
    ]
    partial_dest_dims = []
    partial_indexes_coords = {}
    for i, index_coords in enumerate(indexes_coords.items()):
        partial_indexes_coords.update([index_coords])

        if i == len(dest_dims) - 1:
            break

        partial_dest_dims += [dest_dims[i]]
        other_dims = dest_dims[i + 1 :]

        s_partial_dest_dims = {dim for dims in partial_dest_dims for dim in dims}
        s_other_dims = {dim for dims in other_dims for dim in dims}

        if not s_partial_dest_dims.intersection(s_other_dims):
            # this interpolation is orthogonal to the rest

            yield partial_indexes_coords

            partial_dest_dims = []
            partial_indexes_coords = {}

    yield partial_indexes_coords
