from __future__ import annotations

import itertools
import warnings
from collections import ChainMap
from collections.abc import Callable, Generator, Hashable, Sequence
from functools import partial
from numbers import Number
from typing import TYPE_CHECKING, Any, Generic, TypeVar, get_args

import numpy as np
import pandas as pd

from xarray.core import utils
from xarray.core.common import _contains_datetime_like_objects
from xarray.core.computation import apply_ufunc
from xarray.core.duck_array_ops import (
    datetime_to_numeric,
    push,
    ravel,
    reshape,
    stack,
    timedelta_to_numeric,
    transpose,
)
from xarray.core.options import _get_keep_attrs
from xarray.core.types import (
    Interp1dOptions,
    InterpnOptions,
    InterpOptions,
    LimitAreaOptions,
    LimitDirectionOptions,
    T_GapLength,
    T_Xarray,
)
from xarray.core.utils import OrderedSet, is_scalar
from xarray.core.variable import (
    Variable,
    broadcast_variables,
)
from xarray.namedarray.pycompat import is_chunked_array

if TYPE_CHECKING:
    InterpCallable = Callable[..., np.ndarray]  # interpn
    Interpolator = Callable[..., Callable[..., np.ndarray]]  # *Interpolator
    # interpolator objects return callables that can be evaluated
    SourceDest = dict[Hashable, tuple[Variable, Variable]]

    T = TypeVar("T")


_FILL_MISSING_DOCSTRING_TEMPLATE = """\
Partly fill nan values in this object's data by applying `{name}` to all unmasked values.

Parameters
----------
**kwargs : dict
    Additional keyword arguments passed on to `{name}`.

Returns
-------
filled : same type as caller
    New object with `{name}` applied to all unmasked values.
"""


def _get_gap_left_edge(
    obj: T_Xarray, dim: Hashable, index: Variable, outside=False
) -> T_Xarray:
    left = index.where(~obj.isnull()).ffill(dim).transpose(*obj.dims)
    if outside:
        return left.fillna(index[0])
    return left


def _get_gap_right_edge(
    obj: T_Xarray, dim: Hashable, index: Variable, outside=False
) -> T_Xarray:
    right = index.where(~obj.isnull()).bfill(dim).transpose(*obj.dims)
    if outside:
        return right.fillna(index[-1])
    return right


def _get_gap_dist_to_left_edge(
    obj: T_Xarray, dim: Hashable, index: Variable
) -> T_Xarray:
    return (index - _get_gap_left_edge(obj, dim, index)).transpose(*obj.dims)


def _get_gap_dist_to_right_edge(
    obj: T_Xarray, dim: Hashable, index: Variable
) -> T_Xarray:
    return (_get_gap_right_edge(obj, dim, index) - index).transpose(*obj.dims)


def _get_limit_fill_mask(
    obj: T_Xarray,
    dim: Hashable,
    index: Variable,
    limit: int | float | np.number,
    limit_direction: LimitDirectionOptions,
) -> T_Xarray:
    # At the left boundary, distance to left is nan.
    # For nan, a<=b and ~(a>b) behave differently
    if limit_direction == "forward":
        limit_mask = ~(_get_gap_dist_to_left_edge(obj, dim, index) <= limit)
    elif limit_direction == "backward":
        limit_mask = ~(_get_gap_dist_to_right_edge(obj, dim, index) <= limit)
    elif limit_direction == "both":
        limit_mask = (~(_get_gap_dist_to_left_edge(obj, dim, index) <= limit)) & (
            ~(_get_gap_dist_to_right_edge(obj, dim, index) <= limit)
        )
    else:
        raise ValueError(
            f"limit_direction must be one of 'forward', 'backward', 'both'. Got {limit_direction}"
        )
    return limit_mask


def _get_limit_area_mask(
    obj: T_Xarray, dim: Hashable, index: Variable, limit_area
) -> T_Xarray:
    if limit_area == "inside":
        area_mask = (
            _get_gap_left_edge(obj, dim, index).isnull()
            | _get_gap_right_edge(obj, dim, index).isnull()
        )
    elif limit_area == "outside":
        area_mask = (
            _get_gap_left_edge(obj, dim, index).notnull()
            & _get_gap_right_edge(obj, dim, index).notnull()
        )
        area_mask = area_mask & obj.isnull()
    else:
        raise ValueError(
            f"limit_area must be one of 'inside', 'outside' or None. Got {limit_area}"
        )
    return area_mask


def _get_nan_block_lengths(obj: T_Xarray, dim: Hashable, index: Variable) -> T_Xarray:
    """
    Return an object where each NaN element in 'obj' is replaced by the
    length of the gap the element is in.
    """
    return _get_gap_right_edge(obj, dim, index, outside=True) - _get_gap_left_edge(
        obj, dim, index, outside=True
    )


def _get_max_gap_mask(
    obj: T_Xarray, dim: Hashable, index: Variable, max_gap: int | float | np.number
) -> T_Xarray:
    nan_block_lengths = _get_nan_block_lengths(obj, dim, index)
    return nan_block_lengths > max_gap


def _get_gap_mask(
    obj: T_Xarray,
    dim: Hashable,
    limit: T_GapLength | None = None,
    limit_direction: LimitDirectionOptions = "both",
    limit_area: LimitAreaOptions | None = None,
    limit_use_coordinate=False,
    max_gap: T_GapLength | None = None,
    max_gap_use_coordinate=False,
) -> T_Xarray | None:
    # Input checking
    ##Limit
    if not is_scalar(limit):
        raise ValueError("limit must be a scalar.")

    if limit is None:
        limit = np.inf
    else:
        if limit_use_coordinate is False:
            if not isinstance(limit, Number | np.number):
                raise TypeError(
                    f"Expected integer or floating point limit since limit_use_coordinate=False. Received {type(limit).__name__}."
                )
        if _is_time_index(_get_raw_interp_index(obj, dim, limit_use_coordinate)):
            limit = timedelta_to_numeric(limit)

    ## Max_gap
    if not is_scalar(max_gap):
        raise ValueError("max_gap must be a scalar.")

    if max_gap is None:
        max_gap = np.inf
    else:
        if not max_gap_use_coordinate:
            if not isinstance(max_gap, Number | np.number):
                raise TypeError(
                    f"Expected integer or floating point max_gap since use_coordinate=False. Received {type(max_gap).__name__}."
                )

        if _is_time_index(_get_raw_interp_index(obj, dim, max_gap_use_coordinate)):
            max_gap = timedelta_to_numeric(max_gap)

    # Which masks are really needed?
    need_limit_mask = limit != np.inf or limit_direction != "both"
    need_area_mask = limit_area is not None
    need_max_gap_mask = max_gap != np.inf
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
    if not (need_limit_mask or need_area_mask or need_max_gap_mask):
        return None

    # Calculate individual masks
    masks = []
    if need_limit_mask:
        # due to the dynamic typing of limit, mypy cannot infer the correct type
        masks.append(
            _get_limit_fill_mask(obj, dim, index_limit, limit, limit_direction)  # type: ignore[arg-type]
        )

    if need_area_mask:
        masks.append(_get_limit_area_mask(obj, dim, index_limit, limit_area))

    if need_max_gap_mask:
        masks.append(_get_max_gap_mask(obj, dim, index_max_gap, max_gap))  # type: ignore[arg-type]
    # Combine masks
    mask = masks[0]
    for m in masks[1:]:
        mask |= m
    return mask


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
        axis=-1,
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

        if method == "quintic":
            method = 5

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
            axis=axis,
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


def _get_raw_interp_index(
    arr: T_Xarray, dim: Hashable, use_coordinate: bool | Hashable = True
) -> pd.Index:
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
    arr: T_Xarray,
    dim: Hashable,
    use_coordinate: bool | Hashable = True,
    strict: bool = True,
) -> Variable:
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
    # TODO: index.name is None for multiindexes
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
    if isinstance(index, CFTimeIndex | pd.DatetimeIndex):
        offset = type(index[0])(1970, 1, 1)
        if isinstance(index, CFTimeIndex):
            values = datetime_to_numeric(
                index.values, offset=offset, datetime_unit="ns"
            )
        else:
            values = datetime_to_numeric(index, offset=offset, datetime_unit="ns")
    else:  # if numeric or standard calendar index: try to cast to float
        try:
            values = index.values.astype(np.float64)
        # raise if index cannot be cast to a float (e.g. MultiIndex)
        except (TypeError, ValueError) as err:
            # pandas raises a TypeError
            # xarray/numpy raise a ValueError
            raise TypeError(
                f"Index {index.name!r} must be castable to float64 to support "
                f"interpolation or curve fitting, got {type(index).__name__}."
            ) from err
    var = Variable([dim], values)
    return var


def _is_time_index(index) -> bool:
    from xarray.coding.cftimeindex import CFTimeIndex

    return isinstance(index, pd.DatetimeIndex | CFTimeIndex)


def _interp_na_all(
    obj: T_Xarray,
    dim: Hashable,
    method: InterpOptions = "linear",
    use_coordinate: bool | Hashable = True,
    keep_attrs: bool | None = None,
    **kwargs,
) -> T_Xarray:
    """Interpolate all nan values, without restrictions regarding the gap size."""
    index = get_clean_interp_index(obj, dim, use_coordinate=use_coordinate)
    interp_class, kwargs = _get_interpolator(method, **kwargs)
    interpolator = partial(func_interpolate_na, interp_class, **kwargs)

    if keep_attrs is None:
        keep_attrs = _get_keep_attrs(default=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "overflow", RuntimeWarning)
        warnings.filterwarnings("ignore", "invalid value", RuntimeWarning)
        arr = apply_ufunc(
            interpolator,
            obj,
            index.values,
            input_core_dims=[[dim], [dim]],
            output_core_dims=[[dim]],
            output_dtypes=[obj.dtype],
            dask="parallelized",
            vectorize=True,
            keep_attrs=keep_attrs,
        ).transpose(*obj.dims)
    return arr


class GapMask(Generic[T_Xarray]):
    """An object that allows for flexible masking of gaps. You should use DataArray.fill_gaps() or Dataset.fill_gaps() to construct this object instead of constructing it directly."""

    # Attributes
    # ----------
    _content: T_Xarray
    _dim: Hashable
    _use_coordinate: bool | Hashable
    _limit: T_GapLength | None
    _limit_direction: LimitDirectionOptions | None
    _limit_area: LimitAreaOptions | None
    _max_gap: T_GapLength | None

    def __init__(
        self,
        content: T_Xarray,
        dim: Hashable,
        use_coordinate: bool | Hashable = True,
        limit: T_GapLength | None = None,
        limit_direction: LimitDirectionOptions | None = None,
        limit_area: LimitAreaOptions | None = None,
        max_gap: T_GapLength | None = None,
    ) -> None:
        """An object that allows for flexible masking of gaps. You should use DataArray.fill_gaps() or Dataset.fill_gaps() to construct this object instead of calling this constructor directly.

        Parameters
        ----------
        content : DataArray or Dataset
            The object to be masked.

        Other:
            See xarray.DataArray.fill_gaps or xarray.Dataset.fill_gaps for an explanation of the remaining parameters.

        See Also
        --------
        xarray.DataArray.fill_gaps
        xarray.Dataset.fill_gaps

        """
        self._content = content
        self._dim = dim
        self._use_coordinate = use_coordinate
        self._limit = limit
        self._limit_direction = limit_direction
        self._limit_area = limit_area
        self._max_gap = max_gap

    def _get_mask(self, limit_direction) -> T_Xarray | None:
        mask = _get_gap_mask(
            obj=self._content,
            dim=self._dim,
            limit=self._limit,
            limit_direction=limit_direction,
            limit_area=self._limit_area,
            limit_use_coordinate=self._use_coordinate,
            max_gap=self._max_gap,
            max_gap_use_coordinate=self._use_coordinate,
        )
        return mask

    def _apply_mask(self, filled: T_Xarray, mask: T_Xarray | None) -> T_Xarray:
        if mask is not None:
            filled = filled.where(~mask, other=self._content)
        return filled

    def get_mask(self) -> T_Xarray | None:
        """Return the gap mask.

        Returns
        -------
        mask : DataArray or Dataset
            Boolean gap mask, created based on the parameters passed to DataArray.fill_gaps() or Dataset.fill_gaps(). True values indicate remaining gaps.
        """
        limit_direction = self._limit_direction
        if limit_direction is None:
            limit_direction = "both"
        mask = self._get_mask(limit_direction)
        return mask

    def ffill(self) -> T_Xarray:
        """Partly fill missing values in this object's data by applying ffill to all unmasked values.

        Parameters
        ----------

        Returns
        -------
        filled : same type as caller
            New object with ffill applied to all unmasked values.

        See Also
        --------
        DataArray.ffill
        Dataset.ffill
        """
        if self._limit_direction is None:
            limit_direction = "forward"
        elif self._limit_direction != "forward":
            raise ValueError(
                f"limit_direction='{self._limit_direction}' is not allowed with ffill, must be 'forward'."
            )
        mask = self._get_mask(limit_direction)
        return self._apply_mask(self._content.ffill(self._dim), mask)

    def bfill(self) -> T_Xarray:
        """Partly fill missing values in this object's data by applying bfill to all unmasked values.

        Returns
        -------
        filled : same type as caller
            New object with bfill applied to all unmasked values.

        See Also
        --------
        DataArray.bfill
        Dataset.bfill
        """
        if self._limit_direction is None:
            limit_direction = "backward"
        elif self._limit_direction != "backward":
            raise ValueError(
                f"limit_direction='{self._limit_direction}' is not allowed with bfill, must be 'backward'."
            )
        mask = self._get_mask(limit_direction)
        return self._apply_mask(self._content.bfill(self._dim), mask)

    def fillna(self, value) -> T_Xarray:
        """Partly fill missing values in this object's data by applying fillna to all unmasked values.

        Parameters
        ----------
        value : scalar, ndarray or DataArray
            Used to fill all unmasked values. If the
            argument is a DataArray, it is first aligned with (reindexed to)
            this array.


        Returns
        -------
        filled : same type as caller
            New object with fillna applied to all unmasked values.

        See Also
        --------
        DataArray.fillna
        Dataset.fillna
        """
        mask = self.get_mask()
        return self._apply_mask(self._content.fillna(value), mask)

    def interpolate_na(
        self,
        dim: Hashable | None = None,
        method: InterpOptions = "linear",
        use_coordinate: bool | Hashable = True,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> T_Xarray:
        """Partly fill missing values in this object's data by applying interpolate_na to all unmasked values.

        Parameters
        ----------
        See DataArray.interpolate_na and Dataset.interpolate_na for explanation of parameters.

        Returns
        -------
        filled : same type as caller
            New object with interpolate_na applied to all unmasked values.


        See Also
        --------
        DataArray.interpolate_na
        Dataset.interpolate_na
        """
        if dim is None:
            dim = self._dim
        mask = self.get_mask()
        return self._apply_mask(
            self._content.interpolate_na(
                dim=dim,
                method=method,
                use_coordinate=use_coordinate,
                limit=None,
                max_gap=None,
                keep_attrs=keep_attrs,
                **kwargs,
            ),
            mask,
        )


def interp_na(
    obj: T_Xarray,
    dim: Hashable,
    method: InterpOptions = "linear",
    use_coordinate: bool | Hashable = True,
    limit: T_GapLength | None = None,
    max_gap: T_GapLength | None = None,
    keep_attrs: bool | None = None,
    **kwargs,
):
    """Interpolate values according to different methods."""
    # This was the original behaviour of interp_na and is kept for backward compatibility
    # Limit=None: Fill everything, including both boundaries
    # Limit!=None: Do forward interpolation until limit
    limit_use_coordinate = False
    limit_direction: LimitDirectionOptions = "both" if limit is None else "forward"
    limit_area = None
    mask = _get_gap_mask(
        obj,
        dim,
        limit,
        limit_direction,
        limit_area,
        limit_use_coordinate,
        max_gap,
        use_coordinate,
    )

    arr = _interp_na_all(obj, dim, method, use_coordinate, keep_attrs, **kwargs)
    if mask is not None:
        arr = arr.where(~mask)
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
    interp_class: Interpolator
    interp1d_methods = get_args(Interp1dOptions)
    valid_methods = tuple(vv for v in get_args(InterpOptions) for vv in get_args(v))

    # prefer numpy.interp for 1d linear interpolation. This function cannot
    # take higher dimensional data but scipy.interp1d can.
    if (
        method == "linear"
        and not kwargs.get("fill_value") == "extrapolate"
        and not vectorizeable_only
    ):
        kwargs.update(method=method)
        interp_class = NumpyInterpolator

    elif method in valid_methods:
        if method in interp1d_methods:
            kwargs.update(method=method)
            interp_class = ScipyInterpolator
        elif method == "barycentric":
            kwargs.update(axis=-1)
            interp_class = _import_interpolant("BarycentricInterpolator", method)
        elif method in ["krogh", "krog"]:
            kwargs.update(axis=-1)
            interp_class = _import_interpolant("KroghInterpolator", method)
        elif method == "pchip":
            kwargs.update(axis=-1)
            # pchip default behavior is to extrapolate
            kwargs.setdefault("extrapolate", False)
            interp_class = _import_interpolant("PchipInterpolator", method)
        elif method == "spline":
            utils.emit_user_level_warning(
                "The 1d SplineInterpolator class is performing an incorrect calculation and "
                "is being deprecated. Please use `method=polynomial` for 1D Spline Interpolation.",
                PendingDeprecationWarning,
            )
            if vectorizeable_only:
                raise ValueError(f"{method} is not a vectorizeable interpolator. ")
            kwargs.update(method=method)
            interp_class = SplineInterpolator
        elif method == "akima":
            kwargs.update(axis=-1)
            interp_class = _import_interpolant("Akima1DInterpolator", method)
        elif method == "makima":
            kwargs.update(method="makima", axis=-1)
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
    valid_methods = tuple(get_args(InterpnOptions))
    if method in valid_methods:
        kwargs.update(method=method)
        kwargs.setdefault("bounds_error", False)
        interp_class = _import_interpolant("interpn", method)
    else:
        raise ValueError(
            f"{method} is not a valid interpolator for interpolating "
            "over multiple dimensions."
        )

    return interp_class, kwargs


def _localize(obj: T, indexes_coords: SourceDest) -> tuple[T, SourceDest]:
    """Speed up for linear and nearest neighbor method.
    Only consider a subspace that is needed for the interpolation
    """
    indexes = {}
    for dim, [x, new_x] in indexes_coords.items():
        if is_chunked_array(new_x._data):
            continue
        new_x_loaded = new_x.data
        minval = np.nanmin(new_x_loaded)
        maxval = np.nanmax(new_x_loaded)
        index = x.to_index()
        imin, imax = index.get_indexer([minval, maxval], method="nearest")
        indexes[dim] = slice(max(imin - 2, 0), imax + 2)
        indexes_coords[dim] = (x[indexes[dim]], new_x)
    return obj.isel(indexes), indexes_coords  # type: ignore[attr-defined]


def _floatize_x(
    x: list[Variable], new_x: list[Variable]
) -> tuple[list[Variable], list[Variable]]:
    """Make x and new_x float.
    This is particularly useful for datetime dtype.
    """
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


def interp(
    var: Variable,
    indexes_coords: SourceDest,
    method: InterpOptions,
    **kwargs,
) -> Variable:
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

    result = var

    if method in ["linear", "nearest", "slinear"]:
        # decompose the interpolation into a succession of independent interpolation.
        iter_indexes_coords = decompose_interp(indexes_coords)
    else:
        iter_indexes_coords = (_ for _ in [indexes_coords])

    for indep_indexes_coords in iter_indexes_coords:
        var = result

        # target dimensions
        dims = list(indep_indexes_coords)

        # transpose to make the interpolated axis to the last position
        broadcast_dims = [d for d in var.dims if d not in dims]
        original_dims = broadcast_dims + dims
        result = interpolate_variable(
            var.transpose(*original_dims),
            {k: indep_indexes_coords[k] for k in dims},
            method=method,
            kwargs=kwargs,
        )

        # dimension of the output array
        out_dims: OrderedSet = OrderedSet()
        for d in var.dims:
            if d in dims:
                out_dims.update(indep_indexes_coords[d][1].dims)
            else:
                out_dims.add(d)
        if len(out_dims) > 1:
            result = result.transpose(*out_dims)
    return result


def interpolate_variable(
    var: Variable,
    indexes_coords: SourceDest,
    *,
    method: InterpOptions,
    kwargs: dict[str, Any],
) -> Variable:
    """core routine that returns the interpolated variable."""
    if not indexes_coords:
        return var.copy()

    if len(indexes_coords) == 1:
        func, kwargs = _get_interpolator(method, vectorizeable_only=True, **kwargs)
    else:
        func, kwargs = _get_interpolator_nd(method, **kwargs)

    in_coords, result_coords = zip(*(v for v in indexes_coords.values()), strict=True)

    # input coordinates along which we are interpolation are core dimensions
    # the corresponding output coordinates may or may not have the same name,
    # so `all_in_core_dims` is also `exclude_dims`
    all_in_core_dims = set(indexes_coords)

    result_dims = OrderedSet(itertools.chain(*(_.dims for _ in result_coords)))
    result_sizes = ChainMap(*(_.sizes for _ in result_coords))

    # any dimensions on the output that are present on the input, but are not being
    # interpolated along are dimensions along which we automatically vectorize.
    # Consider the problem in https://github.com/pydata/xarray/issues/6799#issuecomment-2474126217
    # In the following, dimension names are listed out in [].
    # # da[time, q, lat, lon].interp(q=bar[lat,lon]). Here `lat`, `lon`
    # are input dimensions, present on the output, but are not the coordinates
    # we are explicitly interpolating. These are the dimensions along which we vectorize.
    # `q` is the only input core dimensions, and changes size (disappears)
    # so it is in exclude_dims.
    vectorize_dims = (result_dims - all_in_core_dims) & set(var.dims)

    # remove any output broadcast dimensions from the list of core dimensions
    output_core_dims = tuple(d for d in result_dims if d not in vectorize_dims)
    input_core_dims = (
        # all coordinates on the input that we interpolate along
        [tuple(indexes_coords)]
        # the input coordinates are always 1D at the moment, so we just need to list out their names
        + [tuple(_.dims) for _ in in_coords]
        # The last set of inputs are the coordinates we are interpolating to.
        + [
            tuple(d for d in coord.dims if d not in vectorize_dims)
            for coord in result_coords
        ]
    )
    output_sizes = {k: result_sizes[k] for k in output_core_dims}

    # scipy.interpolate.interp1d always forces to float.
    dtype = float if not issubclass(var.dtype.type, np.inexact) else var.dtype
    result = apply_ufunc(
        _interpnd,
        var,
        *in_coords,
        *result_coords,
        input_core_dims=input_core_dims,
        output_core_dims=[output_core_dims],
        exclude_dims=all_in_core_dims,
        dask="parallelized",
        kwargs=dict(
            interp_func=func,
            interp_kwargs=kwargs,
            # we leave broadcasting up to dask if possible
            # but we need broadcasted values in _interpnd, so propagate that
            # context (dimension names), and broadcast there
            # This would be unnecessary if we could tell apply_ufunc
            # to insert size-1 broadcast dimensions
            result_coord_core_dims=input_core_dims[-len(result_coords) :],
        ),
        # TODO: deprecate and have the user rechunk themselves
        dask_gufunc_kwargs=dict(output_sizes=output_sizes, allow_rechunk=True),
        output_dtypes=[dtype],
        vectorize=bool(vectorize_dims),
        keep_attrs=True,
    )
    return result


def _interp1d(
    var: Variable,
    x_: list[Variable],
    new_x_: list[Variable],
    func: Interpolator,
    kwargs,
) -> np.ndarray:
    """Core 1D array interpolation routine."""
    # x, new_x are tuples of size 1.
    x, new_x = x_[0], new_x_[0]
    rslt = func(x.data, var, **kwargs)(ravel(new_x.data))
    if new_x.ndim > 1:
        return reshape(rslt.data, (var.shape[:-1] + new_x.shape))
    if new_x.ndim == 0:
        return rslt[..., -1]
    return rslt


def _interpnd(
    data: np.ndarray,
    *coords: np.ndarray,
    interp_func: Interpolator | InterpCallable,
    interp_kwargs,
    result_coord_core_dims: list[tuple[Hashable, ...]],
) -> np.ndarray:
    """
    Core nD array interpolation routine.
    The first half arrays in `coords` are original coordinates,
    the other half are destination coordinates.
    """
    n_x = len(coords) // 2
    ndim = data.ndim
    nconst = ndim - n_x

    # Convert everything to Variables, since that makes applying
    # `_localize` and `_floatize_x` much easier
    x = [
        Variable([f"dim_{nconst + dim}"], _x, fastpath=True)
        for dim, _x in enumerate(coords[:n_x])
    ]
    new_x = list(
        broadcast_variables(
            *(
                Variable(dims, _x, fastpath=True)
                for dims, _x in zip(result_coord_core_dims, coords[n_x:], strict=True)
            )
        )
    )
    var = Variable([f"dim_{dim}" for dim in range(ndim)], data, fastpath=True)

    if interp_kwargs.get("method") in ["linear", "nearest"]:
        indexes_coords = {
            _x.dims[0]: (_x, _new_x) for _x, _new_x in zip(x, new_x, strict=True)
        }
        # simple speed up for the local interpolation
        var, indexes_coords = _localize(var, indexes_coords)
        x, new_x = tuple(
            list(_)
            for _ in zip(*(indexes_coords[d] for d in indexes_coords), strict=True)
        )

    x_list, new_x_list = _floatize_x(x, new_x)

    if len(x) == 1:
        # TODO: narrow interp_func to interpolator here
        return _interp1d(var, x_list, new_x_list, interp_func, interp_kwargs)  # type: ignore[arg-type]

    # move the interpolation axes to the start position
    data = transpose(var._data, range(-len(x), var.ndim - len(x)))

    # stack new_x to 1 vector, with reshape
    xi = stack([ravel(x1.data) for x1 in new_x_list], axis=-1)
    rslt: np.ndarray = interp_func(x_list, data, xi, **interp_kwargs)  # type: ignore[assignment]
    # move back the interpolation axes to the last position
    rslt = transpose(rslt, range(-rslt.ndim + 1, 1))
    return reshape(rslt, rslt.shape[:-1] + new_x[0].shape)


def decompose_interp(indexes_coords: SourceDest) -> Generator[SourceDest, None]:
    """Decompose the interpolation into a succession of independent interpolation keeping the order"""

    dest_dims = [
        dest[1].dims if dest[1].ndim > 0 else (dim,)
        for dim, dest in indexes_coords.items()
    ]
    partial_dest_dims: list[tuple[Hashable, ...]] = []
    partial_indexes_coords: SourceDest = {}
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
