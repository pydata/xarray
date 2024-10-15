from __future__ import annotations

import datetime as dt
import warnings
from collections.abc import Callable, Generator, Hashable, Mapping, Sequence
from functools import partial
from numbers import Number
from typing import TYPE_CHECKING, Any, get_args

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from xarray.core import utils
from xarray.core.common import _contains_datetime_like_objects, ones_like
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
    ScalarOrArray,
    T_DuckArray,
    T_Xarray,
)
from xarray.core.utils import OrderedSet, is_scalar
from xarray.core.variable import Variable, broadcast_variables
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array

if TYPE_CHECKING:
    from typing import TypeVar

    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset

    # T_DuckArray can't be a function parameter because covariant
    XarrayLike = TypeVar("XarrayLike", Variable, DataArray, Dataset)


def _get_nan_block_lengths(
    obj: Dataset | DataArray, dim: Hashable, index: T_DuckArray | ArrayLike
) -> Any:
    """
    Return an object where each NaN element in 'obj' is replaced by the
    length of the gap the element is in.
    """

    # make variable so that we get broadcasting for free
    index = Variable([dim], index)

    # algorithm from https://github.com/pydata/xarray/pull/3302#discussion_r324707072
    arange = ones_like(obj) * index
    valid = obj.notnull()
    valid_arange = arange.where(valid)
    cumulative_nans = valid_arange.ffill(dim=dim).fillna(index[0])

    nan_block_lengths = (
        cumulative_nans.diff(dim=dim, label="upper")
        .reindex({dim: obj[dim]})
        .where(valid)
        .bfill(dim=dim)
        .where(~valid, 0)
        .fillna(index[-1] - valid_arange.max(dim=[dim]))
    )

    return nan_block_lengths


class BaseInterpolator:
    """Generic interpolator class for normalizing interpolation methods"""

    cons_kwargs: dict[str, Any]
    call_kwargs: dict[str, Any]
    f: Callable
    method: str | int

    def __call__(self, x: ScalarOrArray) -> ArrayLike:  # dask array will return np
        return self.f(x, **self.call_kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: method={self.method}"


class NumpyInterpolator(BaseInterpolator):
    """One-dimensional linear interpolation.

    See Also
    --------
    numpy.interp
    """

    def __init__(
        self,
        xi: T_DuckArray,  # passed to np.asarray
        yi: T_DuckArray,  # passed to np.asarray + requires dtype attribute
        method: str | None = "linear",
        fill_value: float | complex | None = None,
        period: float | None = None,
    ):
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
            self._left = fill_value  # type: ignore[assignment]
            self._right = fill_value  # type: ignore[assignment]
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
        xi: T_DuckArray,
        yi: T_DuckArray,
        method: str | int | None = None,
        fill_value: float | complex | None = None,
        assume_sorted: bool = True,
        copy: bool = False,
        bounds_error: bool = False,
        order: int | None = None,
        axis: int = -1,
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

        self.method: str | int = method

        self.cons_kwargs = kwargs
        self.call_kwargs = {}

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
        xi: T_DuckArray,
        yi: T_DuckArray,
        method: str | int | None = "spline",
        fill_value: float | complex | None = None,
        order: int = 3,
        nu: float | None = 0,
        ext: int | str | None = None,
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


def _apply_over_vars_with_dim(
    func: Callable, self: Dataset, dim: Hashable | None = None, **kwargs
) -> Dataset:
    """Wrapper for datasets"""
    ds = type(self)(coords=self.coords, attrs=self.attrs)

    for name, var in self.data_vars.items():
        if dim in var.dims:
            ds[name] = func(var, dim=dim, **kwargs)
        else:
            ds[name] = var

    return ds


def get_clean_interp_index(
    arr: T_Xarray,
    dim: Hashable,
    use_coordinate: Hashable | bool = True,
    strict: bool = True,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Return index to use for x values in interpolation or curve fitting.

    Parameters
    ----------
    arr : DataArray or Dataset
        Array to interpolate or fit to a curve.
    dim : str
        Name of dimension along which to fit.
    use_coordinate : str or bool
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

    # Question: If use_coordinate is a string, what role does `dim` play?
    from xarray.coding.cftimeindex import CFTimeIndex

    if use_coordinate is False:
        axis = arr.get_axis_num(dim)
        return np.arange(arr.shape[axis], dtype=np.float64)

    if use_coordinate is True:
        index = arr.get_index(dim)

    else:  # string
        index = arr.coords[use_coordinate]  # type: ignore[assignment]
        if index.ndim != 1:
            raise ValueError(
                f"Coordinates used for interpolation must be 1D, "
                f"{use_coordinate} is {index.ndim}D."
            )
        index = index.to_index()  # type: ignore[attr-defined]

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
            index = index.values  # type: ignore[assignment]
        index = Variable(  # type: ignore[assignment]
            data=datetime_to_numeric(index, offset=offset, datetime_unit="ns"),
            dims=(dim,),
        )

    # raise if index cannot be cast to a float (e.g. MultiIndex)
    try:
        # this step ensures output is ndarray
        index = index.values.astype(np.float64)  # type: ignore[assignment]
    except (TypeError, ValueError) as err:
        # pandas raises a TypeError
        # xarray/numpy raise a ValueError
        raise TypeError(
            f"Index {index.name!r} must be castable to float64 to support "
            f"interpolation or curve fitting, got {type(index).__name__}."
        ) from err

    return index  # type: ignore[return-value]


def interp_na(
    self: T_Xarray,
    dim: Hashable | None = None,
    use_coordinate: bool | str = True,
    method: InterpOptions = "linear",
    limit: int | None = None,
    max_gap: (
        int | float | str | pd.Timedelta | np.timedelta64 | dt.timedelta | None
    ) = None,
    keep_attrs: bool | None = None,
    **kwargs,
) -> T_Xarray:
    """Interpolate values according to different methods."""
    from xarray.coding.cftimeindex import CFTimeIndex

    if dim is None:
        raise NotImplementedError("dim is a required argument")

    if limit is not None:
        valids = _get_valid_fill_mask(self, dim, limit)

    if max_gap is not None:
        max_type = type(max_gap).__name__
        if not is_scalar(max_gap):
            raise ValueError("max_gap must be a scalar.")

        if (
            dim in self._indexes
            and isinstance(
                self._indexes[dim].to_pandas_index(), pd.DatetimeIndex | CFTimeIndex
            )
            and use_coordinate
        ):
            # Convert to float
            max_gap = timedelta_to_numeric(max_gap)

        if not use_coordinate:
            if not isinstance(max_gap, Number | np.number):
                raise TypeError(
                    f"Expected integer or floating point max_gap since use_coordinate=False. Received {max_type}."
                )

    # method
    index = get_clean_interp_index(self, dim, use_coordinate=use_coordinate)
    interp_class: type[BaseInterpolator]
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
            index,
            input_core_dims=[[dim], [dim]],
            output_core_dims=[[dim]],
            output_dtypes=[self.dtype],
            dask="parallelized",
            vectorize=True,
            keep_attrs=keep_attrs,
        ).transpose(*self.dims)

    if limit is not None:
        arr = arr.where(valids)

    if max_gap is not None:
        if dim not in self.coords:
            raise NotImplementedError(
                "max_gap not implemented for unlabeled coordinates yet."
            )
        nan_block_lengths = _get_nan_block_lengths(self, dim, index)
        arr = arr.where(nan_block_lengths <= max_gap)

    return arr


def func_interpolate_na(
    interpolator: Callable, y: T_Xarray, x: np.ndarray, **kwargs
) -> T_Xarray:
    """helper function to apply interpolation along 1 dimension"""
    # reversed arguments are so that attrs are preserved from da, not index
    # it would be nice if this wasn't necessary, works around:
    # "ValueError: assignment destination is read-only" in assignment below
    out = y.copy()

    nans = pd.isnull(y)  # type: ignore[call-overload]
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


def ffill(
    arr: T_Xarray, dim: Hashable | None = None, limit: int | None = None
) -> T_Xarray:
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


def bfill(
    arr: T_Xarray, dim: Hashable | None = None, limit: int | None = None
) -> T_Xarray:
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


def _import_interpolant(interpolant: str, method: InterpOptions):
    """Import interpolant from scipy.interpolate."""
    try:
        from scipy import interpolate

        return getattr(interpolate, interpolant)
    except ImportError as e:
        raise ImportError(f"Interpolation with method {method} requires scipy.") from e


def _get_interpolator(
    method: InterpOptions, vectorizeable_only: bool = False, **kwargs
) -> tuple[type[BaseInterpolator], dict[str, Any]]:
    """helper function to select the appropriate interpolator class

    returns interpolator class and keyword arguments for the class
    """
    interp1d_methods = get_args(Interp1dOptions)
    valid_methods = tuple(vv for v in get_args(InterpOptions) for vv in get_args(v))

    interp_class: type[BaseInterpolator]

    # prefer numpy.interp for 1d linear interpolation. This function cannot
    # take higher dimensional data but scipy.interp1d can.
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
        elif method == "barycentric":
            kwargs.update(axis=-1)
            interp_class = _import_interpolant("BarycentricInterpolator", method)
        elif method in ["krogh", "krog"]:
            kwargs.update(axis=-1)
            interp_class = _import_interpolant("KroghInterpolator", method)
        elif method == "pchip":
            kwargs.update(axis=-1)
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
        elif method == "makima":
            kwargs.update(method="makima", axis=-1)
            interp_class = _import_interpolant("Akima1DInterpolator", method)
        else:
            raise ValueError(f"{method} is not a valid scipy interpolator")
    else:
        raise ValueError(f"{method} is not a valid interpolator")

    return interp_class, kwargs


def _get_interpolator_nd(
    method: InterpOptions, **kwargs
) -> tuple[type[BaseInterpolator], dict[str, Any]]:
    """helper function to select the appropriate interpolator class

    returns interpolator class and keyword arguments for the class
    """
    valid_methods = ["linear", "nearest"]

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


def _get_valid_fill_mask(arr: T_Xarray, dim: Hashable, limit: int) -> T_Xarray:
    """helper function to determine values that can be filled when limit is not
    None"""
    kw = {dim: limit + 1}
    # we explicitly use construct method to avoid copy.
    new_dim = utils.get_temp_dimname(arr.dims, "_window")
    return (
        arr.isnull()
        .rolling(
            kw,
            min_periods=1,
        )
        .construct(new_dim, fill_value=False)
        .sum(dim=new_dim, skipna=False)  # type: ignore[arg-type]
    ) <= limit


def _localize(
    var: Variable,
    indexes_coords: dict[
        Any, tuple[Variable, Variable]
    ],  # indexes_coords altered so can't use mapping type
) -> tuple[Variable | T_Xarray, dict[Any, tuple[Variable, Variable]]]:
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
        indexes_coords[dim] = (
            x[indexes[dim]],
            new_x,
        )  # probably the in-place modification is unintentional here?
    return var.isel(indexers=indexes), indexes_coords


def _floatize_x(x: tuple[Variable, ...], new_x: tuple[Variable, ...]):
    """Make x and new_x float.
    This is particularly useful for datetime dtype.
    x, new_x: tuple of np.ndarray
    """
    x = list(x)  # type: ignore[assignment]
    new_x = list(new_x)  # type: ignore[assignment]
    for i in range(len(x)):
        if _contains_datetime_like_objects(x[i]):
            # Scipy casts coordinates to np.float64, which is not accurate
            # enough for datetime64 (uses 64bit integer).
            # We assume that the most of the bits are used to represent the
            # offset (min(x)) and the variation (x - min(x)) can be
            # represented by float.
            xmin = x[i].values.min()
            x[i] = x[i]._to_numeric(offset=xmin, dtype=np.float64)  # type: ignore[index]
            new_x[i] = new_x[i]._to_numeric(offset=xmin, dtype=np.float64)  # type: ignore[index]
    return x, new_x


def interp(
    var: Variable,
    indexes_coords: Mapping[Any, tuple[Any, Any]],
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
    # decompose the interpolation into a succession of independent interpolation
    for indep_indexes_coords in decompose_interp(indexes_coords):
        var = result

        # target dimensions
        dims = list(indep_indexes_coords)
        x, new_x = zip(*[indep_indexes_coords[d] for d in dims], strict=True)
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
                out_dims.update(indep_indexes_coords[d][1].dims)
            else:
                out_dims.add(d)
        if len(out_dims) > 1:
            result = result.transpose(*out_dims)
    return result


def interp_func(
    var: type[T_DuckArray],
    x: tuple[Variable, ...],
    new_x: tuple[Variable, ...],
    method: InterpOptions,
    kwargs: dict[str, Any],
) -> ArrayLike:
    """
    multi-dimensional interpolation for array-like. Interpolated axes should be
    located in the last position.

    Parameters
    ----------
    var : Array to be interpolated. The final dimension is interpolated.
    x : Original coordinates. Should not contain NaN.
    new_x : New coordinates. Should not contain NaN.
    method :
        {'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'pchip', 'akima',
            'makima', 'barycentric', 'krogh'} for 1-dimensional interpolation.
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
        return var.data.copy()

    if len(x) == 1:
        func, kwargs = _get_interpolator(method, vectorizeable_only=True, **kwargs)
    else:
        func, kwargs = _get_interpolator_nd(method, **kwargs)

    # is_chunked_array typed using named_array, unsure of relationship to duck arrays
    if is_chunked_array(var):  # type: ignore[arg-type]
        chunkmanager = get_chunked_array_type(var)  # duck compatible

        ndim = var.ndim
        nconst = ndim - len(x)

        out_ind = list(range(nconst)) + list(range(ndim, ndim + new_x[0].ndim))

        # blockwise args format
        x_arginds = [[_x, (nconst + index,)] for index, _x in enumerate(x)]
        x_arginds = [item for pair in x_arginds for item in pair]  # type: ignore[misc]
        new_x_arginds = [
            [_x, [ndim + index for index in range(_x.ndim)]] for _x in new_x
        ]
        new_x_arginds = [item for pair in new_x_arginds for item in pair]  # type: ignore[misc]

        args = (var, range(ndim), *x_arginds, *new_x_arginds)

        _, rechunked = chunkmanager.unify_chunks(*args)

        args = tuple(
            elem for pair in zip(rechunked, args[1::2], strict=True) for elem in pair
        )

        new_x = rechunked[1 + (len(rechunked) - 1) // 2 :]  # type: ignore[assignment]

        new_x0_chunks = new_x[0].chunks
        new_x0_shape = new_x[0].shape
        new_x0_chunks_is_not_none = new_x0_chunks is not None
        new_axes = {
            ndim + i: new_x0_chunks[i] if new_x0_chunks_is_not_none else new_x0_shape[i]  # type: ignore[index]
            for i in range(new_x[0].ndim)
        }

        # if useful, reuse localize for each chunk of new_x
        localize = (method in ["linear", "nearest"]) and new_x0_chunks_is_not_none

        # scipy.interpolate.interp1d always forces to float.
        # Use the same check for blockwise as well:
        if not issubclass(var.dtype.type, np.inexact):
            dtype = np.dtype(float)
        else:
            dtype = var.dtype

        # mypy may flag this in the future since _meta is not a property
        # of duck arrays--only inside this conditional if a dask array
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
            new_axes=new_axes,  # type: ignore[arg-type]
            meta=meta,
            align_arrays=False,
        )

    return _interpnd(var.data, x, new_x, func, kwargs)


def _interp1d(
    var: XarrayLike,
    x: Variable,
    new_x: Variable,
    func: Callable,
    kwargs: dict[str, Any],
) -> ArrayLike:
    # x, new_x are tuples of size 1.
    rslt = func(x, var, **kwargs)(np.ravel(new_x))
    if new_x.ndim > 1:
        return reshape(rslt, (var.shape[:-1] + new_x.shape))
    if new_x.ndim == 0:
        return rslt[..., -1]
    return rslt


def _interpnd(
    var: XarrayLike,
    x: tuple[Variable, ...],
    new_x: tuple[Variable, ...],
    func: Callable,
    kwargs: dict[str, Any],
) -> ArrayLike:
    x, new_x = _floatize_x(x, new_x)

    if len(x) == 1:
        return _interp1d(var, x[0], new_x[0], func, kwargs)

    # move the interpolation axes to the start position
    var = var.transpose(range(-len(x), var.ndim - len(x)))
    # stack new_x to 1 vector, with reshape
    xi = np.stack([x1.values.ravel() for x1 in new_x], axis=-1)
    rslt = func(x, var, xi, **kwargs)
    # move back the interpolation axes to the last position
    rslt = rslt.transpose(range(-rslt.ndim + 1, 1))
    return reshape(rslt, rslt.shape[:-1] + new_x[0].shape)


def _chunked_aware_interpnd(
    var: Variable,
    *coords,
    interp_func: Callable,
    interp_kwargs: dict[str, Any],
    localize: bool = True,
) -> ArrayLike:
    """Wrapper for `_interpnd` through `blockwise` for chunked arrays.

    The first half arrays in `coords` are original coordinates,
    the other half are destination coordinates
    """
    n_x = len(coords) // 2
    nconst = len(var.shape) - n_x

    # _interpnd expect coords to be Variables
    x = tuple(
        Variable([f"dim_{nconst + dim}"], _x) for dim, _x in enumerate(coords[:n_x])
    )
    new_x = tuple(
        Variable([f"dim_{len(var.shape) + dim}" for dim in range(len(_x.shape))], _x)
        for _x in coords[n_x:]
    )

    if localize:
        # _localize expect var to be a Variable
        var = Variable([f"dim_{dim}" for dim in range(len(var.shape))], var)

        indexes_coords = {
            _x.dims[0]: (_x, _new_x) for _x, _new_x in zip(x, new_x, strict=True)
        }

        # simple speed up for the local interpolation
        var, indexes_coords = _localize(var, indexes_coords)  # type: ignore[assignment]
        x, new_x = zip(*[indexes_coords[d] for d in indexes_coords], strict=True)

        # put var back as a ndarray
        var = var.data

    return _interpnd(var, x, new_x, interp_func, interp_kwargs)


def decompose_interp(
    indexes_coords: Mapping[Any, tuple[Any, Any]]
) -> Generator[Mapping[Any, tuple[Any, Any]]]:
    """Decompose the interpolation into a succession of independent interpolation keeping the order"""

    dest_dims = [
        dest[1].dims if dest[1].ndim > 0 else [dim]
        for dim, dest in indexes_coords.items()
    ]
    partial_dest_dims = []
    partial_indexes_coords: dict[Any, tuple[Any, Any]] = {}
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
