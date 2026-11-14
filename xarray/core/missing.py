from __future__ import annotations

import datetime as dt
import itertools
import warnings
from collections import ChainMap
from collections.abc import Callable, Generator, Hashable, Sequence
from functools import partial
from numbers import Number
from typing import TYPE_CHECKING, Any, TypeVar, get_args, override

import numpy as np
import pandas as pd

from xarray.computation.apply_ufunc import apply_ufunc
from xarray.core import utils
from xarray.core.common import _contains_datetime_like_objects, ones_like
from xarray.core.duck_array_ops import (
    astype,
    datetime_to_numeric,
    permute_dims,
    push,
    ravel,
    reshape,
    stack,
    timedelta_to_numeric,
)
from xarray.core.options import _get_keep_attrs
from xarray.core.types import Interp1dOptions, InterpnOptions, InterpOptions
from xarray.core.utils import OrderedSet, is_scalar
from xarray.core.variable import (
    Variable,
    broadcast_variables,
)
from xarray.namedarray.pycompat import is_chunked_array

if TYPE_CHECKING:
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset

    InterpCallable = Callable[..., np.ndarray]  # interpn
    Interpolator = Callable[..., Callable[..., np.ndarray]]  # *Interpolator
    # interpolator objects return callables that can be evaluated
    SourceDest = dict[Hashable, tuple[Variable, Variable]]

    T = TypeVar("T")


def _get_nan_block_lengths(
    obj: Dataset | DataArray | Variable, dim: Hashable, index: Variable
):
    """Return an object where each NaN element in 'obj' is replaced by its gap length.

    The gap length is the length of the gap the element is in.
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
    """Generic interpolator class for normalizing interpolation methods."""

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

    @override
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
    """Interpolate a 1-D function using Scipy interp1d.

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
    """Apply ``func`` to all data variables of a dataset that have ``dim``."""
    ds = type(self)(coords=self.coords, attrs=self.attrs)

    for name, var in self.data_vars.items():
        if dim in var.dims:
            ds[name] = func(var, dim=dim, **kwargs)
        else:
            ds[name] = var

    return ds


def get_clean_interp_index(
    arr, dim: Hashable, use_coordinate: Hashable | bool = True, strict: bool = True
):
    """Return index to use for x values in interpolation or curve fitting.

    Parameters
    ----------
    arr : DataArray
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
        index = arr.coords[use_coordinate]
        if index.ndim != 1:
            raise ValueError(
                f"Coordinates used for interpolation must be 1D, "
                f"{use_coordinate} is {index.ndim}D."
            )
        index = index.to_index()

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
            index = index.values
        index = Variable(
            data=datetime_to_numeric(index, offset=offset, datetime_unit="ns"),
            dims=(dim,),
        )

    # raise if index cannot be cast to a float (e.g. MultiIndex)
    try:
        index = index.values.astype(np.float64)
    except (TypeError, ValueError) as err:
        # pandas raises a TypeError
        # xarray/numpy raise a ValueError
        raise TypeError(
            f"Index {index.name!r} must be castable to float64 to support "
            f"interpolation or curve fitting, got {type(index).__name__}."
        ) from err

    return index


def interp_na(
    self,
    dim: Hashable | None = None,
    use_coordinate: bool | str = True,
    method: InterpOptions = "linear",
    limit: int | None = None,
    max_gap: (
        int | float | str | pd.Timedelta | np.timedelta64 | dt.timedelta | None
    ) = None,
    keep_attrs: bool | None = None,
    **kwargs,
):
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

        if not use_coordinate and not isinstance(max_gap, Number | np.number):
            raise TypeError(
                f"Expected integer or floating point max_gap since use_coordinate=False. Received {max_type}."
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


def func_interpolate_na(interpolator, y, x, **kwargs):
    """Apply interpolation along 1 dimension."""
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
    """Backfill along an axis (inverse of ffill)."""
    arr = np.flip(arr, axis=axis)

    # fill
    arr = push(arr, axis=axis, n=n)

    # reverse back to original
    return np.flip(arr, axis=axis)


def ffill(arr, dim=None, limit=None):
    """Forward fill missing values."""
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
    """Backfill missing values."""
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
    """Select the appropriate interpolator class.

    Returns interpolator class and keyword arguments for the class.
    """
    interp_class: Interpolator
    interp1d_methods = get_args(Interp1dOptions)
    valid_methods = tuple(vv for v in get_args(InterpOptions) for vv in get_args(v))

    # prefer numpy.interp for 1d linear interpolation. This function cannot
    # take higher dimensional data but scipy.interp1d can.
    if (
        method == "linear"
        and kwargs.get("fill_value") != "extrapolate"
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
                FutureWarning,
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
    """Select the appropriate interpolator class.

    Returns interpolator class and keyword arguments for the class.
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


def _get_valid_fill_mask(arr, dim, limit):
    """Determine values that can be filled when limit is not None."""
    kw = {dim: limit + 1}
    # we explicitly use construct method to avoid copy.
    new_dim = utils.get_temp_dimname(arr.dims, "_window")
    return (
        arr.isnull()
        .rolling(min_periods=1, **kw)
        .construct(new_dim, fill_value=False)
        .sum(new_dim, skipna=False)
    ) <= limit


def _localize[T](obj: T, indexes_coords: SourceDest) -> tuple[T, SourceDest]:
    """Speed up for linear and nearest neighbor method.

    Only consider a subspace that is needed for the interpolation.
    """
    indexes = {}
    for dim, [x, new_x] in indexes_coords.items():
        if is_chunked_array(new_x._data) or new_x.size == 0:
            continue
        new_x_loaded = new_x.data
        minval = np.nanmin(new_x_loaded)
        maxval = np.nanmax(new_x_loaded)
        index = x.to_index()
        imin, imax = index.get_indexer(pd.Index([minval, maxval]), method="nearest")
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
    """Make an interpolation of Variable.

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


def _nearest_indices(src: np.ndarray, dest: np.ndarray) -> np.ndarray:
    """Return indices of the nearest source point, as ``interp1d(kind="nearest")``.

    ``scipy`` compares each destination with the midpoints between source points
    and rounds exact midpoints down. ``nan`` destinations sort after every
    midpoint and therefore select the last source point, which is what
    ``interp1d`` returns for them.
    """
    midpoints = src / 2.0
    midpoints = midpoints[1:] + midpoints[:-1]
    return np.clip(np.searchsorted(midpoints, dest, side="left"), 0, src.size - 1)


def _linear_bracket(
    src: np.ndarray, dest: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Source bracket and weights, as ``interp1d._call_linear`` computes them.

    The interval is located with a ``searchsorted`` that defaults to side
    ``"left"`` and is clipped to ``[1, len(x) - 1]``, so a destination that
    exactly matches a source point uses the interval to its right. Each weight
    is computed with its own division, and they are multiplied with the values
    before being added up.
    """
    upper = np.clip(np.searchsorted(src, dest, side="left"), 1, src.size - 1)
    lower = upper - 1
    span = src[upper] - src[lower]
    return lower, upper, (src[upper] - dest) / span, (dest - src[lower]) / span


def _grid_bracket(
    src: np.ndarray, dest: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Source bracket and normalized distance used by ``RegularGridInterpolator``.

    ``_find_indices`` searches with side ``"right"`` and clips to
    ``[0, len(x) - 2]``; the weight of the lower point is ``1 - norm``.
    """
    lower = np.clip(np.searchsorted(src, dest, side="right") - 1, 0, src.size - 2)
    upper = lower + 1
    return lower, upper, (dest - src[lower]) / (src[upper] - src[lower])


def _outside(src: np.ndarray, dest: np.ndarray) -> np.ndarray:
    """Destinations outside the source range, which ``scipy`` fills with ``nan``."""
    return (dest < src[0]) | (dest > src[-1])


def _unwrap_in_memory(obj: Any, depth: int = 0) -> np.ndarray | None:
    """Return a numpy array held directly by an indexing wrapper, or None."""
    if depth > 5:
        return None
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, pd.Index):
        return obj.to_numpy()
    inner = getattr(obj, "array", None)
    if inner is not None and inner is not obj:
        return _unwrap_in_memory(inner, depth + 1)
    return None


def _in_memory_array(var: Variable) -> np.ndarray | None:
    """Values of a coordinate that is already in memory.

    ``Dataset.interp`` chunks coordinates when any data variable is chunked
    (GH4739), but the chunk is a view of the in-memory index. Reading it must
    not schedule a dask compute: callers build the interpolation graph lazily.
    Returns None when the values live behind a real chunked read.
    """
    data = var._data
    if isinstance(data, np.ndarray):
        return data
    if not is_chunked_array(data):
        try:
            return np.asarray(data)
        except (TypeError, ValueError):
            return None
    chunks = getattr(data, "chunks", None)
    layers = getattr(getattr(data, "dask", None), "layers", None)
    if not chunks or layers is None or any(len(c) != 1 for c in chunks):
        return None
    for layer in layers.values():
        mapping = getattr(layer, "mapping", None)
        if not mapping or len(mapping) != 1:
            continue
        array = _unwrap_in_memory(next(iter(mapping.values())))
        if array is not None and array.shape == data.shape:
            return np.asarray(array)
    return None


def _rgi_uses_compiled_linear(var: Variable) -> bool:
    """Whether ``RegularGridInterpolator`` evaluates 2D linear interp in Cython.

    Integer values are cast to ``float64`` first, so they take the compiled path
    too. That kernel's rounding depends on how SciPy was compiled, so it has to
    be called rather than reimplemented.
    """
    if var.ndim != 2:
        return False
    if var.dtype == np.dtype(np.float64):
        return var.dtype.byteorder == "="
    return var.dtype.kind in "iu"


def _compiled_linear_2d(
    var: Variable,
    dims: list[Hashable],
    dest: Variable,
    brackets: list[tuple[np.ndarray, np.ndarray]],
    weights: list[tuple[np.ndarray, np.ndarray]],
    sources: list[np.ndarray],
) -> Variable | None:
    """Evaluate 2D linear interpolation with SciPy's compiled kernel.

    Corners are gathered first, so a chunked array only loads the chunks that
    contain destination points. The kernel indexes one shared grid, so the
    corners are packed into a grid whose rows are those points.
    """
    try:
        from scipy.interpolate._rgi_cython import evaluate_linear_2d
    except ImportError:
        return None

    dim_0, dim_1 = dims
    pieces = [
        var.isel(
            {
                dim_0: Variable(dest.dims, brackets[0][first]),
                dim_1: Variable(dest.dims, brackets[1][second]),
            }
        )._data
        for first, second in ((0, 0), (0, 1), (1, 0), (1, 1))
    ]
    # The upper weight is the normalized distance the kernel consumes.
    norms = np.vstack((np.ravel(weights[0][1]), np.ravel(weights[1][1])))

    def evaluate(v00, v01, v10, v11):
        corners = [
            np.ravel(np.asarray(piece, dtype=np.float64))
            for piece in (v00, v01, v10, v11)
        ]
        n = corners[0].size
        values = np.empty((2 * n, 2), dtype=np.float64)
        values[0::2, 0] = corners[0]
        values[0::2, 1] = corners[1]
        values[1::2, 0] = corners[2]
        values[1::2, 1] = corners[3]
        indices = np.zeros((2, n), dtype=np.intp)
        indices[0] = np.arange(0, 2 * n, 2)
        out = np.empty(n, dtype=np.float64)
        evaluated = evaluate_linear_2d(values, indices, norms, tuple(sources), out)
        return np.reshape(evaluated, dest.shape)

    try:
        if any(is_chunked_array(piece) for piece in pieces):
            import dask.array as dask_array

            from xarray.namedarray.utils import is_duck_dask_array

            if not all(is_duck_dask_array(piece) for piece in pieces):
                return None
            arrays = [dask_array.asarray(piece).rechunk(-1) for piece in pieces]
            evaluated = dask_array.map_blocks(
                evaluate, *arrays, dtype=np.float64, chunks=arrays[0].chunks
            )
        else:
            evaluated = evaluate(*pieces)
    except (TypeError, ValueError, AttributeError):
        return None
    return Variable(dest.dims, evaluated)


def _interp_sparse(
    var: Variable,
    indexes_coords: SourceDest,
    *,
    method: InterpOptions,
    kwargs: dict[str, Any],
) -> Variable | None:
    """Linear/nearest interpolation that loads only the chunks it needs.

    ``apply_ufunc(..., allow_rechunk=True)`` merges every chunked interpolation
    dimension into a single chunk, so interpolating at a few points scattered
    over a large array loads the whole array (:issue:`6799`). Looking up the
    neighbouring source points up front and indexing with them touches only the
    chunks that contain destination points.

    The arithmetic mirrors the interpolators xarray picks for these methods --
    ``scipy.interpolate.interp1d`` for one interpolation dimension and
    ``scipy.interpolate.interpn`` (``RegularGridInterpolator``) for two -- so the
    result matches the general path.

    Returns None when the inputs do not meet the requirements of this path.
    """
    if method not in ("linear", "nearest") or kwargs:
        return None
    if not is_chunked_array(var._data):
        return None
    if not 1 <= len(indexes_coords) <= 2:
        return None
    if var.dtype.kind not in "iuf":
        return None

    dims = list(indexes_coords)
    # ``interp`` transposes the variable so that the interpolated dimensions come
    # last, in the order of ``indexes_coords``.
    if tuple(var.dims[-len(dims) :]) != tuple(dims):
        return None
    broadcast_dims = var.dims[: -len(dims)]

    sources: list[np.ndarray] = []
    dests: list[Variable] = []
    for dim in dims:
        src, dest = indexes_coords[dim]
        if src.ndim != 1 or src.dims != (dim,):
            return None
        if src.dtype.kind not in "iuf" or dest.dtype.kind not in "iuf":
            return None
        if src.size < 2:
            return None
        if dest.ndim == 0:
            # A single destination point is already cheap: ``_localize`` narrows
            # the source to a few points before the interpolator runs.
            return None
        if dest.chunks is not None and any(len(c) > 1 for c in dest.chunks):
            # A destination that is split over several chunks is loaded lazily by
            # the general path; keep that behaviour instead of materializing it.
            return None
        if not set(dest.dims).isdisjoint(broadcast_dims):
            # The destination is vectorized over a non-interpolated dimension of
            # the variable; leave that broadcasting to ``apply_ufunc``.
            return None
        # Dataset.interp chunks coordinates when any data variable is chunked
        # (GH4739). They are still in memory; scheduling a compute here would
        # break callers that build a lazy graph.
        src_values = _in_memory_array(src)
        if src_values is None:
            return None
        if not np.all(src_values[1:] > src_values[:-1]):
            return None
        sources.append(src_values)
        dests.append(dest)

    if any(dest.dims != dests[0].dims for dest in dests[1:]):
        return None

    if method == "linear" and len(dims) == 1 and var.ndim == 1:
        # ``interp1d`` delegates 1D linear interpolation on these dtypes to the
        # compiled ``np.interp``, whose rounding cannot be reproduced with the
        # numpy expressions used here.
        source_dtype = np.dtype(sources[0].dtype)
        if source_dtype in (np.dtype(np.float64), np.dtype(np.intp)) and (
            var.dtype == np.dtype(np.float64) or var.dtype.kind in "iu"
        ):
            return None

    dest = dests[0]
    # ``RegularGridInterpolator`` casts the grid and the sample points to
    # float64. ``interp1d`` keeps the coordinate dtype, so only the 2D path casts.
    queried = []
    for one in dests:
        values = _in_memory_array(one)
        if values is None:
            return None
        queried.append(values)
    if len(dims) == 2:
        sources = [np.ascontiguousarray(src, dtype=np.float64) for src in sources]
        queried = [np.asarray(values, dtype=np.float64) for values in queried]
    else:
        queried = [np.asarray(values) for values in queried]

    outside = np.zeros(dest.shape, dtype=bool)
    for src_values, dest_values in zip(sources, queried, strict=True):
        outside |= _outside(src_values, dest_values)
    if len(dims) > 1:
        # A NaN on any axis becomes NaN. ``interp1d`` instead sorts a NaN sample
        # last and returns the last source value, so the 1D path leaves it.
        for dest_values in queried:
            if dest_values.dtype.kind == "f":
                outside |= np.isnan(dest_values)
    if method == "nearest":
        indexers: dict[Hashable, Variable] = {}
        for dim, src_values, dest_values in zip(dims, sources, queried, strict=True):
            if len(dims) == 1:
                indices = _nearest_indices(src_values, dest_values)
            else:
                lower, upper, norm = _grid_bracket(src_values, dest_values)
                indices = np.where(norm <= 0.5, lower, upper)
            indexers[dim] = Variable(dest.dims, indices)
        result = var.isel(indexers)
    else:
        brackets: list[tuple[np.ndarray, np.ndarray]] = []
        weights: list[tuple[np.ndarray, np.ndarray]] = []
        for src_values, dest_values in zip(sources, queried, strict=True):
            if len(dims) == 1:
                lower, upper, weight_lo, weight_hi = _linear_bracket(
                    src_values, dest_values
                )
            else:
                lower, upper, norm = _grid_bracket(src_values, dest_values)
                weight_lo, weight_hi = 1 - norm, norm
            brackets.append((lower, upper))
            weights.append((weight_lo, weight_hi))

        if len(dims) == 1:
            (dim,) = dims
            lower, upper = brackets[0]
            weight_lo, weight_hi = weights[0]
            y_lo = var.isel({dim: Variable(dest.dims, lower)})
            y_hi = var.isel({dim: Variable(dest.dims, upper)})
            result = y_hi * Variable(dest.dims, weight_hi) + y_lo * Variable(
                dest.dims, weight_lo
            )
        elif _rgi_uses_compiled_linear(var):
            result = _compiled_linear_2d(var, dims, dest, brackets, weights, sources)
            if result is None:
                return None
        else:
            # ``_evaluate_linear`` multiplies the weights together before the
            # values and sums the corners left to right. A broadcast dimension
            # takes this path; the compiled kernel is only used for a bare 2D array.
            dim_0, dim_1 = dims
            result = None
            for first, second in itertools.product((0, 1), repeat=2):
                corner = var.isel(
                    {
                        dim_0: Variable(dest.dims, brackets[0][first]),
                        dim_1: Variable(dest.dims, brackets[1][second]),
                    }
                )
                weight = Variable(dest.dims, weights[0][first]) * Variable(
                    dest.dims, weights[1][second]
                )
                term = corner * weight
                result = term if result is None else result + term
            assert result is not None

    result = result.where(~Variable(dest.dims, outside))
    # Interpolators upcast integer data to float.
    dtype = np.dtype(float) if not issubclass(var.dtype.type, np.inexact) else var.dtype
    if result.dtype != dtype:
        result = result.astype(dtype)
    result.attrs = var.attrs
    return result


def interpolate_variable(
    var: Variable,
    indexes_coords: SourceDest,
    *,
    method: InterpOptions,
    kwargs: dict[str, Any],
) -> Variable:
    """Core routine that returns the interpolated variable."""
    if not indexes_coords:
        return var.copy()

    sparse_result = _interp_sparse(var, indexes_coords, method=method, kwargs=kwargs)
    if sparse_result is not None:
        return sparse_result

    if len(indexes_coords) == 1:
        func, kwargs = _get_interpolator(method, vectorizeable_only=True, **kwargs)
    else:
        func, kwargs = _get_interpolator_nd(method, **kwargs)

    localize = kwargs.get("method") in ["linear", "nearest"]
    if localize:
        # simple speed up for the local interpolation
        var, indexes_coords = _localize(var, dict(indexes_coords))

    in_coords, result_coords = _floatize_x(
        *(list(_) for _ in zip(*indexes_coords.values(), strict=True))
    )
    # `_localize` skips chunked destination coordinates, so localize each block
    # instead: scipy casts integer data to float, which would otherwise copy
    # the whole block
    localize_blocks = localize and any(
        is_chunked_array(coord._data) for coord in result_coords
    )

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
    # ordered like var.dims, since it also determines the loop order
    vectorize_dims = [
        d for d in var.dims if d in result_dims and d not in all_in_core_dims
    ]

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
    dtype = np.dtype(float) if not issubclass(var.dtype.type, np.inexact) else var.dtype
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
            # the interpolators may upcast, e.g. float32 to float64 depending
            # on the method and the coordinate dtypes, so cast the result to
            # the dtype we promise dask
            dtype=dtype,
            localize=localize_blocks,
        ),
        # TODO: deprecate and have the user rechunk themselves
        dask_gufunc_kwargs=dict(output_sizes=output_sizes, allow_rechunk=True),
        output_dtypes=[dtype],
        # only loop over the dimensions along which the destination coordinates vary,
        # vectorize=True would also loop over all other broadcast dimensions (GH10683)
        vectorize=vectorize_dims,
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
    interp_kwargs: dict[str, Any],
    result_coord_core_dims: list[tuple[Hashable, ...]],
    dtype: np.dtype,
    localize: bool = False,
) -> np.ndarray:
    """Core nD array interpolation routine.

    The first half arrays in `coords` are original coordinates,
    the other half are destination coordinates.

    `data` has the loop dimensions first, followed by the dimensions we
    interpolate along. ``apply_ufunc`` only loops over the loop dimensions along
    which the destination coordinates vary, e.g. ``t`` for
    ``da[t, r, z].interp(z=target[t])``, so the destination coordinates have
    size 1 along all remaining loop dimensions.
    """
    n_x = len(coords) // 2
    ndim = data.ndim
    nconst = ndim - n_x

    # Convert everything to Variables, since that makes broadcasting the
    # destination coordinates much easier
    x: list[Variable] = [
        Variable([f"dim_{nconst + dim}"], _x, fastpath=True)
        for dim, _x in enumerate(coords[:n_x])
    ]
    new_x = list(
        broadcast_variables(
            *(
                # drop the size-1 loop dimensions
                Variable(
                    dims, reshape(_x, _x.shape[_x.ndim - len(dims) :]), fastpath=True
                )
                for dims, _x in zip(result_coord_core_dims, coords[n_x:], strict=True)
            )
        )
    )
    var = Variable([f"dim_{dim}" for dim in range(ndim)], data, fastpath=True)

    if localize:
        indexes_coords = {
            _x.dims[0]: (_x, _new_x) for _x, _new_x in zip(x, new_x, strict=True)
        }
        var, indexes_coords = _localize(var, indexes_coords)
        x, new_x = (list(_) for _ in zip(*indexes_coords.values(), strict=True))

    if len(x) == 1:
        # TODO: narrow interp_func to interpolator here
        return astype(
            _interp1d(var, x, new_x, interp_func, interp_kwargs),  # type: ignore[arg-type]
            dtype,
            copy=False,
        )

    # move the interpolation axes to the start position
    data = permute_dims(var._data, range(-len(x), var.ndim - len(x)))

    # stack new_x to 1 vector, with reshape
    xi = stack([ravel(x1.data) for x1 in new_x], axis=-1)
    rslt: np.ndarray = interp_func(x, data, xi, **interp_kwargs)  # type: ignore[assignment]
    # move back the interpolation axes to the last position
    rslt = permute_dims(rslt, range(-rslt.ndim + 1, 1))
    rslt = reshape(rslt, rslt.shape[:-1] + new_x[0].shape)
    return astype(rslt, dtype, copy=False)


def decompose_interp(indexes_coords: SourceDest) -> Generator[SourceDest, None]:
    """Decompose the interpolation into a succession of independent interpolations.

    The order of the interpolations is kept.

    Yields
    ------
    dict
        The source and destination coordinates of each independent
        interpolation.
    """
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
