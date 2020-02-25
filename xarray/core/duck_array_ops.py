"""Compatibility module defining operations on duck numpy-arrays.

Currently, this means Dask or NumPy arrays. None of these functions should
accept or return xarray objects.
"""
import contextlib
import inspect
import warnings
from functools import partial

import numpy as np
import pandas as pd

from . import dask_array_compat, dask_array_ops, dtypes, npcompat, nputils
from .nputils import nanfirst, nanlast
from .pycompat import dask_array_type

try:
    import dask.array as dask_array
except ImportError:
    dask_array = None  # type: ignore


def _dask_or_eager_func(
    name,
    eager_module=np,
    dask_module=dask_array,
    list_of_args=False,
    array_args=slice(1),
    requires_dask=None,
):
    """Create a function that dispatches to dask for dask array inputs."""
    if dask_module is not None:

        def f(*args, **kwargs):
            if list_of_args:
                dispatch_args = args[0]
            else:
                dispatch_args = args[array_args]
            if any(isinstance(a, dask_array_type) for a in dispatch_args):
                try:
                    wrapped = getattr(dask_module, name)
                except AttributeError as e:
                    raise AttributeError(f"{e}: requires dask >={requires_dask}")
            else:
                wrapped = getattr(eager_module, name)
            return wrapped(*args, **kwargs)

    else:

        def f(*args, **kwargs):
            return getattr(eager_module, name)(*args, **kwargs)

    return f


def fail_on_dask_array_input(values, msg=None, func_name=None):
    if isinstance(values, dask_array_type):
        if msg is None:
            msg = "%r is not yet a valid method on dask arrays"
        if func_name is None:
            func_name = inspect.stack()[1][3]
        raise NotImplementedError(msg % func_name)


# switch to use dask.array / __array_function__ version when dask supports it:
# https://github.com/dask/dask/pull/4822
moveaxis = npcompat.moveaxis

around = _dask_or_eager_func("around")
isclose = _dask_or_eager_func("isclose")


if hasattr(np, "isnat") and (
    dask_array is None or hasattr(dask_array_type, "__array_ufunc__")
):
    # np.isnat is available since NumPy 1.13, so __array_ufunc__ is always
    # supported.
    isnat = np.isnat
else:
    isnat = _dask_or_eager_func("isnull", eager_module=pd)
isnan = _dask_or_eager_func("isnan")
zeros_like = _dask_or_eager_func("zeros_like")


pandas_isnull = _dask_or_eager_func("isnull", eager_module=pd)


def isnull(data):
    data = asarray(data)
    scalar_type = data.dtype.type
    if issubclass(scalar_type, (np.datetime64, np.timedelta64)):
        # datetime types use NaT for null
        # note: must check timedelta64 before integers, because currently
        # timedelta64 inherits from np.integer
        return isnat(data)
    elif issubclass(scalar_type, np.inexact):
        # float types use NaN for null
        return isnan(data)
    elif issubclass(scalar_type, (np.bool_, np.integer, np.character, np.void)):
        # these types cannot represent missing values
        return zeros_like(data, dtype=bool)
    else:
        # at this point, array should have dtype=object
        if isinstance(data, (np.ndarray, dask_array_type)):
            return pandas_isnull(data)
        else:
            # Not reachable yet, but intended for use with other duck array
            # types. For full consistency with pandas, we should accept None as
            # a null value as well as NaN, but it isn't clear how to do this
            # with duck typing.
            return data != data


def notnull(data):
    return ~isnull(data)


transpose = _dask_or_eager_func("transpose")
_where = _dask_or_eager_func("where", array_args=slice(3))
isin = _dask_or_eager_func("isin", array_args=slice(2))
take = _dask_or_eager_func("take")
broadcast_to = _dask_or_eager_func("broadcast_to")
pad = _dask_or_eager_func("pad")

_concatenate = _dask_or_eager_func("concatenate", list_of_args=True)
_stack = _dask_or_eager_func("stack", list_of_args=True)

array_all = _dask_or_eager_func("all")
array_any = _dask_or_eager_func("any")

tensordot = _dask_or_eager_func("tensordot", array_args=slice(2))
einsum = _dask_or_eager_func("einsum", array_args=slice(1, None))


def gradient(x, coord, axis, edge_order):
    if isinstance(x, dask_array_type):
        return dask_array.gradient(x, coord, axis=axis, edge_order=edge_order)
    return np.gradient(x, coord, axis=axis, edge_order=edge_order)


def trapz(y, x, axis):
    if axis < 0:
        axis = y.ndim + axis
    x_sl1 = (slice(1, None),) + (None,) * (y.ndim - axis - 1)
    x_sl2 = (slice(None, -1),) + (None,) * (y.ndim - axis - 1)
    slice1 = (slice(None),) * axis + (slice(1, None),)
    slice2 = (slice(None),) * axis + (slice(None, -1),)
    dx = x[x_sl1] - x[x_sl2]
    integrand = dx * 0.5 * (y[tuple(slice1)] + y[tuple(slice2)])
    return sum(integrand, axis=axis, skipna=False)


masked_invalid = _dask_or_eager_func(
    "masked_invalid", eager_module=np.ma, dask_module=getattr(dask_array, "ma", None)
)


def asarray(data):
    return (
        data
        if (isinstance(data, dask_array_type) or hasattr(data, "__array_function__"))
        else np.asarray(data)
    )


def as_shared_dtype(scalars_or_arrays):
    """Cast a arrays to a shared dtype using xarray's type promotion rules."""
    arrays = [asarray(x) for x in scalars_or_arrays]
    # Pass arrays directly instead of dtypes to result_type so scalars
    # get handled properly.
    # Note that result_type() safely gets the dtype from dask arrays without
    # evaluating them.
    out_type = dtypes.result_type(*arrays)
    return [x.astype(out_type, copy=False) for x in arrays]


def lazy_array_equiv(arr1, arr2):
    """Like array_equal, but doesn't actually compare values.
       Returns True when arr1, arr2 identical or their dask names are equal.
       Returns False when shapes are not equal.
       Returns None when equality cannot determined: one or both of arr1, arr2 are numpy arrays;
       or their dask names are not equal
    """
    if arr1 is arr2:
        return True
    arr1 = asarray(arr1)
    arr2 = asarray(arr2)
    if arr1.shape != arr2.shape:
        return False
    if (
        dask_array
        and isinstance(arr1, dask_array_type)
        and isinstance(arr2, dask_array_type)
    ):
        # GH3068
        if arr1.name == arr2.name:
            return True
        else:
            return None
    return None


def allclose_or_equiv(arr1, arr2, rtol=1e-5, atol=1e-8):
    """Like np.allclose, but also allows values to be NaN in both arrays
    """
    arr1 = asarray(arr1)
    arr2 = asarray(arr2)
    lazy_equiv = lazy_array_equiv(arr1, arr2)
    if lazy_equiv is None:
        return bool(isclose(arr1, arr2, rtol=rtol, atol=atol, equal_nan=True).all())
    else:
        return lazy_equiv


def array_equiv(arr1, arr2):
    """Like np.array_equal, but also allows values to be NaN in both arrays
    """
    arr1 = asarray(arr1)
    arr2 = asarray(arr2)
    lazy_equiv = lazy_array_equiv(arr1, arr2)
    if lazy_equiv is None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "In the future, 'NAT == x'")
            flag_array = (arr1 == arr2) | (isnull(arr1) & isnull(arr2))
            return bool(flag_array.all())
    else:
        return lazy_equiv


def array_notnull_equiv(arr1, arr2):
    """Like np.array_equal, but also allows values to be NaN in either or both
    arrays
    """
    arr1 = asarray(arr1)
    arr2 = asarray(arr2)
    lazy_equiv = lazy_array_equiv(arr1, arr2)
    if lazy_equiv is None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "In the future, 'NAT == x'")
            flag_array = (arr1 == arr2) | isnull(arr1) | isnull(arr2)
            return bool(flag_array.all())
    else:
        return lazy_equiv


def count(data, axis=None):
    """Count the number of non-NA in this array along the given axis or axes
    """
    return np.sum(np.logical_not(isnull(data)), axis=axis)


def where(condition, x, y):
    """Three argument where() with better dtype promotion rules."""
    return _where(condition, *as_shared_dtype([x, y]))


def where_method(data, cond, other=dtypes.NA):
    if other is dtypes.NA:
        other = dtypes.get_fill_value(data.dtype)
    return where(cond, data, other)


def fillna(data, other):
    # we need to pass data first so pint has a chance of returning the
    # correct unit
    # TODO: revert after https://github.com/hgrecco/pint/issues/1019 is fixed
    return where(notnull(data), data, other)


def concatenate(arrays, axis=0):
    """concatenate() with better dtype promotion rules."""
    return _concatenate(as_shared_dtype(arrays), axis=axis)


def stack(arrays, axis=0):
    """stack() with better dtype promotion rules."""
    return _stack(as_shared_dtype(arrays), axis=axis)


@contextlib.contextmanager
def _ignore_warnings_if(condition):
    if condition:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    else:
        yield


def _create_nan_agg_method(name, dask_module=dask_array, coerce_strings=False):
    from . import nanops

    def f(values, axis=None, skipna=None, **kwargs):
        if kwargs.pop("out", None) is not None:
            raise TypeError(f"`out` is not valid for {name}")

        values = asarray(values)

        if coerce_strings and values.dtype.kind in "SU":
            values = values.astype(object)

        func = None
        if skipna or (skipna is None and values.dtype.kind in "cfO"):
            nanname = "nan" + name
            func = getattr(nanops, nanname)
        else:
            func = _dask_or_eager_func(name, dask_module=dask_module)

        try:
            return func(values, axis=axis, **kwargs)
        except AttributeError:
            if isinstance(values, dask_array_type):
                try:  # dask/dask#3133 dask sometimes needs dtype argument
                    # if func does not accept dtype, then raises TypeError
                    return func(values, axis=axis, dtype=values.dtype, **kwargs)
                except (AttributeError, TypeError):
                    msg = "%s is not yet implemented on dask arrays" % name
            else:
                msg = (
                    "%s is not available with skipna=False with the "
                    "installed version of numpy; upgrade to numpy 1.12 "
                    "or newer to use skipna=True or skipna=None" % name
                )
            raise NotImplementedError(msg)

    f.__name__ = name
    return f


# Attributes `numeric_only`, `available_min_count` is used for docs.
# See ops.inject_reduce_methods
argmax = _create_nan_agg_method("argmax", coerce_strings=True)
argmin = _create_nan_agg_method("argmin", coerce_strings=True)
max = _create_nan_agg_method("max", coerce_strings=True)
min = _create_nan_agg_method("min", coerce_strings=True)
sum = _create_nan_agg_method("sum")
sum.numeric_only = True
sum.available_min_count = True
std = _create_nan_agg_method("std")
std.numeric_only = True
var = _create_nan_agg_method("var")
var.numeric_only = True
median = _create_nan_agg_method("median", dask_module=dask_array_compat)
median.numeric_only = True
prod = _create_nan_agg_method("prod")
prod.numeric_only = True
sum.available_min_count = True
cumprod_1d = _create_nan_agg_method("cumprod")
cumprod_1d.numeric_only = True
cumsum_1d = _create_nan_agg_method("cumsum")
cumsum_1d.numeric_only = True


_mean = _create_nan_agg_method("mean")


def _datetime_nanmin(array):
    """nanmin() function for datetime64.

    Caveats that this function deals with:

    - In numpy < 1.18, min() on datetime64 incorrectly ignores NaT
    - numpy nanmin() don't work on datetime64 (all versions at the moment of writing)
    - dask min() does not work on datetime64 (all versions at the moment of writing)
    """
    assert array.dtype.kind in "mM"
    dtype = array.dtype
    # (NaT).astype(float) does not produce NaN...
    array = where(pandas_isnull(array), np.nan, array.astype(float))
    array = min(array, skipna=True)
    if isinstance(array, float):
        array = np.array(array)
    # ...but (NaN).astype("M8") does produce NaT
    return array.astype(dtype)


def datetime_to_numeric(array, offset=None, datetime_unit=None, dtype=float):
    """Convert an array containing datetime-like data to numerical values.

    Convert the datetime array to a timedelta relative to an offset.

    Parameters
    ----------
    da : array-like
      Input data
    offset: None, datetime or cftime.datetime
      Datetime offset. If None, this is set by default to the array's minimum
      value to reduce round off errors.
    datetime_unit: {None, Y, M, W, D, h, m, s, ms, us, ns, ps, fs, as}
      If not None, convert output to a given datetime unit. Note that some
      conversions are not allowed due to non-linear relationships between units.
    dtype: dtype
      Output dtype.

    Returns
    -------
    array
      Numerical representation of datetime object relative to an offset.

    Notes
    -----
    Some datetime unit conversions won't work, for example from days to years, even
    though some calendars would allow for them (e.g. no_leap). This is because there
    is no `cftime.timedelta` object.
    """
    # TODO: make this function dask-compatible?
    # Set offset to minimum if not given
    if offset is None:
        if array.dtype.kind in "Mm":
            offset = _datetime_nanmin(array)
        else:
            offset = min(array)

    # Compute timedelta object.
    # For np.datetime64, this can silently yield garbage due to overflow.
    # One option is to enforce 1970-01-01 as the universal offset.
    array = array - offset

    # Scalar is converted to 0d-array
    if not hasattr(array, "dtype"):
        array = np.array(array)

    # Convert timedelta objects to float by first converting to microseconds.
    if array.dtype.kind in "O":
        return py_timedelta_to_float(array, datetime_unit or "ns").astype(dtype)

    # Convert np.NaT to np.nan
    elif array.dtype.kind in "mM":

        # Convert to specified timedelta units.
        if datetime_unit:
            array = array / np.timedelta64(1, datetime_unit)
        return np.where(isnull(array), np.nan, array.astype(dtype))


def timedelta_to_numeric(value, datetime_unit="ns", dtype=float):
    """Convert a timedelta-like object to numerical values.

    Parameters
    ----------
    value : datetime.timedelta, numpy.timedelta64, pandas.Timedelta, str
      Time delta representation.
    datetime_unit : {Y, M, W, D, h, m, s, ms, us, ns, ps, fs, as}
      The time units of the output values. Note that some conversions are not allowed due to
      non-linear relationships between units.
    dtype : type
      The output data type.

    """
    import datetime as dt

    if isinstance(value, dt.timedelta):
        out = py_timedelta_to_float(value, datetime_unit)
    elif isinstance(value, np.timedelta64):
        out = np_timedelta64_to_float(value, datetime_unit)
    elif isinstance(value, pd.Timedelta):
        out = pd_timedelta_to_float(value, datetime_unit)
    elif isinstance(value, str):
        try:
            a = pd.to_timedelta(value)
        except ValueError:
            raise ValueError(
                f"Could not convert {value!r} to timedelta64 using pandas.to_timedelta"
            )
        return py_timedelta_to_float(a, datetime_unit)
    else:
        raise TypeError(
            f"Expected value of type str, pandas.Timedelta, datetime.timedelta "
            f"or numpy.timedelta64, but received {type(value).__name__}"
        )
    return out.astype(dtype)


def _to_pytimedelta(array, unit="us"):
    index = pd.TimedeltaIndex(array.ravel(), unit=unit)
    return index.to_pytimedelta().reshape(array.shape)


def np_timedelta64_to_float(array, datetime_unit):
    """Convert numpy.timedelta64 to float.

    Notes
    -----
    The array is first converted to microseconds, which is less likely to
    cause overflow errors.
    """
    array = array.astype("timedelta64[ns]").astype(np.float64)
    conversion_factor = np.timedelta64(1, "ns") / np.timedelta64(1, datetime_unit)
    return conversion_factor * array


def pd_timedelta_to_float(value, datetime_unit):
    """Convert pandas.Timedelta to float.

    Notes
    -----
    Built on the assumption that pandas timedelta values are in nanoseconds,
    which is also the numpy default resolution.
    """
    value = value.to_timedelta64()
    return np_timedelta64_to_float(value, datetime_unit)


def py_timedelta_to_float(array, datetime_unit):
    """Convert a timedelta object to a float, possibly at a loss of resolution.
    """
    array = np.asarray(array)
    array = np.reshape([a.total_seconds() for a in array.ravel()], array.shape) * 1e6
    conversion_factor = np.timedelta64(1, "us") / np.timedelta64(1, datetime_unit)
    return conversion_factor * array


def mean(array, axis=None, skipna=None, **kwargs):
    """inhouse mean that can handle np.datetime64 or cftime.datetime
    dtypes"""
    from .common import _contains_cftime_datetimes

    array = asarray(array)
    if array.dtype.kind in "Mm":
        offset = _datetime_nanmin(array)

        # xarray always uses np.datetime64[ns] for np.datetime64 data
        dtype = "timedelta64[ns]"
        return (
            _mean(
                datetime_to_numeric(array, offset), axis=axis, skipna=skipna, **kwargs
            ).astype(dtype)
            + offset
        )
    elif _contains_cftime_datetimes(array):
        if isinstance(array, dask_array_type):
            raise NotImplementedError(
                "Computing the mean of an array containing "
                "cftime.datetime objects is not yet implemented on "
                "dask arrays."
            )
        offset = min(array)
        timedeltas = datetime_to_numeric(array, offset, datetime_unit="us")
        mean_timedeltas = _mean(timedeltas, axis=axis, skipna=skipna, **kwargs)
        return _to_pytimedelta(mean_timedeltas, unit="us") + offset
    else:
        return _mean(array, axis=axis, skipna=skipna, **kwargs)


mean.numeric_only = True  # type: ignore


def _nd_cum_func(cum_func, array, axis, **kwargs):
    array = asarray(array)
    if axis is None:
        axis = tuple(range(array.ndim))
    if isinstance(axis, int):
        axis = (axis,)

    out = array
    for ax in axis:
        out = cum_func(out, axis=ax, **kwargs)
    return out


def cumprod(array, axis=None, **kwargs):
    """N-dimensional version of cumprod."""
    return _nd_cum_func(cumprod_1d, array, axis, **kwargs)


def cumsum(array, axis=None, **kwargs):
    """N-dimensional version of cumsum."""
    return _nd_cum_func(cumsum_1d, array, axis, **kwargs)


_fail_on_dask_array_input_skipna = partial(
    fail_on_dask_array_input,
    msg="%r with skipna=True is not yet implemented on dask arrays",
)


def first(values, axis, skipna=None):
    """Return the first non-NA elements in this array along the given axis
    """
    if (skipna or skipna is None) and values.dtype.kind not in "iSU":
        # only bother for dtypes that can hold NaN
        _fail_on_dask_array_input_skipna(values)
        return nanfirst(values, axis)
    return take(values, 0, axis=axis)


def last(values, axis, skipna=None):
    """Return the last non-NA elements in this array along the given axis
    """
    if (skipna or skipna is None) and values.dtype.kind not in "iSU":
        # only bother for dtypes that can hold NaN
        _fail_on_dask_array_input_skipna(values)
        return nanlast(values, axis)
    return take(values, -1, axis=axis)


def rolling_window(array, axis, window, center, fill_value):
    """
    Make an ndarray with a rolling window of axis-th dimension.
    The rolling dimension will be placed at the last dimension.
    """
    if isinstance(array, dask_array_type):
        return dask_array_ops.rolling_window(array, axis, window, center, fill_value)
    else:  # np.ndarray
        return nputils.rolling_window(array, axis, window, center, fill_value)
