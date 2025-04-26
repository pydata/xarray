from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Generic, Literal

import numpy as np
import pandas as pd
from pandas.core.arrays.datetimelike import dtype_to_unit

from xarray.compat.pdcompat import count_not_none
from xarray.computation.apply_ufunc import apply_ufunc
from xarray.core.common import is_np_datetime_like
from xarray.core.options import _get_keep_attrs
from xarray.core.types import T_DataWithCoords
from xarray.core.utils import module_available


def _get_alpha(
    com: float | None = None,
    span: float | None = None,
    halflife: float | None = None,
    alpha: float | None = None,
) -> float:
    """
    Convert com, span, halflife to alpha.
    """
    valid_count = count_not_none(com, span, halflife, alpha)
    if valid_count > 1:
        raise ValueError("com, span, halflife, and alpha are mutually exclusive")

    # Convert to alpha
    if com is not None:
        if com < 0:
            raise ValueError("commust satisfy: com>= 0")
        return 1 / (com + 1)
    elif span is not None:
        if span < 1:
            raise ValueError("span must satisfy: span >= 1")
        return 2 / (span + 1)
    elif halflife is not None:
        if halflife <= 0:
            raise ValueError("halflife must satisfy: halflife > 0")
        return 1 - np.exp(np.log(0.5) / halflife)
    elif alpha is not None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must satisfy: 0 < alpha <= 1")
        return alpha
    else:
        raise ValueError("Must pass one of comass, span, halflife, or alpha")


def _raise_if_array(alpha: float | np.ndarray):
    """Check if alpha is a float, raise NotImplementedError if not.

    If alpha is an array, it means window_type='halflife' with Timedelta window,
    and the operation is applied on a datetime index. The 'mean' operation is the
    only one supported for this type of operation.

    Parameters
    ----------
    alpha : float or np.ndarray
        If array, only the 'mean' operation is supported.

    Raises
    ------
    NotImplementedError
        If alpha is an array.
    """
    if not isinstance(alpha, float):
        msg = (
            "Operation not supported for window_type='halflife' with 'Timedelta' window. "
            "Only 'mean' operation is supported with those window parameters."
        )
        raise NotImplementedError(msg)


def _calculate_deltas(
    times: np.ndarray,
    halflife: pd.Timedelta,
):
    """
    Return the diff of the times divided by the half-life. These values are used in
    the calculation of the ewm mean.

    Parameters
    ----------
    times : np.ndarray, Series
        Times corresponding to the observations. Must be monotonically increasing
        and ``datetime64[ns]`` dtype.
    halflife : float, str, timedelta, optional
        Half-life specifying the decay

    Returns
    -------
    np.ndarray
        Diff of the times divided by the half-life
    """
    unit = dtype_to_unit(times.dtype)
    _times = np.asarray(times.view(np.int64), dtype=np.float64)
    _halflife = float(pd.Timedelta(halflife).as_unit(unit)._value)
    deltas = np.diff(_times) / _halflife
    deltas = np.insert(deltas, 0, 1)
    return deltas


def _verify_timedelta_requirements(
    window_type: Literal["span", "com", "halflife", "alpha"], dim_type: np.dtype
):
    """
    Check if the window type and dimension type are compatible.

    This function is called when a window with data type 'Timedelta' is used,
    and verifies that the window type is 'halflife' and the dimension type is
    datetime64.

    Parameters
    ----------
    window_type : str
        The type of the window.
    dim_type : np.dtype
        The type of the dimension.

    Raises
    ------
    ValueError
        If the window type is not 'halflife' or the dimension type is not datetime64.
    NotImplementedError
        If the window type is 'halflife' and the dimension type is not datetime64.
    """
    if window_type != "halflife":
        raise ValueError(
            "window with data type 'Timedelta' can only be used with window_type='halflife'"
        )
    if not is_np_datetime_like(dim_type):
        raise NotImplementedError(
            "window with data type 'Timedelta' must be used with a datetime64 coordinate"
        )


class RollingExp(Generic[T_DataWithCoords]):
    """
    Exponentially-weighted moving window object.

    Similar to EWM in pandas. When using a Timedelta window with window_type='halflife',
    the alpha values are computed based on the actual time differences between points,
    allowing for irregular time series. This matches pandas' implementation in
    pd.DataFrame.ewm(halflife=..., times=...).

    Parameters
    ----------
    obj : Dataset or DataArray
        Object to window.
    windows : mapping of hashable to int, float, or pd.Timedelta
        A mapping from the name of the dimension to create the rolling
        exponential window along (e.g. `time`) to the size of the moving window.
        A pd.Timedelta can be provided for datetime dimensions only,
        when using window_type='halflife'.
    window_type : {"span", "com", "halflife", "alpha"}, default: "span"
        The format of the previously supplied window. Each is a simple
        numerical transformation of the others. Described in detail:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
        When using a pd.Timedelta window, only 'halflife' is supported for window_type,
        and it must be applied to a datetime coordinate. In this case, only the 'mean'
        operation is supported.

    Returns
    -------
    RollingExp : type of input argument
    """

    def __init__(
        self,
        obj: T_DataWithCoords,
        windows: Mapping[Any, int | float | pd.Timedelta],
        window_type: Literal["span", "com", "halflife", "alpha"] = "span",
        min_weight: float = 0.0,
    ):
        if not module_available("numbagg"):
            raise ImportError(
                "numbagg >= 0.2.1 is required for rolling_exp but currently numbagg is not installed"
            )

        self.obj: T_DataWithCoords = obj
        dim, window = next(iter(windows.items()))

        if isinstance(window, pd.Timedelta):
            _verify_timedelta_requirements(window_type, self.obj[dim].dtype)
            deltas = _calculate_deltas(self.obj.get_index(dim), window)
            # Equivalent to unweighted alpha=0.5 (like in pandas implementation)
            self.alpha = 1 - (1 - 0.5) ** deltas
        else:
            self.alpha = _get_alpha(**{window_type: window})

        self.dim = dim
        self.min_weight = min_weight
        # Don't pass min_weight=0 so we can support older versions of numbagg
        kwargs = dict(alpha=self.alpha, axis=-1)
        if min_weight > 0:
            kwargs["min_weight"] = min_weight
        self.kwargs = kwargs

    def mean(self, keep_attrs: bool | None = None) -> T_DataWithCoords:
        """
        Exponentially weighted moving average.

        Parameters
        ----------
        keep_attrs : bool, default: None
            If True, the attributes (``attrs``) will be copied from the original
            object to the new one. If False, the new object will be returned
            without attributes. If None uses the global default.

        Examples
        --------
        >>> da = xr.DataArray([1, 1, 2, 2, 2], dims="x")
        >>> da.rolling_exp(x=2, window_type="span").mean()
        <xarray.DataArray (x: 5)> Size: 40B
        array([1.        , 1.        , 1.69230769, 1.9       , 1.96694215])
        Dimensions without coordinates: x
        """

        import numbagg

        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=True)

        dim_order = self.obj.dims

        return apply_ufunc(
            numbagg.move_exp_nanmean,
            self.obj,
            input_core_dims=[[self.dim]],
            kwargs=self.kwargs,
            output_core_dims=[[self.dim]],
            keep_attrs=keep_attrs,
            on_missing_core_dim="copy",
            dask="parallelized",
        ).transpose(*dim_order)

    def sum(self, keep_attrs: bool | None = None) -> T_DataWithCoords:
        """
        Exponentially weighted moving sum.

        Parameters
        ----------
        keep_attrs : bool, default: None
            If True, the attributes (``attrs``) will be copied from the original
            object to the new one. If False, the new object will be returned
            without attributes. If None uses the global default.

        Examples
        --------
        >>> da = xr.DataArray([1, 1, 2, 2, 2], dims="x")
        >>> da.rolling_exp(x=2, window_type="span").sum()
        <xarray.DataArray (x: 5)> Size: 40B
        array([1.        , 1.33333333, 2.44444444, 2.81481481, 2.9382716 ])
        Dimensions without coordinates: x
        """
        _raise_if_array(self.alpha)

        import numbagg

        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=True)

        dim_order = self.obj.dims

        return apply_ufunc(
            numbagg.move_exp_nansum,
            self.obj,
            input_core_dims=[[self.dim]],
            kwargs=self.kwargs,
            output_core_dims=[[self.dim]],
            keep_attrs=keep_attrs,
            on_missing_core_dim="copy",
            dask="parallelized",
        ).transpose(*dim_order)

    def std(self) -> T_DataWithCoords:
        """
        Exponentially weighted moving standard deviation.

        `keep_attrs` is always True for this method. Drop attrs separately to remove attrs.

        Examples
        --------
        >>> da = xr.DataArray([1, 1, 2, 2, 2], dims="x")
        >>> da.rolling_exp(x=2, window_type="span").std()
        <xarray.DataArray (x: 5)> Size: 40B
        array([       nan, 0.        , 0.67936622, 0.42966892, 0.25389527])
        Dimensions without coordinates: x
        """
        _raise_if_array(self.alpha)

        import numbagg

        dim_order = self.obj.dims

        return apply_ufunc(
            numbagg.move_exp_nanstd,
            self.obj,
            input_core_dims=[[self.dim]],
            kwargs=self.kwargs,
            output_core_dims=[[self.dim]],
            keep_attrs=True,
            on_missing_core_dim="copy",
            dask="parallelized",
        ).transpose(*dim_order)

    def var(self) -> T_DataWithCoords:
        """
        Exponentially weighted moving variance.

        `keep_attrs` is always True for this method. Drop attrs separately to remove attrs.

        Examples
        --------
        >>> da = xr.DataArray([1, 1, 2, 2, 2], dims="x")
        >>> da.rolling_exp(x=2, window_type="span").var()
        <xarray.DataArray (x: 5)> Size: 40B
        array([       nan, 0.        , 0.46153846, 0.18461538, 0.06446281])
        Dimensions without coordinates: x
        """
        _raise_if_array(self.alpha)
        dim_order = self.obj.dims
        import numbagg

        return apply_ufunc(
            numbagg.move_exp_nanvar,
            self.obj,
            input_core_dims=[[self.dim]],
            kwargs=self.kwargs,
            output_core_dims=[[self.dim]],
            keep_attrs=True,
            on_missing_core_dim="copy",
            dask="parallelized",
        ).transpose(*dim_order)

    def cov(self, other: T_DataWithCoords) -> T_DataWithCoords:
        """
        Exponentially weighted moving covariance.

        `keep_attrs` is always True for this method. Drop attrs separately to remove attrs.

        Examples
        --------
        >>> da = xr.DataArray([1, 1, 2, 2, 2], dims="x")
        >>> da.rolling_exp(x=2, window_type="span").cov(da**2)
        <xarray.DataArray (x: 5)> Size: 40B
        array([       nan, 0.        , 1.38461538, 0.55384615, 0.19338843])
        Dimensions without coordinates: x
        """
        _raise_if_array(self.alpha)
        dim_order = self.obj.dims
        import numbagg

        return apply_ufunc(
            numbagg.move_exp_nancov,
            self.obj,
            other,
            input_core_dims=[[self.dim], [self.dim]],
            kwargs=self.kwargs,
            output_core_dims=[[self.dim]],
            keep_attrs=True,
            on_missing_core_dim="copy",
            dask="parallelized",
        ).transpose(*dim_order)

    def corr(self, other: T_DataWithCoords) -> T_DataWithCoords:
        """
        Exponentially weighted moving correlation.

        `keep_attrs` is always True for this method. Drop attrs separately to remove attrs.

        Examples
        --------
        >>> da = xr.DataArray([1, 1, 2, 2, 2], dims="x")
        >>> da.rolling_exp(x=2, window_type="span").corr(da.shift(x=1))
        <xarray.DataArray (x: 5)> Size: 40B
        array([       nan,        nan,        nan, 0.4330127 , 0.48038446])
        Dimensions without coordinates: x
        """
        _raise_if_array(self.alpha)
        dim_order = self.obj.dims
        import numbagg

        return apply_ufunc(
            numbagg.move_exp_nancorr,
            self.obj,
            other,
            input_core_dims=[[self.dim], [self.dim]],
            kwargs=self.kwargs,
            output_core_dims=[[self.dim]],
            keep_attrs=True,
            on_missing_core_dim="copy",
            dask="parallelized",
        ).transpose(*dim_order)
