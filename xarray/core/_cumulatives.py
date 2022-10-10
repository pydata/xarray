"""Mixin classes with cumulative operations."""
# This file was generated using xarray.util.generate_cumulatives. Do not edit manually.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

from . import duck_array_ops
from .options import OPTIONS
from .types import Dims
from .utils import contains_only_dask_or_numpy

if TYPE_CHECKING:
    from .dataarray import DataArray
    from .dataset import Dataset

try:
    import flox
except ImportError:
    flox = None  # type: ignore


class DatasetCumulatives:
    __slots__ = ()

    def reduce(
        self,
        func: Callable[..., Any],
        dim: Dims = None,
        *,
        axis: int | Sequence[int] | None = None,
        keep_attrs: bool | None = None,
        keepdims: bool = False,
        **kwargs: Any,
    ) -> Dataset:
        raise NotImplementedError()

    def cumsum(
        self,
        dim: Dims = None,
        *,
        axis: int | Sequence[int] | None = None,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> Dataset:
        """
        Apply ``cumsum`` along some dimension of Dataset.

        Parameters
        ----------
        dim: str or sequence of str, optional
            Dimension over which to apply `cumsum`.
        axis: int or sequence of int, optional
            Axis over which to apply `cumsum`. Only one of the ‘dim‘ and ‘axis’ arguments can be supplied.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, the attributes (`attrs`) will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : Any
            Additional keyword arguments passed on to `cumsum`.

        Returns
        -------
        cumvalue : Dataset
            New Dataset object with `cumsum` applied to its data along the
            indicated dimension.


        See Also
        --------
        numpy.cumsum
        dask.array.cumsum
        DataArray.cumsum

        Examples
        --------

        >>> temperature = np.arange(1.0, 17.0).reshape(4, 4)
        >>> temperature[2, 2] = np.nan
        >>> da = xr.DataArray(
        ...     temperature,
        ...     dims=["x", "y"],
        ...     coords=dict(
        ...         lon=("x", np.arange(10, 30, 5)),
        ...         lat=("y", np.arange(40, 60, 5)),
        ...         labels=("y", ["a", "a", "b", "c"]),
        ...     ),
        ... )

        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset>
        Dimensions:  (x: 4, y: 4)
        Coordinates:
            lon      (x) int64 10 15 20 25
            lat      (y) int64 40 45 50 55
            labels   (y) <U1 'a' 'a' 'b' 'c'
        Dimensions without coordinates: x, y
        Data variables:
            da       (x, y) float64 1.0 2.0 3.0 4.0 5.0 6.0 ... 12.0 13.0 14.0 15.0 16.0


        >>> ds.cumsum()
        <xarray.Dataset>
        Dimensions:  (x: 4, y: 4)
        Dimensions without coordinates: x, y
        Data variables:
            da       (x, y) float64 1.0 3.0 6.0 10.0 6.0 ... 67.0 28.0 60.0 85.0 125.0
        >>> ds.cumsum()["da"]
        <xarray.DataArray 'da' (x: 4, y: 4)>
        array([[  1.,   3.,   6.,  10.],
               [  6.,  14.,  24.,  36.],
               [ 15.,  33.,  43.,  67.],
               [ 28.,  60.,  85., 125.]])
        Dimensions without coordinates: x, y

        """
        return self.reduce(
            func=duck_array_ops.cumsum,
            dim=dim,
            axis=axis,
            skipna=skipna,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def cumprod(
        self,
        dim: Dims = None,
        *,
        axis: int | Sequence[int] | None = None,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> Dataset:
        """
        Apply ``cumprod`` along some dimension of Dataset.

        Parameters
        ----------
        dim: str or sequence of str, optional
            Dimension over which to apply `cumprod`.
        axis: int or sequence of int, optional
            Axis over which to apply `cumprod`. Only one of the ‘dim‘ and ‘axis’ arguments can be supplied.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, the attributes (`attrs`) will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : Any
            Additional keyword arguments passed on to `cumprod`.

        Returns
        -------
        cumvalue : Dataset
            New Dataset object with `cumprod` applied to its data along the
            indicated dimension.


        See Also
        --------
        numpy.cumprod
        dask.array.cumprod
        DataArray.cumprod

        Examples
        --------

        >>> temperature = np.arange(1.0, 17.0).reshape(4, 4)
        >>> temperature[2, 2] = np.nan
        >>> da = xr.DataArray(
        ...     temperature,
        ...     dims=["x", "y"],
        ...     coords=dict(
        ...         lon=("x", np.arange(10, 30, 5)),
        ...         lat=("y", np.arange(40, 60, 5)),
        ...         labels=("y", ["a", "a", "b", "c"]),
        ...     ),
        ... )

        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset>
        Dimensions:  (x: 4, y: 4)
        Coordinates:
            lon      (x) int64 10 15 20 25
            lat      (y) int64 40 45 50 55
            labels   (y) <U1 'a' 'a' 'b' 'c'
        Dimensions without coordinates: x, y
        Data variables:
            da       (x, y) float64 1.0 2.0 3.0 4.0 5.0 6.0 ... 12.0 13.0 14.0 15.0 16.0


        >>> ds.cumprod()
        <xarray.Dataset>
        Dimensions:  (x: 4, y: 4)
        Dimensions without coordinates: x, y
        Data variables:
            da       (x, y) float64 1.0 2.0 6.0 24.0 ... 9.828e+05 3.096e+08 1.902e+12
        >>> ds.cumprod()["da"]
        <xarray.DataArray 'da' (x: 4, y: 4)>
        array([[1.00000000e+00, 2.00000000e+00, 6.00000000e+00, 2.40000000e+01],
               [5.00000000e+00, 6.00000000e+01, 1.26000000e+03, 4.03200000e+04],
               [4.50000000e+01, 5.40000000e+03, 1.13400000e+05, 4.35456000e+07],
               [5.85000000e+02, 9.82800000e+05, 3.09582000e+08, 1.90207181e+12]])
        Dimensions without coordinates: x, y

        """
        return self.reduce(
            func=duck_array_ops.cumprod,
            dim=dim,
            axis=axis,
            skipna=skipna,
            keep_attrs=keep_attrs,
            **kwargs,
        )


class DataArrayCumulatives:
    __slots__ = ()

    def reduce(
        self,
        func: Callable[..., Any],
        dim: Dims = None,
        *,
        axis: int | Sequence[int] | None = None,
        keep_attrs: bool | None = None,
        keepdims: bool = False,
        **kwargs: Any,
    ) -> DataArray:
        raise NotImplementedError()

    def cumsum(
        self,
        dim: Dims = None,
        *,
        axis: int | Sequence[int] | None = None,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> DataArray:
        """
        Apply ``cumsum`` along some dimension of DataArray.

        Parameters
        ----------
        dim: str or sequence of str, optional
            Dimension over which to apply `cumsum`.
        axis: int or sequence of int, optional
            Axis over which to apply `cumsum`. Only one of the ‘dim‘ and ‘axis’ arguments can be supplied.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, the attributes (`attrs`) will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : Any
            Additional keyword arguments passed on to `cumsum`.

        Returns
        -------
        cumvalue : DataArray
            New DataArray object with `cumsum` applied to its data along the
            indicated dimension.


        See Also
        --------
        numpy.cumsum
        dask.array.cumsum
        Dataset.cumsum

        Examples
        --------

        >>> temperature = np.arange(1.0, 17.0).reshape(4, 4)
        >>> temperature[2, 2] = np.nan
        >>> da = xr.DataArray(
        ...     temperature,
        ...     dims=["x", "y"],
        ...     coords=dict(
        ...         lon=("x", np.arange(10, 30, 5)),
        ...         lat=("y", np.arange(40, 60, 5)),
        ...         labels=("y", ["a", "a", "b", "c"]),
        ...     ),
        ... )

        >>> da
        <xarray.DataArray (x: 4, y: 4)>
        array([[ 1.,  2.,  3.,  4.],
               [ 5.,  6.,  7.,  8.],
               [ 9., 10., nan, 12.],
               [13., 14., 15., 16.]])
        Coordinates:
            lon      (x) int64 10 15 20 25
            lat      (y) int64 40 45 50 55
            labels   (y) <U1 'a' 'a' 'b' 'c'
        Dimensions without coordinates: x, y


        >>> da.cumsum()
        <xarray.DataArray (x: 4, y: 4)>
        array([[  1.,   3.,   6.,  10.],
               [  6.,  14.,  24.,  36.],
               [ 15.,  33.,  43.,  67.],
               [ 28.,  60.,  85., 125.]])
        Coordinates:
            lon      (x) int64 10 15 20 25
            lat      (y) int64 40 45 50 55
            labels   (y) <U1 'a' 'a' 'b' 'c'
        Dimensions without coordinates: x, y
        >>> da.cumsum()
        <xarray.DataArray (x: 4, y: 4)>
        array([[  1.,   3.,   6.,  10.],
               [  6.,  14.,  24.,  36.],
               [ 15.,  33.,  43.,  67.],
               [ 28.,  60.,  85., 125.]])
        Coordinates:
            lon      (x) int64 10 15 20 25
            lat      (y) int64 40 45 50 55
            labels   (y) <U1 'a' 'a' 'b' 'c'
        Dimensions without coordinates: x, y

        """
        return self.reduce(
            func=duck_array_ops.cumsum,
            dim=dim,
            axis=axis,
            skipna=skipna,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def cumprod(
        self,
        dim: Dims = None,
        *,
        axis: int | Sequence[int] | None = None,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> DataArray:
        """
        Apply ``cumprod`` along some dimension of DataArray.

        Parameters
        ----------
        dim: str or sequence of str, optional
            Dimension over which to apply `cumprod`.
        axis: int or sequence of int, optional
            Axis over which to apply `cumprod`. Only one of the ‘dim‘ and ‘axis’ arguments can be supplied.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, the attributes (`attrs`) will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : Any
            Additional keyword arguments passed on to `cumprod`.

        Returns
        -------
        cumvalue : DataArray
            New DataArray object with `cumprod` applied to its data along the
            indicated dimension.


        See Also
        --------
        numpy.cumprod
        dask.array.cumprod
        Dataset.cumprod

        Examples
        --------

        >>> temperature = np.arange(1.0, 17.0).reshape(4, 4)
        >>> temperature[2, 2] = np.nan
        >>> da = xr.DataArray(
        ...     temperature,
        ...     dims=["x", "y"],
        ...     coords=dict(
        ...         lon=("x", np.arange(10, 30, 5)),
        ...         lat=("y", np.arange(40, 60, 5)),
        ...         labels=("y", ["a", "a", "b", "c"]),
        ...     ),
        ... )

        >>> da
        <xarray.DataArray (x: 4, y: 4)>
        array([[ 1.,  2.,  3.,  4.],
               [ 5.,  6.,  7.,  8.],
               [ 9., 10., nan, 12.],
               [13., 14., 15., 16.]])
        Coordinates:
            lon      (x) int64 10 15 20 25
            lat      (y) int64 40 45 50 55
            labels   (y) <U1 'a' 'a' 'b' 'c'
        Dimensions without coordinates: x, y


        >>> da.cumprod()
        <xarray.DataArray (x: 4, y: 4)>
        array([[1.00000000e+00, 2.00000000e+00, 6.00000000e+00, 2.40000000e+01],
               [5.00000000e+00, 6.00000000e+01, 1.26000000e+03, 4.03200000e+04],
               [4.50000000e+01, 5.40000000e+03, 1.13400000e+05, 4.35456000e+07],
               [5.85000000e+02, 9.82800000e+05, 3.09582000e+08, 1.90207181e+12]])
        Coordinates:
            lon      (x) int64 10 15 20 25
            lat      (y) int64 40 45 50 55
            labels   (y) <U1 'a' 'a' 'b' 'c'
        Dimensions without coordinates: x, y
        >>> da.cumprod()
        <xarray.DataArray (x: 4, y: 4)>
        array([[1.00000000e+00, 2.00000000e+00, 6.00000000e+00, 2.40000000e+01],
               [5.00000000e+00, 6.00000000e+01, 1.26000000e+03, 4.03200000e+04],
               [4.50000000e+01, 5.40000000e+03, 1.13400000e+05, 4.35456000e+07],
               [5.85000000e+02, 9.82800000e+05, 3.09582000e+08, 1.90207181e+12]])
        Coordinates:
            lon      (x) int64 10 15 20 25
            lat      (y) int64 40 45 50 55
            labels   (y) <U1 'a' 'a' 'b' 'c'
        Dimensions without coordinates: x, y

        """
        return self.reduce(
            func=duck_array_ops.cumprod,
            dim=dim,
            axis=axis,
            skipna=skipna,
            keep_attrs=keep_attrs,
            **kwargs,
        )


class DatasetGroupByCumulatives:
    _obj: Dataset

    def reduce(
        self,
        func: Callable[..., Any],
        dim: Dims | ellipsis = None,
        *,
        axis: int | Sequence[int] | None = None,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        keepdims: bool = False,
        **kwargs: Any,
    ) -> Dataset:
        raise NotImplementedError()

    def _flox_reduce(
        self,
        dim: Dims | ellipsis,
        **kwargs: Any,
    ) -> Dataset:
        raise NotImplementedError()

    def cumsum(
        self,
        dim: Dims = None,
        *,
        axis: int | Sequence[int] | None = None,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> Dataset:
        """
        Apply ``cumsum`` along some dimension of DatasetGroupBy.

        Parameters
        ----------
        dim: str or sequence of str, optional
            Dimension over which to apply `cumsum`.
        axis: int or sequence of int, optional
            Axis over which to apply `cumsum`. Only one of the ‘dim‘ and ‘axis’ arguments can be supplied.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, the attributes (`attrs`) will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : Any
            Additional keyword arguments passed on to `cumsum`.

        Returns
        -------
        cumvalue : Dataset
            New Dataset object with `cumsum` applied to its data along the
            indicated dimension.


        See Also
        --------
        numpy.cumsum
        dask.array.cumsum
        Dataset.cumsum

        Examples
        --------

        >>> temperature = np.arange(1.0, 17.0).reshape(4, 4)
        >>> temperature[2, 2] = np.nan
        >>> da = xr.DataArray(
        ...     temperature,
        ...     dims=["x", "y"],
        ...     coords=dict(
        ...         lon=("x", np.arange(10, 30, 5)),
        ...         lat=("y", np.arange(40, 60, 5)),
        ...         labels=("y", ["a", "a", "b", "c"]),
        ...     ),
        ... )

        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset>
        Dimensions:  (x: 4, y: 4)
        Coordinates:
            lon      (x) int64 10 15 20 25
            lat      (y) int64 40 45 50 55
            labels   (y) <U1 'a' 'a' 'b' 'c'
        Dimensions without coordinates: x, y
        Data variables:
            da       (x, y) float64 1.0 2.0 3.0 4.0 5.0 6.0 ... 12.0 13.0 14.0 15.0 16.0


        >>> ds.groupby("labels").cumsum()
        <xarray.Dataset>
        Dimensions:  (x: 4, y: 4)
        Coordinates:
            lon      (x) int64 10 15 20 25
        Dimensions without coordinates: x, y
        Data variables:
            da       (x, y) float64 1.0 3.0 3.0 4.0 5.0 ... 12.0 13.0 27.0 15.0 16.0
        >>> ds.groupby("labels").cumsum()["da"]
        <xarray.DataArray 'da' (x: 4, y: 4)>
        array([[ 1.,  3.,  3.,  4.],
               [ 5., 11.,  7.,  8.],
               [ 9., 19.,  0., 12.],
               [13., 27., 15., 16.]])
        Coordinates:
            lon      (x) int64 10 15 20 25
        Dimensions without coordinates: x, y

        """
        if flox and OPTIONS["use_flox"] and contains_only_dask_or_numpy(self._obj):
            return self._flox_reduce(
                func="cumsum",
                dim=dim,
                # fill_value=fill_value,
                keep_attrs=keep_attrs,
                **kwargs,
            )
        else:
            return self.reduce(
                func=duck_array_ops.cumsum,
                dim=dim,
                axis=axis,
                skipna=skipna,
                **kwargs,
            )


class DataArrayGroupByCumulatives:
    _obj: DataArray

    def reduce(
        self,
        func: Callable[..., Any],
        dim: Dims | ellipsis = None,
        *,
        axis: int | Sequence[int] | None = None,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        keepdims: bool = False,
        **kwargs: Any,
    ) -> DataArray:
        raise NotImplementedError()

    def _flox_reduce(
        self,
        dim: Dims | ellipsis,
        **kwargs: Any,
    ) -> DataArray:
        raise NotImplementedError()

    def cumsum(
        self,
        dim: Dims = None,
        *,
        axis: int | Sequence[int] | None = None,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> DataArray:
        """
        Apply ``cumsum`` along some dimension of DataArrayGroupBy.

        Parameters
        ----------
        dim: str or sequence of str, optional
            Dimension over which to apply `cumsum`.
        axis: int or sequence of int, optional
            Axis over which to apply `cumsum`. Only one of the ‘dim‘ and ‘axis’ arguments can be supplied.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, the attributes (`attrs`) will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : Any
            Additional keyword arguments passed on to `cumsum`.

        Returns
        -------
        cumvalue : DataArray
            New DataArray object with `cumsum` applied to its data along the
            indicated dimension.


        See Also
        --------
        numpy.cumsum
        dask.array.cumsum
        DataArray.cumsum

        Examples
        --------

        >>> temperature = np.arange(1.0, 17.0).reshape(4, 4)
        >>> temperature[2, 2] = np.nan
        >>> da = xr.DataArray(
        ...     temperature,
        ...     dims=["x", "y"],
        ...     coords=dict(
        ...         lon=("x", np.arange(10, 30, 5)),
        ...         lat=("y", np.arange(40, 60, 5)),
        ...         labels=("y", ["a", "a", "b", "c"]),
        ...     ),
        ... )

        >>> da
        <xarray.DataArray (x: 4, y: 4)>
        array([[ 1.,  2.,  3.,  4.],
               [ 5.,  6.,  7.,  8.],
               [ 9., 10., nan, 12.],
               [13., 14., 15., 16.]])
        Coordinates:
            lon      (x) int64 10 15 20 25
            lat      (y) int64 40 45 50 55
            labels   (y) <U1 'a' 'a' 'b' 'c'
        Dimensions without coordinates: x, y


        >>> da.groupby("labels").cumsum()
        <xarray.DataArray (x: 4, y: 4)>
        array([[ 1.,  3.,  3.,  4.],
               [ 5., 11.,  7.,  8.],
               [ 9., 19.,  0., 12.],
               [13., 27., 15., 16.]])
        Coordinates:
            lon      (x) int64 10 15 20 25
            lat      (y) int64 40 45 50 55
            labels   (y) <U1 'a' 'a' 'b' 'c'
        Dimensions without coordinates: x, y
        >>> da.groupby("labels").cumsum()
        <xarray.DataArray (x: 4, y: 4)>
        array([[ 1.,  3.,  3.,  4.],
               [ 5., 11.,  7.,  8.],
               [ 9., 19.,  0., 12.],
               [13., 27., 15., 16.]])
        Coordinates:
            lon      (x) int64 10 15 20 25
            lat      (y) int64 40 45 50 55
            labels   (y) <U1 'a' 'a' 'b' 'c'
        Dimensions without coordinates: x, y

        """
        if flox and OPTIONS["use_flox"] and contains_only_dask_or_numpy(self._obj):
            return self._flox_reduce(
                func="cumsum",
                dim=dim,
                # fill_value=fill_value,
                keep_attrs=keep_attrs,
                **kwargs,
            )
        else:
            return self.reduce(
                func=duck_array_ops.cumsum,
                dim=dim,
                axis=axis,
                skipna=skipna,
                **kwargs,
            )
