"""Mixin classes with reduction operations."""
# This file was generated using xarray.util.generate_reductions. Do not edit manually.

from typing import TYPE_CHECKING, Any, Callable, Hashable, Optional, Sequence, Union

from . import duck_array_ops

if TYPE_CHECKING:
    from .dataset import Dataset
    from .dataarray import DataArray


class DatasetReductions:
    __slots__ = ()

    def reduce(
        self,
        func: Callable[..., Any],
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        axis: Union[None, int, Sequence[int]] = None,
        keep_attrs: bool = None,
        keepdims: bool = False,
        **kwargs: Any,
    ) -> "Dataset":
        return NotImplemented

    def count(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``count`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``count``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``count`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``count`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.count
        dask.array.count
        DataArray.count
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.count()
        """
        return self.reduce(
            duck_array_ops.count,
            dim=dim,
            numeric_only=False,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def all(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``all`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``all``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``all`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``all`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.all
        dask.array.all
        DataArray.all
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([True, True, True, True, True, False], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.all()
        """
        return self.reduce(
            duck_array_ops.array_all,
            dim=dim,
            numeric_only=False,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def any(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``any`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``any``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``any`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``any`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.any
        dask.array.any
        DataArray.any
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([True, True, True, True, True, False], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.any()
        """
        return self.reduce(
            duck_array_ops.array_any,
            dim=dim,
            numeric_only=False,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def max(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``max`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``max``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``max`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``max`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.max
        dask.array.max
        DataArray.max
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.max()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.max(skipna=False)
        """
        return self.reduce(
            duck_array_ops.max,
            dim=dim,
            skipna=skipna,
            numeric_only=False,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def min(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``min`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``min``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``min`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``min`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.min
        dask.array.min
        DataArray.min
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.min()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.min(skipna=False)
        """
        return self.reduce(
            duck_array_ops.min,
            dim=dim,
            skipna=skipna,
            numeric_only=False,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def mean(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``mean`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``mean``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``mean`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``mean`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.mean
        dask.array.mean
        DataArray.mean
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.mean()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.mean(skipna=False)
        """
        return self.reduce(
            duck_array_ops.mean,
            dim=dim,
            skipna=skipna,
            numeric_only=True,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def prod(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        min_count: Optional[int] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``prod`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``prod``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        min_count : int, default: None
            The required number of valid values to perform the operation. If
            fewer than min_count non-NA values are present the result will be
            NA. Only used if skipna is set to True or defaults to True for the
            array's dtype. Changed in version 0.17.0: if specified on an integer
            array and skipna=True, the result will be a float array.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``prod`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``prod`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.prod
        dask.array.prod
        DataArray.prod
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.prod()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.prod(skipna=False)

        Specify ``min_count`` for finer control over when NaNs are ignored.

        >>> ds.prod(skipna=True, min_count=2)
        """
        return self.reduce(
            duck_array_ops.prod,
            dim=dim,
            skipna=skipna,
            min_count=min_count,
            numeric_only=True,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def sum(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        min_count: Optional[int] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``sum`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``sum``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        min_count : int, default: None
            The required number of valid values to perform the operation. If
            fewer than min_count non-NA values are present the result will be
            NA. Only used if skipna is set to True or defaults to True for the
            array's dtype. Changed in version 0.17.0: if specified on an integer
            array and skipna=True, the result will be a float array.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``sum`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``sum`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.sum
        dask.array.sum
        DataArray.sum
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.sum()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.sum(skipna=False)

        Specify ``min_count`` for finer control over when NaNs are ignored.

        >>> ds.sum(skipna=True, min_count=2)
        """
        return self.reduce(
            duck_array_ops.sum,
            dim=dim,
            skipna=skipna,
            min_count=min_count,
            numeric_only=True,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def std(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        ddof: int = 0,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``std`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``std``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        ddof : int, default: 0
            “Delta Degrees of Freedom”: the divisor used in the calculation is ``N - ddof``,
            where ``N`` represents the number of elements.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``std`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``std`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.std
        dask.array.std
        DataArray.std
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.std()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.std(skipna=False)

        Specify ``ddof=1`` for an unbiased estimate.

        >>> ds.std(skipna=True, ddof=1)
        """
        return self.reduce(
            duck_array_ops.std,
            dim=dim,
            skipna=skipna,
            ddof=ddof,
            numeric_only=True,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def var(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        ddof: int = 0,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``var`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``var``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        ddof : int, default: 0
            “Delta Degrees of Freedom”: the divisor used in the calculation is ``N - ddof``,
            where ``N`` represents the number of elements.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``var`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``var`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.var
        dask.array.var
        DataArray.var
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.var()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.var(skipna=False)

        Specify ``ddof=1`` for an unbiased estimate.

        >>> ds.var(skipna=True, ddof=1)
        """
        return self.reduce(
            duck_array_ops.var,
            dim=dim,
            skipna=skipna,
            ddof=ddof,
            numeric_only=True,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def median(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``median`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``median``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``median`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``median`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.median
        dask.array.median
        DataArray.median
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.median()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.median(skipna=False)
        """
        return self.reduce(
            duck_array_ops.median,
            dim=dim,
            skipna=skipna,
            numeric_only=True,
            keep_attrs=keep_attrs,
            **kwargs,
        )


class DataArrayReductions:
    __slots__ = ()

    def reduce(
        self,
        func: Callable[..., Any],
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        axis: Union[None, int, Sequence[int]] = None,
        keep_attrs: bool = None,
        keepdims: bool = False,
        **kwargs: Any,
    ) -> "DataArray":
        return NotImplemented

    def count(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``count`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``count``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``count`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``count`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.count
        dask.array.count
        Dataset.count
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.count()
        """
        return self.reduce(
            duck_array_ops.count,
            dim=dim,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def all(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``all`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``all``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``all`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``all`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.all
        dask.array.all
        Dataset.all
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([True, True, True, True, True, False], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.all()
        """
        return self.reduce(
            duck_array_ops.array_all,
            dim=dim,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def any(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``any`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``any``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``any`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``any`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.any
        dask.array.any
        Dataset.any
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([True, True, True, True, True, False], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.any()
        """
        return self.reduce(
            duck_array_ops.array_any,
            dim=dim,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def max(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``max`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``max``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``max`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``max`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.max
        dask.array.max
        Dataset.max
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.max()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.max(skipna=False)
        """
        return self.reduce(
            duck_array_ops.max,
            dim=dim,
            skipna=skipna,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def min(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``min`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``min``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``min`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``min`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.min
        dask.array.min
        Dataset.min
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.min()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.min(skipna=False)
        """
        return self.reduce(
            duck_array_ops.min,
            dim=dim,
            skipna=skipna,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def mean(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``mean`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``mean``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``mean`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``mean`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.mean
        dask.array.mean
        Dataset.mean
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.mean()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.mean(skipna=False)
        """
        return self.reduce(
            duck_array_ops.mean,
            dim=dim,
            skipna=skipna,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def prod(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        min_count: Optional[int] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``prod`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``prod``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        min_count : int, default: None
            The required number of valid values to perform the operation. If
            fewer than min_count non-NA values are present the result will be
            NA. Only used if skipna is set to True or defaults to True for the
            array's dtype. Changed in version 0.17.0: if specified on an integer
            array and skipna=True, the result will be a float array.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``prod`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``prod`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.prod
        dask.array.prod
        Dataset.prod
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.prod()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.prod(skipna=False)

        Specify ``min_count`` for finer control over when NaNs are ignored.

        >>> da.prod(skipna=True, min_count=2)
        """
        return self.reduce(
            duck_array_ops.prod,
            dim=dim,
            skipna=skipna,
            min_count=min_count,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def sum(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        min_count: Optional[int] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``sum`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``sum``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        min_count : int, default: None
            The required number of valid values to perform the operation. If
            fewer than min_count non-NA values are present the result will be
            NA. Only used if skipna is set to True or defaults to True for the
            array's dtype. Changed in version 0.17.0: if specified on an integer
            array and skipna=True, the result will be a float array.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``sum`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``sum`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.sum
        dask.array.sum
        Dataset.sum
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.sum()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.sum(skipna=False)

        Specify ``min_count`` for finer control over when NaNs are ignored.

        >>> da.sum(skipna=True, min_count=2)
        """
        return self.reduce(
            duck_array_ops.sum,
            dim=dim,
            skipna=skipna,
            min_count=min_count,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def std(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        ddof: int = 0,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``std`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``std``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        ddof : int, default: 0
            “Delta Degrees of Freedom”: the divisor used in the calculation is ``N - ddof``,
            where ``N`` represents the number of elements.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``std`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``std`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.std
        dask.array.std
        Dataset.std
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.std()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.std(skipna=False)

        Specify ``ddof=1`` for an unbiased estimate.

        >>> da.std(skipna=True, ddof=1)
        """
        return self.reduce(
            duck_array_ops.std,
            dim=dim,
            skipna=skipna,
            ddof=ddof,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def var(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        ddof: int = 0,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``var`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``var``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        ddof : int, default: 0
            “Delta Degrees of Freedom”: the divisor used in the calculation is ``N - ddof``,
            where ``N`` represents the number of elements.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``var`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``var`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.var
        dask.array.var
        Dataset.var
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.var()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.var(skipna=False)

        Specify ``ddof=1`` for an unbiased estimate.

        >>> da.var(skipna=True, ddof=1)
        """
        return self.reduce(
            duck_array_ops.var,
            dim=dim,
            skipna=skipna,
            ddof=ddof,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def median(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``median`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``median``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``median`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``median`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.median
        dask.array.median
        Dataset.median
        :ref:`agg`
            User guide on reduction or aggregation operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.median()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.median(skipna=False)
        """
        return self.reduce(
            duck_array_ops.median,
            dim=dim,
            skipna=skipna,
            keep_attrs=keep_attrs,
            **kwargs,
        )


class DatasetGroupByReductions:
    __slots__ = ()

    def reduce(
        self,
        func: Callable[..., Any],
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        axis: Union[None, int, Sequence[int]] = None,
        keep_attrs: bool = None,
        keepdims: bool = False,
        **kwargs: Any,
    ) -> "Dataset":
        return NotImplemented

    def count(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``count`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``count``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``count`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``count`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.count
        dask.array.count
        Dataset.count
        :ref:`groupby`
            User guide on groupby operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.groupby("labels").count()
        """
        return self.reduce(
            duck_array_ops.count,
            dim=dim,
            numeric_only=False,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def all(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``all`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``all``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``all`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``all`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.all
        dask.array.all
        Dataset.all
        :ref:`groupby`
            User guide on groupby operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([True, True, True, True, True, False], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.groupby("labels").all()
        """
        return self.reduce(
            duck_array_ops.array_all,
            dim=dim,
            numeric_only=False,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def any(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``any`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``any``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``any`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``any`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.any
        dask.array.any
        Dataset.any
        :ref:`groupby`
            User guide on groupby operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([True, True, True, True, True, False], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.groupby("labels").any()
        """
        return self.reduce(
            duck_array_ops.array_any,
            dim=dim,
            numeric_only=False,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def max(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``max`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``max``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``max`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``max`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.max
        dask.array.max
        Dataset.max
        :ref:`groupby`
            User guide on groupby operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.groupby("labels").max()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.groupby("labels").max(skipna=False)
        """
        return self.reduce(
            duck_array_ops.max,
            dim=dim,
            skipna=skipna,
            numeric_only=False,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def min(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``min`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``min``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``min`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``min`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.min
        dask.array.min
        Dataset.min
        :ref:`groupby`
            User guide on groupby operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.groupby("labels").min()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.groupby("labels").min(skipna=False)
        """
        return self.reduce(
            duck_array_ops.min,
            dim=dim,
            skipna=skipna,
            numeric_only=False,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def mean(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``mean`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``mean``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``mean`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``mean`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.mean
        dask.array.mean
        Dataset.mean
        :ref:`groupby`
            User guide on groupby operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.groupby("labels").mean()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.groupby("labels").mean(skipna=False)
        """
        return self.reduce(
            duck_array_ops.mean,
            dim=dim,
            skipna=skipna,
            numeric_only=True,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def prod(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        min_count: Optional[int] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``prod`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``prod``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        min_count : int, default: None
            The required number of valid values to perform the operation. If
            fewer than min_count non-NA values are present the result will be
            NA. Only used if skipna is set to True or defaults to True for the
            array's dtype. Changed in version 0.17.0: if specified on an integer
            array and skipna=True, the result will be a float array.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``prod`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``prod`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.prod
        dask.array.prod
        Dataset.prod
        :ref:`groupby`
            User guide on groupby operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.groupby("labels").prod()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.groupby("labels").prod(skipna=False)

        Specify ``min_count`` for finer control over when NaNs are ignored.

        >>> ds.groupby("labels").prod(skipna=True, min_count=2)
        """
        return self.reduce(
            duck_array_ops.prod,
            dim=dim,
            skipna=skipna,
            min_count=min_count,
            numeric_only=True,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def sum(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        min_count: Optional[int] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``sum`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``sum``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        min_count : int, default: None
            The required number of valid values to perform the operation. If
            fewer than min_count non-NA values are present the result will be
            NA. Only used if skipna is set to True or defaults to True for the
            array's dtype. Changed in version 0.17.0: if specified on an integer
            array and skipna=True, the result will be a float array.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``sum`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``sum`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.sum
        dask.array.sum
        Dataset.sum
        :ref:`groupby`
            User guide on groupby operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.groupby("labels").sum()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.groupby("labels").sum(skipna=False)

        Specify ``min_count`` for finer control over when NaNs are ignored.

        >>> ds.groupby("labels").sum(skipna=True, min_count=2)
        """
        return self.reduce(
            duck_array_ops.sum,
            dim=dim,
            skipna=skipna,
            min_count=min_count,
            numeric_only=True,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def std(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        ddof: int = 0,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``std`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``std``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        ddof : int, default: 0
            “Delta Degrees of Freedom”: the divisor used in the calculation is ``N - ddof``,
            where ``N`` represents the number of elements.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``std`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``std`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.std
        dask.array.std
        Dataset.std
        :ref:`groupby`
            User guide on groupby operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.groupby("labels").std()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.groupby("labels").std(skipna=False)

        Specify ``ddof=1`` for an unbiased estimate.

        >>> ds.groupby("labels").std(skipna=True, ddof=1)
        """
        return self.reduce(
            duck_array_ops.std,
            dim=dim,
            skipna=skipna,
            ddof=ddof,
            numeric_only=True,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def var(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        ddof: int = 0,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``var`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``var``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        ddof : int, default: 0
            “Delta Degrees of Freedom”: the divisor used in the calculation is ``N - ddof``,
            where ``N`` represents the number of elements.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``var`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``var`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.var
        dask.array.var
        Dataset.var
        :ref:`groupby`
            User guide on groupby operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.groupby("labels").var()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.groupby("labels").var(skipna=False)

        Specify ``ddof=1`` for an unbiased estimate.

        >>> ds.groupby("labels").var(skipna=True, ddof=1)
        """
        return self.reduce(
            duck_array_ops.var,
            dim=dim,
            skipna=skipna,
            ddof=ddof,
            numeric_only=True,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def median(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``median`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``median``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``median`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``median`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.median
        dask.array.median
        Dataset.median
        :ref:`groupby`
            User guide on groupby operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.groupby("labels").median()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.groupby("labels").median(skipna=False)
        """
        return self.reduce(
            duck_array_ops.median,
            dim=dim,
            skipna=skipna,
            numeric_only=True,
            keep_attrs=keep_attrs,
            **kwargs,
        )


class DatasetResampleReductions:
    __slots__ = ()

    def reduce(
        self,
        func: Callable[..., Any],
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        axis: Union[None, int, Sequence[int]] = None,
        keep_attrs: bool = None,
        keepdims: bool = False,
        **kwargs: Any,
    ) -> "Dataset":
        return NotImplemented

    def count(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``count`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``count``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``count`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``count`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.count
        dask.array.count
        Dataset.count
        :ref:`resampling`
            User guide on resampling operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.resample(time="3M").count()
        """
        return self.reduce(
            duck_array_ops.count,
            dim=dim,
            numeric_only=False,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def all(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``all`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``all``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``all`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``all`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.all
        dask.array.all
        Dataset.all
        :ref:`resampling`
            User guide on resampling operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([True, True, True, True, True, False], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.resample(time="3M").all()
        """
        return self.reduce(
            duck_array_ops.array_all,
            dim=dim,
            numeric_only=False,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def any(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``any`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``any``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``any`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``any`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.any
        dask.array.any
        Dataset.any
        :ref:`resampling`
            User guide on resampling operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([True, True, True, True, True, False], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.resample(time="3M").any()
        """
        return self.reduce(
            duck_array_ops.array_any,
            dim=dim,
            numeric_only=False,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def max(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``max`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``max``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``max`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``max`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.max
        dask.array.max
        Dataset.max
        :ref:`resampling`
            User guide on resampling operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.resample(time="3M").max()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.resample(time="3M").max(skipna=False)
        """
        return self.reduce(
            duck_array_ops.max,
            dim=dim,
            skipna=skipna,
            numeric_only=False,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def min(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``min`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``min``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``min`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``min`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.min
        dask.array.min
        Dataset.min
        :ref:`resampling`
            User guide on resampling operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.resample(time="3M").min()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.resample(time="3M").min(skipna=False)
        """
        return self.reduce(
            duck_array_ops.min,
            dim=dim,
            skipna=skipna,
            numeric_only=False,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def mean(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``mean`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``mean``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``mean`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``mean`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.mean
        dask.array.mean
        Dataset.mean
        :ref:`resampling`
            User guide on resampling operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.resample(time="3M").mean()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.resample(time="3M").mean(skipna=False)
        """
        return self.reduce(
            duck_array_ops.mean,
            dim=dim,
            skipna=skipna,
            numeric_only=True,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def prod(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        min_count: Optional[int] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``prod`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``prod``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        min_count : int, default: None
            The required number of valid values to perform the operation. If
            fewer than min_count non-NA values are present the result will be
            NA. Only used if skipna is set to True or defaults to True for the
            array's dtype. Changed in version 0.17.0: if specified on an integer
            array and skipna=True, the result will be a float array.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``prod`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``prod`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.prod
        dask.array.prod
        Dataset.prod
        :ref:`resampling`
            User guide on resampling operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.resample(time="3M").prod()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.resample(time="3M").prod(skipna=False)

        Specify ``min_count`` for finer control over when NaNs are ignored.

        >>> ds.resample(time="3M").prod(skipna=True, min_count=2)
        """
        return self.reduce(
            duck_array_ops.prod,
            dim=dim,
            skipna=skipna,
            min_count=min_count,
            numeric_only=True,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def sum(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        min_count: Optional[int] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``sum`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``sum``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        min_count : int, default: None
            The required number of valid values to perform the operation. If
            fewer than min_count non-NA values are present the result will be
            NA. Only used if skipna is set to True or defaults to True for the
            array's dtype. Changed in version 0.17.0: if specified on an integer
            array and skipna=True, the result will be a float array.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``sum`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``sum`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.sum
        dask.array.sum
        Dataset.sum
        :ref:`resampling`
            User guide on resampling operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.resample(time="3M").sum()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.resample(time="3M").sum(skipna=False)

        Specify ``min_count`` for finer control over when NaNs are ignored.

        >>> ds.resample(time="3M").sum(skipna=True, min_count=2)
        """
        return self.reduce(
            duck_array_ops.sum,
            dim=dim,
            skipna=skipna,
            min_count=min_count,
            numeric_only=True,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def std(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        ddof: int = 0,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``std`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``std``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        ddof : int, default: 0
            “Delta Degrees of Freedom”: the divisor used in the calculation is ``N - ddof``,
            where ``N`` represents the number of elements.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``std`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``std`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.std
        dask.array.std
        Dataset.std
        :ref:`resampling`
            User guide on resampling operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.resample(time="3M").std()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.resample(time="3M").std(skipna=False)

        Specify ``ddof=1`` for an unbiased estimate.

        >>> ds.resample(time="3M").std(skipna=True, ddof=1)
        """
        return self.reduce(
            duck_array_ops.std,
            dim=dim,
            skipna=skipna,
            ddof=ddof,
            numeric_only=True,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def var(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        ddof: int = 0,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``var`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``var``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        ddof : int, default: 0
            “Delta Degrees of Freedom”: the divisor used in the calculation is ``N - ddof``,
            where ``N`` represents the number of elements.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``var`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``var`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.var
        dask.array.var
        Dataset.var
        :ref:`resampling`
            User guide on resampling operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.resample(time="3M").var()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.resample(time="3M").var(skipna=False)

        Specify ``ddof=1`` for an unbiased estimate.

        >>> ds.resample(time="3M").var(skipna=True, ddof=1)
        """
        return self.reduce(
            duck_array_ops.var,
            dim=dim,
            skipna=skipna,
            ddof=ddof,
            numeric_only=True,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def median(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "Dataset":
        """
        Reduce this Dataset's data by applying ``median`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``median``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``median`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``median`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.median
        dask.array.median
        Dataset.median
        :ref:`resampling`
            User guide on resampling operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds

        >>> ds.resample(time="3M").median()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.resample(time="3M").median(skipna=False)
        """
        return self.reduce(
            duck_array_ops.median,
            dim=dim,
            skipna=skipna,
            numeric_only=True,
            keep_attrs=keep_attrs,
            **kwargs,
        )


class DataArrayGroupByReductions:
    __slots__ = ()

    def reduce(
        self,
        func: Callable[..., Any],
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        axis: Union[None, int, Sequence[int]] = None,
        keep_attrs: bool = None,
        keepdims: bool = False,
        **kwargs: Any,
    ) -> "DataArray":
        return NotImplemented

    def count(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``count`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``count``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``count`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``count`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.count
        dask.array.count
        DataArray.count
        :ref:`groupby`
            User guide on groupby operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.groupby("labels").count()
        """
        return self.reduce(
            duck_array_ops.count,
            dim=dim,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def all(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``all`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``all``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``all`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``all`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.all
        dask.array.all
        DataArray.all
        :ref:`groupby`
            User guide on groupby operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([True, True, True, True, True, False], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.groupby("labels").all()
        """
        return self.reduce(
            duck_array_ops.array_all,
            dim=dim,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def any(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``any`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``any``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``any`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``any`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.any
        dask.array.any
        DataArray.any
        :ref:`groupby`
            User guide on groupby operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([True, True, True, True, True, False], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.groupby("labels").any()
        """
        return self.reduce(
            duck_array_ops.array_any,
            dim=dim,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def max(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``max`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``max``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``max`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``max`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.max
        dask.array.max
        DataArray.max
        :ref:`groupby`
            User guide on groupby operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.groupby("labels").max()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.groupby("labels").max(skipna=False)
        """
        return self.reduce(
            duck_array_ops.max,
            dim=dim,
            skipna=skipna,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def min(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``min`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``min``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``min`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``min`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.min
        dask.array.min
        DataArray.min
        :ref:`groupby`
            User guide on groupby operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.groupby("labels").min()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.groupby("labels").min(skipna=False)
        """
        return self.reduce(
            duck_array_ops.min,
            dim=dim,
            skipna=skipna,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def mean(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``mean`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``mean``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``mean`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``mean`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.mean
        dask.array.mean
        DataArray.mean
        :ref:`groupby`
            User guide on groupby operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.groupby("labels").mean()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.groupby("labels").mean(skipna=False)
        """
        return self.reduce(
            duck_array_ops.mean,
            dim=dim,
            skipna=skipna,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def prod(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        min_count: Optional[int] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``prod`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``prod``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        min_count : int, default: None
            The required number of valid values to perform the operation. If
            fewer than min_count non-NA values are present the result will be
            NA. Only used if skipna is set to True or defaults to True for the
            array's dtype. Changed in version 0.17.0: if specified on an integer
            array and skipna=True, the result will be a float array.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``prod`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``prod`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.prod
        dask.array.prod
        DataArray.prod
        :ref:`groupby`
            User guide on groupby operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.groupby("labels").prod()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.groupby("labels").prod(skipna=False)

        Specify ``min_count`` for finer control over when NaNs are ignored.

        >>> da.groupby("labels").prod(skipna=True, min_count=2)
        """
        return self.reduce(
            duck_array_ops.prod,
            dim=dim,
            skipna=skipna,
            min_count=min_count,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def sum(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        min_count: Optional[int] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``sum`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``sum``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        min_count : int, default: None
            The required number of valid values to perform the operation. If
            fewer than min_count non-NA values are present the result will be
            NA. Only used if skipna is set to True or defaults to True for the
            array's dtype. Changed in version 0.17.0: if specified on an integer
            array and skipna=True, the result will be a float array.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``sum`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``sum`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.sum
        dask.array.sum
        DataArray.sum
        :ref:`groupby`
            User guide on groupby operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.groupby("labels").sum()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.groupby("labels").sum(skipna=False)

        Specify ``min_count`` for finer control over when NaNs are ignored.

        >>> da.groupby("labels").sum(skipna=True, min_count=2)
        """
        return self.reduce(
            duck_array_ops.sum,
            dim=dim,
            skipna=skipna,
            min_count=min_count,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def std(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        ddof: int = 0,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``std`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``std``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        ddof : int, default: 0
            “Delta Degrees of Freedom”: the divisor used in the calculation is ``N - ddof``,
            where ``N`` represents the number of elements.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``std`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``std`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.std
        dask.array.std
        DataArray.std
        :ref:`groupby`
            User guide on groupby operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.groupby("labels").std()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.groupby("labels").std(skipna=False)

        Specify ``ddof=1`` for an unbiased estimate.

        >>> da.groupby("labels").std(skipna=True, ddof=1)
        """
        return self.reduce(
            duck_array_ops.std,
            dim=dim,
            skipna=skipna,
            ddof=ddof,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def var(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        ddof: int = 0,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``var`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``var``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        ddof : int, default: 0
            “Delta Degrees of Freedom”: the divisor used in the calculation is ``N - ddof``,
            where ``N`` represents the number of elements.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``var`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``var`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.var
        dask.array.var
        DataArray.var
        :ref:`groupby`
            User guide on groupby operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.groupby("labels").var()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.groupby("labels").var(skipna=False)

        Specify ``ddof=1`` for an unbiased estimate.

        >>> da.groupby("labels").var(skipna=True, ddof=1)
        """
        return self.reduce(
            duck_array_ops.var,
            dim=dim,
            skipna=skipna,
            ddof=ddof,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def median(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``median`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``median``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``median`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``median`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.median
        dask.array.median
        DataArray.median
        :ref:`groupby`
            User guide on groupby operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.groupby("labels").median()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.groupby("labels").median(skipna=False)
        """
        return self.reduce(
            duck_array_ops.median,
            dim=dim,
            skipna=skipna,
            keep_attrs=keep_attrs,
            **kwargs,
        )


class DataArrayResampleReductions:
    __slots__ = ()

    def reduce(
        self,
        func: Callable[..., Any],
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        axis: Union[None, int, Sequence[int]] = None,
        keep_attrs: bool = None,
        keepdims: bool = False,
        **kwargs: Any,
    ) -> "DataArray":
        return NotImplemented

    def count(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``count`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``count``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``count`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``count`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.count
        dask.array.count
        DataArray.count
        :ref:`resampling`
            User guide on resampling operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.resample(time="3M").count()
        """
        return self.reduce(
            duck_array_ops.count,
            dim=dim,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def all(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``all`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``all``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``all`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``all`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.all
        dask.array.all
        DataArray.all
        :ref:`resampling`
            User guide on resampling operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([True, True, True, True, True, False], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.resample(time="3M").all()
        """
        return self.reduce(
            duck_array_ops.array_all,
            dim=dim,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def any(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``any`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``any``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``any`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``any`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.any
        dask.array.any
        DataArray.any
        :ref:`resampling`
            User guide on resampling operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([True, True, True, True, True, False], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.resample(time="3M").any()
        """
        return self.reduce(
            duck_array_ops.array_any,
            dim=dim,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def max(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``max`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``max``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``max`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``max`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.max
        dask.array.max
        DataArray.max
        :ref:`resampling`
            User guide on resampling operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.resample(time="3M").max()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.resample(time="3M").max(skipna=False)
        """
        return self.reduce(
            duck_array_ops.max,
            dim=dim,
            skipna=skipna,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def min(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``min`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``min``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``min`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``min`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.min
        dask.array.min
        DataArray.min
        :ref:`resampling`
            User guide on resampling operations.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.resample(time="3M").min()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.resample(time="3M").min(skipna=False)
        """
        return self.reduce(
            duck_array_ops.min,
            dim=dim,
            skipna=skipna,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def mean(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``mean`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``mean``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``mean`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``mean`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.mean
        dask.array.mean
        DataArray.mean
        :ref:`resampling`
            User guide on resampling operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.resample(time="3M").mean()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.resample(time="3M").mean(skipna=False)
        """
        return self.reduce(
            duck_array_ops.mean,
            dim=dim,
            skipna=skipna,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def prod(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        min_count: Optional[int] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``prod`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``prod``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        min_count : int, default: None
            The required number of valid values to perform the operation. If
            fewer than min_count non-NA values are present the result will be
            NA. Only used if skipna is set to True or defaults to True for the
            array's dtype. Changed in version 0.17.0: if specified on an integer
            array and skipna=True, the result will be a float array.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``prod`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``prod`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.prod
        dask.array.prod
        DataArray.prod
        :ref:`resampling`
            User guide on resampling operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.resample(time="3M").prod()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.resample(time="3M").prod(skipna=False)

        Specify ``min_count`` for finer control over when NaNs are ignored.

        >>> da.resample(time="3M").prod(skipna=True, min_count=2)
        """
        return self.reduce(
            duck_array_ops.prod,
            dim=dim,
            skipna=skipna,
            min_count=min_count,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def sum(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        min_count: Optional[int] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``sum`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``sum``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        min_count : int, default: None
            The required number of valid values to perform the operation. If
            fewer than min_count non-NA values are present the result will be
            NA. Only used if skipna is set to True or defaults to True for the
            array's dtype. Changed in version 0.17.0: if specified on an integer
            array and skipna=True, the result will be a float array.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``sum`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``sum`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.sum
        dask.array.sum
        DataArray.sum
        :ref:`resampling`
            User guide on resampling operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.resample(time="3M").sum()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.resample(time="3M").sum(skipna=False)

        Specify ``min_count`` for finer control over when NaNs are ignored.

        >>> da.resample(time="3M").sum(skipna=True, min_count=2)
        """
        return self.reduce(
            duck_array_ops.sum,
            dim=dim,
            skipna=skipna,
            min_count=min_count,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def std(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        ddof: int = 0,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``std`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``std``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        ddof : int, default: 0
            “Delta Degrees of Freedom”: the divisor used in the calculation is ``N - ddof``,
            where ``N`` represents the number of elements.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``std`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``std`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.std
        dask.array.std
        DataArray.std
        :ref:`resampling`
            User guide on resampling operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.resample(time="3M").std()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.resample(time="3M").std(skipna=False)

        Specify ``ddof=1`` for an unbiased estimate.

        >>> da.resample(time="3M").std(skipna=True, ddof=1)
        """
        return self.reduce(
            duck_array_ops.std,
            dim=dim,
            skipna=skipna,
            ddof=ddof,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def var(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        ddof: int = 0,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``var`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``var``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        ddof : int, default: 0
            “Delta Degrees of Freedom”: the divisor used in the calculation is ``N - ddof``,
            where ``N`` represents the number of elements.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``var`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``var`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.var
        dask.array.var
        DataArray.var
        :ref:`resampling`
            User guide on resampling operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.resample(time="3M").var()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.resample(time="3M").var(skipna=False)

        Specify ``ddof=1`` for an unbiased estimate.

        >>> da.resample(time="3M").var(skipna=True, ddof=1)
        """
        return self.reduce(
            duck_array_ops.var,
            dim=dim,
            skipna=skipna,
            ddof=ddof,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def median(
        self,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> "DataArray":
        """
        Reduce this DataArray's data by applying ``median`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``median``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If None, will reduce over all dimensions.
        skipna : bool, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``median`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``median`` applied to its data and the
            indicated dimension(s) removed

        See Also
        --------
        numpy.median
        dask.array.median
        DataArray.median
        :ref:`resampling`
            User guide on resampling operations.

        Notes
        -----
        Non-numeric variables will be removed prior to reducing.

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da

        >>> da.resample(time="3M").median()

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.resample(time="3M").median(skipna=False)
        """
        return self.reduce(
            duck_array_ops.median,
            dim=dim,
            skipna=skipna,
            keep_attrs=keep_attrs,
            **kwargs,
        )
