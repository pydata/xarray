"""Mixin classes with reduction operations."""
# This file was generated using xarray.util.generate_reductions. Do not edit manually.

from typing import Optional

from . import duck_array_ops
from .types import T_DataArray, T_Dataset


class DatasetGroupByReductions:
    __slots__ = ()

    def count(
        self,
        dim=None,
        keep_attrs=None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``count`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``count``.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``count`` on this object's data.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``count`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     [1, 2, 3, 1, 2, np.nan],
        ...     dims="x",
        ...     coords=dict(labels=("x", np.array(["a", "b", "c", "c", "b", "a"]))),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds.groupby("labels").count()
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) int64 1 2 2

        See Also
        --------
        :ref:`groupby`
            User guide on groupby operations.
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
        dim=None,
        keep_attrs=None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``all`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``all``.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``all`` on this object's data.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``all`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([True, True, True, True, True, False], dtype=bool),
        ...     dims="x",
        ...     coords=dict(labels=("x", np.array(["a", "b", "c", "c", "b", "a"]))),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds.groupby("labels").all()
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) bool False True True

        See Also
        --------
        :ref:`groupby`
            User guide on groupby operations.
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
        dim=None,
        keep_attrs=None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``any`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``any``.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``any`` on this object's data.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``any`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([True, True, True, True, True, False], dtype=bool),
        ...     dims="x",
        ...     coords=dict(labels=("x", np.array(["a", "b", "c", "c", "b", "a"]))),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds.groupby("labels").any()
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) bool True True True

        See Also
        --------
        :ref:`groupby`
            User guide on groupby operations.
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
        dim=None,
        skipna: bool = True,
        keep_attrs=None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``max`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``max``.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``max`` on this object's data.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``max`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     [1, 2, 3, 1, 2, np.nan],
        ...     dims="x",
        ...     coords=dict(labels=("x", np.array(["a", "b", "c", "c", "b", "a"]))),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds.groupby("labels").max()
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 1.0 2.0 3.0
        >>> ds.groupby("labels").max(skipna=False)
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 nan 2.0 3.0

        See Also
        --------
        :ref:`groupby`
            User guide on groupby operations.
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
        dim=None,
        skipna: bool = True,
        keep_attrs=None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``min`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``min``.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``min`` on this object's data.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``min`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     [1, 2, 3, 1, 2, np.nan],
        ...     dims="x",
        ...     coords=dict(labels=("x", np.array(["a", "b", "c", "c", "b", "a"]))),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds.groupby("labels").min()
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 1.0 2.0 1.0
        >>> ds.groupby("labels").min(skipna=False)
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 nan 2.0 1.0

        See Also
        --------
        :ref:`groupby`
            User guide on groupby operations.
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
        dim=None,
        skipna: bool = True,
        keep_attrs=None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``mean`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``mean``.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``mean`` on this object's data.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``mean`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     [1, 2, 3, 1, 2, np.nan],
        ...     dims="x",
        ...     coords=dict(labels=("x", np.array(["a", "b", "c", "c", "b", "a"]))),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds.groupby("labels").mean()
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 1.0 2.0 2.0
        >>> ds.groupby("labels").mean(skipna=False)
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 nan 2.0 2.0

        See Also
        --------
        :ref:`groupby`
            User guide on groupby operations.
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
        dim=None,
        skipna: bool = True,
        min_count: Optional[int] = None,
        keep_attrs=None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``prod`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``prod``.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
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

        Returns
        -------
        reduced : Dataset
            New Dataset with ``prod`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     [1, 2, 3, 1, 2, np.nan],
        ...     dims="x",
        ...     coords=dict(labels=("x", np.array(["a", "b", "c", "c", "b", "a"]))),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds.groupby("labels").prod()
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 1.0 4.0 3.0
        >>> ds.groupby("labels").prod(skipna=False)
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 nan 4.0 3.0

        See Also
        --------
        :ref:`groupby`
            User guide on groupby operations.
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
        dim=None,
        skipna: bool = True,
        min_count: Optional[int] = None,
        keep_attrs=None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``sum`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``sum``.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
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

        Returns
        -------
        reduced : Dataset
            New Dataset with ``sum`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     [1, 2, 3, 1, 2, np.nan],
        ...     dims="x",
        ...     coords=dict(labels=("x", np.array(["a", "b", "c", "c", "b", "a"]))),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds.groupby("labels").sum()
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 1.0 4.0 4.0
        >>> ds.groupby("labels").sum(skipna=False)
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 nan 4.0 4.0

        See Also
        --------
        :ref:`groupby`
            User guide on groupby operations.
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
        dim=None,
        skipna: bool = True,
        keep_attrs=None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``std`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``std``.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``std`` on this object's data.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``std`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     [1, 2, 3, 1, 2, np.nan],
        ...     dims="x",
        ...     coords=dict(labels=("x", np.array(["a", "b", "c", "c", "b", "a"]))),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds.groupby("labels").std()
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 0.0 0.0 1.0
        >>> ds.groupby("labels").std(skipna=False)
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 nan 0.0 1.0

        See Also
        --------
        :ref:`groupby`
            User guide on groupby operations.
        """
        return self.reduce(
            duck_array_ops.std,
            dim=dim,
            skipna=skipna,
            numeric_only=True,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def var(
        self,
        dim=None,
        skipna: bool = True,
        keep_attrs=None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``var`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``var``.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``var`` on this object's data.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``var`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     [1, 2, 3, 1, 2, np.nan],
        ...     dims="x",
        ...     coords=dict(labels=("x", np.array(["a", "b", "c", "c", "b", "a"]))),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds.groupby("labels").var()
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 0.0 0.0 1.0
        >>> ds.groupby("labels").var(skipna=False)
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 nan 0.0 1.0

        See Also
        --------
        :ref:`groupby`
            User guide on groupby operations.
        """
        return self.reduce(
            duck_array_ops.var,
            dim=dim,
            skipna=skipna,
            numeric_only=True,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def median(
        self,
        dim=None,
        skipna: bool = True,
        keep_attrs=None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``median`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``median``.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``median`` on this object's data.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``median`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     [1, 2, 3, 1, 2, np.nan],
        ...     dims="x",
        ...     coords=dict(labels=("x", np.array(["a", "b", "c", "c", "b", "a"]))),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds.groupby("labels").median()
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 1.0 2.0 2.0
        >>> ds.groupby("labels").median(skipna=False)
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 nan 2.0 2.0

        See Also
        --------
        :ref:`groupby`
            User guide on groupby operations.
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

    def count(
        self,
        dim=None,
        keep_attrs=None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``count`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``count``.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``count`` on this object's data.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``count`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6))
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds.resample(time="3M").count()
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) int64 1 3 2

        See Also
        --------
        :ref:`resampling`
            User guide on resampling operations.
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
        dim=None,
        keep_attrs=None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``all`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``all``.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``all`` on this object's data.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``all`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([True, True, True, True, True, False], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6))
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds.resample(time="3M").all()
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) bool True True False

        See Also
        --------
        :ref:`resampling`
            User guide on resampling operations.
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
        dim=None,
        keep_attrs=None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``any`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``any``.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``any`` on this object's data.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``any`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([True, True, True, True, True, False], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6))
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds.resample(time="3M").any()
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) bool True True True

        See Also
        --------
        :ref:`resampling`
            User guide on resampling operations.
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
        dim=None,
        skipna: bool = True,
        keep_attrs=None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``max`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``max``.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``max`` on this object's data.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``max`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6))
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds.resample(time="3M").max()
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) bool True True True
        >>> ds.resample(time="3M").max(skipna=False)
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) bool True True True

        See Also
        --------
        :ref:`resampling`
            User guide on resampling operations.
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
        dim=None,
        skipna: bool = True,
        keep_attrs=None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``min`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``min``.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``min`` on this object's data.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``min`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6))
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds.resample(time="3M").min()
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) bool True True True
        >>> ds.resample(time="3M").min(skipna=False)
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) bool True True True

        See Also
        --------
        :ref:`resampling`
            User guide on resampling operations.
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
        dim=None,
        skipna: bool = True,
        keep_attrs=None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``mean`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``mean``.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``mean`` on this object's data.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``mean`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6))
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds.resample(time="3M").mean()
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) float64 1.0 1.0 1.0
        >>> ds.resample(time="3M").mean(skipna=False)
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) float64 1.0 1.0 1.0

        See Also
        --------
        :ref:`resampling`
            User guide on resampling operations.
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
        dim=None,
        skipna: bool = True,
        min_count: Optional[int] = None,
        keep_attrs=None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``prod`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``prod``.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
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

        Returns
        -------
        reduced : Dataset
            New Dataset with ``prod`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6))
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds.resample(time="3M").prod()
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) int64 1 1 1
        >>> ds.resample(time="3M").prod(skipna=False)
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) int64 1 1 1

        See Also
        --------
        :ref:`resampling`
            User guide on resampling operations.
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
        dim=None,
        skipna: bool = True,
        min_count: Optional[int] = None,
        keep_attrs=None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``sum`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``sum``.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
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

        Returns
        -------
        reduced : Dataset
            New Dataset with ``sum`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6))
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds.resample(time="3M").sum()
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) int64 1 3 2
        >>> ds.resample(time="3M").sum(skipna=False)
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) int64 1 3 2

        See Also
        --------
        :ref:`resampling`
            User guide on resampling operations.
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
        dim=None,
        skipna: bool = True,
        keep_attrs=None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``std`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``std``.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``std`` on this object's data.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``std`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6))
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds.resample(time="3M").std()
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) float64 0.0 0.0 0.0
        >>> ds.resample(time="3M").std(skipna=False)
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) float64 0.0 0.0 0.0

        See Also
        --------
        :ref:`resampling`
            User guide on resampling operations.
        """
        return self.reduce(
            duck_array_ops.std,
            dim=dim,
            skipna=skipna,
            numeric_only=True,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def var(
        self,
        dim=None,
        skipna: bool = True,
        keep_attrs=None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``var`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``var``.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``var`` on this object's data.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``var`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6))
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds.resample(time="3M").var()
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) float64 0.0 0.0 0.0
        >>> ds.resample(time="3M").var(skipna=False)
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) float64 0.0 0.0 0.0

        See Also
        --------
        :ref:`resampling`
            User guide on resampling operations.
        """
        return self.reduce(
            duck_array_ops.var,
            dim=dim,
            skipna=skipna,
            numeric_only=True,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def median(
        self,
        dim=None,
        skipna: bool = True,
        keep_attrs=None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``median`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``median``.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``median`` on this object's data.

        Returns
        -------
        reduced : Dataset
            New Dataset with ``median`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6))
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds.resample(time="3M").median()
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) float64 1.0 1.0 1.0
        >>> ds.resample(time="3M").median(skipna=False)
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) float64 1.0 1.0 1.0

        See Also
        --------
        :ref:`resampling`
            User guide on resampling operations.
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

    def count(
        self,
        dim=None,
        keep_attrs=None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``count`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``count``.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``count`` on this object's data.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``count`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     [1, 2, 3, 1, 2, np.nan],
        ...     dims="x",
        ...     coords=dict(labels=("x", np.array(["a", "b", "c", "c", "b", "a"]))),
        ... )
        >>> da.groupby("labels").count()
        <xarray.DataArray (labels: 3)>
        array([1, 2, 2])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'

        See Also
        --------
        :ref:`groupby`
            User guide on groupby operations.
        """
        return self.reduce(
            duck_array_ops.count,
            dim=dim,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def all(
        self,
        dim=None,
        keep_attrs=None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``all`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``all``.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``all`` on this object's data.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``all`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([True, True, True, True, True, False], dtype=bool),
        ...     dims="x",
        ...     coords=dict(labels=("x", np.array(["a", "b", "c", "c", "b", "a"]))),
        ... )
        >>> da.groupby("labels").all()
        <xarray.DataArray (labels: 3)>
        array([False,  True,  True])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'

        See Also
        --------
        :ref:`groupby`
            User guide on groupby operations.
        """
        return self.reduce(
            duck_array_ops.array_all,
            dim=dim,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def any(
        self,
        dim=None,
        keep_attrs=None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``any`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``any``.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``any`` on this object's data.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``any`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([True, True, True, True, True, False], dtype=bool),
        ...     dims="x",
        ...     coords=dict(labels=("x", np.array(["a", "b", "c", "c", "b", "a"]))),
        ... )
        >>> da.groupby("labels").any()
        <xarray.DataArray (labels: 3)>
        array([ True,  True,  True])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'

        See Also
        --------
        :ref:`groupby`
            User guide on groupby operations.
        """
        return self.reduce(
            duck_array_ops.array_any,
            dim=dim,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def max(
        self,
        dim=None,
        skipna: bool = True,
        keep_attrs=None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``max`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``max``.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``max`` on this object's data.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``max`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     [1, 2, 3, 1, 2, np.nan],
        ...     dims="x",
        ...     coords=dict(labels=("x", np.array(["a", "b", "c", "c", "b", "a"]))),
        ... )
        >>> da.groupby("labels").max()
        <xarray.DataArray (labels: 3)>
        array([1., 2., 3.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        >>> da.groupby("labels").max(skipna=False)
        <xarray.DataArray (labels: 3)>
        array([nan,  2.,  3.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'

        See Also
        --------
        :ref:`groupby`
            User guide on groupby operations.
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
        dim=None,
        skipna: bool = True,
        keep_attrs=None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``min`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``min``.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``min`` on this object's data.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``min`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     [1, 2, 3, 1, 2, np.nan],
        ...     dims="x",
        ...     coords=dict(labels=("x", np.array(["a", "b", "c", "c", "b", "a"]))),
        ... )
        >>> da.groupby("labels").min()
        <xarray.DataArray (labels: 3)>
        array([1., 2., 1.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        >>> da.groupby("labels").min(skipna=False)
        <xarray.DataArray (labels: 3)>
        array([nan,  2.,  1.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'

        See Also
        --------
        :ref:`groupby`
            User guide on groupby operations.
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
        dim=None,
        skipna: bool = True,
        keep_attrs=None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``mean`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``mean``.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``mean`` on this object's data.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``mean`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     [1, 2, 3, 1, 2, np.nan],
        ...     dims="x",
        ...     coords=dict(labels=("x", np.array(["a", "b", "c", "c", "b", "a"]))),
        ... )
        >>> da.groupby("labels").mean()
        <xarray.DataArray (labels: 3)>
        array([1., 2., 2.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        >>> da.groupby("labels").mean(skipna=False)
        <xarray.DataArray (labels: 3)>
        array([nan,  2.,  2.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'

        See Also
        --------
        :ref:`groupby`
            User guide on groupby operations.
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
        dim=None,
        skipna: bool = True,
        min_count: Optional[int] = None,
        keep_attrs=None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``prod`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``prod``.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
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

        Returns
        -------
        reduced : DataArray
            New DataArray with ``prod`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     [1, 2, 3, 1, 2, np.nan],
        ...     dims="x",
        ...     coords=dict(labels=("x", np.array(["a", "b", "c", "c", "b", "a"]))),
        ... )
        >>> da.groupby("labels").prod()
        <xarray.DataArray (labels: 3)>
        array([1., 4., 3.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        >>> da.groupby("labels").prod(skipna=False)
        <xarray.DataArray (labels: 3)>
        array([nan,  4.,  3.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'

        See Also
        --------
        :ref:`groupby`
            User guide on groupby operations.
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
        dim=None,
        skipna: bool = True,
        min_count: Optional[int] = None,
        keep_attrs=None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``sum`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``sum``.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
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

        Returns
        -------
        reduced : DataArray
            New DataArray with ``sum`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     [1, 2, 3, 1, 2, np.nan],
        ...     dims="x",
        ...     coords=dict(labels=("x", np.array(["a", "b", "c", "c", "b", "a"]))),
        ... )
        >>> da.groupby("labels").sum()
        <xarray.DataArray (labels: 3)>
        array([1., 4., 4.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        >>> da.groupby("labels").sum(skipna=False)
        <xarray.DataArray (labels: 3)>
        array([nan,  4.,  4.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'

        See Also
        --------
        :ref:`groupby`
            User guide on groupby operations.
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
        dim=None,
        skipna: bool = True,
        keep_attrs=None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``std`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``std``.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``std`` on this object's data.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``std`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     [1, 2, 3, 1, 2, np.nan],
        ...     dims="x",
        ...     coords=dict(labels=("x", np.array(["a", "b", "c", "c", "b", "a"]))),
        ... )
        >>> da.groupby("labels").std()
        <xarray.DataArray (labels: 3)>
        array([0., 0., 1.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        >>> da.groupby("labels").std(skipna=False)
        <xarray.DataArray (labels: 3)>
        array([nan,  0.,  1.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'

        See Also
        --------
        :ref:`groupby`
            User guide on groupby operations.
        """
        return self.reduce(
            duck_array_ops.std,
            dim=dim,
            skipna=skipna,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def var(
        self,
        dim=None,
        skipna: bool = True,
        keep_attrs=None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``var`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``var``.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``var`` on this object's data.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``var`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     [1, 2, 3, 1, 2, np.nan],
        ...     dims="x",
        ...     coords=dict(labels=("x", np.array(["a", "b", "c", "c", "b", "a"]))),
        ... )
        >>> da.groupby("labels").var()
        <xarray.DataArray (labels: 3)>
        array([0., 0., 1.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        >>> da.groupby("labels").var(skipna=False)
        <xarray.DataArray (labels: 3)>
        array([nan,  0.,  1.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'

        See Also
        --------
        :ref:`groupby`
            User guide on groupby operations.
        """
        return self.reduce(
            duck_array_ops.var,
            dim=dim,
            skipna=skipna,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def median(
        self,
        dim=None,
        skipna: bool = True,
        keep_attrs=None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``median`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``median``.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``median`` on this object's data.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``median`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     [1, 2, 3, 1, 2, np.nan],
        ...     dims="x",
        ...     coords=dict(labels=("x", np.array(["a", "b", "c", "c", "b", "a"]))),
        ... )
        >>> da.groupby("labels").median()
        <xarray.DataArray (labels: 3)>
        array([1., 2., 2.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        >>> da.groupby("labels").median(skipna=False)
        <xarray.DataArray (labels: 3)>
        array([nan,  2.,  2.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'

        See Also
        --------
        :ref:`groupby`
            User guide on groupby operations.
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

    def count(
        self,
        dim=None,
        keep_attrs=None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``count`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``count``.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``count`` on this object's data.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``count`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6))
        ...     ),
        ... )

        >>> da.resample(time="3M").count()
        <xarray.DataArray (time: 3)>
        array([1, 3, 2])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31

        See Also
        --------
        :ref:`resampling`
            User guide on resampling operations.
        """
        return self.reduce(
            duck_array_ops.count,
            dim=dim,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def all(
        self,
        dim=None,
        keep_attrs=None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``all`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``all``.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``all`` on this object's data.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``all`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([True, True, True, True, True, False], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6))
        ...     ),
        ... )

        >>> da.resample(time="3M").all()
        <xarray.DataArray (time: 3)>
        array([ True,  True, False])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31

        See Also
        --------
        :ref:`resampling`
            User guide on resampling operations.
        """
        return self.reduce(
            duck_array_ops.array_all,
            dim=dim,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def any(
        self,
        dim=None,
        keep_attrs=None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``any`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``any``.
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``any`` on this object's data.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``any`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([True, True, True, True, True, False], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6))
        ...     ),
        ... )

        >>> da.resample(time="3M").any()
        <xarray.DataArray (time: 3)>
        array([ True,  True,  True])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31

        See Also
        --------
        :ref:`resampling`
            User guide on resampling operations.
        """
        return self.reduce(
            duck_array_ops.array_any,
            dim=dim,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def max(
        self,
        dim=None,
        skipna: bool = True,
        keep_attrs=None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``max`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``max``.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``max`` on this object's data.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``max`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6))
        ...     ),
        ... )

        >>> da.resample(time="3M").max()
        <xarray.DataArray (time: 3)>
        array([ True,  True,  True])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        >>> da.resample(time="3M").max(skipna=False)
        <xarray.DataArray (time: 3)>
        array([ True,  True,  True])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31

        See Also
        --------
        :ref:`resampling`
            User guide on resampling operations.
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
        dim=None,
        skipna: bool = True,
        keep_attrs=None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``min`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``min``.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``min`` on this object's data.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``min`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6))
        ...     ),
        ... )

        >>> da.resample(time="3M").min()
        <xarray.DataArray (time: 3)>
        array([ True,  True,  True])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        >>> da.resample(time="3M").min(skipna=False)
        <xarray.DataArray (time: 3)>
        array([ True,  True,  True])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31

        See Also
        --------
        :ref:`resampling`
            User guide on resampling operations.
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
        dim=None,
        skipna: bool = True,
        keep_attrs=None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``mean`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``mean``.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``mean`` on this object's data.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``mean`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6))
        ...     ),
        ... )

        >>> da.resample(time="3M").mean()
        <xarray.DataArray (time: 3)>
        array([1., 1., 1.])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        >>> da.resample(time="3M").mean(skipna=False)
        <xarray.DataArray (time: 3)>
        array([1., 1., 1.])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31

        See Also
        --------
        :ref:`resampling`
            User guide on resampling operations.
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
        dim=None,
        skipna: bool = True,
        min_count: Optional[int] = None,
        keep_attrs=None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``prod`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``prod``.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
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

        Returns
        -------
        reduced : DataArray
            New DataArray with ``prod`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6))
        ...     ),
        ... )

        >>> da.resample(time="3M").prod()
        <xarray.DataArray (time: 3)>
        array([1, 1, 1])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        >>> da.resample(time="3M").prod(skipna=False)
        <xarray.DataArray (time: 3)>
        array([1, 1, 1])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31

        See Also
        --------
        :ref:`resampling`
            User guide on resampling operations.
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
        dim=None,
        skipna: bool = True,
        min_count: Optional[int] = None,
        keep_attrs=None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``sum`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``sum``.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
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

        Returns
        -------
        reduced : DataArray
            New DataArray with ``sum`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6))
        ...     ),
        ... )

        >>> da.resample(time="3M").sum()
        <xarray.DataArray (time: 3)>
        array([1, 3, 2])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        >>> da.resample(time="3M").sum(skipna=False)
        <xarray.DataArray (time: 3)>
        array([1, 3, 2])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31

        See Also
        --------
        :ref:`resampling`
            User guide on resampling operations.
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
        dim=None,
        skipna: bool = True,
        keep_attrs=None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``std`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``std``.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``std`` on this object's data.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``std`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6))
        ...     ),
        ... )

        >>> da.resample(time="3M").std()
        <xarray.DataArray (time: 3)>
        array([0., 0., 0.])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        >>> da.resample(time="3M").std(skipna=False)
        <xarray.DataArray (time: 3)>
        array([0., 0., 0.])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31

        See Also
        --------
        :ref:`resampling`
            User guide on resampling operations.
        """
        return self.reduce(
            duck_array_ops.std,
            dim=dim,
            skipna=skipna,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def var(
        self,
        dim=None,
        skipna: bool = True,
        keep_attrs=None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``var`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``var``.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``var`` on this object's data.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``var`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6))
        ...     ),
        ... )

        >>> da.resample(time="3M").var()
        <xarray.DataArray (time: 3)>
        array([0., 0., 0.])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        >>> da.resample(time="3M").var(skipna=False)
        <xarray.DataArray (time: 3)>
        array([0., 0., 0.])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31

        See Also
        --------
        :ref:`resampling`
            User guide on resampling operations.
        """
        return self.reduce(
            duck_array_ops.var,
            dim=dim,
            skipna=skipna,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def median(
        self,
        dim=None,
        skipna: bool = True,
        keep_attrs=None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``median`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``median``.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``median`` on this object's data.

        Returns
        -------
        reduced : DataArray
            New DataArray with ``median`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------
        >>> da = xr.DataArray(
        ...     np.array([1, 2, 3, 1, 2, np.nan], dtype=bool),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6))
        ...     ),
        ... )

        >>> da.resample(time="3M").median()
        <xarray.DataArray (time: 3)>
        array([1., 1., 1.])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        >>> da.resample(time="3M").median(skipna=False)
        <xarray.DataArray (time: 3)>
        array([1., 1., 1.])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31

        See Also
        --------
        :ref:`resampling`
            User guide on resampling operations.
        """
        return self.reduce(
            duck_array_ops.median,
            dim=dim,
            skipna=skipna,
            keep_attrs=keep_attrs,
            **kwargs,
        )
