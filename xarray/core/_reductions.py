"""Mixin classes with reduction operations."""
# This file was generated using xarray.util.generate_reductions. Do not edit manually.

import sys
from typing import Any, Callable, Hashable, Optional, Sequence, Union

from . import duck_array_ops
from .types import T_DataArray, T_Dataset

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol


class DatasetReduce(Protocol):
    def reduce(
        self,
        func: Callable[..., Any],
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        axis: Union[None, int, Sequence[int]] = None,
        keep_attrs: bool = None,
        keepdims: bool = False,
        **kwargs: Any,
    ) -> T_Dataset:
        ...


class DatasetGroupByReductions:
    __slots__ = ()

    def count(
        self: DatasetReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``count`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``count``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset>
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) float64 1.0 2.0 3.0 1.0 2.0 nan

        >>> ds.groupby("labels").count()
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) int64 1 2 2

        See Also
        --------
        numpy.count
        Dataset.count
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
        self: DatasetReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``all`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``all``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset>
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) bool True True True True True False

        >>> ds.groupby("labels").all()
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) bool False True True

        See Also
        --------
        numpy.all
        Dataset.all
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
        self: DatasetReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``any`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``any``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset>
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) bool True True True True True False

        >>> ds.groupby("labels").any()
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) bool True True True

        See Also
        --------
        numpy.any
        Dataset.any
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
        self: DatasetReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = True,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``max`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``max``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset>
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) float64 1.0 2.0 3.0 1.0 2.0 nan

        >>> ds.groupby("labels").max()
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 1.0 2.0 3.0

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.groupby("labels").max(skipna=False)
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 nan 2.0 3.0

        See Also
        --------
        numpy.max
        Dataset.max
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
        self: DatasetReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = True,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``min`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``min``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset>
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) float64 1.0 2.0 3.0 1.0 2.0 nan

        >>> ds.groupby("labels").min()
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 1.0 2.0 1.0

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.groupby("labels").min(skipna=False)
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 nan 2.0 1.0

        See Also
        --------
        numpy.min
        Dataset.min
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
        self: DatasetReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = True,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``mean`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``mean``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset>
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) float64 1.0 2.0 3.0 1.0 2.0 nan

        >>> ds.groupby("labels").mean()
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 1.0 2.0 2.0

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.groupby("labels").mean(skipna=False)
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 nan 2.0 2.0

        See Also
        --------
        numpy.mean
        Dataset.mean
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
        self: DatasetReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = True,
        min_count: Optional[int] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``prod`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``prod``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset>
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) float64 1.0 2.0 3.0 1.0 2.0 nan

        >>> ds.groupby("labels").prod()
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 1.0 4.0 3.0

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.groupby("labels").prod(skipna=False)
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 nan 4.0 3.0

        Specify ``min_count`` for finer control over when NaNs are ignored.

        >>> ds.groupby("labels").prod(skipna=True, min_count=2)
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 nan 4.0 3.0

        See Also
        --------
        numpy.prod
        Dataset.prod
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
        self: DatasetReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = True,
        min_count: Optional[int] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``sum`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``sum``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset>
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) float64 1.0 2.0 3.0 1.0 2.0 nan

        >>> ds.groupby("labels").sum()
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 1.0 4.0 4.0

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.groupby("labels").sum(skipna=False)
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 nan 4.0 4.0

        Specify ``min_count`` for finer control over when NaNs are ignored.

        >>> ds.groupby("labels").sum(skipna=True, min_count=2)
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 nan 4.0 4.0

        See Also
        --------
        numpy.sum
        Dataset.sum
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
        self: DatasetReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = True,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``std`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``std``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset>
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) float64 1.0 2.0 3.0 1.0 2.0 nan

        >>> ds.groupby("labels").std()
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 0.0 0.0 1.0

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.groupby("labels").std(skipna=False)
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 nan 0.0 1.0

        See Also
        --------
        numpy.std
        Dataset.std
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
        self: DatasetReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = True,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``var`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``var``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset>
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) float64 1.0 2.0 3.0 1.0 2.0 nan

        >>> ds.groupby("labels").var()
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 0.0 0.0 1.0

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.groupby("labels").var(skipna=False)
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 nan 0.0 1.0

        See Also
        --------
        numpy.var
        Dataset.var
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
        self: DatasetReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = True,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``median`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``median``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset>
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) float64 1.0 2.0 3.0 1.0 2.0 nan

        >>> ds.groupby("labels").median()
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 1.0 2.0 2.0

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.groupby("labels").median(skipna=False)
        <xarray.Dataset>
        Dimensions:  (labels: 3)
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'
        Data variables:
            da       (labels) float64 nan 2.0 2.0

        See Also
        --------
        numpy.median
        Dataset.median
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
        self: DatasetReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``count`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``count``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset>
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) float64 1.0 2.0 3.0 1.0 2.0 nan

        >>> ds.resample(time="3M").count()
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) int64 1 3 1

        See Also
        --------
        numpy.count
        Dataset.count
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
        self: DatasetReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``all`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``all``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset>
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) bool True True True True True False

        >>> ds.resample(time="3M").all()
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) bool True True False

        See Also
        --------
        numpy.all
        Dataset.all
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
        self: DatasetReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``any`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``any``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset>
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) bool True True True True True False

        >>> ds.resample(time="3M").any()
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) bool True True True

        See Also
        --------
        numpy.any
        Dataset.any
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
        self: DatasetReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = True,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``max`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``max``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset>
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) float64 1.0 2.0 3.0 1.0 2.0 nan

        >>> ds.resample(time="3M").max()
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) float64 1.0 3.0 2.0

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.resample(time="3M").max(skipna=False)
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) float64 1.0 3.0 nan

        See Also
        --------
        numpy.max
        Dataset.max
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
        self: DatasetReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = True,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``min`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``min``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset>
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) float64 1.0 2.0 3.0 1.0 2.0 nan

        >>> ds.resample(time="3M").min()
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) float64 1.0 1.0 2.0

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.resample(time="3M").min(skipna=False)
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) float64 1.0 1.0 nan

        See Also
        --------
        numpy.min
        Dataset.min
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
        self: DatasetReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = True,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``mean`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``mean``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset>
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) float64 1.0 2.0 3.0 1.0 2.0 nan

        >>> ds.resample(time="3M").mean()
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) float64 1.0 2.0 2.0

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.resample(time="3M").mean(skipna=False)
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) float64 1.0 2.0 nan

        See Also
        --------
        numpy.mean
        Dataset.mean
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
        self: DatasetReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = True,
        min_count: Optional[int] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``prod`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``prod``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset>
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) float64 1.0 2.0 3.0 1.0 2.0 nan

        >>> ds.resample(time="3M").prod()
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) float64 1.0 6.0 2.0

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.resample(time="3M").prod(skipna=False)
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) float64 1.0 6.0 nan

        Specify ``min_count`` for finer control over when NaNs are ignored.

        >>> ds.resample(time="3M").prod(skipna=True, min_count=2)
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) float64 nan 6.0 nan

        See Also
        --------
        numpy.prod
        Dataset.prod
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
        self: DatasetReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = True,
        min_count: Optional[int] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``sum`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``sum``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset>
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) float64 1.0 2.0 3.0 1.0 2.0 nan

        >>> ds.resample(time="3M").sum()
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) float64 1.0 6.0 2.0

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.resample(time="3M").sum(skipna=False)
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) float64 1.0 6.0 nan

        Specify ``min_count`` for finer control over when NaNs are ignored.

        >>> ds.resample(time="3M").sum(skipna=True, min_count=2)
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) float64 nan 6.0 nan

        See Also
        --------
        numpy.sum
        Dataset.sum
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
        self: DatasetReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = True,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``std`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``std``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset>
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) float64 1.0 2.0 3.0 1.0 2.0 nan

        >>> ds.resample(time="3M").std()
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) float64 0.0 0.8165 0.0

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.resample(time="3M").std(skipna=False)
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) float64 0.0 0.8165 nan

        See Also
        --------
        numpy.std
        Dataset.std
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
        self: DatasetReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = True,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``var`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``var``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset>
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) float64 1.0 2.0 3.0 1.0 2.0 nan

        >>> ds.resample(time="3M").var()
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) float64 0.0 0.6667 0.0

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.resample(time="3M").var(skipna=False)
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) float64 0.0 0.6667 nan

        See Also
        --------
        numpy.var
        Dataset.var
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
        self: DatasetReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = True,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_Dataset:
        """
        Reduce this Dataset's data by applying ``median`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``median``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        <xarray.Dataset>
        Dimensions:  (time: 6)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'
        Data variables:
            da       (time) float64 1.0 2.0 3.0 1.0 2.0 nan

        >>> ds.resample(time="3M").median()
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) float64 1.0 2.0 2.0

        Use ``skipna`` to control whether NaNs are ignored.

        >>> ds.resample(time="3M").median(skipna=False)
        <xarray.Dataset>
        Dimensions:  (time: 3)
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31
        Data variables:
            da       (time) float64 1.0 2.0 nan

        See Also
        --------
        numpy.median
        Dataset.median
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


class DataArrayReduce(Protocol):
    def reduce(
        self,
        func: Callable[..., Any],
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        axis: Union[None, int, Sequence[int]] = None,
        keep_attrs: bool = None,
        keepdims: bool = False,
        **kwargs: Any,
    ) -> T_DataArray:
        ...


class DataArrayGroupByReductions:
    __slots__ = ()

    def count(
        self: DataArrayReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``count`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``count``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da
        <xarray.DataArray (time: 6)>
        array([ 1.,  2.,  3.,  1.,  2., nan])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.groupby("labels").count()
        <xarray.DataArray (labels: 3)>
        array([1, 2, 2])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'

        See Also
        --------
        numpy.count
        DataArray.count
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
        self: DataArrayReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``all`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``all``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da
        <xarray.DataArray (time: 6)>
        array([ True,  True,  True,  True,  True, False])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.groupby("labels").all()
        <xarray.DataArray (labels: 3)>
        array([False,  True,  True])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'

        See Also
        --------
        numpy.all
        DataArray.all
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
        self: DataArrayReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``any`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``any``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da
        <xarray.DataArray (time: 6)>
        array([ True,  True,  True,  True,  True, False])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.groupby("labels").any()
        <xarray.DataArray (labels: 3)>
        array([ True,  True,  True])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'

        See Also
        --------
        numpy.any
        DataArray.any
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
        self: DataArrayReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = True,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``max`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``max``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da
        <xarray.DataArray (time: 6)>
        array([ 1.,  2.,  3.,  1.,  2., nan])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.groupby("labels").max()
        <xarray.DataArray (labels: 3)>
        array([1., 2., 3.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.groupby("labels").max(skipna=False)
        <xarray.DataArray (labels: 3)>
        array([nan,  2.,  3.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'

        See Also
        --------
        numpy.max
        DataArray.max
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
        self: DataArrayReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = True,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``min`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``min``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da
        <xarray.DataArray (time: 6)>
        array([ 1.,  2.,  3.,  1.,  2., nan])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.groupby("labels").min()
        <xarray.DataArray (labels: 3)>
        array([1., 2., 1.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.groupby("labels").min(skipna=False)
        <xarray.DataArray (labels: 3)>
        array([nan,  2.,  1.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'

        See Also
        --------
        numpy.min
        DataArray.min
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
        self: DataArrayReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = True,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``mean`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``mean``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da
        <xarray.DataArray (time: 6)>
        array([ 1.,  2.,  3.,  1.,  2., nan])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.groupby("labels").mean()
        <xarray.DataArray (labels: 3)>
        array([1., 2., 2.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.groupby("labels").mean(skipna=False)
        <xarray.DataArray (labels: 3)>
        array([nan,  2.,  2.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'

        See Also
        --------
        numpy.mean
        DataArray.mean
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
        self: DataArrayReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = True,
        min_count: Optional[int] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``prod`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``prod``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da
        <xarray.DataArray (time: 6)>
        array([ 1.,  2.,  3.,  1.,  2., nan])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.groupby("labels").prod()
        <xarray.DataArray (labels: 3)>
        array([1., 4., 3.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.groupby("labels").prod(skipna=False)
        <xarray.DataArray (labels: 3)>
        array([nan,  4.,  3.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'

        Specify ``min_count`` for finer control over when NaNs are ignored.

        >>> da.groupby("labels").prod(skipna=True, min_count=2)
        <xarray.DataArray (labels: 3)>
        array([nan,  4.,  3.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'

        See Also
        --------
        numpy.prod
        DataArray.prod
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
        self: DataArrayReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = True,
        min_count: Optional[int] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``sum`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``sum``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da
        <xarray.DataArray (time: 6)>
        array([ 1.,  2.,  3.,  1.,  2., nan])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.groupby("labels").sum()
        <xarray.DataArray (labels: 3)>
        array([1., 4., 4.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.groupby("labels").sum(skipna=False)
        <xarray.DataArray (labels: 3)>
        array([nan,  4.,  4.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'

        Specify ``min_count`` for finer control over when NaNs are ignored.

        >>> da.groupby("labels").sum(skipna=True, min_count=2)
        <xarray.DataArray (labels: 3)>
        array([nan,  4.,  4.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'

        See Also
        --------
        numpy.sum
        DataArray.sum
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
        self: DataArrayReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = True,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``std`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``std``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da
        <xarray.DataArray (time: 6)>
        array([ 1.,  2.,  3.,  1.,  2., nan])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.groupby("labels").std()
        <xarray.DataArray (labels: 3)>
        array([0., 0., 1.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.groupby("labels").std(skipna=False)
        <xarray.DataArray (labels: 3)>
        array([nan,  0.,  1.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'

        See Also
        --------
        numpy.std
        DataArray.std
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
        self: DataArrayReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = True,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``var`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``var``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da
        <xarray.DataArray (time: 6)>
        array([ 1.,  2.,  3.,  1.,  2., nan])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.groupby("labels").var()
        <xarray.DataArray (labels: 3)>
        array([0., 0., 1.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.groupby("labels").var(skipna=False)
        <xarray.DataArray (labels: 3)>
        array([nan,  0.,  1.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'

        See Also
        --------
        numpy.var
        DataArray.var
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
        self: DataArrayReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = True,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``median`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``median``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da
        <xarray.DataArray (time: 6)>
        array([ 1.,  2.,  3.,  1.,  2., nan])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.groupby("labels").median()
        <xarray.DataArray (labels: 3)>
        array([1., 2., 2.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.groupby("labels").median(skipna=False)
        <xarray.DataArray (labels: 3)>
        array([nan,  2.,  2.])
        Coordinates:
          * labels   (labels) object 'a' 'b' 'c'

        See Also
        --------
        numpy.median
        DataArray.median
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
        self: DataArrayReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``count`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``count``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da
        <xarray.DataArray (time: 6)>
        array([ 1.,  2.,  3.,  1.,  2., nan])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.resample(time="3M").count()
        <xarray.DataArray (time: 3)>
        array([1, 3, 1])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31

        See Also
        --------
        numpy.count
        DataArray.count
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
        self: DataArrayReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``all`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``all``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da
        <xarray.DataArray (time: 6)>
        array([ True,  True,  True,  True,  True, False])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.resample(time="3M").all()
        <xarray.DataArray (time: 3)>
        array([ True,  True, False])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31

        See Also
        --------
        numpy.all
        DataArray.all
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
        self: DataArrayReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``any`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``any``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da
        <xarray.DataArray (time: 6)>
        array([ True,  True,  True,  True,  True, False])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.resample(time="3M").any()
        <xarray.DataArray (time: 3)>
        array([ True,  True,  True])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31

        See Also
        --------
        numpy.any
        DataArray.any
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
        self: DataArrayReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = True,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``max`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``max``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da
        <xarray.DataArray (time: 6)>
        array([ 1.,  2.,  3.,  1.,  2., nan])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.resample(time="3M").max()
        <xarray.DataArray (time: 3)>
        array([1., 3., 2.])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.resample(time="3M").max(skipna=False)
        <xarray.DataArray (time: 3)>
        array([ 1.,  3., nan])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31

        See Also
        --------
        numpy.max
        DataArray.max
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
        self: DataArrayReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = True,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``min`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``min``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da
        <xarray.DataArray (time: 6)>
        array([ 1.,  2.,  3.,  1.,  2., nan])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.resample(time="3M").min()
        <xarray.DataArray (time: 3)>
        array([1., 1., 2.])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.resample(time="3M").min(skipna=False)
        <xarray.DataArray (time: 3)>
        array([ 1.,  1., nan])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31

        See Also
        --------
        numpy.min
        DataArray.min
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
        self: DataArrayReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = True,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``mean`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``mean``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da
        <xarray.DataArray (time: 6)>
        array([ 1.,  2.,  3.,  1.,  2., nan])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.resample(time="3M").mean()
        <xarray.DataArray (time: 3)>
        array([1., 2., 2.])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.resample(time="3M").mean(skipna=False)
        <xarray.DataArray (time: 3)>
        array([ 1.,  2., nan])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31

        See Also
        --------
        numpy.mean
        DataArray.mean
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
        self: DataArrayReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = True,
        min_count: Optional[int] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``prod`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``prod``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da
        <xarray.DataArray (time: 6)>
        array([ 1.,  2.,  3.,  1.,  2., nan])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.resample(time="3M").prod()
        <xarray.DataArray (time: 3)>
        array([1., 6., 2.])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.resample(time="3M").prod(skipna=False)
        <xarray.DataArray (time: 3)>
        array([ 1.,  6., nan])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31

        Specify ``min_count`` for finer control over when NaNs are ignored.

        >>> da.resample(time="3M").prod(skipna=True, min_count=2)
        <xarray.DataArray (time: 3)>
        array([nan,  6., nan])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31

        See Also
        --------
        numpy.prod
        DataArray.prod
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
        self: DataArrayReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = True,
        min_count: Optional[int] = None,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``sum`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``sum``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da
        <xarray.DataArray (time: 6)>
        array([ 1.,  2.,  3.,  1.,  2., nan])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.resample(time="3M").sum()
        <xarray.DataArray (time: 3)>
        array([1., 6., 2.])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.resample(time="3M").sum(skipna=False)
        <xarray.DataArray (time: 3)>
        array([ 1.,  6., nan])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31

        Specify ``min_count`` for finer control over when NaNs are ignored.

        >>> da.resample(time="3M").sum(skipna=True, min_count=2)
        <xarray.DataArray (time: 3)>
        array([nan,  6., nan])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31

        See Also
        --------
        numpy.sum
        DataArray.sum
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
        self: DataArrayReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = True,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``std`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``std``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da
        <xarray.DataArray (time: 6)>
        array([ 1.,  2.,  3.,  1.,  2., nan])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.resample(time="3M").std()
        <xarray.DataArray (time: 3)>
        array([0.        , 0.81649658, 0.        ])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.resample(time="3M").std(skipna=False)
        <xarray.DataArray (time: 3)>
        array([0.        , 0.81649658,        nan])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31

        See Also
        --------
        numpy.std
        DataArray.std
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
        self: DataArrayReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = True,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``var`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``var``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da
        <xarray.DataArray (time: 6)>
        array([ 1.,  2.,  3.,  1.,  2., nan])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.resample(time="3M").var()
        <xarray.DataArray (time: 3)>
        array([0.        , 0.66666667, 0.        ])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.resample(time="3M").var(skipna=False)
        <xarray.DataArray (time: 3)>
        array([0.        , 0.66666667,        nan])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31

        See Also
        --------
        numpy.var
        DataArray.var
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
        self: DataArrayReduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        skipna: bool = True,
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_DataArray:
        """
        Reduce this DataArray's data by applying ``median`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``median``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If ``None``, will reduce over all dimensions
            present in the grouped variable.
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
        ...     np.array([1, 2, 3, 1, 2, np.nan]),
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )
        >>> da
        <xarray.DataArray (time: 6)>
        array([ 1.,  2.,  3.,  1.,  2., nan])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-02-28 ... 2001-06-30
            labels   (time) <U1 'a' 'b' 'c' 'c' 'b' 'a'

        >>> da.resample(time="3M").median()
        <xarray.DataArray (time: 3)>
        array([1., 2., 2.])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31

        Use ``skipna`` to control whether NaNs are ignored.

        >>> da.resample(time="3M").median(skipna=False)
        <xarray.DataArray (time: 3)>
        array([ 1.,  2., nan])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-31 2001-04-30 2001-07-31

        See Also
        --------
        numpy.median
        DataArray.median
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
