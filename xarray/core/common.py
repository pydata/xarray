import warnings
from collections import OrderedDict
from contextlib import suppress
from textwrap import dedent
from typing import (
    Any,
    Callable,
    Hashable,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd

from . import dtypes, duck_array_ops, formatting, ops
from .arithmetic import SupportsArithmetic
from .npcompat import DTypeLike
from .options import _get_keep_attrs
from .pycompat import dask_array_type
from .rolling_exp import RollingExp
from .utils import Frozen, ReprObject, SortedKeysDict, either_dict_or_kwargs

# Used as a sentinel value to indicate a all dimensions
ALL_DIMS = ReprObject("<all-dims>")


C = TypeVar("C")
T = TypeVar("T")


class ImplementsArrayReduce:
    __slots__ = ()

    @classmethod
    def _reduce_method(cls, func: Callable, include_skipna: bool, numeric_only: bool):
        if include_skipna:

            def wrapped_func(self, dim=None, axis=None, skipna=None, **kwargs):
                return self.reduce(
                    func, dim, axis, skipna=skipna, allow_lazy=True, **kwargs
                )

        else:

            def wrapped_func(self, dim=None, axis=None, **kwargs):  # type: ignore
                return self.reduce(func, dim, axis, allow_lazy=True, **kwargs)

        return wrapped_func

    _reduce_extra_args_docstring = dedent(
        """\
        dim : str or sequence of str, optional
            Dimension(s) over which to apply `{name}`.
        axis : int or sequence of int, optional
            Axis(es) over which to apply `{name}`. Only one of the 'dim'
            and 'axis' arguments can be supplied. If neither are supplied, then
            `{name}` is calculated over axes."""
    )

    _cum_extra_args_docstring = dedent(
        """\
        dim : str or sequence of str, optional
            Dimension over which to apply `{name}`.
        axis : int or sequence of int, optional
            Axis over which to apply `{name}`. Only one of the 'dim'
            and 'axis' arguments can be supplied."""
    )


class ImplementsDatasetReduce:
    __slots__ = ()

    @classmethod
    def _reduce_method(cls, func: Callable, include_skipna: bool, numeric_only: bool):
        if include_skipna:

            def wrapped_func(self, dim=None, skipna=None, **kwargs):
                return self.reduce(
                    func,
                    dim,
                    skipna=skipna,
                    numeric_only=numeric_only,
                    allow_lazy=True,
                    **kwargs
                )

        else:

            def wrapped_func(self, dim=None, **kwargs):  # type: ignore
                return self.reduce(
                    func, dim, numeric_only=numeric_only, allow_lazy=True, **kwargs
                )

        return wrapped_func

    _reduce_extra_args_docstring = """dim : str or sequence of str, optional
            Dimension(s) over which to apply `{name}`.  By default `{name}` is
            applied over all dimensions."""

    _cum_extra_args_docstring = """dim : str or sequence of str, optional
            Dimension over which to apply `{name}`.
        axis : int or sequence of int, optional
            Axis over which to apply `{name}`. Only one of the 'dim'
            and 'axis' arguments can be supplied."""


class AbstractArray(ImplementsArrayReduce):
    """Shared base class for DataArray and Variable.
    """

    __slots__ = ()

    def __bool__(self: Any) -> bool:
        return bool(self.values)

    def __float__(self: Any) -> float:
        return float(self.values)

    def __int__(self: Any) -> int:
        return int(self.values)

    def __complex__(self: Any) -> complex:
        return complex(self.values)

    def __array__(self: Any, dtype: DTypeLike = None) -> np.ndarray:
        return np.asarray(self.values, dtype=dtype)

    def __repr__(self) -> str:
        return formatting.array_repr(self)

    def _iter(self: Any) -> Iterator[Any]:
        for n in range(len(self)):
            yield self[n]

    def __iter__(self: Any) -> Iterator[Any]:
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d array")
        return self._iter()

    def get_axis_num(
        self, dim: Union[Hashable, Iterable[Hashable]]
    ) -> Union[int, Tuple[int, ...]]:
        """Return axis number(s) corresponding to dimension(s) in this array.

        Parameters
        ----------
        dim : str or iterable of str
            Dimension name(s) for which to lookup axes.

        Returns
        -------
        int or tuple of int
            Axis number or numbers corresponding to the given dimensions.
        """
        if isinstance(dim, Iterable) and not isinstance(dim, str):
            return tuple(self._get_axis_num(d) for d in dim)
        else:
            return self._get_axis_num(dim)

    def _get_axis_num(self: Any, dim: Hashable) -> int:
        try:
            return self.dims.index(dim)
        except ValueError:
            raise ValueError("%r not found in array dimensions %r" % (dim, self.dims))

    @property
    def sizes(self: Any) -> Mapping[Hashable, int]:
        """Ordered mapping from dimension names to lengths.

        Immutable.

        See also
        --------
        Dataset.sizes
        """
        return Frozen(OrderedDict(zip(self.dims, self.shape)))


class AttrAccessMixin:
    """Mixin class that allows getting keys with attribute access
    """

    __slots__ = ()

    def __init_subclass__(cls):
        """Verify that all subclasses explicitly define ``__slots__``. If they don't,
        raise error in the core xarray module and a FutureWarning in third-party
        extensions.
        This check is only triggered in Python 3.6+.
        """
        if not hasattr(object.__new__(cls), "__dict__"):
            cls.__setattr__ = cls._setattr_slots
        elif cls.__module__.startswith("xarray."):
            raise AttributeError("%s must explicitly define __slots__" % cls.__name__)
        else:
            cls.__setattr__ = cls._setattr_dict
            warnings.warn(
                "xarray subclass %s should explicitly define __slots__" % cls.__name__,
                FutureWarning,
                stacklevel=2,
            )

    @property
    def _attr_sources(self) -> List[Mapping[Hashable, Any]]:
        """List of places to look-up items for attribute-style access
        """
        return []

    @property
    def _item_sources(self) -> List[Mapping[Hashable, Any]]:
        """List of places to look-up items for key-autocompletion
        """
        return []

    def __getattr__(self, name: str) -> Any:
        if name not in {"__dict__", "__setstate__"}:
            # this avoids an infinite loop when pickle looks for the
            # __setstate__ attribute before the xarray object is initialized
            for source in self._attr_sources:
                with suppress(KeyError):
                    return source[name]
        raise AttributeError(
            "%r object has no attribute %r" % (type(self).__name__, name)
        )

    # This complicated three-method design boosts overall performance of simple
    # operations - particularly DataArray methods that perform a _to_temp_dataset()
    # round-trip - by a whopping 8% compared to a single method that checks
    # hasattr(self, "__dict__") at runtime before every single assignment (like
    # _setattr_py35 does). All of this is just temporary until the FutureWarning can be
    # changed into a hard crash.
    def _setattr_dict(self, name: str, value: Any) -> None:
        """Deprecated third party subclass (see ``__init_subclass__`` above)
        """
        object.__setattr__(self, name, value)
        if name in self.__dict__:
            # Custom, non-slotted attr, or improperly assigned variable?
            warnings.warn(
                "Setting attribute %r on a %r object. Explicitly define __slots__ "
                "to suppress this warning for legitimate custom attributes and "
                "raise an error when attempting variables assignments."
                % (name, type(self).__name__),
                FutureWarning,
                stacklevel=2,
            )

    def _setattr_slots(self, name: str, value: Any) -> None:
        """Objects with ``__slots__`` raise AttributeError if you try setting an
        undeclared attribute. This is desirable, but the error message could use some
        improvement.
        """
        try:
            object.__setattr__(self, name, value)
        except AttributeError as e:
            # Don't accidentally shadow custom AttributeErrors, e.g.
            # DataArray.dims.setter
            if str(e) != "%r object has no attribute %r" % (type(self).__name__, name):
                raise
            raise AttributeError(
                "cannot set attribute %r on a %r object. Use __setitem__ style"
                "assignment (e.g., `ds['name'] = ...`) instead of assigning variables."
                % (name, type(self).__name__)
            ) from e

    def _setattr_py35(self, name: str, value: Any) -> None:
        if hasattr(self, "__dict__"):
            return self._setattr_dict(name, value)
        return self._setattr_slots(name, value)

    # Overridden in Python >=3.6 by __init_subclass__
    __setattr__ = _setattr_py35

    def __dir__(self) -> List[str]:
        """Provide method name lookup and completion. Only provide 'public'
        methods.
        """
        extra_attrs = [
            item
            for sublist in self._attr_sources
            for item in sublist
            if isinstance(item, str)
        ]
        return sorted(set(dir(type(self)) + extra_attrs))

    def _ipython_key_completions_(self) -> List[str]:
        """Provide method for the key-autocompletions in IPython.
        See http://ipython.readthedocs.io/en/stable/config/integrating.html#tab-completion
        For the details.
        """  # noqa
        item_lists = [
            item
            for sublist in self._item_sources
            for item in sublist
            if isinstance(item, str)
        ]
        return list(set(item_lists))


def get_squeeze_dims(
    xarray_obj,
    dim: Union[Hashable, Iterable[Hashable], None] = None,
    axis: Union[int, Iterable[int], None] = None,
) -> List[Hashable]:
    """Get a list of dimensions to squeeze out.
    """
    if dim is not None and axis is not None:
        raise ValueError("cannot use both parameters `axis` and `dim`")
    if dim is None and axis is None:
        return [d for d, s in xarray_obj.sizes.items() if s == 1]

    if isinstance(dim, Iterable) and not isinstance(dim, str):
        dim = list(dim)
    elif dim is not None:
        dim = [dim]
    else:
        assert axis is not None
        if isinstance(axis, int):
            axis = [axis]
        axis = list(axis)
        if any(not isinstance(a, int) for a in axis):
            raise TypeError("parameter `axis` must be int or iterable of int.")
        alldims = list(xarray_obj.sizes.keys())
        dim = [alldims[a] for a in axis]

    if any(xarray_obj.sizes[k] > 1 for k in dim):
        raise ValueError(
            "cannot select a dimension to squeeze out "
            "which has length greater than one"
        )
    return dim


class DataWithCoords(SupportsArithmetic, AttrAccessMixin):
    """Shared base class for Dataset and DataArray."""

    __slots__ = ()

    _rolling_exp_cls = RollingExp

    def squeeze(
        self,
        dim: Union[Hashable, Iterable[Hashable], None] = None,
        drop: bool = False,
        axis: Union[int, Iterable[int], None] = None,
    ):
        """Return a new object with squeezed data.

        Parameters
        ----------
        dim : None or Hashable or iterable of Hashable, optional
            Selects a subset of the length one dimensions. If a dimension is
            selected with length greater than one, an error is raised. If
            None, all length one dimensions are squeezed.
        drop : bool, optional
            If ``drop=True``, drop squeezed coordinates instead of making them
            scalar.
        axis : None or int or iterable of int, optional
            Like dim, but positional.

        Returns
        -------
        squeezed : same type as caller
            This object, but with with all or a subset of the dimensions of
            length 1 removed.

        See Also
        --------
        numpy.squeeze
        """
        dims = get_squeeze_dims(self, dim, axis)
        return self.isel(drop=drop, **{d: 0 for d in dims})

    def get_index(self, key: Hashable) -> pd.Index:
        """Get an index for a dimension, with fall-back to a default RangeIndex
        """
        if key not in self.dims:
            raise KeyError(key)

        try:
            return self.indexes[key]
        except KeyError:
            # need to ensure dtype=int64 in case range is empty on Python 2
            return pd.Index(range(self.sizes[key]), name=key, dtype=np.int64)

    def _calc_assign_results(
        self: C, kwargs: Mapping[Hashable, Union[T, Callable[[C], T]]]
    ) -> MutableMapping[Hashable, T]:
        results = SortedKeysDict()  # type: SortedKeysDict[Hashable, T]
        for k, v in kwargs.items():
            if callable(v):
                results[k] = v(self)
            else:
                results[k] = v
        return results

    def assign_coords(self, coords=None, **coords_kwargs):
        """Assign new coordinates to this object.

        Returns a new object with all the original data in addition to the new
        coordinates.

        Parameters
        ----------
        coords : dict, optional
            A dict with keys which are variables names. If the values are
            callable, they are computed on this object and assigned to new
            coordinate variables. If the values are not callable,
            (e.g. a ``DataArray``, scalar, or array), they are simply assigned.

        **coords_kwargs : keyword, value pairs, optional
            The keyword arguments form of ``coords``.
            One of ``coords`` or ``coords_kwargs`` must be provided.

        Returns
        -------
        assigned : same type as caller
            A new object with the new coordinates in addition to the existing
            data.

        Examples
        --------
        Convert longitude coordinates from 0-359 to -180-179:

        >>> da = xr.DataArray(np.random.rand(4),
        ...                   coords=[np.array([358, 359, 0, 1])],
        ...                   dims='lon')
        >>> da
        <xarray.DataArray (lon: 4)>
        array([0.28298 , 0.667347, 0.657938, 0.177683])
        Coordinates:
          * lon      (lon) int64 358 359 0 1
        >>> da.assign_coords(lon=(((da.lon + 180) % 360) - 180))
        <xarray.DataArray (lon: 4)>
        array([0.28298 , 0.667347, 0.657938, 0.177683])
        Coordinates:
          * lon      (lon) int64 -2 -1 0 1

        The function also accepts dictionary arguments:

        >>> da.assign_coords({'lon': (((da.lon + 180) % 360) - 180)})
        <xarray.DataArray (lon: 4)>
        array([0.28298 , 0.667347, 0.657938, 0.177683])
        Coordinates:
          * lon      (lon) int64 -2 -1 0 1

        Notes
        -----
        Since ``coords_kwargs`` is a dictionary, the order of your arguments may
        not be preserved, and so the order of the new variables is not well
        defined. Assigning multiple variables within the same ``assign_coords``
        is possible, but you cannot reference other variables created within
        the same ``assign_coords`` call.

        See also
        --------
        Dataset.assign
        Dataset.swap_dims
        """
        coords_kwargs = either_dict_or_kwargs(coords, coords_kwargs, "assign_coords")
        data = self.copy(deep=False)
        results = self._calc_assign_results(coords_kwargs)
        data.coords.update(results)
        return data

    def assign_attrs(self, *args, **kwargs):
        """Assign new attrs to this object.

        Returns a new object equivalent to self.attrs.update(*args, **kwargs).

        Parameters
        ----------
        args : positional arguments passed into ``attrs.update``.
        kwargs : keyword arguments passed into ``attrs.update``.

        Returns
        -------
        assigned : same type as caller
            A new object with the new attrs in addition to the existing data.

        See also
        --------
        Dataset.assign
        """
        out = self.copy(deep=False)
        out.attrs.update(*args, **kwargs)
        return out

    def pipe(
        self,
        func: Union[Callable[..., T], Tuple[Callable[..., T], str]],
        *args,
        **kwargs
    ) -> T:
        """
        Apply func(self, *args, **kwargs)

        This method replicates the pandas method of the same name.

        Parameters
        ----------
        func : function
            function to apply to this xarray object (Dataset/DataArray).
            ``args``, and ``kwargs`` are passed into ``func``.
            Alternatively a ``(callable, data_keyword)`` tuple where
            ``data_keyword`` is a string indicating the keyword of
            ``callable`` that expects the xarray object.
        args : positional arguments passed into ``func``.
        kwargs : a dictionary of keyword arguments passed into ``func``.

        Returns
        -------
        object : the return type of ``func``.

        Notes
        -----

        Use ``.pipe`` when chaining together functions that expect
        xarray or pandas objects, e.g., instead of writing

        >>> f(g(h(ds), arg1=a), arg2=b, arg3=c)

        You can write

        >>> (ds.pipe(h)
        ...    .pipe(g, arg1=a)
        ...    .pipe(f, arg2=b, arg3=c)
        ... )

        If you have a function that takes the data as (say) the second
        argument, pass a tuple indicating which keyword expects the
        data. For example, suppose ``f`` takes its data as ``arg2``:

        >>> (ds.pipe(h)
        ...    .pipe(g, arg1=a)
        ...    .pipe((f, 'arg2'), arg1=a, arg3=c)
        ...  )

        See Also
        --------
        pandas.DataFrame.pipe
        """
        if isinstance(func, tuple):
            func, target = func
            if target in kwargs:
                raise ValueError(
                    "%s is both the pipe target and a keyword " "argument" % target
                )
            kwargs[target] = self
            return func(*args, **kwargs)
        else:
            return func(self, *args, **kwargs)

    def groupby(self, group, squeeze: bool = True, restore_coord_dims: bool = None):
        """Returns a GroupBy object for performing grouped operations.

        Parameters
        ----------
        group : str, DataArray or IndexVariable
            Array whose unique values should be used to group this array. If a
            string, must be the name of a variable contained in this dataset.
        squeeze : boolean, optional
            If "group" is a dimension of any arrays in this dataset, `squeeze`
            controls whether the subarrays have a dimension of length 1 along
            that dimension or if the dimension is squeezed out.
        restore_coord_dims : bool, optional
            If True, also restore the dimension order of multi-dimensional
            coordinates.

        Returns
        -------
        grouped : GroupBy
            A `GroupBy` object patterned after `pandas.GroupBy` that can be
            iterated over in the form of `(unique_value, grouped_array)` pairs.

        Examples
        --------
        Calculate daily anomalies for daily data:

        >>> da = xr.DataArray(np.linspace(0, 1826, num=1827),
        ...                   coords=[pd.date_range('1/1/2000', '31/12/2004',
        ...                           freq='D')],
        ...                   dims='time')
        >>> da
        <xarray.DataArray (time: 1827)>
        array([0.000e+00, 1.000e+00, 2.000e+00, ..., 1.824e+03, 1.825e+03, 1.826e+03])
        Coordinates:
          * time     (time) datetime64[ns] 2000-01-01 2000-01-02 2000-01-03 ...
        >>> da.groupby('time.dayofyear') - da.groupby('time.dayofyear').mean('time')
        <xarray.DataArray (time: 1827)>
        array([-730.8, -730.8, -730.8, ...,  730.2,  730.2,  730.5])
        Coordinates:
          * time       (time) datetime64[ns] 2000-01-01 2000-01-02 2000-01-03 ...
            dayofyear  (time) int64 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 ...

        See Also
        --------
        core.groupby.DataArrayGroupBy
        core.groupby.DatasetGroupBy
        """  # noqa
        return self._groupby_cls(
            self, group, squeeze=squeeze, restore_coord_dims=restore_coord_dims
        )

    def groupby_bins(
        self,
        group,
        bins,
        right: bool = True,
        labels=None,
        precision: int = 3,
        include_lowest: bool = False,
        squeeze: bool = True,
        restore_coord_dims: bool = None,
    ):
        """Returns a GroupBy object for performing grouped operations.

        Rather than using all unique values of `group`, the values are discretized
        first by applying `pandas.cut` [1]_ to `group`.

        Parameters
        ----------
        group : str, DataArray or IndexVariable
            Array whose binned values should be used to group this array. If a
            string, must be the name of a variable contained in this dataset.
        bins : int or array of scalars
            If bins is an int, it defines the number of equal-width bins in the
            range of x. However, in this case, the range of x is extended by .1%
            on each side to include the min or max values of x. If bins is a
            sequence it defines the bin edges allowing for non-uniform bin
            width. No extension of the range of x is done in this case.
        right : boolean, optional
            Indicates whether the bins include the rightmost edge or not. If
            right == True (the default), then the bins [1,2,3,4] indicate
            (1,2], (2,3], (3,4].
        labels : array or boolean, default None
            Used as labels for the resulting bins. Must be of the same length as
            the resulting bins. If False, string bin labels are assigned by
            `pandas.cut`.
        precision : int
            The precision at which to store and display the bins labels.
        include_lowest : bool
            Whether the first interval should be left-inclusive or not.
        squeeze : boolean, optional
            If "group" is a dimension of any arrays in this dataset, `squeeze`
            controls whether the subarrays have a dimension of length 1 along
            that dimension or if the dimension is squeezed out.
        restore_coord_dims : bool, optional
            If True, also restore the dimension order of multi-dimensional
            coordinates.

        Returns
        -------
        grouped : GroupBy
            A `GroupBy` object patterned after `pandas.GroupBy` that can be
            iterated over in the form of `(unique_value, grouped_array)` pairs.
            The name of the group has the added suffix `_bins` in order to
            distinguish it from the original variable.

        References
        ----------
        .. [1] http://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
        """  # noqa
        return self._groupby_cls(
            self,
            group,
            squeeze=squeeze,
            bins=bins,
            restore_coord_dims=restore_coord_dims,
            cut_kwargs={
                "right": right,
                "labels": labels,
                "precision": precision,
                "include_lowest": include_lowest,
            },
        )

    def rolling(
        self,
        dim: Mapping[Hashable, int] = None,
        min_periods: int = None,
        center: bool = False,
        **window_kwargs: int
    ):
        """
        Rolling window object.

        Parameters
        ----------
        dim: dict, optional
            Mapping from the dimension name to create the rolling iterator
            along (e.g. `time`) to its moving window size.
        min_periods : int, default None
            Minimum number of observations in window required to have a value
            (otherwise result is NA). The default, None, is equivalent to
            setting min_periods equal to the size of the window.
        center : boolean, default False
            Set the labels at the center of the window.
        **window_kwargs : optional
            The keyword arguments form of ``dim``.
            One of dim or window_kwargs must be provided.

        Returns
        -------
        Rolling object (core.rolling.DataArrayRolling for DataArray,
        core.rolling.DatasetRolling for Dataset.)

        Examples
        --------
        Create rolling seasonal average of monthly data e.g. DJF, JFM, ..., SON:

        >>> da = xr.DataArray(np.linspace(0, 11, num=12),
        ...                   coords=[pd.date_range('15/12/1999',
        ...                           periods=12, freq=pd.DateOffset(months=1))],
        ...                   dims='time')
        >>> da
        <xarray.DataArray (time: 12)>
        array([  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7., 8.,   9.,  10.,  11.])
        Coordinates:
          * time     (time) datetime64[ns] 1999-12-15 2000-01-15 2000-02-15 ...
        >>> da.rolling(time=3, center=True).mean()
        <xarray.DataArray (time: 12)>
        array([nan,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., nan])
        Coordinates:
          * time     (time) datetime64[ns] 1999-12-15 2000-01-15 2000-02-15 ...

        Remove the NaNs using ``dropna()``:

        >>> da.rolling(time=3, center=True).mean().dropna('time')
        <xarray.DataArray (time: 10)>
        array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])
        Coordinates:
          * time     (time) datetime64[ns] 2000-01-15 2000-02-15 2000-03-15 ...

        See Also
        --------
        core.rolling.DataArrayRolling
        core.rolling.DatasetRolling
        """  # noqa
        dim = either_dict_or_kwargs(dim, window_kwargs, "rolling")
        return self._rolling_cls(self, dim, min_periods=min_periods, center=center)

    def rolling_exp(
        self,
        window: Mapping[Hashable, int] = None,
        window_type: str = "span",
        **window_kwargs
    ):
        """
        Exponentially-weighted moving window.
        Similar to EWM in pandas

        Requires the optional Numbagg dependency.

        Parameters
        ----------
        window : A single mapping from a dimension name to window value,
                 optional
            dim : str
                Name of the dimension to create the rolling exponential window
                along (e.g., `time`).
            window : int
                Size of the moving window. The type of this is specified in
                `window_type`
        window_type : str, one of ['span', 'com', 'halflife', 'alpha'],
                      default 'span'
            The format of the previously supplied window. Each is a simple
            numerical transformation of the others. Described in detail:
            https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.ewm.html
        **window_kwargs : optional
            The keyword arguments form of ``window``.
            One of window or window_kwargs must be provided.

        See Also
        --------
        core.rolling_exp.RollingExp
        """
        window = either_dict_or_kwargs(window, window_kwargs, "rolling_exp")

        return self._rolling_exp_cls(self, window, window_type)

    def coarsen(
        self,
        dim: Mapping[Hashable, int] = None,
        boundary: str = "exact",
        side: Union[str, Mapping[Hashable, str]] = "left",
        coord_func: str = "mean",
        **window_kwargs: int
    ):
        """
        Coarsen object.

        Parameters
        ----------
        dim: dict, optional
            Mapping from the dimension name to the window size.
            dim : str
                Name of the dimension to create the rolling iterator
                along (e.g., `time`).
            window : int
                Size of the moving window.
        boundary : 'exact' | 'trim' | 'pad'
            If 'exact', a ValueError will be raised if dimension size is not a
            multiple of the window size. If 'trim', the excess entries are
            dropped. If 'pad', NA will be padded.
        side : 'left' or 'right' or mapping from dimension to 'left' or 'right'
        coord_func: function (name) that is applied to the coordintes,
            or a mapping from coordinate name to function (name).

        Returns
        -------
        Coarsen object (core.rolling.DataArrayCoarsen for DataArray,
        core.rolling.DatasetCoarsen for Dataset.)

        Examples
        --------
        Coarsen the long time series by averaging over every four days.

        >>> da = xr.DataArray(np.linspace(0, 364, num=364),
        ...                   dims='time',
        ...                   coords={'time': pd.date_range(
        ...                       '15/12/1999', periods=364)})
        >>> da
        <xarray.DataArray (time: 364)>
        array([  0.      ,   1.002755,   2.00551 , ..., 361.99449 , 362.997245,
               364.      ])
        Coordinates:
          * time     (time) datetime64[ns] 1999-12-15 1999-12-16 ... 2000-12-12
        >>>
        >>> da.coarsen(time=3, boundary='trim').mean()
        <xarray.DataArray (time: 121)>
        array([  1.002755,   4.011019,   7.019284,  ...,  358.986226,
               361.99449 ])
        Coordinates:
          * time     (time) datetime64[ns] 1999-12-16 1999-12-19 ... 2000-12-10
        >>>

        See Also
        --------
        core.rolling.DataArrayCoarsen
        core.rolling.DatasetCoarsen
        """
        dim = either_dict_or_kwargs(dim, window_kwargs, "coarsen")
        return self._coarsen_cls(
            self, dim, boundary=boundary, side=side, coord_func=coord_func
        )

    def resample(
        self,
        indexer: Mapping[Hashable, str] = None,
        skipna=None,
        closed: str = None,
        label: str = None,
        base: int = 0,
        keep_attrs: bool = None,
        loffset=None,
        restore_coord_dims: bool = None,
        **indexer_kwargs: str
    ):
        """Returns a Resample object for performing resampling operations.

        Handles both downsampling and upsampling. If any intervals contain no
        values from the original object, they will be given the value ``NaN``.

        Parameters
        ----------
        indexer : {dim: freq}, optional
            Mapping from the dimension name to resample frequency.
        skipna : bool, optional
            Whether to skip missing values when aggregating in downsampling.
        closed : 'left' or 'right', optional
            Side of each interval to treat as closed.
        label : 'left or 'right', optional
            Side of each interval to use for labeling.
        base : int, optional
            For frequencies that evenly subdivide 1 day, the "origin" of the
            aggregated intervals. For example, for '24H' frequency, base could
            range from 0 through 23.
        loffset : timedelta or str, optional
            Offset used to adjust the resampled time labels. Some pandas date
            offset strings are supported.
        keep_attrs : bool, optional
            If True, the object's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        restore_coord_dims : bool, optional
            If True, also restore the dimension order of multi-dimensional
            coordinates.
        **indexer_kwargs : {dim: freq}
            The keyword arguments form of ``indexer``.
            One of indexer or indexer_kwargs must be provided.

        Returns
        -------
        resampled : same type as caller
            This object resampled.

        Examples
        --------
        Downsample monthly time-series data to seasonal data:

        >>> da = xr.DataArray(np.linspace(0, 11, num=12),
        ...                   coords=[pd.date_range('15/12/1999',
        ...                           periods=12, freq=pd.DateOffset(months=1))],
        ...                   dims='time')
        >>> da
        <xarray.DataArray (time: 12)>
        array([  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7., 8.,   9.,  10.,  11.])
        Coordinates:
          * time     (time) datetime64[ns] 1999-12-15 2000-01-15 2000-02-15 ...
        >>> da.resample(time="QS-DEC").mean()
        <xarray.DataArray (time: 4)>
        array([ 1.,  4.,  7., 10.])
        Coordinates:
          * time     (time) datetime64[ns] 1999-12-01 2000-03-01 2000-06-01 2000-09-01

        Upsample monthly time-series data to daily data:

        >>> da.resample(time='1D').interpolate('linear')
        <xarray.DataArray (time: 337)>
        array([ 0.      ,  0.032258,  0.064516, ..., 10.935484, 10.967742, 11.      ])
        Coordinates:
          * time     (time) datetime64[ns] 1999-12-15 1999-12-16 1999-12-17 ...

        Limit scope of upsampling method
        >>> da.resample(time='1D').nearest(tolerance='1D')
        <xarray.DataArray (time: 337)>
        array([ 0.,  0., nan, ..., nan, 11., 11.])
        Coordinates:
          * time     (time) datetime64[ns] 1999-12-15 1999-12-16 ... 2000-11-15

        References
        ----------

        .. [1] http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
        """  # noqa
        # TODO support non-string indexer after removing the old API.

        from .dataarray import DataArray
        from .resample import RESAMPLE_DIM
        from ..coding.cftimeindex import CFTimeIndex

        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=False)

        # note: the second argument (now 'skipna') use to be 'dim'
        if (
            (skipna is not None and not isinstance(skipna, bool))
            or ("how" in indexer_kwargs and "how" not in self.dims)
            or ("dim" in indexer_kwargs and "dim" not in self.dims)
        ):
            raise TypeError(
                "resample() no longer supports the `how` or "
                "`dim` arguments. Instead call methods on resample "
                "objects, e.g., data.resample(time='1D').mean()"
            )

        indexer = either_dict_or_kwargs(indexer, indexer_kwargs, "resample")
        if len(indexer) != 1:
            raise ValueError("Resampling only supported along single dimensions.")
        dim, freq = next(iter(indexer.items()))

        dim_name = dim
        dim_coord = self[dim]

        if isinstance(self.indexes[dim_name], CFTimeIndex):
            from .resample_cftime import CFTimeGrouper

            grouper = CFTimeGrouper(freq, closed, label, base, loffset)
        else:
            # TODO: to_offset() call required for pandas==0.19.2
            grouper = pd.Grouper(
                freq=freq,
                closed=closed,
                label=label,
                base=base,
                loffset=pd.tseries.frequencies.to_offset(loffset),
            )
        group = DataArray(
            dim_coord, coords=dim_coord.coords, dims=dim_coord.dims, name=RESAMPLE_DIM
        )
        resampler = self._resample_cls(
            self,
            group=group,
            dim=dim_name,
            grouper=grouper,
            resample_dim=RESAMPLE_DIM,
            restore_coord_dims=restore_coord_dims,
        )

        return resampler

    def where(self, cond, other=dtypes.NA, drop: bool = False):
        """Filter elements from this object according to a condition.

        This operation follows the normal broadcasting and alignment rules that
        xarray uses for binary arithmetic.

        Parameters
        ----------
        cond : DataArray or Dataset with boolean dtype
            Locations at which to preserve this object's values.
        other : scalar, DataArray or Dataset, optional
            Value to use for locations in this object where ``cond`` is False.
            By default, these locations filled with NA.
        drop : boolean, optional
            If True, coordinate labels that only correspond to False values of
            the condition are dropped from the result. Mutually exclusive with
            ``other``.

        Returns
        -------
        Same type as caller.

        Examples
        --------

        >>> import numpy as np
        >>> a = xr.DataArray(np.arange(25).reshape(5, 5), dims=('x', 'y'))
        >>> a.where(a.x + a.y < 4)
        <xarray.DataArray (x: 5, y: 5)>
        array([[  0.,   1.,   2.,   3.,  nan],
               [  5.,   6.,   7.,  nan,  nan],
               [ 10.,  11.,  nan,  nan,  nan],
               [ 15.,  nan,  nan,  nan,  nan],
               [ nan,  nan,  nan,  nan,  nan]])
        Dimensions without coordinates: x, y
        >>> a.where(a.x + a.y < 5, -1)
        <xarray.DataArray (x: 5, y: 5)>
        array([[ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8, -1],
               [10, 11, 12, -1, -1],
               [15, 16, -1, -1, -1],
               [20, -1, -1, -1, -1]])
        Dimensions without coordinates: x, y
        >>> a.where(a.x + a.y < 4, drop=True)
        <xarray.DataArray (x: 4, y: 4)>
        array([[  0.,   1.,   2.,   3.],
               [  5.,   6.,   7.,  nan],
               [ 10.,  11.,  nan,  nan],
               [ 15.,  nan,  nan,  nan]])
        Dimensions without coordinates: x, y

        See also
        --------
        numpy.where : corresponding numpy function
        where : equivalent function
        """
        from .alignment import align
        from .dataarray import DataArray
        from .dataset import Dataset

        if drop:
            if other is not dtypes.NA:
                raise ValueError("cannot set `other` if drop=True")

            if not isinstance(cond, (Dataset, DataArray)):
                raise TypeError(
                    "cond argument is %r but must be a %r or %r"
                    % (cond, Dataset, DataArray)
                )

            # align so we can use integer indexing
            self, cond = align(self, cond)

            # get cond with the minimal size needed for the Dataset
            if isinstance(cond, Dataset):
                clipcond = cond.to_array().any("variable")
            else:
                clipcond = cond

            # clip the data corresponding to coordinate dims that are not used
            nonzeros = zip(clipcond.dims, np.nonzero(clipcond.values))
            indexers = {k: np.unique(v) for k, v in nonzeros}

            self = self.isel(**indexers)
            cond = cond.isel(**indexers)

        return ops.where_method(self, cond, other)

    def close(self: Any) -> None:
        """Close any files linked to this object
        """
        if self._file_obj is not None:
            self._file_obj.close()
        self._file_obj = None

    def isin(self, test_elements):
        """Tests each value in the array for whether it is in test elements.

        Parameters
        ----------
        test_elements : array_like
            The values against which to test each value of `element`.
            This argument is flattened if an array or array_like.
            See numpy notes for behavior with non-array-like parameters.

        Returns
        -------
        isin : same as object, bool
            Has the same shape as this object.

        Examples
        --------

        >>> array = xr.DataArray([1, 2, 3], dims='x')
        >>> array.isin([1, 3])
        <xarray.DataArray (x: 3)>
        array([ True, False,  True])
        Dimensions without coordinates: x

        See also
        --------
        numpy.isin
        """
        from .computation import apply_ufunc
        from .dataset import Dataset
        from .dataarray import DataArray
        from .variable import Variable

        if isinstance(test_elements, Dataset):
            raise TypeError(
                "isin() argument must be convertible to an array: {}".format(
                    test_elements
                )
            )
        elif isinstance(test_elements, (Variable, DataArray)):
            # need to explicitly pull out data to support dask arrays as the
            # second argument
            test_elements = test_elements.data

        return apply_ufunc(
            duck_array_ops.isin,
            self,
            kwargs=dict(test_elements=test_elements),
            dask="allowed",
        )

    def __enter__(self: T) -> T:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __getitem__(self, value):
        # implementations of this class should implement this method
        raise NotImplementedError


def full_like(other, fill_value, dtype: DTypeLike = None):
    """Return a new object with the same shape and type as a given object.

    Parameters
    ----------
    other : DataArray, Dataset, or Variable
        The reference object in input
    fill_value : scalar
        Value to fill the new object with before returning it.
    dtype : dtype, optional
        dtype of the new array. If omitted, it defaults to other.dtype.

    Returns
    -------
    out : same as object
        New object with the same shape and type as other, with the data
        filled with fill_value. Coords will be copied from other.
        If other is based on dask, the new one will be as well, and will be
        split in the same chunks.
    """
    from .dataarray import DataArray
    from .dataset import Dataset
    from .variable import Variable

    if isinstance(other, Dataset):
        data_vars = OrderedDict(
            (k, _full_like_variable(v, fill_value, dtype))
            for k, v in other.data_vars.items()
        )
        return Dataset(data_vars, coords=other.coords, attrs=other.attrs)
    elif isinstance(other, DataArray):
        return DataArray(
            _full_like_variable(other.variable, fill_value, dtype),
            dims=other.dims,
            coords=other.coords,
            attrs=other.attrs,
            name=other.name,
        )
    elif isinstance(other, Variable):
        return _full_like_variable(other, fill_value, dtype)
    else:
        raise TypeError("Expected DataArray, Dataset, or Variable")


def _full_like_variable(other, fill_value, dtype: DTypeLike = None):
    """Inner function of full_like, where other must be a variable
    """
    from .variable import Variable

    if isinstance(other.data, dask_array_type):
        import dask.array

        if dtype is None:
            dtype = other.dtype
        data = dask.array.full(
            other.shape, fill_value, dtype=dtype, chunks=other.data.chunks
        )
    else:
        data = np.full_like(other, fill_value, dtype=dtype)

    return Variable(dims=other.dims, data=data, attrs=other.attrs)


def zeros_like(other, dtype: DTypeLike = None):
    """Shorthand for full_like(other, 0, dtype)
    """
    return full_like(other, 0, dtype)


def ones_like(other, dtype: DTypeLike = None):
    """Shorthand for full_like(other, 1, dtype)
    """
    return full_like(other, 1, dtype)


def is_np_datetime_like(dtype: DTypeLike) -> bool:
    """Check if a dtype is a subclass of the numpy datetime types
    """
    return np.issubdtype(dtype, np.datetime64) or np.issubdtype(dtype, np.timedelta64)


def _contains_cftime_datetimes(array) -> bool:
    """Check if an array contains cftime.datetime objects
    """
    try:
        from cftime import datetime as cftime_datetime
    except ImportError:
        return False
    else:
        if array.dtype == np.dtype("O") and array.size > 0:
            sample = array.ravel()[0]
            if isinstance(sample, dask_array_type):
                sample = sample.compute()
                if isinstance(sample, np.ndarray):
                    sample = sample.item()
            return isinstance(sample, cftime_datetime)
        else:
            return False


def contains_cftime_datetimes(var) -> bool:
    """Check if an xarray.Variable contains cftime.datetime objects
    """
    return _contains_cftime_datetimes(var.data)


def _contains_datetime_like_objects(var) -> bool:
    """Check if a variable contains datetime like objects (either
    np.datetime64, np.timedelta64, or cftime.datetime)
    """
    return is_np_datetime_like(var.dtype) or contains_cftime_datetimes(var)
