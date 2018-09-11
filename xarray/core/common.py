from __future__ import absolute_import, division, print_function

import warnings
from distutils.version import LooseVersion
from textwrap import dedent

import numpy as np
import pandas as pd

from . import duck_array_ops, dtypes, formatting, ops
from .arithmetic import SupportsArithmetic
from .pycompat import OrderedDict, basestring, dask_array_type, suppress
from .utils import either_dict_or_kwargs, Frozen, SortedKeysDict


class ImplementsArrayReduce(object):
    @classmethod
    def _reduce_method(cls, func, include_skipna, numeric_only):
        if include_skipna:
            def wrapped_func(self, dim=None, axis=None, skipna=None,
                             keep_attrs=False, **kwargs):
                return self.reduce(func, dim, axis, keep_attrs=keep_attrs,
                                   skipna=skipna, allow_lazy=True, **kwargs)
        else:
            def wrapped_func(self, dim=None, axis=None, keep_attrs=False,
                             **kwargs):
                return self.reduce(func, dim, axis, keep_attrs=keep_attrs,
                                   allow_lazy=True, **kwargs)
        return wrapped_func

    _reduce_extra_args_docstring = dedent("""\
        dim : str or sequence of str, optional
            Dimension(s) over which to apply `{name}`.
        axis : int or sequence of int, optional
            Axis(es) over which to apply `{name}`. Only one of the 'dim'
            and 'axis' arguments can be supplied. If neither are supplied, then
            `{name}` is calculated over axes.""")

    _cum_extra_args_docstring = dedent("""\
        dim : str or sequence of str, optional
            Dimension over which to apply `{name}`.
        axis : int or sequence of int, optional
            Axis over which to apply `{name}`. Only one of the 'dim'
            and 'axis' arguments can be supplied.""")


class ImplementsDatasetReduce(object):
    @classmethod
    def _reduce_method(cls, func, include_skipna, numeric_only):
        if include_skipna:
            def wrapped_func(self, dim=None, keep_attrs=False, skipna=None,
                             **kwargs):
                return self.reduce(func, dim, keep_attrs, skipna=skipna,
                                   numeric_only=numeric_only, allow_lazy=True,
                                   **kwargs)
        else:
            def wrapped_func(self, dim=None, keep_attrs=False, **kwargs):
                return self.reduce(func, dim, keep_attrs,
                                   numeric_only=numeric_only, allow_lazy=True,
                                   **kwargs)
        return wrapped_func

    _reduce_extra_args_docstring = \
        """dim : str or sequence of str, optional
            Dimension(s) over which to apply `{name}`.  By default `{name}` is
            applied over all dimensions."""

    _cum_extra_args_docstring = \
        """dim : str or sequence of str, optional
            Dimension over which to apply `{name}`.
        axis : int or sequence of int, optional
            Axis over which to apply `{name}`. Only one of the 'dim'
            and 'axis' arguments can be supplied."""


class AbstractArray(ImplementsArrayReduce, formatting.ReprMixin):
    """Shared base class for DataArray and Variable."""

    def __bool__(self):
        return bool(self.values)

    # Python 3 uses __bool__, Python 2 uses __nonzero__
    __nonzero__ = __bool__

    def __float__(self):
        return float(self.values)

    def __int__(self):
        return int(self.values)

    def __complex__(self):
        return complex(self.values)

    def __long__(self):
        return long(self.values)  # flake8: noqa

    def __array__(self, dtype=None):
        return np.asarray(self.values, dtype=dtype)

    def __repr__(self):
        return formatting.array_repr(self)

    def _iter(self):
        for n in range(len(self)):
            yield self[n]

    def __iter__(self):
        if self.ndim == 0:
            raise TypeError('iteration over a 0-d array')
        return self._iter()

    @property
    def T(self):
        return self.transpose()

    def get_axis_num(self, dim):
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
        if isinstance(dim, basestring):
            return self._get_axis_num(dim)
        else:
            return tuple(self._get_axis_num(d) for d in dim)

    def _get_axis_num(self, dim):
        try:
            return self.dims.index(dim)
        except ValueError:
            raise ValueError("%r not found in array dimensions %r" %
                             (dim, self.dims))

    @property
    def sizes(self):
        """Ordered mapping from dimension names to lengths.

        Immutable.

        See also
        --------
        Dataset.sizes
        """
        return Frozen(OrderedDict(zip(self.dims, self.shape)))


class AttrAccessMixin(object):
    """Mixin class that allows getting keys with attribute access
    """
    _initialized = False

    @property
    def _attr_sources(self):
        """List of places to look-up items for attribute-style access"""
        return []

    @property
    def _item_sources(self):
        """List of places to look-up items for key-autocompletion """
        return []

    def __getattr__(self, name):
        if name != '__setstate__':
            # this avoids an infinite loop when pickle looks for the
            # __setstate__ attribute before the xarray object is initialized
            for source in self._attr_sources:
                with suppress(KeyError):
                    return source[name]
        raise AttributeError("%r object has no attribute %r" %
                             (type(self).__name__, name))

    def __setattr__(self, name, value):
        if self._initialized:
            try:
                # Allow setting instance variables if they already exist
                # (e.g., _attrs). We use __getattribute__ instead of hasattr
                # to avoid key lookups with attribute-style access.
                self.__getattribute__(name)
            except AttributeError:
                raise AttributeError(
                    "cannot set attribute %r on a %r object. Use __setitem__ "
                    "style assignment (e.g., `ds['name'] = ...`) instead to "
                    "assign variables." % (name, type(self).__name__))
        object.__setattr__(self, name, value)

    def __dir__(self):
        """Provide method name lookup and completion. Only provide 'public'
        methods.
        """
        extra_attrs = [item
                       for sublist in self._attr_sources
                       for item in sublist
                       if isinstance(item, basestring)]
        return sorted(set(dir(type(self)) + extra_attrs))

    def _ipython_key_completions_(self):
        """Provide method for the key-autocompletions in IPython.
        See http://ipython.readthedocs.io/en/stable/config/integrating.html#tab-completion
        For the details.
        """
        item_lists = [item
                      for sublist in self._item_sources
                      for item in sublist
                      if isinstance(item, basestring)]
        return list(set(item_lists))


def get_squeeze_dims(xarray_obj, dim, axis=None):
    """Get a list of dimensions to squeeze out.
    """
    if dim is not None and axis is not None:
        raise ValueError('cannot use both parameters `axis` and `dim`')

    if dim is None and axis is None:
        dim = [d for d, s in xarray_obj.sizes.items() if s == 1]
    else:
        if isinstance(dim, basestring):
            dim = [dim]
        if isinstance(axis, int):
            axis = (axis, )
        if isinstance(axis, tuple):
            for a in axis:
                if not isinstance(a, int):
                    raise ValueError(
                        'parameter `axis` must be int or tuple of int.')
            alldims = list(xarray_obj.sizes.keys())
            dim = [alldims[a] for a in axis]
        if any(xarray_obj.sizes[k] > 1 for k in dim):
            raise ValueError('cannot select a dimension to squeeze out '
                             'which has length greater than one')
    return dim


class DataWithCoords(SupportsArithmetic, AttrAccessMixin):
    """Shared base class for Dataset and DataArray."""

    def squeeze(self, dim=None, drop=False, axis=None):
        """Return a new object with squeezed data.

        Parameters
        ----------
        dim : None or str or tuple of str, optional
            Selects a subset of the length one dimensions. If a dimension is
            selected with length greater than one, an error is raised. If
            None, all length one dimensions are squeezed.
        drop : bool, optional
            If ``drop=True``, drop squeezed coordinates instead of making them
            scalar.
        axis : int, optional
            Select the dimension to squeeze. Added for compatibility reasons.

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

    def get_index(self, key):
        """Get an index for a dimension, with fall-back to a default RangeIndex
        """
        if key not in self.dims:
            raise KeyError(key)

        try:
            return self.indexes[key]
        except KeyError:
            # need to ensure dtype=int64 in case range is empty on Python 2
            return pd.Index(range(self.sizes[key]), name=key, dtype=np.int64)

    def _calc_assign_results(self, kwargs):
        results = SortedKeysDict()
        for k, v in kwargs.items():
            if callable(v):
                results[k] = v(self)
            else:
                results[k] = v
        return results

    def assign_coords(self, **kwargs):
        """Assign new coordinates to this object.

        Returns a new object with all the original data in addition to the new
        coordinates.

        Parameters
        ----------
        kwargs : keyword, value pairs
            keywords are the variables names. If the values are callable, they
            are computed on this object and assigned to new coordinate
            variables. If the values are not callable, (e.g. a DataArray,
            scalar, or array), they are simply assigned.

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

        Notes
        -----
        Since ``kwargs`` is a dictionary, the order of your arguments may not
        be preserved, and so the order of the new variables is not well
        defined. Assigning multiple variables within the same ``assign_coords``
        is possible, but you cannot reference other variables created within
        the same ``assign_coords`` call.

        See also
        --------
        Dataset.assign
        """
        data = self.copy(deep=False)
        results = self._calc_assign_results(kwargs)
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

    def pipe(self, func, *args, **kwargs):
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
                msg = '%s is both the pipe target and a keyword argument' % target
                raise ValueError(msg)
            kwargs[target] = self
            return func(*args, **kwargs)
        else:
            return func(self, *args, **kwargs)

    def groupby(self, group, squeeze=True):
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
        """
        return self._groupby_cls(self, group, squeeze=squeeze)

    def groupby_bins(self, group, bins, right=True, labels=None, precision=3,
                     include_lowest=False, squeeze=True):
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
        """
        return self._groupby_cls(self, group, squeeze=squeeze, bins=bins,
                                 cut_kwargs={'right': right, 'labels': labels,
                                             'precision': precision,
                                             'include_lowest': include_lowest})

    def rolling(self, dim=None, min_periods=None, center=False, **dim_kwargs):
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
        **dim_kwargs : optional
            The keyword arguments form of ``dim``.
            One of dim or dim_kwarg must be provided.

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
        """
        dim = either_dict_or_kwargs(dim, dim_kwargs, 'rolling')
        return self._rolling_cls(self, dim, min_periods=min_periods,
                                 center=center)

    def resample(self, freq=None, dim=None, how=None, skipna=None,
                 closed=None, label=None, base=0, keep_attrs=False, **indexer):
        """Returns a Resample object for performing resampling operations.

        Handles both downsampling and upsampling. If any intervals contain no
        values from the original object, they will be given the value ``NaN``.

        Parameters
        ----------
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
        keep_attrs : bool, optional
            If True, the object's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        **indexer : {dim: freq}
            Dictionary with a key indicating the dimension name to resample
            over and a value corresponding to the resampling frequency.

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

        References
        ----------

        .. [1] http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
        """
        # TODO support non-string indexer after removing the old API.

        from .dataarray import DataArray
        from .resample import RESAMPLE_DIM

        if dim is not None:
            if how is None:
                how = 'mean'
            return self._resample_immediately(freq, dim, how, skipna, closed,
                                              label, base, keep_attrs)

        if (how is not None) and indexer:
            raise TypeError("If passing an 'indexer' then 'dim' "
                            "and 'how' should not be used")

        # More than one indexer is ambiguous, but we do in fact need one if
        # "dim" was not provided, until the old API is fully deprecated
        if len(indexer) != 1:
            raise ValueError(
                "Resampling only supported along single dimensions."
            )
        dim, freq = indexer.popitem()

        if isinstance(dim, basestring):
            dim_name = dim
            dim = self[dim]
        else:
            raise TypeError("Dimension name should be a string; "
                            "was passed %r" % dim)
        group = DataArray(dim, [(dim.dims, dim)], name=RESAMPLE_DIM)
        grouper = pd.Grouper(freq=freq, closed=closed, label=label, base=base)
        resampler = self._resample_cls(self, group=group, dim=dim_name,
                                       grouper=grouper,
                                       resample_dim=RESAMPLE_DIM)

        return resampler

    def _resample_immediately(self, freq, dim, how, skipna,
                              closed, label, base, keep_attrs):
        """Implement the original version of .resample() which immediately
        executes the desired resampling operation. """
        from .dataarray import DataArray
        RESAMPLE_DIM = '__resample_dim__'

        warnings.warn("\n.resample() has been modified to defer "
                      "calculations. Instead of passing 'dim' and "
                      "how=\"{how}\", instead consider using "
                      ".resample({dim}=\"{freq}\").{how}('{dim}') ".format(
                      dim=dim, freq=freq, how=how),
                      FutureWarning, stacklevel=3)

        if isinstance(dim, basestring):
            dim = self[dim]
        group = DataArray(dim, [(dim.dims, dim)], name=RESAMPLE_DIM)
        grouper = pd.Grouper(freq=freq, how=how, closed=closed, label=label,
                             base=base)
        gb = self._groupby_cls(self, group, grouper=grouper)
        if isinstance(how, basestring):
            f = getattr(gb, how)
            if how in ['first', 'last']:
                result = f(skipna=skipna, keep_attrs=keep_attrs)
            elif how == 'count':
                result = f(dim=dim.name, keep_attrs=keep_attrs)
            else:
                result = f(dim=dim.name, skipna=skipna, keep_attrs=keep_attrs)
        else:
            result = gb.reduce(how, dim=dim.name, keep_attrs=keep_attrs)
        result = result.rename({RESAMPLE_DIM: dim.name})
        return result

    def where(self, cond, other=dtypes.NA, drop=False):
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
                raise ValueError('cannot set `other` if drop=True')

            if not isinstance(cond, (Dataset, DataArray)):
                raise TypeError("cond argument is %r but must be a %r or %r" %
                                (cond, Dataset, DataArray))

            # align so we can use integer indexing
            self, cond = align(self, cond)

            # get cond with the minimal size needed for the Dataset
            if isinstance(cond, Dataset):
                clipcond = cond.to_array().any('variable')
            else:
                clipcond = cond

            # clip the data corresponding to coordinate dims that are not used
            nonzeros = zip(clipcond.dims, np.nonzero(clipcond.values))
            indexers = {k: np.unique(v) for k, v in nonzeros}

            self = self.isel(**indexers)
            cond = cond.isel(**indexers)

        return ops.where_method(self, cond, other)

    def close(self):
        """Close any files linked to this object
        """
        if self._file_obj is not None:
            self._file_obj.close()
        self._file_obj = None

    def isin(self, test_elements):
        """Tests each value in the array for whether it is in the supplied list.

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
                'isin() argument must be convertible to an array: {}'
                .format(test_elements))
        elif isinstance(test_elements, (Variable, DataArray)):
            # need to explicitly pull out data to support dask arrays as the
            # second argument
            test_elements = test_elements.data

        return apply_ufunc(
            duck_array_ops.isin,
            self,
            kwargs=dict(test_elements=test_elements),
            dask='allowed',
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def full_like(other, fill_value, dtype=None):
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
            for k, v in other.data_vars.items())
        return Dataset(data_vars, coords=other.coords, attrs=other.attrs)
    elif isinstance(other, DataArray):
        return DataArray(
            _full_like_variable(other.variable, fill_value, dtype),
            dims=other.dims, coords=other.coords, attrs=other.attrs,
            name=other.name)
    elif isinstance(other, Variable):
        return _full_like_variable(other, fill_value, dtype)
    else:
        raise TypeError("Expected DataArray, Dataset, or Variable")


def _full_like_variable(other, fill_value, dtype=None):
    """Inner function of full_like, where other must be a variable
    """
    from .variable import Variable

    if isinstance(other.data, dask_array_type):
        import dask.array
        if dtype is None:
            dtype = other.dtype
        data = dask.array.full(other.shape, fill_value, dtype=dtype,
                               chunks=other.data.chunks)
    else:
        data = np.full_like(other, fill_value, dtype=dtype)

    return Variable(dims=other.dims, data=data, attrs=other.attrs)


def zeros_like(other, dtype=None):
    """Shorthand for full_like(other, 0, dtype)
    """
    return full_like(other, 0, dtype)


def ones_like(other, dtype=None):
    """Shorthand for full_like(other, 1, dtype)
    """
    return full_like(other, 1, dtype)


def is_np_datetime_like(dtype):
    """Check if a dtype is a subclass of the numpy datetime types
    """
    return (np.issubdtype(dtype, np.datetime64) or
            np.issubdtype(dtype, np.timedelta64))


def contains_cftime_datetimes(var):
    """Check if a variable contains cftime datetime objects"""
    try:
        from cftime import datetime as cftime_datetime
    except ImportError:
        return False
    else:
        if var.dtype == np.dtype('O') and var.data.size > 0:
            sample = var.data.ravel()[0]
            if isinstance(sample, dask_array_type):
                sample = sample.compute()
                if isinstance(sample, np.ndarray):
                    sample = sample.item()
            return isinstance(sample, cftime_datetime)
        else:
            return False


def _contains_datetime_like_objects(var):
    """Check if a variable contains datetime like objects (either
    np.datetime64, np.timedelta64, or cftime.datetime)"""
    return is_np_datetime_like(var.dtype) or contains_cftime_datetimes(var)
