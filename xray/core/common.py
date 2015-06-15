import functools

import numpy as np
import pandas as pd

from .pycompat import basestring, iteritems
from . import formatting
from .utils import SortedKeysDict


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

    _reduce_extra_args_docstring = \
        """dim : str or sequence of str, optional
            Dimension(s) over which to apply `{name}`.
        axis : int or sequence of int, optional
            Axis(es) over which to apply `{name}`. Only one of the 'dim'
            and 'axis' arguments can be supplied. If neither are supplied, then
            `{name}` is calculated over axes."""


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
            Dimension(s) over which to apply `func`.  By default `func` is
            applied over all dimensions."""


class AbstractArray(ImplementsArrayReduce):
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
        return long(self.values)

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


class AttrAccessMixin(object):
    """Mixin class that allow getting keys with attribute access
    """
    @property
    def __attr_sources__(self):
        """List of places to look-up items for attribute-style access"""
        return [self, self.attrs]

    def __getattr__(self, name):
        for source in self.__attr_sources__:
            try:
                return source[name]
            except KeyError:
                pass
        raise AttributeError("%r object has no attribute %r" %
                             (type(self).__name__, name))

    def __dir__(self):
        """Provide method name lookup and completion. Only provide 'public'
        methods.
        """
        extra_attrs = [item for sublist in self.__attr_sources__
                       for item in sublist]
        return sorted(set(dir(type(self)) + extra_attrs))


class BaseDataObject(AttrAccessMixin):
    def _calc_assign_results(self, kwargs):
        results = SortedKeysDict()
        for k, v in kwargs.items():
            if callable(v):
                results[k] = v(self)
            else:
                results[k] = v
        return results

    def assign_coords(self, **kwargs):
        """Assign new coordinates to this object, returning a new object
        with all the original data in addition to the new coordinates.

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
        results = data._calc_assign_results(kwargs)
        data.coords.update(results)
        return data

    def pipe(self, func, *args, **kwargs):
        """
        Apply func(self, *args, **kwargs)

        This method replicates the pandas method of the same name.

        Parameters
        ----------
        func : function
            function to apply to this xray object (Dataset/DataArray).
            ``args``, and ``kwargs`` are passed into ``func``.
            Alternatively a ``(callable, data_keyword)`` tuple where
            ``data_keyword`` is a string indicating the keyword of
            ``callable`` that expects the xray object.
        args : positional arguments passed into ``func``.
        kwargs : a dictionary of keyword arguments passed into ``func``.

        Returns
        -------
        object : the return type of ``func``.

        Notes
        -----

        Use ``.pipe`` when chaining together functions that expect
        xray or pandas objects, e.g., instead of writing

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
        group : str, DataArray or Coordinate
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
        """
        if isinstance(group, basestring):
            group = self[group]
        return self.groupby_cls(self, group, squeeze=squeeze)

    def resample(self, freq, dim, how='mean', skipna=None, closed=None,
                 label=None, base=0):
        """Resample this object to a new temporal resolution.

        Handles both downsampling and upsampling. Upsampling with filling is
        not yet supported; if any intervals contain no values in the original
        object, they will be given the value ``NaN``.

        Parameters
        ----------
        freq : str
            String in the '#offset' to specify the step-size along the
            resampled dimension, where '#' is an (optional) integer multipler
            (default 1) and 'offset' is any pandas date offset alias. Examples
            of valid offsets include:

            * 'AS': year start
            * 'Q-DEC': quarter, starting on December 1
            * 'MS': month start
            * 'D': day
            * 'H': hour
            * 'Min': minute

            The full list of these offset aliases is documented in pandas [1]_.
        dim : str
            Name of the dimension to resample along (e.g., 'time').
        how : str or func, optional
            Used for downsampling. If a string, ``how`` must be a valid
            aggregation operation supported by xray. Otherwise, ``how`` must be
            a function that can be called like ``how(values, axis)`` to reduce
            ndarray values along the given axis. Valid choices that can be
            provided as a string include all the usual Dataset/DataArray
            aggregations (``all``, ``any``, ``argmax``, ``argmin``, ``max``,
            ``mean``, ``median``, ``min``, ``prod``, ``sum``, ``std`` and
            ``var``), as well as ``first`` and ``last``.
        skipna : bool, optional
            Whether to skip missing values when aggregating in downsampling.
        closed : 'left' or 'right', optional
            Side of each interval to treat as closed.
        label : 'left or 'right', optional
            Side of each interval to use for labeling.
        base : int, optionalt
            For frequencies that evenly subdivide 1 day, the "origin" of the
            aggregated intervals. For example, for '24H' frequency, base could
            range from 0 through 23.

        Returns
        -------
        resampled : same type as caller
            This object resampled.

        References
        ----------

        .. [1] http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
        """
        from .dataarray import DataArray

        RESAMPLE_DIM = '__resample_dim__'
        if isinstance(dim, basestring):
            dim = self[dim]
        group = DataArray(dim, name=RESAMPLE_DIM)
        time_grouper = pd.TimeGrouper(freq=freq, how=how, closed=closed,
                                      label=label, base=base)
        gb = self.groupby_cls(self, group, grouper=time_grouper)
        if isinstance(how, basestring):
            f = getattr(gb, how)
            if how in ['first', 'last']:
                result = f(skipna=skipna)
            else:
                result = f(dim=dim.name, skipna=skipna)
        else:
            result = gb.reduce(how, dim=dim.name)
        result = result.rename({RESAMPLE_DIM: dim.name})
        return result


def squeeze(xray_obj, dims, dim=None):
    """Squeeze the dims of an xray object."""
    if dim is None:
        dim = [d for d, s in iteritems(dims) if s == 1]
    else:
        if isinstance(dim, basestring):
            dim = [dim]
        if any(dims[k] > 1 for k in dim):
            raise ValueError('cannot select a dimension to squeeze out '
                             'which has length greater than one')
    return xray_obj.isel(**dict((d, 0) for d in dim))


def _maybe_promote(dtype):
    """Simpler equivalent of pandas.core.common._maybe_promote"""
    # N.B. these casting rules should match pandas
    if np.issubdtype(dtype, float):
        fill_value = np.nan
    elif np.issubdtype(dtype, int):
        # convert to floating point so NaN is valid
        dtype = float
        fill_value = np.nan
    elif np.issubdtype(dtype, np.datetime64):
        fill_value = np.datetime64('NaT')
    elif np.issubdtype(dtype, np.timedelta64):
        fill_value = np.timedelta64('NaT')
    else:
        dtype = object
        fill_value = np.nan
    return np.dtype(dtype), fill_value


def _possibly_convert_objects(values):
    """Convert arrays of datetime.datetime and datetime.timedelta objects into
    datetime64 and timedelta64
    """
    try:
        converter = functools.partial(pd.core.common._possibly_convert_objects,
                                      convert_numeric=False)
    except AttributeError:
        # our fault for using a private pandas API that has gone missing
        # this should do the same coercion (though it will be slower)
        converter = lambda x: np.asarray(pd.Series(x))
    return converter(values.ravel()).reshape(values.shape)
