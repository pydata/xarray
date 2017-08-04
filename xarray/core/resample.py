from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy.interpolate import interp1d

from . import ops
from .groupby import DataArrayGroupBy, DatasetGroupBy, GroupBy
from .utils import maybe_wrap_array

RESAMPLE_DIM = '__resample_dim__'


class Resample(GroupBy):
    """An object that extends the `GroupBy` object with additional logic
    for handling specialized re-sampling operations.

    You should create a `Resample` object by using the `DataArray.resample` or
    `Dataset.resample` methods.

    See Also
    --------
    DataArray.resample
    Dataset.resample

    """

    def _upsample(self, method, *args, **kwargs):
        """Dispatch function to call appropriate up-sampling methods on
        data.

        This method should not be called directly; instead, use one of the
        wrapper functions supplied by `Resample`.

        Parameters
        ----------
        method : str {'asfreq', 'pad', 'ffill', 'backfill', 'bfill', 'nearest',
                 'interpolate'}
            Method to use for up-sampling

        See Also
        --------
        Resample.asfreq
        Resample.pad
        Resample.backfill
        Resample.interpolate

        """

        _upsampled_means = self.mean()

        if method == 'asfreq':
            return _upsampled_means

        elif method in ['pad', 'ffill', 'backfill', 'bfill', 'nearest']:
            kwargs = kwargs.copy()
            kwargs.update(**{self._dim: _upsampled_means[self._dim]})
            return self._obj.reindex(method=method, *args, **kwargs)

        elif method == 'interpolate':
            return self._interpolate(*args, **kwargs)

        else:
            raise ValueError('Specified method was "{}" but must be one of'
                             '"asfreq", "ffill", "bfill", or "interpolate"'
                             .format(method))

    def asfreq(self):
        """Return values of original object at the new up-sampling frequency;
        essentially a re-index with new times set to NaN.
        """
        return self._upsample('asfreq')

    def pad(self):
        """Forward fill new values at up-sampled frequency.
        """
        return self._upsample('pad')
    ffill = pad

    def backfill(self):
        """Backward fill new values at up-sampled frequency.
        """
        return self._upsample('backfill')
    bfill = backfill

    def interpolate(self, kind='linear'):
        """Interpolate up-sampled data using the original data
        as knots.

        Parameters
        ----------
        kind : str {'linear', 'nearest', 'zero', 'slinear',
               'quadratic', 'cubic'}
            Interpolation method to use.

        """
        return self._interpolate(kind=kind)

    def _interpolate(self, kind='linear'):
        raise NotImplementedError


class DataArrayResample(DataArrayGroupBy, Resample):
    """DataArrayGroupBy object specialized to time resampling operations over a
    specified dimension
    """

    def __init__(self, *args, **kwargs):

        self._dim = kwargs.pop('dim', None)
        self._resample_dim = kwargs.pop('resample_dim', None)

        if self._dim == self._resample_dim:
            raise ValueError("Proxy resampling dimension ('{_resample_dim}') "
                             "cannot have the same name as actual dimension "
                             "('{_dim}')! ".format(self))
        super(DataArrayResample, self).__init__(*args, **kwargs)


    def apply(self, func, shortcut=False, **kwargs):
        """Apply a function over each array in the group and concatenate them
        together into a new array.

        `func` is called like `func(ar, *args, **kwargs)` for each array `ar`
        in this group.

        Apply uses heuristics (like `pandas.GroupBy.apply`) to figure out how
        to stack together the array. The rule is:
        1. If the dimension along which the group coordinate is defined is
           still in the first grouped array after applying `func`, then stack
           over this dimension.
        2. Otherwise, stack over the new dimension given by name of this
           grouping (the argument to the `groupby` function).

        Parameters
        ----------
        func : function
            Callable to apply to each array.
        shortcut : bool, optional
            Whether or not to shortcut evaluation under the assumptions that:
            (1) The action of `func` does not depend on any of the array
                metadata (attributes or coordinates) but only on the data and
                dimensions.
            (2) The action of `func` creates arrays with homogeneous metadata,
                that is, with the same dimensions and attributes.
            If these conditions are satisfied `shortcut` provides significant
            speedup. This should be the case for many common groupby operations
            (e.g., applying numpy ufuncs).
        **kwargs
            Used to call `func(ar, **kwargs)` for each array `ar`.

        Returns
        -------
        applied : DataArray or DataArray
            The result of splitting, applying and combining this array.
        """
        if shortcut:
            grouped = self._iter_grouped_shortcut()
        else:
            grouped = self._iter_grouped()
        applied = (maybe_wrap_array(arr, func(arr, **kwargs))
                   for arr in grouped)
        combined = self._combine(applied, shortcut=shortcut)

        # If the aggregation function didn't drop the original resampling
        # dimension, then we need to do so before we can rename the proxy
        # dimension we used.
        if self._dim in combined:
            combined = combined.drop(self._dim)

        if self._resample_dim in combined.dims:
            combined = combined.rename({self._resample_dim: self._dim})

        return combined

    def reduce(self, func, dim=None, axis=None, shortcut=True,
               keep_attrs=False, **kwargs):
        """Reduce the items in this group by applying `func` along the
        pre-defined resampling dimension.

        Note that `dim` and `axis` are set by default here and are ignored
        if passed by the user; this ensures compatibility with the existing
        reduce interface.

        Parameters
        ----------
        func : function
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of collapsing an
            np.ndarray over an integer valued axis.
        keep_attrs : bool, optional
            If True, the datasets's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : Array
            Array with summarized data and the indicated dimension(s)
            removed.
        """

        def reduce_array(ar):
            return ar.reduce(func, self._dim, axis=None, keep_attrs=keep_attrs,
                             **kwargs)
        return self.apply(reduce_array, shortcut=shortcut)

    def _interpolate(self, kind='linear'):
        _upsampled = self.mean()

        x = self._obj[self._dim].astype('float')
        y = self._obj.values
        axis = self._obj.get_axis_num(self._dim)

        f = interp1d(x, y, kind=kind, axis=axis, bounds_error=True,
                     assume_sorted=True)
        new_x = _upsampled[self._dim].astype('float')
        _upsampled.values[:] = f(new_x)

        return _upsampled

ops.inject_reduce_methods(DataArrayResample)
ops.inject_binary_ops(DataArrayResample)


class DatasetResample(DatasetGroupBy, Resample):
    """DatasetGroupBy object specialized to resampling a specified dimension
    """

    def __init__(self, *args, **kwargs):

        self._dim = kwargs.pop('dim', None)
        self._resample_dim = kwargs.pop('resample_dim', None)

        if self._dim == self._resample_dim:
            raise ValueError("Proxy resampling dimension ('{_resample_dim}') "
                             "cannot have the same name as actual dimension "
                             "('{_dim}')! ".format(self))
        super(DatasetResample, self).__init__(*args, **kwargs)

    def apply(self, func, **kwargs):
        """Apply a function over each Dataset in the groups generated for
        resampling  and concatenate them together into a new Dataset.

        `func` is called like `func(ds, *args, **kwargs)` for each dataset `ds`
        in this group.

        Apply uses heuristics (like `pandas.GroupBy.apply`) to figure out how
        to stack together the datasets. The rule is:
        1. If the dimension along which the group coordinate is defined is
           still in the first grouped item after applying `func`, then stack
           over this dimension.
        2. Otherwise, stack over the new dimension given by name of this
           grouping (the argument to the `groupby` function).

        Parameters
        ----------
        func : function
            Callable to apply to each sub-dataset.
        **kwargs
            Used to call `func(ds, **kwargs)` for each sub-dataset `ar`.

        Returns
        -------
        applied : Dataset or DataArray
            The result of splitting, applying and combining this dataset.
        """
        kwargs.pop('shortcut', None)  # ignore shortcut if set (for now)
        applied = (func(ds, **kwargs) for ds in self._iter_grouped())
        combined = self._combine(applied)

        return combined.rename({self._resample_dim: self._dim})

    def reduce(self, func, dim=None, keep_attrs=False, **kwargs):
        """Reduce the items in this group by applying `func` along the
        pre-defined resampling dimension.

        Note that `dim` is by default here and ignored if passed by the user;
        this ensures compatibility with the existing reduce interface.

        Parameters
        ----------
        func : function
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of collapsing an
            np.ndarray over an integer valued axis.
        keep_attrs : bool, optional
            If True, the datasets's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : Array
            Array with summarized data and the indicated dimension(s)
            removed.
        """

        def reduce_dataset(ds):
            return ds.reduce(func, self._dim, keep_attrs=keep_attrs, **kwargs)
        return self.apply(reduce_dataset)

        # return result.rename({self._resample_dim: self._dim})

ops.inject_reduce_methods(DatasetResample)
ops.inject_binary_ops(DatasetResample)
