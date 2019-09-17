from . import ops
from .groupby import DataArrayGroupBy, DatasetGroupBy

RESAMPLE_DIM = "__resample_dim__"


class Resample:
    """An object that extends the `GroupBy` object with additional logic
    for handling specialized re-sampling operations.

    You should create a `Resample` object by using the `DataArray.resample` or
    `Dataset.resample` methods. The dimension along re-sampling

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

        upsampled_index = self._full_index

        # Drop non-dimension coordinates along the resampled dimension
        for k, v in self._obj.coords.items():
            if k == self._dim:
                continue
            if self._dim in v.dims:
                self._obj = self._obj.drop(k)

        if method == "asfreq":
            return self.mean(self._dim)

        elif method in ["pad", "ffill", "backfill", "bfill", "nearest"]:
            kwargs = kwargs.copy()
            kwargs.update(**{self._dim: upsampled_index})
            return self._obj.reindex(method=method, *args, **kwargs)

        elif method == "interpolate":
            return self._interpolate(*args, **kwargs)

        else:
            raise ValueError(
                'Specified method was "{}" but must be one of'
                '"asfreq", "ffill", "bfill", or "interpolate"'.format(method)
            )

    def asfreq(self):
        """Return values of original object at the new up-sampling frequency;
        essentially a re-index with new times set to NaN.
        """
        return self._upsample("asfreq")

    def pad(self, tolerance=None):
        """Forward fill new values at up-sampled frequency.

        Parameters
        ----------
        tolerance : optional
            Maximum distance between original and new labels to limit
            the up-sampling method.
            Up-sampled data with indices that satisfy the equation
            ``abs(index[indexer] - target) <= tolerance`` are filled by
            new values. Data with indices that are outside the given
            tolerance are filled with ``NaN``  s
        """
        return self._upsample("pad", tolerance=tolerance)

    ffill = pad

    def backfill(self, tolerance=None):
        """Backward fill new values at up-sampled frequency.

        Parameters
        ----------
        tolerance : optional
            Maximum distance between original and new labels to limit
            the up-sampling method.
            Up-sampled data with indices that satisfy the equation
            ``abs(index[indexer] - target) <= tolerance`` are filled by
            new values. Data with indices that are outside the given
            tolerance are filled with ``NaN`` s
        """
        return self._upsample("backfill", tolerance=tolerance)

    bfill = backfill

    def nearest(self, tolerance=None):
        """Take new values from nearest original coordinate to up-sampled
        frequency coordinates.

        Parameters
        ----------
        tolerance : optional
            Maximum distance between original and new labels to limit
            the up-sampling method.
            Up-sampled data with indices that satisfy the equation
            ``abs(index[indexer] - target) <= tolerance`` are filled by
            new values. Data with indices that are outside the given
            tolerance are filled with ``NaN`` s
        """
        return self._upsample("nearest", tolerance=tolerance)

    def interpolate(self, kind="linear"):
        """Interpolate up-sampled data using the original data
        as knots.

        Parameters
        ----------
        kind : str {'linear', 'nearest', 'zero', 'slinear',
               'quadratic', 'cubic'}
            Interpolation scheme to use

        See Also
        --------
        scipy.interpolate.interp1d

        """
        return self._interpolate(kind=kind)

    def _interpolate(self, kind="linear"):
        """Apply scipy.interpolate.interp1d along resampling dimension."""
        # drop any existing non-dimension coordinates along the resampling
        # dimension
        dummy = self._obj.copy()
        for k, v in self._obj.coords.items():
            if k != self._dim and self._dim in v.dims:
                dummy = dummy.drop(k)
        return dummy.interp(
            assume_sorted=True,
            method=kind,
            kwargs={"bounds_error": False},
            **{self._dim: self._full_index}
        )


class DataArrayResample(DataArrayGroupBy, Resample):
    """DataArrayGroupBy object specialized to time resampling operations over a
    specified dimension
    """

    def __init__(self, *args, dim=None, resample_dim=None, **kwargs):

        if dim == resample_dim:
            raise ValueError(
                "Proxy resampling dimension ('{}') "
                "cannot have the same name as actual dimension "
                "('{}')! ".format(resample_dim, dim)
            )
        self._dim = dim
        self._resample_dim = resample_dim

        super().__init__(*args, **kwargs)

    def apply(self, func, shortcut=False, args=(), **kwargs):
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
        args : tuple, optional
            Positional arguments passed on to `func`.
        **kwargs
            Used to call `func(ar, **kwargs)` for each array `ar`.

        Returns
        -------
        applied : DataArray or DataArray
            The result of splitting, applying and combining this array.
        """
        combined = super().apply(func, shortcut=shortcut, args=args, **kwargs)

        # If the aggregation function didn't drop the original resampling
        # dimension, then we need to do so before we can rename the proxy
        # dimension we used.
        if self._dim in combined.coords:
            combined = combined.drop(self._dim)

        if self._resample_dim in combined.dims:
            combined = combined.rename({self._resample_dim: self._dim})

        return combined


ops.inject_reduce_methods(DataArrayResample)
ops.inject_binary_ops(DataArrayResample)


class DatasetResample(DatasetGroupBy, Resample):
    """DatasetGroupBy object specialized to resampling a specified dimension
    """

    def __init__(self, *args, dim=None, resample_dim=None, **kwargs):

        if dim == resample_dim:
            raise ValueError(
                "Proxy resampling dimension ('{}') "
                "cannot have the same name as actual dimension "
                "('{}')! ".format(resample_dim, dim)
            )
        self._dim = dim
        self._resample_dim = resample_dim

        super().__init__(*args, **kwargs)

    def apply(self, func, args=(), shortcut=None, **kwargs):
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
        args : tuple, optional
            Positional arguments passed on to `func`.
        **kwargs
            Used to call `func(ds, **kwargs)` for each sub-dataset `ar`.

        Returns
        -------
        applied : Dataset or DataArray
            The result of splitting, applying and combining this dataset.
        """
        # ignore shortcut if set (for now)
        applied = (func(ds, *args, **kwargs) for ds in self._iter_grouped())
        combined = self._combine(applied)

        return combined.rename({self._resample_dim: self._dim})

    def reduce(self, func, dim=None, keep_attrs=None, **kwargs):
        """Reduce the items in this group by applying `func` along the
        pre-defined resampling dimension.

        Parameters
        ----------
        func : function
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of collapsing
            an np.ndarray over an integer valued axis.
        dim : str or sequence of str, optional
            Dimension(s) over which to apply `func`.
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
        return super().reduce(func, dim, keep_attrs, **kwargs)


ops.inject_reduce_methods(DatasetResample)
ops.inject_binary_ops(DatasetResample)
