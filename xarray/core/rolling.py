from __future__ import absolute_import, division, print_function

import warnings
from distutils.version import LooseVersion

import numpy as np

from . import dtypes
from .dask_array_ops import dask_rolling_wrapper
from .ops import (
    bn, has_bottleneck, inject_bottleneck_rolling_methods,
    inject_datasetrolling_methods)
from .pycompat import OrderedDict, dask_array_type, zip


def _get_new_dimname(dims, new_dim):
    """ Get an new dimension name based on new_dim, that is not used in dims.
    If the same name exists, we add an underscore(s) in the head.

    Example1:
        dims: ['a', 'b', 'c']
        new_dim: ['_rolling']
        -> ['_rolling']
    Example2:
        dims: ['a', 'b', 'c', '_rolling']
        new_dim: ['_rolling']
        -> ['__rolling']
    """
    while new_dim in dims:
        new_dim = '_' + new_dim
    return new_dim


class Rolling(object):
    """A object that implements the moving window pattern.

    See Also
    --------
    Dataset.groupby
    DataArray.groupby
    Dataset.rolling
    DataArray.rolling
    """

    _attributes = ['window', 'min_periods', 'center', 'dim']

    def __init__(self, obj, windows, min_periods=None, center=False):
        """
        Moving window object.

        Parameters
        ----------
        obj : Dataset or DataArray
            Object to window.
        windows : A mapping from a dimension name to window size
            dim : str
                Name of the dimension to create the rolling iterator
                along (e.g., `time`).
            window : int
                Size of the moving window.
        min_periods : int, default None
            Minimum number of observations in window required to have a value
            (otherwise result is NA). The default, None, is equivalent to
            setting min_periods equal to the size of the window.
        center : boolean, default False
            Set the labels at the center of the window.

        Returns
        -------
        rolling : type of input argument
        """

        if (has_bottleneck and
                (LooseVersion(bn.__version__) < LooseVersion('1.0'))):
            warnings.warn('xarray requires bottleneck version of 1.0 or '
                          'greater for rolling operations. Rolling '
                          'aggregation methods will use numpy instead'
                          'of bottleneck.')

        if len(windows) != 1:
            raise ValueError('exactly one dim/window should be provided')

        dim, window = next(iter(windows.items()))

        if window <= 0:
            raise ValueError('window must be > 0')

        self.obj = obj

        # attributes
        self.window = window
        self.min_periods = min_periods
        if min_periods is None:
            self._min_periods = window
        else:
            if min_periods <= 0:
                raise ValueError(
                    'min_periods must be greater than zero or None')

            self._min_periods = min_periods
        self.center = center
        self.dim = dim

    def __repr__(self):
        """provide a nice str repr of our rolling object"""

        attrs = ["{k}->{v}".format(k=k, v=getattr(self, k))
                 for k in self._attributes
                 if getattr(self, k, None) is not None]
        return "{klass} [{attrs}]".format(klass=self.__class__.__name__,
                                          attrs=','.join(attrs))

    def __len__(self):
        return self.obj.sizes[self.dim]


class DataArrayRolling(Rolling):
    def __init__(self, obj, windows, min_periods=None, center=False):
        """
        Moving window object for DataArray.
        You should use DataArray.rolling() method to construct this object
        instead of the class constructor.

        Parameters
        ----------
        obj : DataArray
            Object to window.
        windows : A mapping from a dimension name to window size
            dim : str
                Name of the dimension to create the rolling iterator
                along (e.g., `time`).
            window : int
                Size of the moving window.
        min_periods : int, default None
            Minimum number of observations in window required to have a value
            (otherwise result is NA). The default, None, is equivalent to
            setting min_periods equal to the size of the window.
        center : boolean, default False
            Set the labels at the center of the window.

        Returns
        -------
        rolling : type of input argument

        See Also
        --------
        DataArray.rolling
        DataArray.groupby
        Dataset.rolling
        Dataset.groupby
        """
        super(DataArrayRolling, self).__init__(
            obj, windows, min_periods=min_periods, center=center)

        self.window_labels = self.obj[self.dim]

    def __iter__(self):
        stops = np.arange(1, len(self.window_labels) + 1)
        starts = stops - int(self.window)
        starts[:int(self.window)] = 0
        for (label, start, stop) in zip(self.window_labels, starts, stops):
            window = self.obj.isel(**{self.dim: slice(start, stop)})

            counts = window.count(dim=self.dim)
            window = window.where(counts >= self._min_periods)

            yield (label, window)

    def construct(self, window_dim, stride=1, fill_value=dtypes.NA):
        """
        Convert this rolling object to xr.DataArray,
        where the window dimension is stacked as a new dimension

        Parameters
        ----------
        window_dim: str
            New name of the window dimension.
        stride: integer, optional
            Size of stride for the rolling window.
        fill_value: optional. Default dtypes.NA
            Filling value to match the dimension size.

        Returns
        -------
        DataArray that is a view of the original array. The returned array is
        not writeable.

        Examples
        --------
        >>> da = DataArray(np.arange(8).reshape(2, 4), dims=('a', 'b'))
        >>>
        >>> rolling = da.rolling(a=3)
        >>> rolling.to_datarray('window_dim')
        <xarray.DataArray (a: 2, b: 4, window_dim: 3)>
        array([[[np.nan, np.nan, 0], [np.nan, 0, 1], [0, 1, 2], [1, 2, 3]],
               [[np.nan, np.nan, 4], [np.nan, 4, 5], [4, 5, 6], [5, 6, 7]]])
        Dimensions without coordinates: a, b, window_dim
        >>>
        >>> rolling = da.rolling(a=3, center=True)
        >>> rolling.to_datarray('window_dim')
        <xarray.DataArray (a: 2, b: 4, window_dim: 3)>
        array([[[np.nan, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, np.nan]],
               [[np.nan, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, np.nan]]])
        Dimensions without coordinates: a, b, window_dim
        """

        from .dataarray import DataArray

        window = self.obj.variable.rolling_window(self.dim, self.window,
                                                  window_dim, self.center,
                                                  fill_value=fill_value)
        result = DataArray(window, dims=self.obj.dims + (window_dim,),
                           coords=self.obj.coords)
        return result.isel(**{self.dim: slice(None, None, stride)})

    def reduce(self, func, **kwargs):
        """Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : function
            Function which can be called in the form
            `func(x, **kwargs)` to return the result of collapsing an
            np.ndarray over an the rolling dimension.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : DataArray
            Array with summarized data.
        """
        rolling_dim = _get_new_dimname(self.obj.dims, '_rolling_dim')
        windows = self.construct(rolling_dim)
        result = windows.reduce(func, dim=rolling_dim, **kwargs)

        # Find valid windows based on count.
        counts = self._counts()
        return result.where(counts >= self._min_periods)

    def _counts(self):
        """ Number of non-nan entries in each rolling window. """

        rolling_dim = _get_new_dimname(self.obj.dims, '_rolling_dim')
        # We use False as the fill_value instead of np.nan, since boolean
        # array is faster to be reduced than object array.
        # The use of skipna==False is also faster since it does not need to
        # copy the strided array.
        counts = (self.obj.notnull()
                  .rolling(center=self.center, **{self.dim: self.window})
                  .construct(rolling_dim, fill_value=False)
                  .sum(dim=rolling_dim, skipna=False))
        return counts

    @classmethod
    def _reduce_method(cls, func):
        """
        Methods to return a wrapped function for any function `func` for
        numpy methods.
        """

        def wrapped_func(self, **kwargs):
            return self.reduce(func, **kwargs)
        return wrapped_func

    @classmethod
    def _bottleneck_reduce(cls, func):
        """
        Methods to return a wrapped function for any function `func` for
        bottoleneck method, except for `median`.
        """

        def wrapped_func(self, **kwargs):
            from .dataarray import DataArray

            # bottleneck doesn't allow min_count to be 0, although it should
            # work the same as if min_count = 1
            if self.min_periods is not None and self.min_periods == 0:
                min_count = 1
            else:
                min_count = self.min_periods

            axis = self.obj.get_axis_num(self.dim)

            padded = self.obj.variable
            if self.center:
                if (LooseVersion(np.__version__) < LooseVersion('1.13') and
                        self.obj.dtype.kind == 'b'):
                    # with numpy < 1.13 bottleneck cannot handle np.nan-Boolean
                    # mixed array correctly. We cast boolean array to float.
                    padded = padded.astype(float)

                if isinstance(padded.data, dask_array_type):
                    # Workaround to make the padded chunk size is larger than
                    # self.window-1
                    shift = - (self.window + 1) // 2
                    offset = (self.window - 1) // 2
                    valid = (slice(None), ) * axis + (
                        slice(offset, offset + self.obj.shape[axis]), )
                else:
                    shift = (-self.window // 2) + 1
                    valid = (slice(None), ) * axis + (slice(-shift, None), )
                padded = padded.pad_with_fill_value(**{self.dim: (0, -shift)})

            if isinstance(padded.data, dask_array_type):
                values = dask_rolling_wrapper(func, padded,
                                              window=self.window,
                                              min_count=min_count,
                                              axis=axis)
            else:
                values = func(padded.data, window=self.window,
                              min_count=min_count, axis=axis)

            if self.center:
                values = values[valid]
            result = DataArray(values, self.obj.coords)

            return result
        return wrapped_func


class DatasetRolling(Rolling):
    def __init__(self, obj, windows, min_periods=None, center=False):
        """
        Moving window object for Dataset.
        You should use Dataset.rolling() method to construct this object
        instead of the class constructor.

        Parameters
        ----------
        obj : Dataset
            Object to window.
        windows : A mapping from a dimension name to window size
            dim : str
                Name of the dimension to create the rolling iterator
                along (e.g., `time`).
            window : int
                Size of the moving window.
        min_periods : int, default None
            Minimum number of observations in window required to have a value
            (otherwise result is NA). The default, None, is equivalent to
            setting min_periods equal to the size of the window.
        center : boolean, default False
            Set the labels at the center of the window.

        Returns
        -------
        rolling : type of input argument

        See Also
        --------
        Dataset.rolling
        DataArray.rolling
        Dataset.groupby
        DataArray.groupby
        """
        super(DatasetRolling, self).__init__(obj, windows, min_periods, center)
        if self.dim not in self.obj.dims:
            raise KeyError(self.dim)
        # Keep each Rolling object as an OrderedDict
        self.rollings = OrderedDict()
        for key, da in self.obj.data_vars.items():
            # keeps rollings only for the dataset depending on slf.dim
            if self.dim in da.dims:
                self.rollings[key] = DataArrayRolling(
                    da, windows, min_periods, center)

    def reduce(self, func, **kwargs):
        """Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : function
            Function which can be called in the form
            `func(x, **kwargs)` to return the result of collapsing an
            np.ndarray over an the rolling dimension.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : DataArray
            Array with summarized data.
        """
        from .dataset import Dataset
        reduced = OrderedDict()
        for key, da in self.obj.data_vars.items():
            if self.dim in da.dims:
                reduced[key] = self.rollings[key].reduce(func, **kwargs)
            else:
                reduced[key] = self.obj[key]
        return Dataset(reduced, coords=self.obj.coords)

    def _counts(self):
        from .dataset import Dataset
        reduced = OrderedDict()
        for key, da in self.obj.data_vars.items():
            if self.dim in da.dims:
                reduced[key] = self.rollings[key]._counts()
            else:
                reduced[key] = self.obj[key]
        return Dataset(reduced, coords=self.obj.coords)

    @classmethod
    def _reduce_method(cls, func):
        """
        Return a wrapped function for injecting numpy and bottoleneck methods.
        see ops.inject_datasetrolling_methods
        """

        def wrapped_func(self, **kwargs):
            from .dataset import Dataset
            reduced = OrderedDict()
            for key, da in self.obj.data_vars.items():
                if self.dim in da.dims:
                    reduced[key] = getattr(self.rollings[key],
                                           func.__name__)(**kwargs)
                else:
                    reduced[key] = self.obj[key]
            return Dataset(reduced, coords=self.obj.coords)
        return wrapped_func

    def construct(self, window_dim, stride=1, fill_value=dtypes.NA):
        """
        Convert this rolling object to xr.Dataset,
        where the window dimension is stacked as a new dimension

        Parameters
        ----------
        window_dim: str
            New name of the window dimension.
        stride: integer, optional
            size of stride for the rolling window.
        fill_value: optional. Default dtypes.NA
            Filling value to match the dimension size.

        Returns
        -------
        Dataset with variables converted from rolling object.
        """

        from .dataset import Dataset

        dataset = OrderedDict()
        for key, da in self.obj.data_vars.items():
            if self.dim in da.dims:
                dataset[key] = self.rollings[key].construct(
                    window_dim, fill_value=fill_value)
            else:
                dataset[key] = da
        return Dataset(dataset, coords=self.obj.coords).isel(
            **{self.dim: slice(None, None, stride)})


inject_bottleneck_rolling_methods(DataArrayRolling)
inject_datasetrolling_methods(DatasetRolling)
