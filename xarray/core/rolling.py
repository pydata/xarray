import functools

import numpy as np

from . import dtypes, duck_array_ops, utils
from .dask_array_ops import dask_rolling_wrapper
from .ops import inject_coarsen_methods
from .pycompat import dask_array_type

try:
    import bottleneck
except ImportError:
    # use numpy methods instead
    bottleneck = None


_ROLLING_REDUCE_DOCSTRING_TEMPLATE = """\
Reduce this object's data windows by applying `{name}` along its dimension.

Parameters
----------
**kwargs : dict
    Additional keyword arguments passed on to `{name}`.

Returns
-------
reduced : same type as caller
    New object with `{name}` applied along its rolling dimnension.
"""


class Rolling:
    """A object that implements the moving window pattern.

    See Also
    --------
    Dataset.groupby
    DataArray.groupby
    Dataset.rolling
    DataArray.rolling
    """

    __slots__ = ("obj", "window", "min_periods", "center", "dim")
    _attributes = ("window", "min_periods", "center", "dim")

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
        if len(windows) != 1:
            raise ValueError("exactly one dim/window should be provided")

        dim, window = next(iter(windows.items()))

        if window <= 0:
            raise ValueError("window must be > 0")

        self.obj = obj

        # attributes
        self.window = window
        if min_periods is not None and min_periods <= 0:
            raise ValueError("min_periods must be greater than zero or None")
        self.min_periods = min_periods

        self.center = center
        self.dim = dim

    @property
    def _min_periods(self):
        return self.min_periods if self.min_periods is not None else self.window

    def __repr__(self):
        """provide a nice str repr of our rolling object"""

        attrs = [
            "{k}->{v}".format(k=k, v=getattr(self, k))
            for k in self._attributes
            if getattr(self, k, None) is not None
        ]
        return "{klass} [{attrs}]".format(
            klass=self.__class__.__name__, attrs=",".join(attrs)
        )

    def __len__(self):
        return self.obj.sizes[self.dim]

    def _reduce_method(name):
        array_agg_func = getattr(duck_array_ops, name)
        bottleneck_move_func = getattr(bottleneck, "move_" + name, None)

        def method(self, **kwargs):
            return self._numpy_or_bottleneck_reduce(
                array_agg_func, bottleneck_move_func, **kwargs
            )

        method.__name__ = name
        method.__doc__ = _ROLLING_REDUCE_DOCSTRING_TEMPLATE.format(name=name)
        return method

    argmax = _reduce_method("argmax")
    argmin = _reduce_method("argmin")
    max = _reduce_method("max")
    min = _reduce_method("min")
    mean = _reduce_method("mean")
    prod = _reduce_method("prod")
    sum = _reduce_method("sum")
    std = _reduce_method("std")
    var = _reduce_method("var")
    median = _reduce_method("median")

    def count(self):
        rolling_count = self._counts()
        enough_periods = rolling_count >= self._min_periods
        return rolling_count.where(enough_periods)

    count.__doc__ = _ROLLING_REDUCE_DOCSTRING_TEMPLATE.format(name="count")


class DataArrayRolling(Rolling):
    __slots__ = ("window_labels",)

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
        super().__init__(obj, windows, min_periods=min_periods, center=center)

        self.window_labels = self.obj[self.dim]

    def __iter__(self):
        stops = np.arange(1, len(self.window_labels) + 1)
        starts = stops - int(self.window)
        starts[: int(self.window)] = 0
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
        >>> rolling = da.rolling(b=3)
        >>> rolling.construct('window_dim')
        <xarray.DataArray (a: 2, b: 4, window_dim: 3)>
        array([[[np.nan, np.nan, 0], [np.nan, 0, 1], [0, 1, 2], [1, 2, 3]],
               [[np.nan, np.nan, 4], [np.nan, 4, 5], [4, 5, 6], [5, 6, 7]]])
        Dimensions without coordinates: a, b, window_dim
        >>>
        >>> rolling = da.rolling(b=3, center=True)
        >>> rolling.construct('window_dim')
        <xarray.DataArray (a: 2, b: 4, window_dim: 3)>
        array([[[np.nan, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, np.nan]],
               [[np.nan, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, np.nan]]])
        Dimensions without coordinates: a, b, window_dim
        """

        from .dataarray import DataArray

        window = self.obj.variable.rolling_window(
            self.dim, self.window, window_dim, self.center, fill_value=fill_value
        )
        result = DataArray(
            window, dims=self.obj.dims + (window_dim,), coords=self.obj.coords
        )
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

        Examples
        --------
        >>> da = DataArray(np.arange(8).reshape(2, 4), dims=('a', 'b'))
        >>>
        >>> rolling = da.rolling(b=3)
        >>> rolling.construct('window_dim')
        <xarray.DataArray (a: 2, b: 4, window_dim: 3)>
        array([[[np.nan, np.nan, 0], [np.nan, 0, 1], [0, 1, 2], [1, 2, 3]],
               [[np.nan, np.nan, 4], [np.nan, 4, 5], [4, 5, 6], [5, 6, 7]]])
        Dimensions without coordinates: a, b, window_dim
        >>>
        >>> rolling.reduce(np.sum)
        <xarray.DataArray (a: 2, b: 4)>
        array([[nan, nan,  3.,  6.],
               [nan, nan, 15., 18.]])
        Dimensions without coordinates: a, b
        >>>
        >>> rolling = da.rolling(b=3, min_periods=1)
        >>> rolling.reduce(np.nansum)
        <xarray.DataArray (a: 2, b: 4)>
        array([[ 0.,  1.,  3.,  6.],
               [ 4.,  9., 15., 18.]])
        """
        rolling_dim = utils.get_temp_dimname(self.obj.dims, "_rolling_dim")
        windows = self.construct(rolling_dim)
        result = windows.reduce(func, dim=rolling_dim, **kwargs)

        # Find valid windows based on count.
        counts = self._counts()
        return result.where(counts >= self._min_periods)

    def _counts(self):
        """ Number of non-nan entries in each rolling window. """

        rolling_dim = utils.get_temp_dimname(self.obj.dims, "_rolling_dim")
        # We use False as the fill_value instead of np.nan, since boolean
        # array is faster to be reduced than object array.
        # The use of skipna==False is also faster since it does not need to
        # copy the strided array.
        counts = (
            self.obj.notnull()
            .rolling(center=self.center, **{self.dim: self.window})
            .construct(rolling_dim, fill_value=False)
            .sum(dim=rolling_dim, skipna=False)
        )
        return counts

    def _bottleneck_reduce(self, func, **kwargs):
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
            if isinstance(padded.data, dask_array_type):
                # Workaround to make the padded chunk size is larger than
                # self.window-1
                shift = -(self.window + 1) // 2
                offset = (self.window - 1) // 2
                valid = (slice(None),) * axis + (
                    slice(offset, offset + self.obj.shape[axis]),
                )
            else:
                shift = (-self.window // 2) + 1
                valid = (slice(None),) * axis + (slice(-shift, None),)
            padded = padded.pad_with_fill_value({self.dim: (0, -shift)})

        if isinstance(padded.data, dask_array_type):
            raise AssertionError("should not be reachable")
            values = dask_rolling_wrapper(
                func, padded.data, window=self.window, min_count=min_count, axis=axis
            )
        else:
            values = func(
                padded.data, window=self.window, min_count=min_count, axis=axis
            )

        if self.center:
            values = values[valid]
        result = DataArray(values, self.obj.coords)

        return result

    def _numpy_or_bottleneck_reduce(
        self, array_agg_func, bottleneck_move_func, **kwargs
    ):
        if bottleneck_move_func is not None and not isinstance(
            self.obj.data, dask_array_type
        ):
            # TODO: renable bottleneck with dask after the issues
            # underlying https://github.com/pydata/xarray/issues/2940 are
            # fixed.
            return self._bottleneck_reduce(bottleneck_move_func, **kwargs)
        else:
            return self.reduce(array_agg_func, **kwargs)


class DatasetRolling(Rolling):
    __slots__ = ("rollings",)

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
        super().__init__(obj, windows, min_periods, center)
        if self.dim not in self.obj.dims:
            raise KeyError(self.dim)
        # Keep each Rolling object as a dictionary
        self.rollings = {}
        for key, da in self.obj.data_vars.items():
            # keeps rollings only for the dataset depending on slf.dim
            if self.dim in da.dims:
                self.rollings[key] = DataArrayRolling(da, windows, min_periods, center)

    def _dataset_implementation(self, func, **kwargs):
        from .dataset import Dataset

        reduced = {}
        for key, da in self.obj.data_vars.items():
            if self.dim in da.dims:
                reduced[key] = func(self.rollings[key], **kwargs)
            else:
                reduced[key] = self.obj[key]
        return Dataset(reduced, coords=self.obj.coords)

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
        return self._dataset_implementation(
            functools.partial(DataArrayRolling.reduce, func=func), **kwargs
        )

    def _counts(self):
        return self._dataset_implementation(DataArrayRolling._counts)

    def _numpy_or_bottleneck_reduce(
        self, array_agg_func, bottleneck_move_func, **kwargs
    ):
        return self._dataset_implementation(
            functools.partial(
                DataArrayRolling._numpy_or_bottleneck_reduce,
                array_agg_func=array_agg_func,
                bottleneck_move_func=bottleneck_move_func,
            ),
            **kwargs
        )

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

        dataset = {}
        for key, da in self.obj.data_vars.items():
            if self.dim in da.dims:
                dataset[key] = self.rollings[key].construct(
                    window_dim, fill_value=fill_value
                )
            else:
                dataset[key] = da
        return Dataset(dataset, coords=self.obj.coords).isel(
            **{self.dim: slice(None, None, stride)}
        )


class Coarsen:
    """A object that implements the coarsen.

    See Also
    --------
    Dataset.coarsen
    DataArray.coarsen
    """

    __slots__ = ("obj", "boundary", "coord_func", "windows", "side", "trim_excess")
    _attributes = ("windows", "side", "trim_excess")

    def __init__(self, obj, windows, boundary, side, coord_func):
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
        boundary : 'exact' | 'trim' | 'pad'
            If 'exact', a ValueError will be raised if dimension size is not a
            multiple of window size. If 'trim', the excess indexes are trimed.
            If 'pad', NA will be padded.
        side : 'left' or 'right' or mapping from dimension to 'left' or 'right'
        coord_func: mapping from coordinate name to func.

        Returns
        -------
        coarsen
        """
        self.obj = obj
        self.windows = windows
        self.side = side
        self.boundary = boundary

        if not utils.is_dict_like(coord_func):
            coord_func = {d: coord_func for d in self.obj.dims}
        for c in self.obj.coords:
            if c not in coord_func:
                coord_func[c] = duck_array_ops.mean
        self.coord_func = coord_func

    def __repr__(self):
        """provide a nice str repr of our coarsen object"""

        attrs = [
            "{k}->{v}".format(k=k, v=getattr(self, k))
            for k in self._attributes
            if getattr(self, k, None) is not None
        ]
        return "{klass} [{attrs}]".format(
            klass=self.__class__.__name__, attrs=",".join(attrs)
        )


class DataArrayCoarsen(Coarsen):
    __slots__ = ()

    @classmethod
    def _reduce_method(cls, func):
        """
        Return a wrapped function for injecting numpy methods.
        see ops.inject_coarsen_methods
        """

        def wrapped_func(self, **kwargs):
            from .dataarray import DataArray

            reduced = self.obj.variable.coarsen(
                self.windows, func, self.boundary, self.side
            )
            coords = {}
            for c, v in self.obj.coords.items():
                if c == self.obj.name:
                    coords[c] = reduced
                else:
                    if any(d in self.windows for d in v.dims):
                        coords[c] = v.variable.coarsen(
                            self.windows, self.coord_func[c], self.boundary, self.side
                        )
                    else:
                        coords[c] = v
            return DataArray(reduced, dims=self.obj.dims, coords=coords)

        return wrapped_func


class DatasetCoarsen(Coarsen):
    __slots__ = ()

    @classmethod
    def _reduce_method(cls, func):
        """
        Return a wrapped function for injecting numpy methods.
        see ops.inject_coarsen_methods
        """

        def wrapped_func(self, **kwargs):
            from .dataset import Dataset

            reduced = {}
            for key, da in self.obj.data_vars.items():
                reduced[key] = da.variable.coarsen(
                    self.windows, func, self.boundary, self.side
                )

            coords = {}
            for c, v in self.obj.coords.items():
                if any(d in self.windows for d in v.dims):
                    coords[c] = v.variable.coarsen(
                        self.windows, self.coord_func[c], self.boundary, self.side
                    )
                else:
                    coords[c] = v.variable
            return Dataset(reduced, coords=coords)

        return wrapped_func


inject_coarsen_methods(DataArrayCoarsen)
inject_coarsen_methods(DatasetCoarsen)
