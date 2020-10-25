import functools
import warnings
from typing import Any, Callable, Dict

import numpy as np

from . import dtypes, duck_array_ops, utils
from .dask_array_ops import dask_rolling_wrapper
from .ops import inject_reduce_methods
from .options import _get_keep_attrs
from .pycompat import is_duck_dask_array

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

    __slots__ = ("obj", "window", "min_periods", "center", "dim", "keep_attrs")
    _attributes = ("window", "min_periods", "center", "dim", "keep_attrs")

    def __init__(self, obj, windows, min_periods=None, center=False, keep_attrs=None):
        """
        Moving window object.

        Parameters
        ----------
        obj : Dataset or DataArray
            Object to window.
        windows : mapping of hashable to int
            A mapping from the name of the dimension to create the rolling
            exponential window along (e.g. `time`) to the size of the moving window.
        min_periods : int, default: None
            Minimum number of observations in window required to have a value
            (otherwise result is NA). The default, None, is equivalent to
            setting min_periods equal to the size of the window.
        center : bool, default: False
            Set the labels at the center of the window.
        keep_attrs : bool, optional
            If True, the object's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.

        Returns
        -------
        rolling : type of input argument
        """
        self.dim, self.window = [], []
        for d, w in windows.items():
            self.dim.append(d)
            if w <= 0:
                raise ValueError("window must be > 0")
            self.window.append(w)

        self.center = self._mapping_to_list(center, default=False)
        self.obj = obj

        # attributes
        if min_periods is not None and min_periods <= 0:
            raise ValueError("min_periods must be greater than zero or None")

        self.min_periods = np.prod(self.window) if min_periods is None else min_periods

        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=False)
        self.keep_attrs = keep_attrs

    def __repr__(self):
        """provide a nice str repr of our rolling object"""

        attrs = [
            "{k}->{v}{c}".format(k=k, v=w, c="(center)" if c else "")
            for k, w, c in zip(self.dim, self.window, self.center)
        ]
        return "{klass} [{attrs}]".format(
            klass=self.__class__.__name__, attrs=",".join(attrs)
        )

    def __len__(self):
        return self.obj.sizes[self.dim]

    def _reduce_method(name: str) -> Callable:  # type: ignore
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
        enough_periods = rolling_count >= self.min_periods
        return rolling_count.where(enough_periods)

    count.__doc__ = _ROLLING_REDUCE_DOCSTRING_TEMPLATE.format(name="count")

    def _mapping_to_list(
        self, arg, default=None, allow_default=True, allow_allsame=True
    ):
        if utils.is_dict_like(arg):
            if allow_default:
                return [arg.get(d, default) for d in self.dim]
            else:
                for d in self.dim:
                    if d not in arg:
                        raise KeyError(f"argument has no key {d}.")
                return [arg[d] for d in self.dim]
        elif allow_allsame:  # for single argument
            return [arg] * len(self.dim)
        elif len(self.dim) == 1:
            return [arg]
        else:
            raise ValueError(
                "Mapping argument is necessary for {}d-rolling.".format(len(self.dim))
            )


class DataArrayRolling(Rolling):
    __slots__ = ("window_labels",)

    def __init__(self, obj, windows, min_periods=None, center=False, keep_attrs=None):
        """
        Moving window object for DataArray.
        You should use DataArray.rolling() method to construct this object
        instead of the class constructor.

        Parameters
        ----------
        obj : DataArray
            Object to window.
        windows : mapping of hashable to int
            A mapping from the name of the dimension to create the rolling
            exponential window along (e.g. `time`) to the size of the moving window.
        min_periods : int, default: None
            Minimum number of observations in window required to have a value
            (otherwise result is NA). The default, None, is equivalent to
            setting min_periods equal to the size of the window.
        center : bool, default: False
            Set the labels at the center of the window.
        keep_attrs : bool, optional
            If True, the object's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.

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
        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=False)
        super().__init__(
            obj, windows, min_periods=min_periods, center=center, keep_attrs=keep_attrs
        )

        # TODO legacy attribute
        self.window_labels = self.obj[self.dim[0]]

    def __iter__(self):
        if len(self.dim) > 1:
            raise ValueError("__iter__ is only supported for 1d-rolling")
        stops = np.arange(1, len(self.window_labels) + 1)
        starts = stops - int(self.window[0])
        starts[: int(self.window[0])] = 0
        for (label, start, stop) in zip(self.window_labels, starts, stops):
            window = self.obj.isel(**{self.dim[0]: slice(start, stop)})

            counts = window.count(dim=self.dim[0])
            window = window.where(counts >= self.min_periods)

            yield (label, window)

    def construct(
        self, window_dim=None, stride=1, fill_value=dtypes.NA, **window_dim_kwargs
    ):
        """
        Convert this rolling object to xr.DataArray,
        where the window dimension is stacked as a new dimension

        Parameters
        ----------
        window_dim : str or mapping, optional
            A mapping from dimension name to the new window dimension names.
            Just a string can be used for 1d-rolling.
        stride : int or mapping of int, optional
            Size of stride for the rolling window.
        fill_value : default: dtypes.NA
            Filling value to match the dimension size.
        **window_dim_kwargs : {dim: new_name, ...}, optional
            The keyword arguments form of ``window_dim``.

        Returns
        -------
        DataArray that is a view of the original array. The returned array is
        not writeable.

        Examples
        --------
        >>> da = xr.DataArray(np.arange(8).reshape(2, 4), dims=("a", "b"))

        >>> rolling = da.rolling(b=3)
        >>> rolling.construct("window_dim")
        <xarray.DataArray (a: 2, b: 4, window_dim: 3)>
        array([[[nan, nan,  0.],
                [nan,  0.,  1.],
                [ 0.,  1.,  2.],
                [ 1.,  2.,  3.]],
        <BLANKLINE>
               [[nan, nan,  4.],
                [nan,  4.,  5.],
                [ 4.,  5.,  6.],
                [ 5.,  6.,  7.]]])
        Dimensions without coordinates: a, b, window_dim

        >>> rolling = da.rolling(b=3, center=True)
        >>> rolling.construct("window_dim")
        <xarray.DataArray (a: 2, b: 4, window_dim: 3)>
        array([[[nan,  0.,  1.],
                [ 0.,  1.,  2.],
                [ 1.,  2.,  3.],
                [ 2.,  3., nan]],
        <BLANKLINE>
               [[nan,  4.,  5.],
                [ 4.,  5.,  6.],
                [ 5.,  6.,  7.],
                [ 6.,  7., nan]]])
        Dimensions without coordinates: a, b, window_dim

        """

        from .dataarray import DataArray

        if window_dim is None:
            if len(window_dim_kwargs) == 0:
                raise ValueError(
                    "Either window_dim or window_dim_kwargs need to be specified."
                )
            window_dim = {d: window_dim_kwargs[d] for d in self.dim}

        window_dim = self._mapping_to_list(
            window_dim, allow_default=False, allow_allsame=False
        )
        stride = self._mapping_to_list(stride, default=1)

        window = self.obj.variable.rolling_window(
            self.dim, self.window, window_dim, self.center, fill_value=fill_value
        )
        result = DataArray(
            window, dims=self.obj.dims + tuple(window_dim), coords=self.obj.coords
        )
        return result.isel(
            **{d: slice(None, None, s) for d, s in zip(self.dim, stride)}
        )

    def reduce(self, func, **kwargs):
        """Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : callable
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
        >>> da = xr.DataArray(np.arange(8).reshape(2, 4), dims=("a", "b"))
        >>> rolling = da.rolling(b=3)
        >>> rolling.construct("window_dim")
        <xarray.DataArray (a: 2, b: 4, window_dim: 3)>
        array([[[nan, nan,  0.],
                [nan,  0.,  1.],
                [ 0.,  1.,  2.],
                [ 1.,  2.,  3.]],
        <BLANKLINE>
               [[nan, nan,  4.],
                [nan,  4.,  5.],
                [ 4.,  5.,  6.],
                [ 5.,  6.,  7.]]])
        Dimensions without coordinates: a, b, window_dim

        >>> rolling.reduce(np.sum)
        <xarray.DataArray (a: 2, b: 4)>
        array([[nan, nan,  3.,  6.],
               [nan, nan, 15., 18.]])
        Dimensions without coordinates: a, b

        >>> rolling = da.rolling(b=3, min_periods=1)
        >>> rolling.reduce(np.nansum)
        <xarray.DataArray (a: 2, b: 4)>
        array([[ 0.,  1.,  3.,  6.],
               [ 4.,  9., 15., 18.]])
        Dimensions without coordinates: a, b
        """
        rolling_dim = {
            d: utils.get_temp_dimname(self.obj.dims, f"_rolling_dim_{d}")
            for d in self.dim
        }
        windows = self.construct(rolling_dim)
        result = windows.reduce(func, dim=list(rolling_dim.values()), **kwargs)

        # Find valid windows based on count.
        counts = self._counts()
        return result.where(counts >= self.min_periods)

    def _counts(self):
        """ Number of non-nan entries in each rolling window. """

        rolling_dim = {
            d: utils.get_temp_dimname(self.obj.dims, f"_rolling_dim_{d}")
            for d in self.dim
        }
        # We use False as the fill_value instead of np.nan, since boolean
        # array is faster to be reduced than object array.
        # The use of skipna==False is also faster since it does not need to
        # copy the strided array.
        counts = (
            self.obj.notnull()
            .rolling(
                center={d: self.center[i] for i, d in enumerate(self.dim)},
                **{d: w for d, w in zip(self.dim, self.window)},
            )
            .construct(rolling_dim, fill_value=False)
            .sum(dim=list(rolling_dim.values()), skipna=False)
        )
        return counts

    def _bottleneck_reduce(self, func, **kwargs):
        from .dataarray import DataArray

        # bottleneck doesn't allow min_count to be 0, although it should
        # work the same as if min_count = 1
        # Note bottleneck only works with 1d-rolling.
        if self.min_periods is not None and self.min_periods == 0:
            min_count = 1
        else:
            min_count = self.min_periods

        axis = self.obj.get_axis_num(self.dim[0])

        padded = self.obj.variable
        if self.center[0]:
            if is_duck_dask_array(padded.data):
                # Workaround to make the padded chunk size is larger than
                # self.window-1
                shift = -(self.window[0] + 1) // 2
                offset = (self.window[0] - 1) // 2
                valid = (slice(None),) * axis + (
                    slice(offset, offset + self.obj.shape[axis]),
                )
            else:
                shift = (-self.window[0] // 2) + 1
                valid = (slice(None),) * axis + (slice(-shift, None),)
            padded = padded.pad({self.dim[0]: (0, -shift)}, mode="constant")

        if is_duck_dask_array(padded.data):
            raise AssertionError("should not be reachable")
            values = dask_rolling_wrapper(
                func, padded.data, window=self.window[0], min_count=min_count, axis=axis
            )
        else:
            values = func(
                padded.data, window=self.window[0], min_count=min_count, axis=axis
            )

        if self.center[0]:
            values = values[valid]
        result = DataArray(values, self.obj.coords)

        return result

    def _numpy_or_bottleneck_reduce(
        self, array_agg_func, bottleneck_move_func, **kwargs
    ):
        if "dim" in kwargs:
            warnings.warn(
                f"Reductions will be applied along the rolling dimension '{self.dim}'. Passing the 'dim' kwarg to reduction operations has no effect and will raise an error in xarray 0.16.0.",
                DeprecationWarning,
                stacklevel=3,
            )
            del kwargs["dim"]

        if (
            bottleneck_move_func is not None
            and not is_duck_dask_array(self.obj.data)
            and len(self.dim) == 1
        ):
            # TODO: renable bottleneck with dask after the issues
            # underlying https://github.com/pydata/xarray/issues/2940 are
            # fixed.
            return self._bottleneck_reduce(bottleneck_move_func, **kwargs)
        else:
            return self.reduce(array_agg_func, **kwargs)


class DatasetRolling(Rolling):
    __slots__ = ("rollings",)

    def __init__(self, obj, windows, min_periods=None, center=False, keep_attrs=None):
        """
        Moving window object for Dataset.
        You should use Dataset.rolling() method to construct this object
        instead of the class constructor.

        Parameters
        ----------
        obj : Dataset
            Object to window.
        windows : mapping of hashable to int
            A mapping from the name of the dimension to create the rolling
            exponential window along (e.g. `time`) to the size of the moving window.
        min_periods : int, default: None
            Minimum number of observations in window required to have a value
            (otherwise result is NA). The default, None, is equivalent to
            setting min_periods equal to the size of the window.
        center : bool or mapping of hashable to bool, default: False
            Set the labels at the center of the window.
        keep_attrs : bool, optional
            If True, the object's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.

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
        super().__init__(obj, windows, min_periods, center, keep_attrs)
        if any(d not in self.obj.dims for d in self.dim):
            raise KeyError(self.dim)
        # Keep each Rolling object as a dictionary
        self.rollings = {}
        for key, da in self.obj.data_vars.items():
            # keeps rollings only for the dataset depending on slf.dim
            dims, center = [], {}
            for i, d in enumerate(self.dim):
                if d in da.dims:
                    dims.append(d)
                    center[d] = self.center[i]

            if len(dims) > 0:
                w = {d: windows[d] for d in dims}
                self.rollings[key] = DataArrayRolling(
                    da, w, min_periods, center, keep_attrs
                )

    def _dataset_implementation(self, func, **kwargs):
        from .dataset import Dataset

        reduced = {}
        for key, da in self.obj.data_vars.items():
            if any(d in da.dims for d in self.dim):
                reduced[key] = func(self.rollings[key], **kwargs)
            else:
                reduced[key] = self.obj[key]
        attrs = self.obj.attrs if self.keep_attrs else {}
        return Dataset(reduced, coords=self.obj.coords, attrs=attrs)

    def reduce(self, func, **kwargs):
        """Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : callable
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
            **kwargs,
        )

    def construct(
        self,
        window_dim=None,
        stride=1,
        fill_value=dtypes.NA,
        keep_attrs=None,
        **window_dim_kwargs,
    ):
        """
        Convert this rolling object to xr.Dataset,
        where the window dimension is stacked as a new dimension

        Parameters
        ----------
        window_dim : str or mapping, optional
            A mapping from dimension name to the new window dimension names.
            Just a string can be used for 1d-rolling.
        stride : int, optional
            size of stride for the rolling window.
        fill_value : Any, default: dtypes.NA
            Filling value to match the dimension size.
        **window_dim_kwargs : {dim: new_name, ...}, optional
            The keyword arguments form of ``window_dim``.

        Returns
        -------
        Dataset with variables converted from rolling object.
        """

        from .dataset import Dataset

        if window_dim is None:
            if len(window_dim_kwargs) == 0:
                raise ValueError(
                    "Either window_dim or window_dim_kwargs need to be specified."
                )
            window_dim = {d: window_dim_kwargs[d] for d in self.dim}

        window_dim = self._mapping_to_list(
            window_dim, allow_default=False, allow_allsame=False
        )
        stride = self._mapping_to_list(stride, default=1)

        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=True)

        dataset = {}
        for key, da in self.obj.data_vars.items():
            # keeps rollings only for the dataset depending on slf.dim
            dims = [d for d in self.dim if d in da.dims]
            if len(dims) > 0:
                wi = {d: window_dim[i] for i, d in enumerate(self.dim) if d in da.dims}
                st = {d: stride[i] for i, d in enumerate(self.dim) if d in da.dims}
                dataset[key] = self.rollings[key].construct(
                    window_dim=wi, fill_value=fill_value, stride=st
                )
            else:
                dataset[key] = da
        return Dataset(dataset, coords=self.obj.coords).isel(
            **{d: slice(None, None, s) for d, s in zip(self.dim, stride)}
        )


class Coarsen:
    """A object that implements the coarsen.

    See Also
    --------
    Dataset.coarsen
    DataArray.coarsen
    """

    __slots__ = (
        "obj",
        "boundary",
        "coord_func",
        "windows",
        "side",
        "trim_excess",
        "keep_attrs",
    )
    _attributes = ("windows", "side", "trim_excess")

    def __init__(self, obj, windows, boundary, side, coord_func, keep_attrs):
        """
        Moving window object.

        Parameters
        ----------
        obj : Dataset or DataArray
            Object to window.
        windows : mapping of hashable to int
            A mapping from the name of the dimension to create the rolling
            exponential window along (e.g. `time`) to the size of the moving window.
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
        self.keep_attrs = keep_attrs

        absent_dims = [dim for dim in windows.keys() if dim not in self.obj.dims]
        if absent_dims:
            raise ValueError(
                f"Dimensions {absent_dims!r} not found in {self.obj.__class__.__name__}."
            )
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

    _reduce_extra_args_docstring = """"""

    @classmethod
    def _reduce_method(cls, func: Callable, include_skipna: bool, numeric_only: bool):
        """
        Return a wrapped function for injecting reduction methods.
        see ops.inject_reduce_methods
        """
        kwargs: Dict[str, Any] = {}
        if include_skipna:
            kwargs["skipna"] = None

        def wrapped_func(self, **kwargs):
            from .dataarray import DataArray

            reduced = self.obj.variable.coarsen(
                self.windows, func, self.boundary, self.side, self.keep_attrs, **kwargs
            )
            coords = {}
            for c, v in self.obj.coords.items():
                if c == self.obj.name:
                    coords[c] = reduced
                else:
                    if any(d in self.windows for d in v.dims):
                        coords[c] = v.variable.coarsen(
                            self.windows,
                            self.coord_func[c],
                            self.boundary,
                            self.side,
                            self.keep_attrs,
                            **kwargs,
                        )
                    else:
                        coords[c] = v
            return DataArray(reduced, dims=self.obj.dims, coords=coords)

        return wrapped_func


class DatasetCoarsen(Coarsen):
    __slots__ = ()

    _reduce_extra_args_docstring = """"""

    @classmethod
    def _reduce_method(cls, func: Callable, include_skipna: bool, numeric_only: bool):
        """
        Return a wrapped function for injecting reduction methods.
        see ops.inject_reduce_methods
        """
        kwargs: Dict[str, Any] = {}
        if include_skipna:
            kwargs["skipna"] = None

        def wrapped_func(self, **kwargs):
            from .dataset import Dataset

            if self.keep_attrs:
                attrs = self.obj.attrs
            else:
                attrs = {}

            reduced = {}
            for key, da in self.obj.data_vars.items():
                reduced[key] = da.variable.coarsen(
                    self.windows, func, self.boundary, self.side, **kwargs
                )

            coords = {}
            for c, v in self.obj.coords.items():
                if any(d in self.windows for d in v.dims):
                    coords[c] = v.variable.coarsen(
                        self.windows,
                        self.coord_func[c],
                        self.boundary,
                        self.side,
                        **kwargs,
                    )
                else:
                    coords[c] = v.variable
            return Dataset(reduced, coords=coords, attrs=attrs)

        return wrapped_func


inject_reduce_methods(DataArrayCoarsen)
inject_reduce_methods(DatasetCoarsen)
