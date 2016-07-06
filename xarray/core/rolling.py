import numpy as np

from .pycompat import OrderedDict, zip
from .common import ImplementsRollingArrayReduce, _full_like
from .combine import concat
from .ops import inject_bottleneck_rolling_methods


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

    def __init__(self, obj, min_periods=None, center=False, **windows):
        """
        Moving window object.

        Parameters
        ----------
        obj : Dataset or DataArray
            Object to window.
        min_periods : int, default None
            Minimum number of observations in window required to have a value
            (otherwise result is NA). The default, None, is equivalent to
            setting min_periods equal to the size of the window.
        center : boolean, default False
            Set the labels at the center of the window.
        **windows : dim=window
            dim : str
                Name of the dimension to create the rolling iterator
                along (e.g., `time`).
            window : int
                Size of the moving window.

        Returns
        -------
        rolling : type of input argument
        """

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
                raise ValueError('min_periods must be greater than zero or None')

            self._min_periods = min_periods
        self.center = center
        self.dim = dim

        self._axis_num = self.obj.get_axis_num(self.dim)

        self._windows = None
        self._valid_windows = None
        self.window_indices = None
        self.window_labels = None

        self._setup_windows()

    def __repr__(self):
        """provide a nice str repr of our rolling object"""

        attrs = ["{k}->{v}".format(k=k, v=getattr(self, k))
                 for k in self._attributes if getattr(self, k, None) is not None]
        return "{klass} [{attrs}]".format(klass=self.__class__.__name__,
                                          attrs=','.join(attrs))

    @property
    def windows(self):
        if self._windows is None:
            self._windows = OrderedDict(zip(self.window_labels,
                                            self.window_indices))
        return self._windows

    def __len__(self):
        return len(self.obj[self.dim])

    def __iter__(self):
        for (label, indices, valid) in zip(self.window_labels,
                                           self.window_indices,
                                           self._valid_windows):

            window = self.obj.isel(**{self.dim: indices})

            if not valid:
                window = _full_like(window, fill_value=True)

            yield (label, window)

    def _setup_windows(self):
        """
        Find the indices and labels for each window
        """
        from .dataarray import DataArray

        self.window_labels = self.obj[self.dim]

        window = int(self.window)

        dim_size = self.obj[self.dim].size

        stops = np.arange(dim_size) + 1
        starts = np.maximum(stops - window, 0)

        if self._min_periods > 1:
            valid_windows = (stops - starts) >= self._min_periods
        else:
            # No invalid windows
            valid_windows = np.ones(dim_size, dtype=bool)
        self._valid_windows = DataArray(valid_windows, dims=(self.dim, ),
                                        coords=self.obj[self.dim].coords)

        self.window_indices = [slice(start, stop)
                               for start, stop in zip(starts, stops)]

    def _center_result(self, result):
        """center result"""
        shift = (-self.window // 2) + 1
        return result.shift(**{self.dim: shift})

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

        windows = [window.reduce(func, dim=self.dim, **kwargs)
                   for _, window in self]

        # Find valid windows based on count
        counts = concat([window.count(dim=self.dim) for _, window in self],
                        dim=self.obj[self.dim])

        result = concat(windows, dim=self.window_labels)

        result = result.where(counts >= self._min_periods)

        if self.center:
            result = self._center_result(result)

        return result


class DataArrayRolling(Rolling, ImplementsRollingArrayReduce):
    """Rolling object for DataArrays"""


inject_bottleneck_rolling_methods(DataArrayRolling)
