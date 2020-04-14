import numpy as np

from .pdcompat import count_not_none
from .pycompat import dask_array_type


def _get_alpha(com=None, span=None, halflife=None, alpha=None):
    # pandas defines in terms of com (converting to alpha in the algo)
    # so use its function to get a com and then convert to alpha

    com = _get_center_of_mass(com, span, halflife, alpha)
    return 1 / (1 + com)


def move_exp_nanmean(array, *, axis, alpha):
    if isinstance(array, dask_array_type):
        raise TypeError("rolling_exp is not currently support for dask arrays")
    import numbagg

    if axis == ():
        return array.astype(np.float64)
    else:
        return numbagg.move_exp_nanmean(array, axis=axis, alpha=alpha)


def _get_center_of_mass(comass, span, halflife, alpha):
    """
    Vendored from pandas.core.window.common._get_center_of_mass

    See licenses/PANDAS_LICENSE for the function's license
    """
    valid_count = count_not_none(comass, span, halflife, alpha)
    if valid_count > 1:
        raise ValueError("comass, span, halflife, and alpha " "are mutually exclusive")

    # Convert to center of mass; domain checks ensure 0 < alpha <= 1
    if comass is not None:
        if comass < 0:
            raise ValueError("comass must satisfy: comass >= 0")
    elif span is not None:
        if span < 1:
            raise ValueError("span must satisfy: span >= 1")
        comass = (span - 1) / 2.0
    elif halflife is not None:
        if halflife <= 0:
            raise ValueError("halflife must satisfy: halflife > 0")
        decay = 1 - np.exp(np.log(0.5) / halflife)
        comass = 1 / decay - 1
    elif alpha is not None:
        if alpha <= 0 or alpha > 1:
            raise ValueError("alpha must satisfy: 0 < alpha <= 1")
        comass = (1.0 - alpha) / alpha
    else:
        raise ValueError("Must pass one of comass, span, halflife, or alpha")

    return float(comass)


class RollingExp:
    """
    Exponentially-weighted moving window object.
    Similar to EWM in pandas

    Parameters
    ----------
    obj : Dataset or DataArray
        Object to window.
    windows : A single mapping from a single dimension name to window value
        dim : str
            Name of the dimension to create the rolling exponential window
            along (e.g., `time`).
        window : int
            Size of the moving window. The type of this is specified in
            `window_type`
    window_type : str, one of ['span', 'com', 'halflife', 'alpha'], default 'span'
        The format of the previously supplied window. Each is a simple
        numerical transformation of the others. Described in detail:
        https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.ewm.html

    Returns
    -------
    RollingExp : type of input argument
    """

    def __init__(self, obj, windows, window_type="span"):
        self.obj = obj
        dim, window = next(iter(windows.items()))
        self.dim = dim
        self.alpha = _get_alpha(**{window_type: window})

    def mean(self):
        """
        Exponentially weighted moving average

        Examples
        --------
        >>> da = xr.DataArray([1, 1, 2, 2, 2], dims="x")
        >>> da.rolling_exp(x=2, window_type="span").mean()
        <xarray.DataArray (x: 5)>
        array([1.      , 1.      , 1.692308, 1.9     , 1.966942])
        Dimensions without coordinates: x
        """

        return self.obj.reduce(move_exp_nanmean, dim=self.dim, alpha=self.alpha)
