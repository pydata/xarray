
import numpy as np
from pandas.core.window import _get_center_of_mass


def _get_alpha(com=None, span=None, halflife=None, alpha=None):
    # pandas defines in terms of com (converting to alpha in the algo)
    # so use its function to get a com and then convert to alpha

    com = _get_center_of_mass(com, span, halflife, alpha)
    return 1 / (1 + com)


def rolling_exp_nanmean(array, *, axis, window):
    import numbagg
    if axis == ():
        return array.astype(np.float)
    else:
        return numbagg.moving.rolling_exp_nanmean(
            array, axis=axis, window=window)


class RollingExp(object):
    """
    Exponentially-weighted moving window object.

    Parameters
    ----------
    obj : Dataset or DataArray
        Object to window.
    windows : A mapping from a dimension name to window value
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
    """  # noqa
    _attributes = ['alpha', 'dim']

    def __init__(self, obj, windows, window_type='span'):
        self.obj = obj
        dim, window = next(iter(windows.items()))
        self.dim = dim
        self.alpha = _get_alpha(**{window_type: window})

    def mean(self):
        """Exponentially weighted average"""
        return self.obj.reduce(
            rolling_exp_nanmean, dim=self.dim, window=self.alpha)
