
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
    _attributes = ['alpha', 'dim']

    def __init__(self, obj, windows, window_type='span'):
        self.obj = obj
        dim, window = next(iter(windows.items()))
        self.dim = dim
        self.alpha = _get_alpha(**{window_type: window})

    def mean(self):
        return self.obj.reduce(
            rolling_exp_nanmean, dim=self.dim, window=self.alpha)
