
from pandas.core.window import _get_center_of_mass

from .computation import apply_ufunc


def _get_alpha(com=None, span=None, halflife=None, alpha=None):
    # pandas defines in terms of comass, so use its function
    # but then convert to alpha

    comass = _get_center_of_mass(com, span, halflife, alpha)
    return 1 / (1 + comass)


class RollingExp(object):
    _attributes = ['alpha', 'dim']

    def __init__(self, obj, windows, window_type='span'):
        self.obj = obj
        dim, window = next(iter(windows.items()))
        self.dim = dim
        self.alpha = _get_alpha(**{window_type: window})


class DataArrayRollingExp(RollingExp):
    def mean(self):
        from numbagg.moving import rolling_exp_nanmean

        da = apply_ufunc(
            rolling_exp_nanmean,
            self.obj,
            input_core_dims=[[self.dim]],
            output_core_dims=[[self.dim]],
            kwargs=dict(window=self.alpha),
        )
        return da.transpose(*self.obj.dims)
