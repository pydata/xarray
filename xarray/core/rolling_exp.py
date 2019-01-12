from numbagg.moving import ewm_nanmean

from .computation import apply_ufunc


def _rolling_exp_nanmean(array, com):
    # wrapper b/c of kwarg
    # potentially this fuction should be in numbagg?
    return ewm_nanmean(array, com)


class RollingExp(object):
    _attributes = ['com', 'dim']

    def __init__(self, obj, spans):
        # TODO: add alternatives to com
        self.obj = obj
        dim, span = next(iter(spans.items()))
        self.dim = dim
        self.span = span


class DataArrayRollingExp(RollingExp):
    def mean(self):
        da = apply_ufunc(
            _rolling_exp_nanmean,
            self.obj,
            input_core_dims=[[self.dim]],
            output_core_dims=[[self.dim]],
            kwargs=dict(com=self.span),
        )
        return da.transpose(*self.obj.dims)
