from numbagg.moving import ewm_nanmean

from .computation import apply_ufunc


def _ewm_nanmean(array, com):
    # wrapper b/c of kwarg
    # potentially this fuction should be in numbagg?
    return ewm_nanmean(array, com)


class EWM(object):
    _attributes = ['com', 'dim']

    def __init__(self, obj, spans):
        # TODO: add alternatives to com
        self.obj = obj
        dim, span = next(iter(spans.items()))
        self.dim = dim
        self.span = span


class DataArrayEWM(EWM):
    def mean(self):
        return apply_ufunc(
            _ewm_nanmean,
            self.obj,
            input_core_dims=[[self.dim]],
            output_core_dims=[[self.dim]],
            kwargs=dict(com=self.span),
        )
