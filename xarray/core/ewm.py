from numbagg.moving import ewm_nanmean

from .computation import apply_ufunc


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
            ewm_nanmean,
            [self.obj],
            input_core_dims=(self.dim,),
            kwargs=dict(com=self.span),


        )
