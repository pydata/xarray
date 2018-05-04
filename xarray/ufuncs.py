"""xarray specific universal functions

Handles unary and binary operations for the following types, in ascending
priority order:
- scalars
- numpy.ndarray
- dask.array.Array
- xarray.Variable
- xarray.DataArray
- xarray.Dataset
- xarray.core.groupby.GroupBy

Once NumPy 1.10 comes out with support for overriding ufuncs, this module will
hopefully no longer be necessary.
"""
from __future__ import absolute_import, division, print_function

import warnings as _warnings

import numpy as _np

from .core.dataarray import DataArray as _DataArray
from .core.dataset import Dataset as _Dataset
from .core.duck_array_ops import _dask_or_eager_func
from .core.groupby import GroupBy as _GroupBy
from .core.pycompat import dask_array_type as _dask_array_type
from .core.variable import Variable as _Variable

_xarray_types = (_Variable, _DataArray, _Dataset, _GroupBy)
_dispatch_order = (_np.ndarray, _dask_array_type) + _xarray_types


def _dispatch_priority(obj):
    for priority, cls in enumerate(_dispatch_order):
        if isinstance(obj, cls):
            return priority
    return -1


class _UFuncDispatcher(object):
    """Wrapper for dispatching ufuncs."""

    def __init__(self, name):
        self._name = name

    def __call__(self, *args, **kwargs):
        _warnings.warn(
            'xarray.ufuncs will be deprecated when xarray no longer supports '
            'versions of numpy older than v1.13. Instead, use numpy ufuncs '
            'directly.', PendingDeprecationWarning, stacklevel=2)

        new_args = args
        f = _dask_or_eager_func(self._name, array_args=slice(len(args)))
        if len(args) > 2 or len(args) == 0:
            raise TypeError('cannot handle %s arguments for %r' %
                            (len(args), self._name))
        elif len(args) == 1:
            if isinstance(args[0], _xarray_types):
                f = args[0]._unary_op(self)
        else:  # len(args) = 2
            p1, p2 = map(_dispatch_priority, args)
            if p1 >= p2:
                if isinstance(args[0], _xarray_types):
                    f = args[0]._binary_op(self)
            else:
                if isinstance(args[1], _xarray_types):
                    f = args[1]._binary_op(self, reflexive=True)
                    new_args = tuple(reversed(args))
        res = f(*new_args, **kwargs)
        if res is NotImplemented:
            raise TypeError('%r not implemented for types (%r, %r)'
                            % (self._name, type(args[0]), type(args[1])))
        return res


def _create_op(name):
    func = _UFuncDispatcher(name)
    func.__name__ = name
    doc = getattr(_np, name).__doc__
    func.__doc__ = ('xarray specific variant of numpy.%s. Handles '
                    'xarray.Dataset, xarray.DataArray, xarray.Variable, '
                    'numpy.ndarray and dask.array.Array objects with '
                    'automatic dispatching.\n\n'
                    'Documentation from numpy:\n\n%s' % (name, doc))
    return func


__all__ = """logaddexp logaddexp2 conj exp log log2 log10 log1p expm1 sqrt
             square sin cos tan arcsin arccos arctan arctan2 hypot sinh cosh
             tanh arcsinh arccosh arctanh deg2rad rad2deg logical_and
             logical_or logical_xor logical_not maximum minimum fmax fmin
             isreal iscomplex isfinite isinf isnan signbit copysign nextafter
             ldexp fmod floor ceil trunc degrees radians rint fix angle real
             imag fabs sign frexp fmod
             """.split()

for name in __all__:
    globals()[name] = _create_op(name)
