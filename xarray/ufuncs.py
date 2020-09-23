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
import textwrap
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


class _UFuncDispatcher:
    """Wrapper for dispatching ufuncs."""

    def __init__(self, name):
        self._name = name

    def __call__(self, *args, **kwargs):
        if self._name not in ["angle", "iscomplex"]:
            _warnings.warn(
                "xarray.ufuncs will be deprecated when xarray no longer "
                "supports versions of numpy older than v1.17. Instead, use "
                "numpy ufuncs directly.",
                PendingDeprecationWarning,
                stacklevel=2,
            )

        new_args = args
        f = _dask_or_eager_func(self._name, array_args=slice(len(args)))
        if len(args) > 2 or len(args) == 0:
            raise TypeError(
                "cannot handle {} arguments for {!r}".format(len(args), self._name)
            )
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
            raise TypeError(
                "%r not implemented for types (%r, %r)"
                % (self._name, type(args[0]), type(args[1]))
            )
        return res


def _skip_signature(doc, name):
    if not isinstance(doc, str):
        return doc

    if doc.startswith(name):
        signature_end = doc.find("\n\n")
        doc = doc[signature_end + 2 :]

    return doc


def _remove_unused_reference_labels(doc):
    if not isinstance(doc, str):
        return doc

    max_references = 5
    for num in range(max_references):
        label = f".. [{num}]"
        reference = f"[{num}]_"
        index = f"{num}.    "

        if label not in doc or reference in doc:
            continue

        doc = doc.replace(label, index)

    return doc


def _dedent(doc):
    if not isinstance(doc, str):
        return doc

    return textwrap.dedent(doc)


def _create_op(name):
    func = _UFuncDispatcher(name)
    func.__name__ = name
    doc = getattr(_np, name).__doc__

    doc = _remove_unused_reference_labels(_skip_signature(_dedent(doc), name))

    func.__doc__ = (
        "xarray specific variant of numpy.%s. Handles "
        "xarray.Dataset, xarray.DataArray, xarray.Variable, "
        "numpy.ndarray and dask.array.Array objects with "
        "automatic dispatching.\n\n"
        "Documentation from numpy:\n\n%s" % (name, doc)
    )
    return func


__all__ = (  # noqa: F822
    "angle",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctan2",
    "arctanh",
    "ceil",
    "conj",
    "copysign",
    "cos",
    "cosh",
    "deg2rad",
    "degrees",
    "exp",
    "expm1",
    "fabs",
    "fix",
    "floor",
    "fmax",
    "fmin",
    "fmod",
    "fmod",
    "frexp",
    "hypot",
    "imag",
    "iscomplex",
    "isfinite",
    "isinf",
    "isnan",
    "isreal",
    "ldexp",
    "log",
    "log10",
    "log1p",
    "log2",
    "logaddexp",
    "logaddexp2",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "maximum",
    "minimum",
    "nextafter",
    "rad2deg",
    "radians",
    "real",
    "rint",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "sqrt",
    "square",
    "tan",
    "tanh",
    "trunc",
)


for name in __all__:
    globals()[name] = _create_op(name)
