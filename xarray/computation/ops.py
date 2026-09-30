"""Define core operations for xarray objects.

TODO(shoyer): rewrite this module, making use of xarray.computation.computation,
NumPy's __array_ufunc__ and mixin classes instead of the unintuitive "inject"
functions.
"""

from __future__ import annotations

import operator
from typing import Literal

import numpy as np

from xarray.core import dtypes, duck_array_ops

try:
    import bottleneck as bn

    has_bottleneck = True
except ImportError:
    # use numpy methods instead
    bn = np
    has_bottleneck = False


NUM_BINARY_OPS = [
    "add",
    "sub",
    "mul",
    "truediv",
    "floordiv",
    "mod",
    "pow",
    "and",
    "xor",
    "or",
    "lshift",
    "rshift",
]

# methods which pass on the numpy return value unchanged
# be careful not to list methods that we would want to wrap later
NUMPY_SAME_METHODS = ["item", "searchsorted"]


def fillna(data, other, join="left", dataset_join="left"):
    """Fill missing values in this object with data from the other object.
    Follows normal broadcasting and alignment rules.

    Parameters
    ----------
    join : {"outer", "inner", "left", "right"}, optional
        Method for joining the indexes of the passed objects along each
        dimension
        - "outer": use the union of object indexes
        - "inner": use the intersection of object indexes
        - "left": use indexes from the first object with each dimension
        - "right": use indexes from the last object with each dimension
        - "exact": raise `ValueError` instead of aligning when indexes to be
          aligned are not equal
    dataset_join : {"outer", "inner", "left", "right"}, optional
        Method for joining variables of Dataset objects with mismatched
        data variables.
        - "outer": take variables from both Dataset objects
        - "inner": take only overlapped variables
        - "left": take only variables from the first object
        - "right": take only variables from the last object
    """
    from xarray.computation.apply_ufunc import apply_ufunc

    return apply_ufunc(
        duck_array_ops.fillna,
        data,
        other,
        join=join,
        dask="allowed",
        dataset_join=dataset_join,
        dataset_fill_value=np.nan,
        keep_attrs=True,
    )


# TODO: type this properly
def where_method(self, cond, other=dtypes.NA):  # type: ignore[unused-ignore,has-type]
    """Return elements from `self` or `other` depending on `cond`.

    Parameters
    ----------
    cond : DataArray or Dataset with boolean dtype
        Locations at which to preserve this objects values.
    other : scalar, DataArray or Dataset, optional
        Value to use for locations in this object where ``cond`` is False.
        By default, inserts missing values.

    Returns
    -------
    Same type as caller.
    """
    from xarray.computation.apply_ufunc import apply_ufunc

    # alignment for three arguments is complicated, so don't support it yet
    join: Literal["inner", "exact"] = "inner" if other is dtypes.NA else "exact"
    return apply_ufunc(
        duck_array_ops.where_method,
        self,
        cond,
        other,
        join=join,
        dataset_join=join,
        dask="allowed",
        keep_attrs=True,
    )


def _call_possibly_missing_method(arg, name, args, kwargs):
    try:
        method = getattr(arg, name)
    except AttributeError:
        duck_array_ops.fail_on_dask_array_input(arg, func_name=name)
        if hasattr(arg, "data"):
            duck_array_ops.fail_on_dask_array_input(arg.data, func_name=name)
        raise
    else:
        return method(*args, **kwargs)


def _values_method_wrapper(name):
    def func(self, *args, **kwargs):
        return _call_possibly_missing_method(self.data, name, args, kwargs)

    func.__name__ = name
    func.__doc__ = getattr(np.ndarray, name).__doc__
    return func


def _method_wrapper(name):
    def func(self, *args, **kwargs):
        return _call_possibly_missing_method(self, name, args, kwargs)

    func.__name__ = name
    func.__doc__ = getattr(np.ndarray, name).__doc__
    return func


def _func_slash_method_wrapper(f, name=None):
    # try to wrap a method, but if not found use the function
    # this is useful when patching in a function as both a DataArray and
    # Dataset method
    if name is None:
        name = f.__name__

    def func(self, *args, **kwargs):
        try:
            return getattr(self, name)(*args, **kwargs)
        except AttributeError:
            return f(self, *args, **kwargs)

    func.__name__ = name
    func.__doc__ = f.__doc__
    return func


def op_str(name):
    return f"__{name}__"


def get_op(name):
    return getattr(operator, op_str(name))


NON_INPLACE_OP = {get_op("i" + name): get_op(name) for name in NUM_BINARY_OPS}


def inplace_to_noninplace_op(f):
    return NON_INPLACE_OP[f]


# _typed_ops.py uses the following wrapped functions as a kind of unary operator
argsort = _method_wrapper("argsort")
conj = _method_wrapper("conj")
conjugate = _method_wrapper("conj")
round_ = _func_slash_method_wrapper(duck_array_ops.around, name="round")


def inject_numpy_same(cls):
    # these methods don't return arrays of the same shape as the input, so
    # don't try to patch these in for Dataset objects
    for name in NUMPY_SAME_METHODS:
        setattr(cls, name, _values_method_wrapper(name))


class IncludeNumpySameMethods:
    __slots__ = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        inject_numpy_same(cls)  # some methods not applicable to Dataset objects
