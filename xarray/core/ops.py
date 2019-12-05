"""Define core operations for xarray objects.

TODO(shoyer): rewrite this module, making use of xarray.core.computation,
NumPy's __array_ufunc__ and mixin classes instead of the unintuitive "inject"
functions.
"""

import operator

import numpy as np

from . import dtypes, duck_array_ops
from .nputils import array_eq, array_ne

try:
    import bottleneck as bn

    has_bottleneck = True
except ImportError:
    # use numpy methods instead
    bn = np
    has_bottleneck = False


UNARY_OPS = ["neg", "pos", "abs", "invert"]
CMP_BINARY_OPS = ["lt", "le", "ge", "gt"]
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
]

# methods which pass on the numpy return value unchanged
# be careful not to list methods that we would want to wrap later
NUMPY_SAME_METHODS = ["item", "searchsorted"]
# methods which don't modify the data shape, so the result should still be
# wrapped in an Variable/DataArray
NUMPY_UNARY_METHODS = ["astype", "argsort", "clip", "conj", "conjugate"]
PANDAS_UNARY_FUNCTIONS = ["isnull", "notnull"]
# methods which remove an axis
REDUCE_METHODS = ["all", "any"]
NAN_REDUCE_METHODS = [
    "argmax",
    "argmin",
    "max",
    "min",
    "mean",
    "prod",
    "sum",
    "std",
    "var",
    "median",
]
NAN_CUM_METHODS = ["cumsum", "cumprod"]
# TODO: wrap take, dot, sort


_CUM_DOCSTRING_TEMPLATE = """\
Apply `{name}` along some dimension of {cls}.

Parameters
----------
{extra_args}
skipna : bool, optional
    If True, skip missing values (as marked by NaN). By default, only
    skips missing values for float dtypes; other dtypes either do not
    have a sentinel missing value (int) or skipna=True has not been
    implemented (object, datetime64 or timedelta64).
keep_attrs : bool, optional
    If True, the attributes (`attrs`) will be copied from the original
    object to the new one.  If False (default), the new object will be
    returned without attributes.
**kwargs : dict
    Additional keyword arguments passed on to `{name}`.

Returns
-------
cumvalue : {cls}
    New {cls} object with `{name}` applied to its data along the
    indicated dimension.
"""

_REDUCE_DOCSTRING_TEMPLATE = """\
Reduce this {cls}'s data by applying `{name}` along some dimension(s).

Parameters
----------
{extra_args}
skipna : bool, optional
    If True, skip missing values (as marked by NaN). By default, only
    skips missing values for float dtypes; other dtypes either do not
    have a sentinel missing value (int) or skipna=True has not been
    implemented (object, datetime64 or timedelta64).{min_count_docs}
keep_attrs : bool, optional
    If True, the attributes (`attrs`) will be copied from the original
    object to the new one.  If False (default), the new object will be
    returned without attributes.
**kwargs : dict
    Additional keyword arguments passed on to the appropriate array
    function for calculating `{name}` on this object's data.

Returns
-------
reduced : {cls}
    New {cls} object with `{name}` applied to its data and the
    indicated dimension(s) removed.
"""

_MINCOUNT_DOCSTRING = """
min_count : int, default None
    The required number of valid values to perform the operation.
    If fewer than min_count non-NA values are present the result will
    be NA. New in version 0.10.8: Added with the default being None."""

_COARSEN_REDUCE_DOCSTRING_TEMPLATE = """\
Coarsen this object by applying `{name}` along its dimensions.

Parameters
----------
**kwargs : dict
    Additional keyword arguments passed on to `{name}`.

Returns
-------
reduced : DataArray or Dataset
    New object with `{name}` applied along its coasen dimnensions.
"""


def fillna(data, other, join="left", dataset_join="left"):
    """Fill missing values in this object with data from the other object.
    Follows normal broadcasting and alignment rules.

    Parameters
    ----------
    join : {'outer', 'inner', 'left', 'right'}, optional
        Method for joining the indexes of the passed objects along each
        dimension
        - 'outer': use the union of object indexes
        - 'inner': use the intersection of object indexes
        - 'left': use indexes from the first object with each dimension
        - 'right': use indexes from the last object with each dimension
        - 'exact': raise `ValueError` instead of aligning when indexes to be
          aligned are not equal
    dataset_join : {'outer', 'inner', 'left', 'right'}, optional
        Method for joining variables of Dataset objects with mismatched
        data variables.
        - 'outer': take variables from both Dataset objects
        - 'inner': take only overlapped variables
        - 'left': take only variables from the first object
        - 'right': take only variables from the last object
    """
    from .computation import apply_ufunc

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


def where_method(self, cond, other=dtypes.NA):
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
    from .computation import apply_ufunc

    # alignment for three arguments is complicated, so don't support it yet
    join = "inner" if other is dtypes.NA else "exact"
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


def inject_reduce_methods(cls):
    methods = (
        [
            (name, getattr(duck_array_ops, "array_%s" % name), False)
            for name in REDUCE_METHODS
        ]
        + [(name, getattr(duck_array_ops, name), True) for name in NAN_REDUCE_METHODS]
        + [("count", duck_array_ops.count, False)]
    )
    for name, f, include_skipna in methods:
        numeric_only = getattr(f, "numeric_only", False)
        available_min_count = getattr(f, "available_min_count", False)
        min_count_docs = _MINCOUNT_DOCSTRING if available_min_count else ""

        func = cls._reduce_method(f, include_skipna, numeric_only)
        func.__name__ = name
        func.__doc__ = _REDUCE_DOCSTRING_TEMPLATE.format(
            name=name,
            cls=cls.__name__,
            extra_args=cls._reduce_extra_args_docstring.format(name=name),
            min_count_docs=min_count_docs,
        )
        setattr(cls, name, func)


def inject_cum_methods(cls):
    methods = [(name, getattr(duck_array_ops, name), True) for name in NAN_CUM_METHODS]
    for name, f, include_skipna in methods:
        numeric_only = getattr(f, "numeric_only", False)
        func = cls._reduce_method(f, include_skipna, numeric_only)
        func.__name__ = name
        func.__doc__ = _CUM_DOCSTRING_TEMPLATE.format(
            name=name,
            cls=cls.__name__,
            extra_args=cls._cum_extra_args_docstring.format(name=name),
        )
        setattr(cls, name, func)


def op_str(name):
    return "__%s__" % name


def get_op(name):
    return getattr(operator, op_str(name))


NON_INPLACE_OP = {get_op("i" + name): get_op(name) for name in NUM_BINARY_OPS}


def inplace_to_noninplace_op(f):
    return NON_INPLACE_OP[f]


def inject_binary_ops(cls, inplace=False):
    for name in CMP_BINARY_OPS + NUM_BINARY_OPS:
        setattr(cls, op_str(name), cls._binary_op(get_op(name)))

    for name, f in [("eq", array_eq), ("ne", array_ne)]:
        setattr(cls, op_str(name), cls._binary_op(f))

    for name in NUM_BINARY_OPS:
        # only numeric operations have in-place and reflexive variants
        setattr(cls, op_str("r" + name), cls._binary_op(get_op(name), reflexive=True))
        if inplace:
            setattr(cls, op_str("i" + name), cls._inplace_binary_op(get_op("i" + name)))


def inject_all_ops_and_reduce_methods(cls, priority=50, array_only=True):
    # prioritize our operations over those of numpy.ndarray (priority=1)
    # and numpy.matrix (priority=10)
    cls.__array_priority__ = priority

    # patch in standard special operations
    for name in UNARY_OPS:
        setattr(cls, op_str(name), cls._unary_op(get_op(name)))
    inject_binary_ops(cls, inplace=True)

    # patch in numpy/pandas methods
    for name in NUMPY_UNARY_METHODS:
        setattr(cls, name, cls._unary_op(_method_wrapper(name)))

    for name in PANDAS_UNARY_FUNCTIONS:
        f = _func_slash_method_wrapper(getattr(duck_array_ops, name), name=name)
        setattr(cls, name, cls._unary_op(f))

    f = _func_slash_method_wrapper(duck_array_ops.around, name="round")
    setattr(cls, "round", cls._unary_op(f))

    if array_only:
        # these methods don't return arrays of the same shape as the input, so
        # don't try to patch these in for Dataset objects
        for name in NUMPY_SAME_METHODS:
            setattr(cls, name, _values_method_wrapper(name))

    inject_reduce_methods(cls)
    inject_cum_methods(cls)
