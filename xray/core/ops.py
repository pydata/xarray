from functools import partial
import contextlib
import inspect
import operator
import warnings

import numpy as np
import pandas as pd

from . import npcompat
from .pycompat import PY3, range, dask_array_type
from .nputils import (
    nanfirst, nanlast, interleaved_concat as _interleaved_concat_numpy,
    array_eq, array_ne, _validate_axis, _calc_concat_shape
)


try:
    import bottleneck as bn
except ImportError:
    # use numpy methods instead
    bn = np

try:
    import dask.array as da
    has_dask = True
except ImportError:
    has_dask = False


UNARY_OPS = ['neg', 'pos', 'abs', 'invert']
CMP_BINARY_OPS = ['lt', 'le', 'ge', 'gt']
NUM_BINARY_OPS = ['add', 'sub', 'mul', 'truediv', 'floordiv', 'mod',
                  'pow', 'and', 'xor', 'or']
if not PY3:
    NUM_BINARY_OPS.append('div')

# methods which pass on the numpy return value unchanged
# be careful not to list methods that we would want to wrap later
NUMPY_SAME_METHODS = ['item', 'searchsorted']
# methods which don't modify the data shape, so the result should still be
# wrapped in an Variable/DataArray
NUMPY_UNARY_METHODS = ['astype', 'argsort', 'clip', 'conj', 'conjugate']
PANDAS_UNARY_FUNCTIONS = ['isnull', 'notnull']
# methods which remove an axis
NUMPY_REDUCE_METHODS = ['all', 'any']
NAN_REDUCE_METHODS = ['argmax', 'argmin', 'max', 'min', 'mean', 'prod', 'sum',
                      'std', 'var', 'median']
# TODO: wrap cumprod/cumsum, take, dot, sort


def _dask_or_eager_func(name, eager_module=np, dispatch_elemwise=False):
    if has_dask:
        def f(data, *args, **kwargs):
            target = data[0] if dispatch_elemwise else data
            module = da if isinstance(target, da.Array) else eager_module
            return getattr(module, name)(data, *args, **kwargs)
    else:
        def f(data, *args, **kwargs):
            return getattr(eager_module, name)(data, *args, **kwargs)
    return f


def _fail_on_dask_array_input(values, msg=None, func_name=None):
    if isinstance(values, dask_array_type):
        if msg is None:
            msg = '%r is not a valid method on dask arrays'
        if func_name is None:
            func_name = inspect.stack()[1][3]
        raise NotImplementedError(msg % func_name)


around = _dask_or_eager_func('around')
isclose = _dask_or_eager_func('isclose')
isnull = _dask_or_eager_func('isnull', pd)
notnull = _dask_or_eager_func('notnull', pd)

transpose = _dask_or_eager_func('transpose')
where = _dask_or_eager_func('where')
insert = _dask_or_eager_func('insert')
take = _dask_or_eager_func('take')
broadcast_to = _dask_or_eager_func('broadcast_to', npcompat)

concatenate = _dask_or_eager_func('concatenate', dispatch_elemwise=True)
stack = _dask_or_eager_func('stack', npcompat, dispatch_elemwise=True)


def _interleaved_indices_required(indices):
    """With dask, we care about data locality and would rather avoid splitting
    splitting up each arrays into single elements. This routine checks to see
    if we really need the "interleaved" part of interleaved_concat.

    We don't use for the pure numpy version of interleaved_concat, because it's
    just as fast or faster to directly do the interleaved concatenate rather
    than check if we could simply it.
    """
    next_expected = 0
    for ind in indices:
        if isinstance(ind, slice):
            if ((ind.start or 0) != next_expected
                    or ind.step not in (1, None)):
                return True
            next_expected = ind.stop
        else:
            ind = np.asarray(ind)
            expected = np.arange(next_expected, next_expected + ind.size)
            if (ind != expected).any():
                return True
            next_expected = ind[-1] + 1
    return False


def _interleaved_concat_slow(arrays, indices, axis=0):
    """A slow version of interleaved_concat that also works on dask arrays
    """
    axis = _validate_axis(arrays[0], axis)

    result_shape = _calc_concat_shape(arrays, axis=axis)
    length = result_shape[axis]
    array_lookup = np.empty(length, dtype=int)
    element_lookup = np.empty(length, dtype=int)

    for n, ind in enumerate(indices):
        if isinstance(ind, slice):
            ind = np.arange(*ind.indices(length))
        for m, i in enumerate(ind):
            array_lookup[i] = n
            element_lookup[i] = m

    split_arrays = [arrays[n][(slice(None),) * axis + (slice(m, m + 1),)]
                    for (n, m) in zip(array_lookup, element_lookup)]
    return concatenate(split_arrays, axis)


def interleaved_concat(arrays, indices, axis=0):
    """Concatenate each array along the given axis, but also assign each array
    element into the location given by indices. This operation is used for
    groupby.transform.
    """
    if has_dask and isinstance(arrays[0], da.Array):
        if not _interleaved_indices_required(indices):
            return da.concatenate(arrays, axis)
        else:
            return _interleaved_concat_slow(arrays, indices, axis)
    else:
        return _interleaved_concat_numpy(arrays, indices, axis)


def asarray(data):
    return data if isinstance(data, dask_array_type) else np.asarray(data)


def as_like_arrays(*data):
    if all(isinstance(d, dask_array_type) for d in data):
        return data
    else:
        return tuple(np.asarray(d) for d in data)


def allclose_or_equiv(arr1, arr2, rtol=1e-5, atol=1e-8):
    """Like np.allclose, but also allows values to be NaN in both arrays
    """
    arr1, arr2 = as_like_arrays(arr1, arr2)
    if arr1.shape != arr2.shape:
        return False
    return bool(isclose(arr1, arr2, rtol=rtol, atol=atol, equal_nan=True).all())


def array_equiv(arr1, arr2):
    """Like np.array_equal, but also allows values to be NaN in both arrays
    """
    arr1, arr2 = as_like_arrays(arr1, arr2)
    if arr1.shape != arr2.shape:
        return False
    return bool(((arr1 == arr2) | (isnull(arr1) & isnull(arr2))).all())


def _call_possibly_missing_method(arg, name, args, kwargs):
    try:
        method = getattr(arg, name)
    except AttributeError:
        _fail_on_dask_array_input(arg, func_name=name)
        if hasattr(arg, 'data'):
            _fail_on_dask_array_input(arg.data, func_name=name)
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


_REDUCE_DOCSTRING_TEMPLATE = \
        """Reduce this {cls}'s data by applying `{name}` along some
        dimension(s).

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
        reduced : {cls}
            New {cls} object with `{name}` applied to its data and the
            indicated dimension(s) removed.
        """


def count(values, axis=None):
    """Count the number of non-NA in this array along the given axis or axes
    """
    return sum(~isnull(values), axis=axis)


def fillna(values, other):
    """Fill missing values in this object with values from the other object.
    Follows normal broadcasting and alignment rules.
    """
    return where(isnull(values), other, values)


@contextlib.contextmanager
def _ignore_warnings_if(condition):
    if condition:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            yield
    else:
        yield


def _create_nan_agg_method(name, numeric_only=False):
    def f(values, axis=None, skipna=None, **kwargs):
        # ignore keyword args inserted by np.mean and other numpy aggreagators
        # automatically:
        kwargs.pop('dtype', None)
        kwargs.pop('out', None)

        values = asarray(values)
        if skipna or (skipna is None and values.dtype.kind == 'f'):
            if values.dtype.kind not in ['i', 'f']:
                raise NotImplementedError(
                    'skipna=True not yet implemented for %s with dtype %s'
                    % (name, values.dtype))
            nanname = 'nan' + name
            eager_module = np if isinstance(axis, tuple) else bn
            func = _dask_or_eager_func(nanname, eager_module)
            using_numpy_nan_func = eager_module is np
        else:
            func = _dask_or_eager_func(name)
            using_numpy_nan_func = False
        with _ignore_warnings_if(using_numpy_nan_func):
            try:
                return func(values, axis=axis, **kwargs)
            except AttributeError:
                if isinstance(values, dask_array_type):
                    msg = '%s is not yet implemented on dask arrays' % name
                else:
                    assert using_numpy_nan_func
                    msg = ('%s is not available with skipna=False with the '
                           'installed version of numpy; upgrade to numpy 1.9 '
                           'or newer to use skipna=True or skipna=None' % name)
                raise NotImplementedError(msg)
    f.numeric_only = numeric_only
    return f


argmax = _create_nan_agg_method('argmax')
argmin = _create_nan_agg_method('argmin')
max = _create_nan_agg_method('max')
min = _create_nan_agg_method('min')
sum = _create_nan_agg_method('sum', numeric_only=True)
mean = _create_nan_agg_method('mean', numeric_only=True)
std = _create_nan_agg_method('std', numeric_only=True)
var = _create_nan_agg_method('var', numeric_only=True)
median = _create_nan_agg_method('median', numeric_only=True)


_fail_on_dask_array_input_skipna = partial(
    _fail_on_dask_array_input,
    msg='%r with skipna=True is not yet implemented on dask arrays')


_prod = _dask_or_eager_func('prod')

def prod(values, axis=None, skipna=None, **kwargs):
    if skipna or (skipna is None and values.dtype.kind == 'f'):
        if values.dtype.kind not in ['i', 'f']:
            raise NotImplementedError(
                'skipna=True not yet implemented for prod with dtype %s'
                % values.dtype)
        _fail_on_dask_array_input_skipna(values)
        return npcompat.nanprod(values, axis=axis, **kwargs)
    return _prod(values, axis=axis, **kwargs)
prod.numeric_only = True


def first(values, axis, skipna=None):
    """Return the first non-NA elements in this array along the given axis
    """
    if (skipna or skipna is None) and values.dtype.kind not in 'iSU':
        # only bother for dtypes that can hold NaN
        _fail_on_dask_array_input_skipna(values)
        return nanfirst(values, axis)
    return take(values, 0, axis=axis)


def last(values, axis, skipna=None):
    """Return the last non-NA elements in this array along the given axis
    """
    if (skipna or skipna is None) and values.dtype.kind not in 'iSU':
        # only bother for dtypes that can hold NaN
        _fail_on_dask_array_input_skipna(values)
        return nanlast(values, axis)
    return take(values, -1, axis=axis)


def inject_reduce_methods(cls):
    methods = ([(name, getattr(np, name), False) for name
               in NUMPY_REDUCE_METHODS]
               + [(name, globals()[name], True) for name
                  in NAN_REDUCE_METHODS]
               + [('count', count, False)])
    for name, f, include_skipna in methods:
        numeric_only = getattr(f, 'numeric_only', False)
        func = cls._reduce_method(f, include_skipna, numeric_only)
        func.__name__ = name
        func.__doc__ = _REDUCE_DOCSTRING_TEMPLATE.format(
            name=name, cls=cls.__name__,
            extra_args=cls._reduce_extra_args_docstring)
        setattr(cls, name, func)


def op_str(name):
    return '__%s__' % name


def get_op(name):
    return getattr(operator, op_str(name))


NON_INPLACE_OP = dict((get_op('i' + name), get_op(name))
                      for name in NUM_BINARY_OPS)

def inplace_to_noninplace_op(f):
    return NON_INPLACE_OP[f]


def inject_binary_ops(cls, inplace=False):
    for name in CMP_BINARY_OPS + NUM_BINARY_OPS:
        setattr(cls, op_str(name), cls._binary_op(get_op(name)))

    for name, f in [('eq', array_eq), ('ne', array_ne)]:
        setattr(cls, op_str(name), cls._binary_op(f))

    # patch in fillna
    f = _func_slash_method_wrapper(fillna)
    method = cls._binary_op(f, join='left', drop_na_vars=False)
    setattr(cls, '_fillna', method)

    for name in NUM_BINARY_OPS:
        # only numeric operations have in-place and reflexive variants
        setattr(cls, op_str('r' + name),
                cls._binary_op(get_op(name), reflexive=True))
        if inplace:
            setattr(cls, op_str('i' + name),
                    cls._inplace_binary_op(get_op('i' + name)))


def inject_all_ops_and_reduce_methods(cls, priority=50, array_only=True):
    # priortize our operations over those of numpy.ndarray (priority=1)
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
        f = _func_slash_method_wrapper(getattr(pd, name))
        setattr(cls, name, cls._unary_op(f))

    f = _func_slash_method_wrapper(around, name='round')
    setattr(cls, 'round', cls._unary_op(f))

    if array_only:
        # these methods don't return arrays of the same shape as the input, so
        # don't try to patch these in for Dataset objects
        for name in NUMPY_SAME_METHODS:
            setattr(cls, name, _values_method_wrapper(name))

    inject_reduce_methods(cls)
