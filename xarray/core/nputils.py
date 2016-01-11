import numpy as np
import pandas as pd
import warnings

from .pycompat import builtins, reduce


def _validate_axis(data, axis):
    ndim = data.ndim
    if not -ndim <= axis < ndim:
        raise IndexError('axis %r out of bounds [-%r, %r)'
                         % (axis, ndim, ndim))
    if axis < 0:
        axis += ndim
    return axis


def _select_along_axis(values, idx, axis):
    other_ind = np.ix_(*[np.arange(s) for s in idx.shape])
    sl = other_ind[:axis] + (idx,) + other_ind[axis:]
    return values[sl]


def nanfirst(values, axis):
    axis = _validate_axis(values, axis)
    idx_first = np.argmax(~pd.isnull(values), axis=axis)
    return _select_along_axis(values, idx_first, axis)


def nanlast(values, axis):
    axis = _validate_axis(values, axis)
    rev = (slice(None),) * axis + (slice(None, None, -1),)
    idx_last = -1 - np.argmax(~pd.isnull(values)[rev], axis=axis)
    return _select_along_axis(values, idx_last, axis)


def _calc_concat_shape(arrays, axis=0):
    first_shape = arrays[0].shape
    length = builtins.sum(a.shape[axis] for a in arrays)
    result_shape = first_shape[:axis] + (length,) + first_shape[(axis + 1):]
    return result_shape


def interleaved_concat(arrays, indices, axis=0):
    arrays = [np.asarray(a) for a in arrays]
    axis = _validate_axis(arrays[0], axis)
    result_shape = _calc_concat_shape(arrays, axis=axis)
    dtype = reduce(np.promote_types, [a.dtype for a in arrays])
    result = np.empty(result_shape, dtype)
    key = [slice(None)] * result.ndim
    for a, ind in zip(arrays, indices):
        key[axis] = ind
        result[key] = a
    return result


def _ensure_bool_is_ndarray(result, *args):
    # numpy will sometimes return a scalar value from binary comparisons if it
    # can't handle the comparison instead of broadcasting, e.g.,
    # In [10]: 1 == np.array(['a', 'b'])
    # Out[10]: False
    # This function ensures that the result is the appropriate shape in these
    # cases
    if isinstance(result, bool):
        shape = np.broadcast(*args).shape
        constructor = np.ones if result else np.zeros
        result = constructor(shape, dtype=bool)
    return result


def array_eq(self, other):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'elementwise comparison failed')
        return _ensure_bool_is_ndarray(self == other, self, other)


def array_ne(self, other):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'elementwise comparison failed')
        return _ensure_bool_is_ndarray(self != other, self, other)
