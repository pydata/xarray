from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import warnings


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


def inverse_permutation(indices):
    """Return indices for an inverse permutation.

    Parameters
    ----------
    indices : 1D np.ndarray with dtype=int
        Integer positions to assign elements to.

    Returns
    -------
    inverse_permutation : 1D np.ndarray with dtype=int
        Integer indices to take from the original array to create the
        permutation.
    """
    # use intp instead of int64 because of windows :(
    inverse_permutation = np.empty(len(indices), dtype=np.intp)
    inverse_permutation[indices] = np.arange(len(indices), dtype=np.intp)
    return inverse_permutation


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


def _is_contiguous(positions):
    """Given a non-empty list, does it consist of contiguous integers?"""
    previous = positions[0]
    for current in positions[1:]:
        if current != previous + 1:
            return False
        previous = current
    return True


def _advanced_indexer_subspaces(key):
    """Indices of the advanced indexes subspaces for mixed indexing and vindex.
    """
    if not isinstance(key, tuple):
        key = (key,)
    advanced_index_positions = [i for i, k in enumerate(key)
                                if not isinstance(k, slice)]

    if (not advanced_index_positions or
            not _is_contiguous(advanced_index_positions)):
        # Nothing to reorder: dimensions on the indexing result are already
        # ordered like vindex. See NumPy's rule for "Combining advanced and
        # basic indexing":
        # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#combining-advanced-and-basic-indexing
        return (), ()

    non_slices = [k for k in key if not isinstance(k, slice)]
    ndim = len(np.broadcast(*non_slices).shape)
    mixed_positions = advanced_index_positions[0] + np.arange(ndim)
    vindex_positions = np.arange(ndim)
    return mixed_positions, vindex_positions


class NumpyVIndexAdapter(object):
    """Object that implements indexing like vindex on a np.ndarray.

    This is a pure Python implementation of (some of) the logic in this NumPy
    proposal: https://github.com/numpy/numpy/pull/6256
    """
    def __init__(self, array):
        self._array = array

    def __getitem__(self, key):
        mixed_positions, vindex_positions = _advanced_indexer_subspaces(key)
        return np.moveaxis(self._array[key], mixed_positions, vindex_positions)

    def __setitem__(self, key, value):
        """Value must have dimensionality matching the key."""
        mixed_positions, vindex_positions = _advanced_indexer_subspaces(key)
        self._array[key] = np.moveaxis(value, vindex_positions,
                                       mixed_positions)
