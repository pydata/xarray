import numpy as np
from . import utils
from .pycompat import iteritems, xrange


def expanded_indexer(key, ndim):
    """Given a key for indexing an ndarray, return an equivalent key which is a
    tuple with length equal to the number of dimensions.

    The expansion is done by replacing all `Ellipsis` items with the right
    number of full slices and then padding the key with full slices so that it
    reaches the appropriate dimensionality.
    """
    if not isinstance(key, tuple):
        # numpy treats non-tuple keys equivalent to tuples of length 1
        key = (key,)
    new_key = []
    # handling Ellipsis right is a little tricky, see:
    # http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing
    found_ellipsis = False
    for k in key:
        if k is Ellipsis:
            if not found_ellipsis:
                new_key.extend((ndim + 1 - len(key)) * [slice(None)])
                found_ellipsis = True
            else:
                new_key.append(slice(None))
        else:
            new_key.append(k)
    if len(new_key) > ndim:
        raise IndexError('too many indices')
    new_key.extend((ndim - len(new_key)) * [slice(None)])
    return tuple(new_key)


def canonicalize_indexer(key, ndim):
    """Given an indexer for orthogonal array indexing, return an indexer that
    is a tuple composed entirely of slices, integer ndarrays and native python
    ints.
    """
    def canonicalize(indexer):
        if not isinstance(indexer, slice):
            indexer = np.asarray(indexer)
            if indexer.ndim == 0:
                indexer = int(np.asscalar(indexer))
            if isinstance(indexer, np.ndarray):
                if indexer.ndim != 1:
                    raise ValueError('orthogonal array indexing only supports '
                                     '1d arrays')
                if indexer.dtype.kind == 'b':
                    indexer, = np.nonzero(indexer)
                elif indexer.dtype.kind != 'i':
                    raise ValueError('invalid subkey %r for integer based '
                                     'array indexing; all subkeys must be '
                                     'slices, integers or sequences of '
                                     'integers or Booleans' % indexer)
        return indexer

    return tuple(canonicalize(k) for k in expanded_indexer(key, ndim))


def orthogonal_indexer(key, shape):
    """Given a key for orthogonal array indexing, returns an equivalent key
    suitable for indexing a numpy.ndarray with fancy indexing.
    """
    def expand_key(k, length):
        if isinstance(k, slice):
            return np.arange(k.start or 0, k.stop or length, k.step or 1)
        else:
            return k

    # replace Ellipsis objects with slices
    key = list(canonicalize_indexer(key, len(shape)))
    # replace 1d arrays and slices with broadcast compatible arrays
    # note: we treat integers separately (instead of turning them into 1d
    # arrays) because integers (and only integers) collapse axes when used with
    # __getitem__
    non_int_keys = [n for n, k in enumerate(key) if not isinstance(k, (int, np.integer))]

    def full_slices_unselected(n_list):
        def all_full_slices(key_index):
            return all(isinstance(key[n], slice) and key[n] == slice(None)
                       for n in key_index)
        if not n_list:
            return n_list
        elif all_full_slices(range(n_list[0] + 1)):
            return full_slices_unselected(n_list[1:])
        elif all_full_slices(range(n_list[-1], len(key))):
            return full_slices_unselected(n_list[:-1])
        else:
            return n_list

    # However, testing suggests it is OK to keep contiguous sequences of full
    # slices at the start or the end of the key. Keeping slices around (when
    # possible) instead of converting slices to arrays significantly speeds up
    # indexing.
    # (Honestly, I don't understand when it's not OK to keep slices even in
    # between integer indices if as array is somewhere in the key, but such are
    # the admittedly mind-boggling ways of numpy's advanced indexing.)
    array_keys = full_slices_unselected(non_int_keys)

    array_indexers = np.ix_(*(expand_key(key[n], shape[n])
                              for n in array_keys))
    for i, n in enumerate(array_keys):
        key[n] = array_indexers[i]
    return tuple(key)


def convert_label_indexer(index, label, index_name=''):
    """Given a pandas.Index (or xray.Coordinate) and labels (e.g., from
    __getitem__) for one dimension, return an indexer suitable for indexing an
    ndarray along that dimension
    """
    if isinstance(label, slice):
        indexer = index.slice_indexer(label.start, label.stop, label.step)
    else:
        label = np.asarray(label)
        if label.ndim == 0:
            indexer = index.get_loc(np.asscalar(label))
        elif label.dtype.kind == 'b':
            indexer, = np.nonzero(label)
        else:
            indexer = index.get_indexer(label)
            if np.any(indexer < 0):
                raise ValueError('not all values found in index %r'
                                 % index_name)
    return indexer


def remap_label_indexers(data_obj, indexers):
    """Given an xray data object and label based indexers, return a mapping
    of equivalent location based indexers.
    """
    return dict((dim, convert_label_indexer(data_obj.coords[dim], label, dim))
                for dim, label in iteritems(indexers))


def _expand_slice(slice_, size):
    return np.arange(*slice_.indices(size))


def slice_slice(old_slice, applied_slice, size):
    """Given a slice and the size of the dimension to which it will be applied,
    index it with another slice to return a new slice equivalent to applying
    the slices sequentially
    """
    step = (old_slice.step or 1) * (applied_slice.step or 1)

    # For now, use the hack of turning old_slice into an ndarray to reconstruct
    # the slice start and stop. This is not entirely ideal, but it is still
    # definitely better than leaving the indexer as an array.
    items = _expand_slice(old_slice, size)[applied_slice]
    if len(items) > 0:
        start = items[0]
        stop = items[-1] + step
        if stop < 0:
            stop = None
    else:
        start = 0
        stop = 0
    return slice(start, stop, step)


def _index_indexer_1d(old_indexer, applied_indexer, size):
    assert isinstance(applied_indexer, (int, np.integer, slice, np.ndarray))
    if isinstance(applied_indexer, slice) and applied_indexer == slice(None):
        # shortcut for the usual case
        return old_indexer
    if isinstance(old_indexer, slice):
        if isinstance(applied_indexer, slice):
            indexer = slice_slice(old_indexer, applied_indexer, size)
        else:
            indexer = _expand_slice(old_indexer, size)[applied_indexer]
    else:
        indexer = old_indexer[applied_indexer]
    return indexer


class LazilyIndexedArray(utils.NDArrayMixin):
    """Wrap an array that handles orthogonal indexing to make indexing lazy
    """
    def __init__(self, array, key=None):
        """
        Parameters
        ----------
        array : array_like
            Array like object to index.
        key : tuple, optional
            Array indexer. If provided, it is assumed to already be in
            canonical expanded form.
        """
        if key is None:
            key = (slice(None),) * array.ndim
        self.array = array
        self.key = key

    def _updated_key(self, new_key):
        new_key = iter(canonicalize_indexer(new_key, self.ndim))
        key = []
        for size, k in zip(self.array.shape, self.key):
            if isinstance(k, (int, np.integer)):
                key.append(k)
            else:
                key.append(_index_indexer_1d(k, next(new_key), size))
        return tuple(key)

    @property
    def shape(self):
        shape = []
        for size, k in zip(self.array.shape, self.key):
            if isinstance(k, slice):
                shape.append(len(xrange(*k.indices(size))))
            elif isinstance(k, np.ndarray):
                shape.append(k.size)
        return tuple(shape)

    def __array__(self, dtype=None):
        return np.asarray(self.array[self.key], dtype=None)

    def __getitem__(self, key):
        return type(self)(self.array, self._updated_key(key))

    def __setitem__(self, key, value):
        key = self._updated_key(key)
        self.array[key] = value

    def __repr__(self):
        return ('%s(array=%r, key=%r)' %
                (type(self).__name__, self.array, self.key))
