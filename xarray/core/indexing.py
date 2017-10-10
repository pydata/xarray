from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import timedelta
from collections import defaultdict, Hashable
import numpy as np
import pandas as pd

from . import nputils
from . import utils
from .pycompat import (iteritems, range, integer_types, dask_array_type,
                       suppress)
from .utils import is_dict_like


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


def _expand_slice(slice_, size):
    return np.arange(*slice_.indices(size))


def _try_get_item(x):
    try:
        return x.item()
    except AttributeError:
        return x


def _asarray_tuplesafe(values):
    """
    Convert values into a numpy array of at most 1-dimension, while preserving
    tuples.

    Adapted from pandas.core.common._asarray_tuplesafe
    """
    if isinstance(values, tuple):
        result = utils.to_0d_object_array(values)
    else:
        result = np.asarray(values)
        if result.ndim == 2:
            result = np.empty(len(values), dtype=object)
            result[:] = values

    return result


def _is_nested_tuple(possible_tuple):
    return (isinstance(possible_tuple, tuple)
            and any(isinstance(value, (tuple, list, slice))
                    for value in possible_tuple))


def _index_method_kwargs(method, tolerance):
    # backwards compatibility for pandas<0.16 (method) or pandas<0.17
    # (tolerance)
    kwargs = {}
    if method is not None:
        kwargs['method'] = method
    if tolerance is not None:
        kwargs['tolerance'] = tolerance
    return kwargs


def get_loc(index, label, method=None, tolerance=None):
    kwargs = _index_method_kwargs(method, tolerance)
    return index.get_loc(label, **kwargs)


def get_indexer_nd(index, labels, method=None, tolerance=None):
    """ Call pd.Index.get_indexer(labels). """
    kwargs = _index_method_kwargs(method, tolerance)

    flat_labels = np.ravel(labels)
    flat_indexer = index.get_indexer(flat_labels, **kwargs)
    indexer = flat_indexer.reshape(labels.shape)
    return indexer


def convert_label_indexer(index, label, index_name='', method=None,
                          tolerance=None):
    """Given a pandas.Index and labels (e.g., from __getitem__) for one
    dimension, return an indexer suitable for indexing an ndarray along that
    dimension. If `index` is a pandas.MultiIndex and depending on `label`,
    return a new pandas.Index or pandas.MultiIndex (otherwise return None).
    """
    new_index = None

    if isinstance(label, slice):
        if method is not None or tolerance is not None:
            raise NotImplementedError(
                'cannot use ``method`` argument if any indexers are '
                'slice objects')
        indexer = index.slice_indexer(_try_get_item(label.start),
                                      _try_get_item(label.stop),
                                      _try_get_item(label.step))
        if not isinstance(indexer, slice):
            # unlike pandas, in xarray we never want to silently convert a slice
            # indexer into an array indexer
            raise KeyError('cannot represent labeled-based slice indexer for '
                           'dimension %r with a slice over integer positions; '
                           'the index is unsorted or non-unique' % index_name)

    elif is_dict_like(label):
        is_nested_vals = _is_nested_tuple(tuple(label.values()))
        if not isinstance(index, pd.MultiIndex):
            raise ValueError('cannot use a dict-like object for selection on a '
                             'dimension that does not have a MultiIndex')
        elif len(label) == index.nlevels and not is_nested_vals:
            indexer = index.get_loc(tuple((label[k] for k in index.names)))
        else:
            for k, v in label.items():
                # index should be an item (i.e. Hashable) not an array-like
                if not isinstance(v, Hashable):
                    raise ValueError('Vectorized selection is not '
                                     'available along level variable: ' + k)
            indexer, new_index = index.get_loc_level(
                        tuple(label.values()), level=tuple(label.keys()))

    elif isinstance(label, tuple) and isinstance(index, pd.MultiIndex):
        if _is_nested_tuple(label):
            indexer = index.get_locs(label)
        elif len(label) == index.nlevels:
            indexer = index.get_loc(label)
        else:
            indexer, new_index = index.get_loc_level(
                label, level=list(range(len(label)))
            )

    else:
        label = (label if getattr(label, 'ndim', 1) > 1  # vectorized-indexing
                 else _asarray_tuplesafe(label))
        if label.ndim == 0:
            if isinstance(index, pd.MultiIndex):
                indexer, new_index = index.get_loc_level(label.item(), level=0)
            else:
                indexer = get_loc(index, label.item(), method, tolerance)
        elif label.dtype.kind == 'b':
            indexer = label
        else:
            if isinstance(index, pd.MultiIndex) and label.ndim > 1:
                raise ValueError('Vectorized selection is not available along '
                                 'MultiIndex variable: ' + index_name)
            indexer = get_indexer_nd(index, label, method, tolerance)
            if np.any(indexer < 0):
                raise KeyError('not all values found in index %r'
                               % index_name)
    return indexer, new_index


def get_dim_indexers(data_obj, indexers):
    """Given a xarray data object and label based indexers, return a mapping
    of label indexers with only dimension names as keys.

    It groups multiple level indexers given on a multi-index dimension
    into a single, dictionary indexer for that dimension (Raise a ValueError
    if it is not possible).
    """
    invalid = [k for k in indexers
               if k not in data_obj.dims and k not in data_obj._level_coords]
    if invalid:
        raise ValueError("dimensions or multi-index levels %r do not exist"
                         % invalid)

    level_indexers = defaultdict(dict)
    dim_indexers = {}
    for key, label in iteritems(indexers):
        dim, = data_obj[key].dims
        if key != dim:
            # assume here multi-index level indexer
            level_indexers[dim][key] = label
        else:
            dim_indexers[key] = label

    for dim, level_labels in iteritems(level_indexers):
        if dim_indexers.get(dim, False):
            raise ValueError("cannot combine multi-index level indexers "
                             "with an indexer for dimension %s" % dim)
        dim_indexers[dim] = level_labels

    return dim_indexers


def remap_label_indexers(data_obj, indexers, method=None, tolerance=None):
    """Given an xarray data object and label based indexers, return a mapping
    of equivalent location based indexers. Also return a mapping of updated
    pandas index objects (in case of multi-index level drop).
    """
    if method is not None and not isinstance(method, str):
        raise TypeError('``method`` must be a string')

    pos_indexers = {}
    new_indexes = {}

    dim_indexers = get_dim_indexers(data_obj, indexers)
    for dim, label in iteritems(dim_indexers):
        try:
            index = data_obj.indexes[dim]
        except KeyError:
            # no index for this dimension: reuse the provided labels
            if method is not None or tolerance is not None:
                raise ValueError('cannot supply ``method`` or ``tolerance`` '
                                 'when the indexed dimension does not have '
                                 'an associated coordinate.')
            pos_indexers[dim] = label
        else:
            idxr, new_idx = convert_label_indexer(index, label,
                                                  dim, method, tolerance)
            pos_indexers[dim] = idxr
            if new_idx is not None:
                new_indexes[dim] = new_idx

    return pos_indexers, new_indexes


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
    assert isinstance(applied_indexer, integer_types + (slice, np.ndarray))
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


class IndexerTuple(tuple):
    """ Base class for xarray indexing tuples """

    def __repr__(self):
        return type(self).__name__ + super(IndexerTuple, self).__repr__()


def to_tuple(key):
    """ Converts our indexer tuple to a native python tuple  """
    return tuple(key) if isinstance(key, IndexerTuple) else key


class BasicIndexer(IndexerTuple):
    """ Tuple for basic indexing. """


class OuterIndexer(IndexerTuple):
    """ Tuple for outer/orthogonal indexing.
    All the items are one of integer, slice, and 1d-np.ndarray.
    """


class VectorizedIndexer(IndexerTuple):
    """ Tuple for vectorized indexing """


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
        # We need to avoid doubly wrapping.
        if isinstance(array, type(self)):
            self.array = array.array
            self.key = array.key
            if key is not None:
                self.key = self._updated_key(key)

        else:
            if key is None:
                key = (slice(None),) * array.ndim
                key = OuterIndexer(key)
            self.array = array
            self.key = key

    def _updated_key(self, new_key):
        # TODO should suport VectorizedIndexer
        if isinstance(new_key, VectorizedIndexer):
            raise NotImplementedError(
                'Vectorized indexing for {} is not implemented. Load your '
                'data first with .load() or .compute().'.format(type(self)))
        new_key = iter(expanded_indexer(new_key, self.ndim))
        key = []
        for size, k in zip(self.array.shape, self.key):
            if isinstance(k, integer_types):
                key.append(k)
            else:
                key.append(_index_indexer_1d(k, next(new_key), size))
        return OuterIndexer(key)

    @property
    def shape(self):
        shape = []
        for size, k in zip(self.array.shape, self.key):
            if isinstance(k, slice):
                shape.append(len(range(*k.indices(size))))
            elif isinstance(k, np.ndarray):
                shape.append(k.size)
        return tuple(shape)

    def __array__(self, dtype=None):
        array = xarray_indexable(self.array)
        return np.asarray(array[self.key], dtype=None)

    def __getitem__(self, key):
        return type(self)(self.array, self._updated_key(key))

    def __setitem__(self, key, value):
        key = self._updated_key(key)
        self.array[key] = value

    def __repr__(self):
        return ('%s(array=%r, key=%r)' %
                (type(self).__name__, self.array, self.key))


def _wrap_numpy_scalars(array):
    """Wrap NumPy scalars in 0d arrays."""
    if np.isscalar(array):
        return np.array(array)
    else:
        return array


class CopyOnWriteArray(utils.NDArrayMixin):
    def __init__(self, array):
        self.array = array
        self._copied = False

    def _ensure_copied(self):
        if not self._copied:
            self.array = np.array(self.array)
            self._copied = True

    def __array__(self, dtype=None):
        return np.asarray(self.array, dtype=dtype)

    def __getitem__(self, key):
        return type(self)(_wrap_numpy_scalars(self.array[key]))

    def __setitem__(self, key, value):
        self._ensure_copied()
        self.array[key] = value


class MemoryCachedArray(utils.NDArrayMixin):
    def __init__(self, array):
        self.array = _wrap_numpy_scalars(array)

    def _ensure_cached(self):
        if not isinstance(self.array, np.ndarray):
            self.array = np.asarray(self.array)

    def __array__(self, dtype=None):
        self._ensure_cached()
        return np.asarray(self.array, dtype=dtype)

    def __getitem__(self, key):
        return type(self)(_wrap_numpy_scalars(self.array[key]))

    def __setitem__(self, key, value):
        self.array[key] = value


def xarray_indexable(array):
    if isinstance(array, np.ndarray):
        return NumpyIndexingAdapter(array)
    if isinstance(array, pd.Index):
        return PandasIndexAdapter(array)
    if isinstance(array, dask_array_type):
        return DaskIndexingAdapter(array)
    return array


def _outer_to_numpy_indexer(key, shape):
    """Convert an OuterIndexer into an indexer for NumPy.

    Parameters
    ----------
    key : OuterIndexer
        Outer indexing tuple to convert.
    shape : tuple
        Shape of the array subject to the indexing.

    Returns
    -------
    tuple
        Base tuple suitable for use to index a NumPy array.
    """
    if len([k for k in key if not isinstance(k, slice)]) <= 1:
        # If there is only one vector and all others are slice,
        # it can be safely used in mixed basic/advanced indexing.
        # Boolean index should already be converted to integer array.
        return tuple(key)

    n_dim = len([k for k in key if not isinstance(k, integer_types)])
    i_dim = 0
    new_key = []
    for k, size in zip(key, shape):
        if isinstance(k, integer_types):
            new_key.append(k)
        else:  # np.ndarray or slice
            if isinstance(k, slice):
                k = np.arange(*k.indices(size))
            assert k.dtype.kind in {'i', 'u'}
            shape = [(1,) * i_dim + (k.size, ) +
                     (1,) * (n_dim - i_dim - 1)]
            new_key.append(k.reshape(*shape))
            i_dim += 1
    return tuple(new_key)


class NumpyIndexingAdapter(utils.NDArrayMixin):
    """Wrap a NumPy array to use broadcasted indexing
    """
    def __init__(self, array):
        self.array = array

    def _ensure_ndarray(self, value):
        # We always want the result of indexing to be a NumPy array. If it's
        # not, then it really should be a 0d array. Doing the coercion here
        # instead of inside variable.as_compatible_data makes it less error
        # prone.
        if not isinstance(value, np.ndarray):
            value = utils.to_0d_array(value)
        return value

    def _indexing_array_and_key(self, key):
        if isinstance(key, OuterIndexer):
            key = _outer_to_numpy_indexer(key, self.array.shape)

        if isinstance(key, VectorizedIndexer):
            array = nputils.NumpyVIndexAdapter(self.array)
        else:
            array = self.array

        return array, to_tuple(key)

    def __getitem__(self, key):
        array, key = self._indexing_array_and_key(key)
        return self._ensure_ndarray(array[key])

    def __setitem__(self, key, value):
        array, key = self._indexing_array_and_key(key)
        array[key] = value


class DaskIndexingAdapter(utils.NDArrayMixin):
    """Wrap a dask array to support xarray-style indexing.
    """
    def __init__(self, array):
        """ This adapter is usually called in Variable.__getitem__ with
        array=Variable._broadcast_indexes
        """
        self.array = array

    def __getitem__(self, key):
        def to_int_tuple(key):
            # workaround for uint64 indexer (GH:1406)
            # TODO remove here after next dask release (0.15.3)
            return tuple([k.astype(int) if isinstance(k, np.ndarray)
                          else k for k in key])

        if isinstance(key, BasicIndexer):
            return self.array[to_int_tuple(key)]
        elif isinstance(key, VectorizedIndexer):
            return self.array.vindex[to_int_tuple(tuple(key))]
        else:
            assert isinstance(key, OuterIndexer)
            key = to_int_tuple(tuple(key))
            try:
                return self.array[key]
            except NotImplementedError:
                # manual orthogonal indexing.
                # TODO: port this upstream into dask in a saner way.
                value = self.array
                for axis, subkey in reversed(list(enumerate(key))):
                    value = value[(slice(None),) * axis + (subkey,)]
                return value

    def __setitem__(self, key, value):
        raise TypeError("this variable's data is stored in a dask array, "
                        'which does not support item assignment. To '
                        'assign to this variable, you must first load it '
                        'into memory explicitly using the .load() '
                        'method or accessing its .values attribute.')


class PandasIndexAdapter(utils.NDArrayMixin):
    """Wrap a pandas.Index to be better about preserving dtypes and to handle
    indexing by length 1 tuples like numpy
    """
    def __init__(self, array, dtype=None):
        self.array = utils.safe_cast_to_index(array)
        if dtype is None:
            if isinstance(array, pd.PeriodIndex):
                dtype = np.dtype('O')
            elif hasattr(array, 'categories'):
                # category isn't a real numpy dtype
                dtype = array.categories.dtype
            elif not utils.is_valid_numpy_dtype(array.dtype):
                dtype = np.dtype('O')
            else:
                dtype = array.dtype
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    def __array__(self, dtype=None):
        if dtype is None:
            dtype = self.dtype
        array = self.array
        if isinstance(array, pd.PeriodIndex):
            with suppress(AttributeError):
                # this might not be public API
                array = array.asobject
        return np.asarray(array.values, dtype=dtype)

    @property
    def shape(self):
        # .shape is broken on pandas prior to v0.15.2
        return (len(self.array),)

    def __getitem__(self, tuple_key):
        key = to_tuple(tuple_key)
        if isinstance(key, tuple) and len(key) == 1:
            # unpack key so it can index a pandas.Index object (pandas.Index
            # objects don't like tuples)
            key, = key

        if getattr(key, 'ndim', 0) > 1:  # Return np-array if multidimensional
            return NumpyIndexingAdapter(self.array.values)[tuple_key]

        result = self.array[key]

        if isinstance(result, pd.Index):
            result = PandasIndexAdapter(result, dtype=self.dtype)
        else:
            # result is a scalar
            if result is pd.NaT:
                # work around the impossibility of casting NaT with asarray
                # note: it probably would be better in general to return
                # pd.Timestamp rather np.than datetime64 but this is easier
                # (for now)
                result = np.datetime64('NaT', 'ns')
            elif isinstance(result, timedelta):
                result = np.timedelta64(getattr(result, 'value', result), 'ns')
            elif self.dtype != object:
                result = np.asarray(result, dtype=self.dtype)

            # as for numpy.ndarray indexing, we always want the result to be
            # a NumPy array.
            result = utils.to_0d_array(result)

        return result

    def __repr__(self):
        return ('%s(array=%r, dtype=%r)'
                % (type(self).__name__, self.array, self.dtype))
