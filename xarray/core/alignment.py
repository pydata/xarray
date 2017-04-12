from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import operator
from collections import defaultdict

import numpy as np

from . import duck_array_ops, utils
from .common import _maybe_promote
from .indexing import get_indexer
from .pycompat import iteritems, OrderedDict, suppress
from .utils import is_full_slice, is_dict_like
from .variable import Variable, IndexVariable


def _get_joiner(join):
    if join == 'outer':
        return functools.partial(functools.reduce, operator.or_)
    elif join == 'inner':
        return functools.partial(functools.reduce, operator.and_)
    elif join == 'left':
        return operator.itemgetter(0)
    elif join == 'right':
        return operator.itemgetter(-1)
    else:
        raise ValueError('invalid value for join: %s' % join)


_DEFAULT_EXCLUDE = frozenset()


def align(*objects, **kwargs):
    """align(*objects, join='inner', copy=True, indexes=None,
             exclude=frozenset())

    Given any number of Dataset and/or DataArray objects, returns new
    objects with aligned indexes and dimension sizes.

    Array from the aligned objects are suitable as input to mathematical
    operators, because along each dimension they have the same index and size.

    Missing values (if ``join != 'inner'``) are filled with NaN.

    Parameters
    ----------
    *objects : Dataset or DataArray
        Objects to align.
    join : {'outer', 'inner', 'left', 'right'}, optional
        Method for joining the indexes of the passed objects along each
        dimension:
        - 'outer': use the union of object indexes
        - 'inner': use the intersection of object indexes
        - 'left': use indexes from the first object with each dimension
        - 'right': use indexes from the last object with each dimension
    copy : bool, optional
        If ``copy=True``, data in the return values is always copied. If
        ``copy=False`` and reindexing is unnecessary, or can be performed with
        only slice operations, then the output may share memory with the input.
        In either case, new xarray objects are always returned.
    exclude : sequence of str, optional
        Dimensions that must be excluded from alignment
    indexes : dict-like, optional
        Any indexes explicitly provided with the `indexes` argument should be
        used in preference to the aligned indexes.

    Returns
    -------
    aligned : same as *objects
        Tuple of objects with aligned coordinates.

    Raises
    ------
    ValueError
        If any dimensions without labels on the arguments have different sizes,
        or a different size than the size of the aligned dimension labels.
    """
    join = kwargs.pop('join', 'inner')
    copy = kwargs.pop('copy', True)
    indexes = kwargs.pop('indexes', None)
    exclude = kwargs.pop('exclude', _DEFAULT_EXCLUDE)
    if indexes is None:
        indexes = {}
    if kwargs:
        raise TypeError('align() got unexpected keyword arguments: %s'
                        % list(kwargs))

    if not indexes and len(objects) == 1:
        # fast path for the trivial case
        obj, = objects
        return (obj.copy(deep=copy),)

    all_indexes = defaultdict(list)
    unlabeled_dim_sizes = defaultdict(set)
    for obj in objects:
        for dim in obj.dims:
            if dim not in exclude:
                try:
                    index = obj.indexes[dim]
                except KeyError:
                    unlabeled_dim_sizes[dim].add(obj.sizes[dim])
                else:
                    all_indexes[dim].append(index)

    # We don't reindex over dimensions with all equal indexes for two reasons:
    # - It's faster for the usual case (already aligned objects).
    # - It ensures it's possible to do operations that don't require alignment
    #   on indexes with duplicate values (which cannot be reindexed with
    #   pandas). This is useful, e.g., for overwriting such duplicate indexes.
    joiner = _get_joiner(join)
    joined_indexes = {}
    for dim, matching_indexes in iteritems(all_indexes):
        if dim in indexes:
            index = utils.safe_cast_to_index(indexes[dim])
            if (any(not index.equals(other) for other in matching_indexes) or
                    dim in unlabeled_dim_sizes):
                joined_indexes[dim] = index
        else:
            if (any(not matching_indexes[0].equals(other)
                    for other in matching_indexes[1:]) or
                    dim in unlabeled_dim_sizes):
                index = joiner(matching_indexes)
                joined_indexes[dim] = index
            else:
                index = matching_indexes[0]

        if dim in unlabeled_dim_sizes:
            unlabeled_sizes = unlabeled_dim_sizes[dim]
            labeled_size = index.size
            if len(unlabeled_sizes | {labeled_size}) > 1:
                raise ValueError(
                    'arguments without labels along dimension %r cannot be '
                    'aligned because they have different dimension size(s) %r '
                    'than the size of the aligned dimension labels: %r'
                    % (dim, unlabeled_sizes, labeled_size))

    for dim in unlabeled_dim_sizes:
        if dim not in all_indexes:
            sizes = unlabeled_dim_sizes[dim]
            if len(sizes) > 1:
                raise ValueError(
                    'arguments without labels along dimension %r cannot be '
                    'aligned because they have different dimension sizes: %r'
                    % (dim, sizes))

    result = []
    for obj in objects:
        valid_indexers = {k: v for k, v in joined_indexes.items()
                          if k in obj.dims}
        if not valid_indexers:
            # fast path for no reindexing necessary
            new_obj = obj.copy(deep=copy)
        else:
            new_obj = obj.reindex(copy=copy, **valid_indexers)
        result.append(new_obj)

    return tuple(result)


def deep_align(objects, join='inner', copy=True, indexes=None,
               exclude=frozenset(), raise_on_invalid=True):
    """Align objects for merging, recursing into dictionary values.

    This function is not public API.
    """
    if indexes is None:
        indexes = {}

    def is_alignable(obj):
        return hasattr(obj, 'indexes') and hasattr(obj, 'reindex')

    positions = []
    keys = []
    out = []
    targets = []
    no_key = object()
    not_replaced = object()
    for n, variables in enumerate(objects):
        if is_alignable(variables):
            positions.append(n)
            keys.append(no_key)
            targets.append(variables)
            out.append(not_replaced)
        elif is_dict_like(variables):
            for k, v in variables.items():
                if is_alignable(v) and k not in indexes:
                    # Skip variables in indexes for alignment, because these
                    # should to be overwritten instead:
                    # https://github.com/pydata/xarray/issues/725
                    positions.append(n)
                    keys.append(k)
                    targets.append(v)
            out.append(OrderedDict(variables))
        elif raise_on_invalid:
            raise ValueError('object to align is neither an xarray.Dataset, '
                             'an xarray.DataArray nor a dictionary: %r'
                             % variables)
        else:
            out.append(variables)

    aligned = align(*targets, join=join, copy=copy, indexes=indexes,
                    exclude=exclude)

    for position, key, aligned_obj in zip(positions, keys, aligned):
        if key is no_key:
            out[position] = aligned_obj
        else:
            out[position][key] = aligned_obj

    # something went wrong: we should have replaced all sentinel values
    assert all(arg is not not_replaced for arg in out)

    return out


def reindex_like_indexers(target, other):
    """Extract indexers to align target with other.

    Not public API.

    Parameters
    ----------
    target : Dataset or DataArray
        Object to be aligned.
    other : Dataset or DataArray
        Object to be aligned with.

    Returns
    -------
    Dict[Any, pandas.Index] providing indexes for reindex keyword arguments.

    Raises
    ------
    ValueError
        If any dimensions without labels have different sizes.
    """
    indexers = {k: v for k, v in other.indexes.items() if k in target.dims}

    for dim in other.dims:
        if dim not in indexers and dim in target.dims:
            other_size = other.sizes[dim]
            target_size = target.sizes[dim]
            if other_size != target_size:
                raise ValueError('different size for unlabeled '
                                 'dimension on argument %r: %r vs %r'
                                 % (dim, other_size, target_size))
    return indexers


def reindex_variables(variables, sizes, indexes, indexers, method=None,
                      tolerance=None, copy=True):
    """Conform a dictionary of aligned variables onto a new set of variables,
    filling in missing values with NaN.

    Not public API.

    Parameters
    ----------
    variables : dict-like
        Dictionary of xarray.Variable objects.
    sizes : dict-like
        Dictionary from dimension names to integer sizes.
    indexes : dict-like
        Dictionary of xarray.IndexVariable objects associated with variables.
    indexers : dict
        Dictionary with keys given by dimension names and values given by
        arrays of coordinates tick labels. Any mis-matched coordinate values
        will be filled in with NaN, and any mis-matched dimension names will
        simply be ignored.
    method : {None, 'nearest', 'pad'/'ffill', 'backfill'/'bfill'}, optional
        Method to use for filling index values in ``indexers`` not found in
        this dataset:
          * None (default): don't fill gaps
          * pad / ffill: propagate last valid index value forward
          * backfill / bfill: propagate next valid index value backward
          * nearest: use nearest valid index value
    tolerance : optional
        Maximum distance between original and new labels for inexact matches.
        The values of the index at the matching locations most satisfy the
        equation ``abs(index[indexer] - target) <= tolerance``.
    copy : bool, optional
        If ``copy=True``, data in the return values is always copied. If
        ``copy=False`` and reindexing is unnecessary, or can be performed
        with only slice operations, then the output may share memory with
        the input. In either case, new xarray objects are always returned.

    Returns
    -------
    reindexed : OrderedDict
        Another dict, with the items in variables but replaced indexes.
    """
    # build up indexers for assignment along each dimension
    to_indexers = {}
    from_indexers = {}
    # size of reindexed dimensions
    new_sizes = {}

    for name, index in iteritems(indexes):
        if name in indexers:
            target = utils.safe_cast_to_index(indexers[name])
            if not index.is_unique:
                raise ValueError(
                    'cannot reindex or align along dimension %r because the '
                    'index has duplicate values' % name)
            indexer = get_indexer(index, target, method, tolerance)

            new_sizes[name] = len(target)
            # Note pandas uses negative values from get_indexer to signify
            # values that are missing in the index
            # The non-negative values thus indicate the non-missing values
            to_indexers[name] = indexer >= 0
            if to_indexers[name].all():
                # If an indexer includes no negative values, then the
                # assignment can be to a full-slice (which is much faster,
                # and means we won't need to fill in any missing values)
                to_indexers[name] = slice(None)

            from_indexers[name] = indexer[to_indexers[name]]
            if np.array_equal(from_indexers[name], np.arange(len(index))):
                # If the indexer is equal to the original index, use a full
                # slice object to speed up selection and so we can avoid
                # unnecessary copies
                from_indexers[name] = slice(None)

    for dim in sizes:
        if dim not in indexes and dim in indexers:
            existing_size = sizes[dim]
            new_size = utils.safe_cast_to_index(indexers[dim]).size
            if existing_size != new_size:
                raise ValueError(
                    'cannot reindex or align along dimension %r without an '
                    'index because its size %r is different from the size of '
                    'the new index %r' % (dim, existing_size, new_size))

    def any_not_full_slices(indexers):
        return any(not is_full_slice(idx) for idx in indexers)

    def var_indexers(var, indexers):
        return tuple(indexers.get(d, slice(None)) for d in var.dims)

    # create variables for the new dataset
    reindexed = OrderedDict()

    for dim, indexer in indexers.items():
        if dim in variables:
            var = variables[dim]
            args = (var.attrs, var.encoding)
        else:
            args = ()
        reindexed[dim] = IndexVariable((dim,), indexers[dim], *args)

    for name, var in iteritems(variables):
        if name not in indexers:
            assign_to = var_indexers(var, to_indexers)
            assign_from = var_indexers(var, from_indexers)

            if any_not_full_slices(assign_to):
                # there are missing values to in-fill
                data = var[assign_from].data
                dtype, fill_value = _maybe_promote(var.dtype)

                if isinstance(data, np.ndarray):
                    shape = tuple(new_sizes.get(dim, size)
                                  for dim, size in zip(var.dims, var.shape))
                    new_data = np.empty(shape, dtype=dtype)
                    new_data[...] = fill_value
                    # create a new Variable so we can use orthogonal indexing
                    # use fastpath=True to avoid dtype inference
                    new_var = Variable(var.dims, new_data, var.attrs,
                                       fastpath=True)
                    new_var[assign_to] = data

                else:  # dask array
                    data = data.astype(dtype, copy=False)
                    for axis, indexer in enumerate(assign_to):
                        if not is_full_slice(indexer):
                            indices = np.cumsum(indexer)[~indexer]
                            data = duck_array_ops.insert(
                                data, indices, fill_value, axis=axis)
                    new_var = Variable(var.dims, data, var.attrs,
                                       fastpath=True)

            elif any_not_full_slices(assign_from):
                # type coercion is not necessary as there are no missing
                # values
                new_var = var[assign_from]

            else:
                # no reindexing is necessary
                # here we need to manually deal with copying data, since
                # we neither created a new ndarray nor used fancy indexing
                new_var = var.copy(deep=copy)

            reindexed[name] = new_var
    return reindexed


def broadcast(*args, **kwargs):
    """Explicitly broadcast any number of DataArray or Dataset objects against
    one another.

    xarray objects automatically broadcast against each other in arithmetic
    operations, so this function should not be necessary for normal use.

    If no change is needed, the input data is returned to the output without
    being copied.

    Parameters
    ----------
    *args : DataArray or Dataset objects
        Arrays to broadcast against each other.
    exclude : sequence of str, optional
        Dimensions that must not be broadcasted

    Returns
    -------
    broadcast : tuple of xarray objects
        The same data as the input arrays, but with additional dimensions
        inserted so that all data arrays have the same dimensions and shape.

    Examples
    --------

    Broadcast two data arrays against one another to fill out their dimensions:

    >>> a = xr.DataArray([1, 2, 3], dims='x')
    >>> b = xr.DataArray([5, 6], dims='y')
    >>> a
    <xarray.DataArray (x: 3)>
    array([1, 2, 3])
    Coordinates:
      * x        (x) int64 0 1 2
    >>> b
    <xarray.DataArray (y: 2)>
    array([5, 6])
    Coordinates:
      * y        (y) int64 0 1
    >>> a2, b2 = xr.broadcast(a, b)
    >>> a2
    <xarray.DataArray (x: 3, y: 2)>
    array([[1, 1],
           [2, 2],
           [3, 3]])
    Coordinates:
      * x        (x) int64 0 1 2
      * y        (y) int64 0 1
    >>> b2
    <xarray.DataArray (x: 3, y: 2)>
    array([[5, 6],
           [5, 6],
           [5, 6]])
    Coordinates:
      * y        (y) int64 0 1
      * x        (x) int64 0 1 2

    Fill out the dimensions of all data variables in a dataset:

    >>> ds = xr.Dataset({'a': a, 'b': b})
    >>> ds2, = xr.broadcast(ds)  # use tuple unpacking to extract one dataset
    >>> ds2
    <xarray.Dataset>
    Dimensions:  (x: 3, y: 2)
    Coordinates:
      * x        (x) int64 0 1 2
      * y        (y) int64 0 1
    Data variables:
        a        (x, y) int64 1 1 2 2 3 3
        b        (x, y) int64 5 6 5 6 5 6
    """
    from .dataarray import DataArray
    from .dataset import Dataset

    exclude = kwargs.pop('exclude', None)
    if exclude is None:
        exclude = set()
    if kwargs:
        raise TypeError('broadcast() got unexpected keyword arguments: %s'
                        % list(kwargs))

    args = align(*args, join='outer', copy=False, exclude=exclude)

    common_coords = OrderedDict()
    dims_map = OrderedDict()
    for arg in args:
        for dim in arg.dims:
            if dim not in common_coords and dim not in exclude:
                dims_map[dim] = arg.sizes[dim]
                if dim in arg.coords:
                    common_coords[dim] = arg.coords[dim].variable

    def _set_dims(var):
        # Add excluded dims to a copy of dims_map
        var_dims_map = dims_map.copy()
        for dim in exclude:
            with suppress(ValueError):
                # ignore dim not in var.dims
                var_dims_map[dim] = var.shape[var.dims.index(dim)]

        return var.set_dims(var_dims_map)

    def _broadcast_array(array):
        data = _set_dims(array.variable)
        coords = OrderedDict(array.coords)
        coords.update(common_coords)
        return DataArray(data, coords, data.dims, name=array.name,
                         attrs=array.attrs, encoding=array.encoding)

    def _broadcast_dataset(ds):
        data_vars = OrderedDict(
            (k, _set_dims(ds.variables[k]))
            for k in ds.data_vars)
        coords = OrderedDict(ds.coords)
        coords.update(common_coords)
        return Dataset(data_vars, coords, ds.attrs)

    result = []
    for arg in args:
        if isinstance(arg, DataArray):
            result.append(_broadcast_array(arg))
        elif isinstance(arg, Dataset):
            result.append(_broadcast_dataset(arg))
        else:
            raise ValueError('all input must be Dataset or DataArray objects')

    return tuple(result)


def broadcast_arrays(*args):
    import warnings
    warnings.warn('xarray.broadcast_arrays is deprecated: use '
                  'xarray.broadcast instead', DeprecationWarning, stacklevel=2)
    return broadcast(*args)
