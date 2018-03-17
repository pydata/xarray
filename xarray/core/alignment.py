from __future__ import absolute_import, division, print_function

import functools
import operator
import warnings
from collections import defaultdict

import numpy as np

from . import utils
from .indexing import get_indexer_nd
from .pycompat import OrderedDict, iteritems, suppress
from .utils import is_dict_like, is_full_slice
from .variable import IndexVariable


def _get_joiner(join):
    if join == 'outer':
        return functools.partial(functools.reduce, operator.or_)
    elif join == 'inner':
        return functools.partial(functools.reduce, operator.and_)
    elif join == 'left':
        return operator.itemgetter(0)
    elif join == 'right':
        return operator.itemgetter(-1)
    elif join == 'exact':
        # We cannot return a function to "align" in this case, because it needs
        # access to the dimension name to give a good error message.
        return None
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
    join : {'outer', 'inner', 'left', 'right', 'exact'}, optional
        Method for joining the indexes of the passed objects along each
        dimension:
        - 'outer': use the union of object indexes
        - 'inner': use the intersection of object indexes
        - 'left': use indexes from the first object with each dimension
        - 'right': use indexes from the last object with each dimension
        - 'exact': instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
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
                if join == 'exact':
                    raise ValueError(
                        'indexes along dimension {!r} are not equal'
                        .format(dim))
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
        new_obj.encoding = obj.encoding
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
    from .dataarray import DataArray

    # build up indexers for assignment along each dimension
    int_indexers = {}
    targets = {}
    masked_dims = set()
    unchanged_dims = set()

    # size of reindexed dimensions
    new_sizes = {}

    for name, index in iteritems(indexes):
        if name in indexers:
            if not index.is_unique:
                raise ValueError(
                    'cannot reindex or align along dimension %r because the '
                    'index has duplicate values' % name)

            target = utils.safe_cast_to_index(indexers[name])
            new_sizes[name] = len(target)

            int_indexer = get_indexer_nd(index, target, method, tolerance)

            # We uses negative values from get_indexer_nd to signify
            # values that are missing in the index.
            if (int_indexer < 0).any():
                masked_dims.add(name)
            elif np.array_equal(int_indexer, np.arange(len(index))):
                unchanged_dims.add(name)

            int_indexers[name] = int_indexer
            targets[name] = target

    for dim in sizes:
        if dim not in indexes and dim in indexers:
            existing_size = sizes[dim]
            new_size = indexers[dim].size
            if existing_size != new_size:
                raise ValueError(
                    'cannot reindex or align along dimension %r without an '
                    'index because its size %r is different from the size of '
                    'the new index %r' % (dim, existing_size, new_size))

    # create variables for the new dataset
    reindexed = OrderedDict()

    for dim, indexer in indexers.items():
        if isinstance(indexer, DataArray) and indexer.dims != (dim,):
            warnings.warn(
                "Indexer has dimensions {0:s} that are different "
                "from that to be indexed along {1:s}. "
                "This will behave differently in the future.".format(
                    str(indexer.dims), dim),
                FutureWarning, stacklevel=3)

        if dim in variables:
            var = variables[dim]
            args = (var.attrs, var.encoding)
        else:
            args = ()
        reindexed[dim] = IndexVariable((dim,), indexers[dim], *args)

    for name, var in iteritems(variables):
        if name not in indexers:
            key = tuple(slice(None)
                        if d in unchanged_dims
                        else int_indexers.get(d, slice(None))
                        for d in var.dims)
            needs_masking = any(d in masked_dims for d in var.dims)

            if needs_masking:
                new_var = var._getitem_with_mask(key)
            elif all(is_full_slice(k) for k in key):
                # no reindexing necessary
                # here we need to manually deal with copying data, since
                # we neither created a new ndarray nor used fancy indexing
                new_var = var.copy(deep=copy)
            else:
                new_var = var[key]

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
