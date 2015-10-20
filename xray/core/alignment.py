import functools
import operator
import pandas as pd
from collections import defaultdict

import numpy as np

from . import ops, utils
from .common import _maybe_promote
from .pycompat import iteritems, OrderedDict
from .utils import is_full_slice
from .variable import Variable, Coordinate, broadcast_variables


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


def _get_all_indexes(objects, exclude=set()):
    all_indexes = defaultdict(list)
    for obj in objects:
        for k, v in iteritems(obj.indexes):
            if k not in exclude:
                all_indexes[k].append(v)
    return all_indexes


def _join_indexes(join, objects, exclude=set()):
    joiner = _get_joiner(join)
    indexes = _get_all_indexes(objects, exclude=exclude)
    # exclude dimensions with all equal indices (the usual case) to avoid
    # unnecessary reindexing work.
    # TODO: don't bother to check equals for left or right joins
    joined_indexes = dict((k, joiner(v)) for k, v in iteritems(indexes)
                          if any(not v[0].equals(idx) for idx in v[1:]))
    return joined_indexes


def align(*objects, **kwargs):
    """align(*objects, join='inner', copy=True)

    Given any number of Dataset and/or DataArray objects, returns new
    objects with aligned indexes.

    Array from the aligned objects are suitable as input to mathematical
    operators, because along each dimension they have the same indexes.

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
        If ``copy=True``, the returned objects contain all new variables. If
        ``copy=False`` and no reindexing is required then the aligned objects
        will include original variables.

    Returns
    -------
    aligned : same as *objects
        Tuple of objects with aligned coordinates.
    """
    join = kwargs.pop('join', 'inner')
    copy = kwargs.pop('copy', True)
    if kwargs:
        raise TypeError('align() got unexpected keyword arguments: %s'
                        % list(kwargs))

    joined_indexes = _join_indexes(join, objects)
    return tuple(obj.reindex(copy=copy, **joined_indexes) for obj in objects)


def partial_align(*objects, **kwargs):
    """partial_align(*objects, join='inner', copy=True, exclude=set()

    Like align, but don't align along dimensions in exclude. Not public API.
    """
    join = kwargs.pop('join', 'inner')
    copy = kwargs.pop('copy', True)
    exclude = kwargs.pop('exclude', set())
    assert not kwargs
    joined_indexes = _join_indexes(join, objects, exclude=exclude)
    return tuple(obj.reindex(copy=copy, **joined_indexes) for obj in objects)


def reindex_variables(variables, indexes, indexers, method=None,
                      tolerance=None, copy=True):
    """Conform a dictionary of aligned variables onto a new set of variables,
    filling in missing values with NaN.

    WARNING: This method is not public API. Don't use it directly.

    Parameters
    ----------
    variables : dict-like
        Dictionary of xray.Variable objects.
    indexes : dict-like
        Dictionary of xray.Coordinate objects associated with variables.
    indexers : dict
        Dictionary with keys given by dimension names and values given by
        arrays of coordinates tick labels. Any mis-matched coordinate values
        will be filled in with NaN, and any mis-matched dimension names will
        simply be ignored.
    method : {None, 'nearest', 'pad'/'ffill', 'backfill'/'bfill'}, optional
        Method to use for filling index values in ``indexers`` not found in
        this dataset:
          * None (default): don't fill gaps
          * pad / ffill: propgate last valid index value forward
          * backfill / bfill: propagate next valid index value backward
          * nearest: use nearest valid index value
    tolerance : optional
        Maximum distance between original and new labels for inexact matches.
        The values of the index at the matching locations most satisfy the
        equation ``abs(index[indexer] - target) <= tolerance``.
    copy : bool, optional
        If `copy=True`, the returned dataset contains only copied
        variables. If `copy=False` and no reindexing is required then
        original variables from this dataset are returned.

    Returns
    -------
    reindexed : OrderedDict
        Another dict, with the items in variables but replaced indexes.
    """
    # build up indexers for assignment along each index
    to_indexers = {}
    to_shape = {}
    from_indexers = {}

    # for compat with older versions of pandas that don't support tolerance
    get_indexer_kwargs = {}
    if tolerance is not None:
        if pd.__version__ < '0.17':
            raise NotImplementedError(
                'the tolerance argument requires pandas v0.17 or newer')
        get_indexer_kwargs['tolerance'] = tolerance

    for name, index in iteritems(indexes):
        to_shape[name] = index.size
        if name in indexers:
            target = utils.safe_cast_to_index(indexers[name])
            indexer = index.get_indexer(target, method=method,
                                        **get_indexer_kwargs)

            to_shape[name] = len(target)
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
            if np.array_equal(from_indexers[name], np.arange(index.size)):
                # If the indexer is equal to the original index, use a full
                # slice object to speed up selection and so we can avoid
                # unnecessary copies
                from_indexers[name] = slice(None)

    def any_not_full_slices(indexers):
        return any(not is_full_slice(idx) for idx in indexers)

    def var_indexers(var, indexers):
        return tuple(indexers.get(d, slice(None)) for d in var.dims)

    # create variables for the new dataset
    reindexed = OrderedDict()
    for name, var in iteritems(variables):
        if name in indexers:
            # no need to copy, because index data is immutable
            new_var = Coordinate(var.dims, indexers[name], var.attrs,
                                 var.encoding)
        else:
            assign_to = var_indexers(var, to_indexers)
            assign_from = var_indexers(var, from_indexers)

            if any_not_full_slices(assign_to):
                # there are missing values to in-fill
                data = var[assign_from].data
                dtype, fill_value = _maybe_promote(var.dtype)

                if isinstance(data, np.ndarray):
                    shape = tuple(to_shape[dim] for dim in var.dims)
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
                            data = ops.insert(data, indices, fill_value,
                                              axis=axis)
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
                new_var = var.copy() if copy else var

        reindexed[name] = new_var
    return reindexed


def broadcast_arrays(*args):
    """Explicitly broadcast any number of DataArrays against one another.

    xray objects automatically broadcast against each other in arithmetic
    operations, so this function should not be necessary for normal use.

    Parameters
    ----------
    *args: DataArray
        Arrays to broadcast against each other.

    Returns
    -------
    broadcast: tuple of DataArray
        The same data as the input arrays, but with additional dimensions
        inserted so that all arrays have the same dimensions and shape.

    Raises
    ------
    ValueError
        If indexes on the different arrays are not aligned.
    """
    # TODO: fixme for coordinate arrays

    from .dataarray import DataArray

    all_indexes = _get_all_indexes(args)
    for k, v in all_indexes.items():
        if not all(v[0].equals(vi) for vi in v[1:]):
            raise ValueError('cannot broadcast arrays: the %s index is not '
                             'aligned (use xray.align first)' % k)

    vars = broadcast_variables(*[a.variable for a in args])
    indexes = dict((k, all_indexes[k][0]) for k in vars[0].dims)

    arrays = []
    for a, v in zip(args, vars):
        arr = DataArray(v.values, indexes, v.dims, a.name, a.attrs, a.encoding)
        for k, v in a.coords.items():
            arr.coords[k] = v
        arrays.append(arr)

    return tuple(arrays)
