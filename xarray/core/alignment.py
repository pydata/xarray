import functools
import operator
from collections import defaultdict

import numpy as np
import pandas as pd

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
    result = []
    for obj in objects:
        valid_indexers = dict((k, v) for k, v in joined_indexes.items()
                              if k in obj.dims)
        result.append(obj.reindex(copy=copy, **valid_indexers))
    return tuple(result)


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


def align_variables(variables, join='outer', copy=False):
    """Align all DataArrays in the provided dict, leaving other values alone.
    """
    from .dataarray import DataArray
    from pandas import Series, DataFrame, Panel

    new_variables = OrderedDict(variables)
    # if an item is a Series / DataFrame / Panel, try and wrap it in a DataArray constructor
    new_variables.update((
        (k, DataArray(v)) for k, v in variables.items()
        if isinstance(v, (Series, DataFrame, Panel))
    ))

    alignable = [k for k, v in new_variables.items() if hasattr(v, 'indexes')]
    aligned = align(*[new_variables[a] for a in alignable], join=join, copy=copy)
    new_variables.update(zip(alignable, aligned))
    return new_variables


def reindex_variables(variables, indexes, indexers, method=None,
                      tolerance=None, copy=True):
    """Conform a dictionary of aligned variables onto a new set of variables,
    filling in missing values with NaN.

    WARNING: This method is not public API. Don't use it directly.

    Parameters
    ----------
    variables : dict-like
        Dictionary of xarray.Variable objects.
    indexes : dict-like
        Dictionary of xarray.Coordinate objects associated with variables.
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


def broadcast(*args):
    """Explicitly broadcast any number of DataArray or Dataset objects against
    one another.

    xarray objects automatically broadcast against each other in arithmetic
    operations, so this function should not be necessary for normal use.

    Parameters
    ----------
    *args: DataArray or Dataset objects
        Arrays to broadcast against each other.

    Returns
    -------
    broadcast: tuple of xarray objects
        The same data as the input arrays, but with additional dimensions
        inserted so that all data arrays have the same dimensions and shape.

    Raises
    ------
    ValueError
        If indexes on the different objects are not aligned.

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

    all_indexes = _get_all_indexes(args)
    for k, v in all_indexes.items():
        if not all(v[0].equals(vi) for vi in v[1:]):
            raise ValueError('cannot broadcast arrays: the %s index is not '
                             'aligned (use xarray.align first)' % k)

    common_coords = OrderedDict()
    dims_map = OrderedDict()
    for arg in args:
        for dim in arg.dims:
            if dim not in common_coords:
                common_coords[dim] = arg.coords[dim].variable
                dims_map[dim] = common_coords[dim].size

    def _broadcast_array(array):
        data = array.variable.expand_dims(dims_map)
        coords = OrderedDict(array.coords)
        coords.update(common_coords)
        dims = tuple(common_coords)
        return DataArray(data, coords, dims, name=array.name,
                         attrs=array.attrs, encoding=array.encoding)

    def _broadcast_dataset(ds):
        data_vars = OrderedDict()
        for k in ds.data_vars:
            data_vars[k] = ds.variables[k].expand_dims(dims_map)

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
    warnings.warn('xarray.broadcast_arrays is deprecated: use '
                  'xarray.broadcast instead', DeprecationWarning, stacklevel=2)
    return broadcast(*args)
