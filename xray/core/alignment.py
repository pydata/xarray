import functools
import operator
from collections import defaultdict

import numpy as np

from . import ops, utils
from .common import _maybe_promote
from .pycompat import iteritems, OrderedDict, reduce
from .utils import is_full_slice
from .variable import as_variable, Variable, Coordinate, broadcast_variables


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


def reindex_variables(variables, indexes, indexers, method=None, copy=True):
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
          * default: don't fill gaps
          * pad / ffill: propgate last valid index value forward
          * backfill / bfill: propagate next valid index value backward
          * nearest: use nearest valid index value
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
    for name, index in iteritems(indexes):
        to_shape[name] = index.size
        if name in indexers:
            target = utils.safe_cast_to_index(indexers[name])
            indexer = index.get_indexer(target, method=method)

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


def concat(objs, dim='concat_dim', indexers=None, mode='different',
           concat_over=None, compat='equals'):
    """Concatenate xray objects along a new or existing dimension.

    Parameters
    ----------
    objs : sequence of Dataset and DataArray objects
        xray objects to concatenate together. Each object is expected to
        consist of variables and coordinates with matching shapes except for
        along the concatenated dimension.
    dim : str or DataArray or Index, optional
        Name of the dimension to concatenate along. This can either be a new
        dimension name, in which case it is added along axis=0, or an existing
        dimension name, in which case the location of the dimension is
        unchanged. If dimension is provided as a DataArray or Index, its name
        is used as the dimension to concatenate along and the values are added
        as a coordinate.
    indexers : None or iterable of indexers, optional
        Iterable of indexers of the same length as datasets which
        specifies how to assign variables from each dataset along the given
        dimension. If not supplied, indexers is inferred from the length of
        each variable along the dimension, and the variables are stacked in
        the given order.
    mode : {'minimal', 'different', 'all'}, optional
        Decides which variables are concatenated.  Choices are 'minimal'
        in which only variables in which dimension already appears are
        included, 'different' in which all variables which are not equal
        (ignoring attributes) across all datasets are concatenated (as well
        as all for which dimension already appears), and 'all' for which all
        variables are concatenated.
    concat_over : None or str or iterable of str, optional
        Names of additional variables to concatenate, in which the provided
        parameter ``dim`` does not already appear as a dimension. The default
        value includes all data variables.
    compat : {'equals', 'identical'}, optional
        String indicating how to compare non-concatenated variables and
        dataset global attributes for potential conflicts. 'equals' means
        that all variable values and dimensions must be the same;
        'identical' means that variable attributes and global attributes
        must also be equal.

    Returns
    -------
    concatenated : type of objs

    See also
    --------
    auto_combine
    """
    # TODO: add join and ignore_index arguments copied from pandas.concat
    # TODO: support concatenating scaler coordinates even if the concatenated
    # dimension already exists
    try:
        first_obj, objs = utils.peek_at(objs)
    except StopIteration:
        raise ValueError('must supply at least one object to concatenate')
    cls = type(first_obj)
    return cls._concat(objs, dim, indexers, mode, concat_over, compat)


def _auto_concat(datasets, dim=None):
    if len(datasets) == 1:
        return datasets[0]
    else:
        if dim is None:
            ds0 = datasets[0]
            ds1 = datasets[1]
            concat_dims = set(ds0.dims)
            if ds0.dims != ds1.dims:
                dim_tuples = set(ds0.dims.items()) - set(ds1.dims.items())
                concat_dims = set(i for i, _ in dim_tuples)
            if len(concat_dims) > 1:
                concat_dims = set(d for d in concat_dims
                                  if not ds0[d].equals(ds1[d]))
            if len(concat_dims) > 1:
                raise ValueError('too many different dimensions to '
                                 'concatenate: %s' % concat_dims)
            elif len(concat_dims) == 0:
                raise ValueError('cannot infer dimension to concatenate: '
                                 'supply the ``concat_dim`` argument '
                                 'explicitly')
            dim, = concat_dims
        return concat(datasets, dim=dim)


def auto_combine(datasets, concat_dim=None):
    """Attempt to auto-magically combine the given datasets into one.

    This method attempts to combine a list of datasets into a single entity by
    inspecting metadata and using a combination of concat and merge.

    It does not concatenate along more than one dimension or align or sort data
    under any circumstances. It will fail in complex cases, for which you
    should use ``concat`` and ``merge`` explicitly.

    When ``auto_combine`` may succeed:

    * You have N years of data and M data variables. Each combination of a
      distinct time period and test of data variables is saved its own dataset.

    Examples of when ``auto_combine`` fails:

    * In the above scenario, one file is missing, containing the data for one
      year's data for one variable.
    * In the most recent year, there is an additional data variable.
    * Your data includes "time" and "station" dimensions, and each year's data
      has a different set of stations.

    Parameters
    ----------
    datasets : sequence of xray.Dataset
        Dataset objects to merge.
    concat_dim : str or DataArray or Index, optional
        Dimension along which to concatenate variables, as used by
        :py:func:`xray.concat`. You only need to provide this argument if the
        dimension along which you want to concatenate is not a dimension in
        the original datasets, e.g., if you want to stack a collection of
        2D arrays along a third dimension.

    Returns
    -------
    combined : xray.Dataset

    See also
    --------
    concat
    Dataset.merge
    """
    from toolz import itertoolz
    grouped = itertoolz.groupby(lambda ds: tuple(sorted(ds.data_vars)),
                                datasets).values()
    concatenated = [_auto_concat(ds, dim=concat_dim) for ds in grouped]
    merged = reduce(lambda ds, other: ds.merge(other), concatenated)
    return merged


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
