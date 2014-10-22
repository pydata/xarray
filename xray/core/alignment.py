import functools
import operator
import warnings
from collections import defaultdict

import numpy as np

from . import utils
from .pycompat import iteritems, OrderedDict
from .variable import as_variable, Variable, Coordinate, broadcast_variables


def _get_all_indexes(objects):
    all_indexes = defaultdict(list)
    for obj in objects:
        for k, v in iteritems(obj.indexes):
            all_indexes[k].append(v)
    return all_indexes


def align(*objects, **kwargs):
    """align(*objects, join='inner', copy=True)

    Given any number of Dataset and/or DataArray objects, returns new
    objects with aligned indexes.

    Array from the aligned objects are suitable as input to mathematical
    operators, because along each dimension they are indexed by the same
    indexes.

    Missing values (if ``join != 'inner'``) are filled with NaN.

    Parameters
    ----------
    *objects : Dataset or DataArray
        Objects to align.
    join : {'outer', 'inner', 'left', 'right'}, optional
        Method for joining the indexes of the passed objects along each
        dimension:
         - 'outer': use the union of object indexes
         - 'outer': use the intersection of object indexes
         - 'left': use indexes from the first object with each dimension
         - 'right': use indexes from the last object with each dimension
    copy : bool, optional
        If `copy=True`, the returned objects contain all new variables. If
        `copy=False` and no reindexing is required then the aligned objects
        will include original variables.

    Returns
    -------
    aligned : same as *objects
        Tuple of objects with aligned coordinates.
    """
    join = kwargs.pop('join', 'inner')
    copy = kwargs.pop('copy', True)

    if join == 'outer':
        join_indices = functools.partial(functools.reduce, operator.or_)
    elif join == 'inner':
        join_indices = functools.partial(functools.reduce, operator.and_)
    elif join == 'left':
        join_indices = operator.itemgetter(0)
    elif join == 'right':
        join_indices = operator.itemgetter(-1)

    all_indexes = _get_all_indexes(objects)

    # Exclude dimensions with all equal indices to avoid unnecessary reindexing
    # work.
    joined_indexes = dict((k, join_indices(v)) for k, v in iteritems(all_indexes)
                          if any(not v[0].equals(idx) for idx in v[1:]))

    return tuple(obj.reindex(copy=copy, **joined_indexes) for obj in objects)


def reindex_variables(variables, indexes, indexers, copy=True):
    """Conform a dictionary of variables onto a new set of coordinates, filling
    in missing values with NaN.

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
    from_indexers = {}
    for name, index in iteritems(indexes):
        index = utils.safe_cast_to_index(index)
        if name in indexers:
            target = utils.safe_cast_to_index(indexers[name])
            indexer = index.get_indexer(target)

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

    def is_full_slice(idx):
        return isinstance(idx, slice) and idx == slice(None)

    def any_not_full_slices(indexers):
        return any(not is_full_slice(idx) for idx in indexers)

    def var_indexers(var, indexers):
        return tuple(indexers.get(d, slice(None)) for d in var.dims)

    def get_fill_value_and_dtype(dtype):
        # N.B. these casting rules should match pandas
        if np.issubdtype(dtype, np.datetime64):
            fill_value = np.datetime64('NaT')
        elif any(np.issubdtype(dtype, t) for t in (int, float)):
            # convert to floating point so NaN is valid
            dtype = float
            fill_value = np.nan
        else:
            dtype = object
            fill_value = np.nan
        return fill_value, dtype

    # create variables for the new dataset
    reindexed = OrderedDict()
    for name, var in iteritems(variables):
        if name in indexers:
            new_var = indexers[name]
            if not hasattr(new_var, 'dims') or not hasattr(new_var, 'values'):
                new_var = Coordinate(var.dims, new_var, var.attrs,
                                     var.encoding)
            elif copy:
                new_var = as_variable(new_var).copy()
        else:
            assign_to = var_indexers(var, to_indexers)
            assign_from = var_indexers(var, from_indexers)

            if any_not_full_slices(assign_to):
                # there are missing values to in-fill
                fill_value, dtype = get_fill_value_and_dtype(var.dtype)
                shape = tuple(length if is_full_slice(idx) else idx.size
                              for idx, length in zip(assign_to, var.shape))
                data = np.empty(shape, dtype=dtype)
                data[:] = fill_value
                # create a new Variable so we can use orthogonal indexing
                new_var = Variable(var.dims, data, var.attrs)
                new_var[assign_to] = var[assign_from].values
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
        parameter ``dim`` does not already appear as a dimension.
    compat : {'equals', 'identical'}, optional
        String indicating how to compare non-concatenated variables and
        dataset global attributes for potential conflicts. 'equals' means
        that all variable values and dimensions must be the same;
        'identical' means that variable attributes and global attributes
        must also be equal.

    Returns
    -------
    concatenated : type of objs
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
