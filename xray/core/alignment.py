import functools
import operator
import warnings
from collections import defaultdict

import numpy as np

from . import utils
from .pycompat import iteritems, OrderedDict
from .variable import as_variable, Variable, Coordinate


def align(*objects, **kwargs):
    """align(*objects, join='inner', copy=True)

    Given any number of Dataset and/or DataArray objects, returns new
    objects with aligned coordinates.

    Array from the aligned objects are suitable as input to mathematical
    operators, because along each dimension they are indexed by the same
    coordinates.

    Missing values (if ``join != 'inner'``) are filled with NaN.

    Parameters
    ----------
    *objects : Dataset or DataArray
        Objects to align.
    join : {'outer', 'inner', 'left', 'right'}, optional
        Method for joining the coordinates of the passed objects along each
        dimension:
         - 'outer': use the union of object coordinates
         - 'outer': use the intersection of object coordinates
         - 'left': use coordinates from the first object with each dimension
         - 'right': use coordinates from the last object with each dimension
    copy : bool, optional
        If `copy=True`, the returned objects contain all new variables. If
        `copy=False` and no reindexing is required then the aligned objects
        will include original variables.

    Returns
    -------
    aligned : same as *objects
        Tuple of objects with aligned coordinates.
    """
    # TODO: automatically align when doing math with dataset arrays?
    # TODO: change this to default to join='outer' like pandas?
    if 'join' not in kwargs:
        warnings.warn('using align without setting explicitly setting the '
                      "'join' keyword argument. In future versions of xray, "
                      "the default will likely change from join='inner' to "
                      "join='outer', to match pandas.",
                      FutureWarning, stacklevel=2)

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

    all_indexes = defaultdict(list)
    for obj in objects:
        for k, v in iteritems(obj.coords):
            all_indexes[k].append(v.to_index())

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
        Another dataset, with this dataset's data but replaced coordinates.
    """
    # build up indexers for assignment along each index
    to_indexers = {}
    from_indexers = {}
    for name, coord in iteritems(indexes):
        if name in indexers:
            target = utils.safe_cast_to_index(indexers[name])
            indexer = coord.get_indexer(target)

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
            if np.array_equal(from_indexers[name], np.arange(coord.size)):
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

