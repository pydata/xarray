import itertools

from common import ImplementsReduce
from ops import inject_reduce_methods
import variable
import dataset
import numpy as np


def unique_value_groups(ar):
    """Group an array by its unique values.

    Parameters
    ----------
    ar : array-like
        Input array. This will be flattened if it is not already 1-D.

    Returns
    -------
    values : np.ndarray
        Sorted, unique values as returned by `np.unique`.
    indices : list of lists of int
        Each element provides the integer indices in `ar` with values given by
        the corresponding value in `unique_values`.
    """
    values, inverse = np.unique(ar, return_inverse=True)
    groups = [[] for _ in range(len(values))]
    for n, g in enumerate(inverse):
        groups[g].append(n)
    return values, groups


def peek_at(iterable):
    """Returns the first value from iterable, as well as a new iterable with
    the same content as the original iterable
    """
    gen = iter(iterable)
    peek = gen.next()
    return peek, itertools.chain([peek], gen)


class GroupBy(object):
    """A object that implements the split-apply-combine pattern.

    Modeled after `pandas.GroupBy`. The `GroupBy` object can be iterated over
    (unique_value, grouped_array) pairs, but the main way to interact with a
    groupby object are with the `apply` or `reduce` methods. You can also
    directly call numpy methods like `mean` or `std`.

    You should create a GroupBy object by using the `DataArray.groupby` or
    `Dataset.groupby` methods.

    See Also
    --------
    XArray.groupby
    DataArray.groupby
    """
    def __init__(self, obj, group_coord, squeeze=True):
        """Create a GroupBy object

        Parameters
        ----------
        obj : Dataset or DataArray
            Object to group.
        group_coord : DataArray
            1-dimensional array with the group values.
        squeeze : boolean, optional
            If "group" is a coordinate of object, `squeeze` controls whether
            the subarrays have a dimension of length 1 along that coordinate or
            if the dimension is squeezed out.
        """
        if group_coord.ndim != 1:
            # TODO: remove this limitation?
            raise ValueError('`group_coord` must be 1 dimensional')

        self.obj = obj
        self.group_coord = group_coord
        self.group_dim, = group_coord.dimensions

        expected_size = dataset.as_dataset(obj).dimensions[self.group_dim]
        if group_coord.size != expected_size:
            raise ValueError('the group variable\'s length does not '
                             'match the length of this variable along its '
                             'dimension')

        if group_coord.name in obj.dimensions:
            # assume that group_coord already has sorted, unique values
            if group_coord.dimensions != (group_coord.name,):
                raise ValueError('`group_coord` is required to be a '
                                 'coordinate variable if `group_coord.name` '
                                 'is a dimension in `obj`')
            group_indices = np.arange(group_coord.size)
            if not squeeze:
                # group_indices = group_indices.reshape(-1, 1)
                # use slices to do views instead of fancy indexing
                group_indices = [slice(i, i + 1) for i in group_indices]
            unique_coord = group_coord
        else:
            # look through group_coord to find the unique values
            unique_values, group_indices = unique_value_groups(group_coord)
            # TODO: switch this to using the new DataArray constructor when we
            # get around to writing it:
            # unique_coord = xary.DataArray(unique_values, name=group_coord.name)
            variables = {group_coord.name: (group_coord.name, unique_values)}
            unique_coord = dataset.Dataset(variables)[group_coord.name]

        self.group_indices = group_indices
        self.unique_coord = unique_coord
        self._groups = None

    @property
    def groups(self):
        # provided to mimic pandas.groupby
        if self._groups is None:
            self._groups = dict(zip(self.unique_coord.values,
                                    self.group_indices))
        return self._groups

    def __len__(self):
        return self.unique_coord.size

    def __iter__(self):
        return itertools.izip(self.unique_coord.values, self._iter_grouped())

    def _iter_grouped(self):
        """Iterate over each element in this group"""
        for indices in self.group_indices:
            yield self.obj.indexed(**{self.group_dim: indices})

    def _infer_concat_args(self, applied_example):
        if self.group_dim in applied_example.dimensions:
            concat_dim = self.group_coord
            indexers = self.group_indices
        else:
            concat_dim = self.unique_coord
            indexers = np.arange(self.unique_coord.size)
        return concat_dim, indexers

    @property
    def _combine(self):
        return type(self.obj).concat


class ArrayGroupBy(GroupBy, ImplementsReduce):
    """GroupBy object specialized to grouping DataArray objects
    """
    def _iter_grouped_shortcut(self):
        """Fast version of `_iter_grouped` that yields XArrays without metadata
        """
        array = variable.as_variable(self.obj)

        # build the new dimensions
        if isinstance(self.group_indices[0], int):
            # group_dim is squeezed out
            dims = tuple(d for d in array.dimensions if d != self.group_dim)
        else:
            dims = array.dimensions

        # slice the data and build the new Arrays directly
        indexer = [slice(None)] * array.ndim
        group_axis = array.get_axis_num(self.group_dim)
        for indices in self.group_indices:
            indexer[group_axis] = indices
            data = array.values[tuple(indexer)]
            yield variable.Variable(dims, data)

    def _combine_shortcut(self, applied, concat_dim, indexers):
        stacked = variable.Variable.concat(
            applied, concat_dim, indexers, shortcut=True)
        stacked.attrs.update(self.obj.attrs)

        name = self.obj.name
        ds = self.obj.dataset.unselect(name)
        ds[concat_dim.name] = concat_dim
        # remove extraneous dimensions
        for dim in self.obj.dimensions:
            if dim not in stacked.dimensions:
                del ds[dim]
        ds[name] = stacked
        return ds[name]

    def _restore_dim_order(self, stacked, concat_dim):
        def lookup_order(dimension):
            if dimension == self.group_coord.name:
                dimension, = concat_dim.dimensions
            if dimension in self.obj.dimensions:
                axis = self.obj.get_axis_num(dimension)
            else:
                axis = 1e6  # some arbitrarily high value
            return axis

        new_order = sorted(stacked.dimensions, key=lookup_order)
        return stacked.transpose(*new_order)

    def apply(self, func, shortcut=False, **kwargs):
        """Apply a function over each array in the group and concatenate them
        together into a new array.

        `func` is called like `func(ar, *args, **kwargs)` for each array `ar`
        in this group.

        Apply uses heuristics (like `pandas.GroupBy.apply`) to figure out how
        to stack together the array. The rule is:
        1. If the dimension along which the group coordinate is defined is
           still in the first grouped array after applying `func`, then stack
           over this dimension.
        2. Otherwise, stack over the new dimension given by name of this
           grouping (the argument to the `groupby` function).

        Parameters
        ----------
        func : function
            Callable to apply to each array.
        shortcut : bool, optional
            Whether or not to shortcut evaluation under the assumptions that:
            (1) The action of `func` does not depend on any of the array
                metadata (attributes, indices or other contained arrays) but
                only on the data and dimensions.
            (2) The action of `func` creates arrays with homogeneous metadata,
                that is, with the same dimensions and attributes.
            If these conditions are satisfied `shortcut` provides significant
            speedup. This should be the case for many common groupby operations
            (e.g., applying numpy ufuncs).
        **kwargs
            Used to call `func(ar, **kwargs)` for each array `ar.

        Returns
        -------
        applied : DataArray
            The result of splitting, applying and combining this array.
        """
        if shortcut:
            grouped = self._iter_grouped_shortcut()
        else:
            grouped = self._iter_grouped()
        applied = (func(arr, **kwargs) for arr in grouped)

        # peek at applied to determine which coordinate to stack over
        applied_example, applied = peek_at(applied)
        concat_dim, indexers = self._infer_concat_args(applied_example)
        if shortcut:
            combined = self._combine_shortcut(applied, concat_dim, indexers)
        else:
            combined = self._combine(applied, concat_dim, indexers)

        reordered = self._restore_dim_order(combined, concat_dim)
        return reordered

    def reduce(self, func, dimension=None, axis=None, shortcut=True,
               **kwargs):
        """Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : function
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of collapsing an
            np.ndarray over an integer valued axis.
        dimension : str or sequence of str, optional
            Dimension(s) over which to apply `func`.
        axis : int or sequence of int, optional
            Axis(es) over which to apply `func`. Only one of the 'dimension'
            and 'axis' arguments can be supplied. If neither are supplied, then
            `func` is calculated over all dimension for each group item.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : Array
            Array with summarized data and the indicated dimension(s)
            removed.
        """
        def reduce_array(ar):
            return ar.reduce(func, dimension, axis, **kwargs)
        return self.apply(reduce_array, shortcut=shortcut)

    _reduce_method_docstring = \
        """Reduce the items in this group by applying `{name}` along some
        dimension(s).

        Parameters
        ----------
        dimension : str or sequence of str, optional
            Dimension(s) over which to apply `{name}`.
        axis : int or sequence of int, optional
            Axis(es) over which to apply `{name}`. Only one of the 'dimension'
            and 'axis' arguments can be supplied. If neither are supplied, then
            `{name}` is calculated over all dimension for each group item.
        **kwargs : dict
            Additional keyword arguments passed on to `{name}`.

        Returns
        -------
        reduced : {cls}
            New {cls} object with `{name}` applied to its data and the
            indicated dimension(s) removed.
        """

inject_reduce_methods(ArrayGroupBy)


class DatasetGroupBy(GroupBy):
    def apply(self, func, **kwargs):
        """Apply a function over each Dataset in the group and concatenate them
        together into a new Dataset.

        `func` is called like `func(ds, *args, **kwargs)` for each dataset `ds`
        in this group.

        Apply uses heuristics (like `pandas.GroupBy.apply`) to figure out how
        to stack together the datasets. The rule is:
        1. If the dimension along which the group coordinate is defined is
           still in the first grouped item after applying `func`, then stack
           over this dimension.
        2. Otherwise, stack over the new dimension given by name of this
           grouping (the argument to the `groupby` function).

        Parameters
        ----------
        func : function
            Callable to apply to each sub-dataset.
        **kwargs
            Used to call `func(ds, **kwargs)` for each sub-dataset `ar`.

        Returns
        -------
        applied : Dataset
            The result of splitting, applying and combining this dataset.
        """
        applied = [func(ds, **kwargs) for ds in self._iter_grouped()]
        concat_dim, indexers = self._infer_concat_args(applied[0])
        combined = self._combine(applied, concat_dim, indexers)
        return combined
