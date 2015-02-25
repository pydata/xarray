import functools
import numpy as np
import pandas as pd

from . import ops
from .alignment import concat
from .common import ImplementsArrayReduce, ImplementsDatasetReduce
from .pycompat import zip
from .utils import peek_at, maybe_wrap_array
from .variable import Variable, Coordinate


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
    inverse, values = pd.factorize(ar, sort=True)
    groups = [[] for _ in range(len(values))]
    for n, g in enumerate(inverse):
        if g >= 0:
            # pandas uses -1 to mark NaN, but doesn't include them in values
            groups[g].append(n)
    return values, groups


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
    Dataset.groupby
    DataArray.groupby
    """
    def __init__(self, obj, group, squeeze=True):
        """Create a GroupBy object

        Parameters
        ----------
        obj : Dataset or DataArray
            Object to group.
        group : DataArray or Coordinate
            1-dimensional array with the group values.
        squeeze : boolean, optional
            If "group" is a coordinate of object, `squeeze` controls whether
            the subarrays have a dimension of length 1 along that coordinate or
            if the dimension is squeezed out.
        """
        if group.ndim != 1:
            # TODO: remove this limitation?
            raise ValueError('`group` must be 1 dimensional')
        if getattr(group, 'name', None) is None:
            raise ValueError('`group` must have a name')
        if not hasattr(group, 'dims'):
            raise ValueError("`group` must have a 'dims' attribute")

        self.obj = obj
        self.group = group
        self.group_dim, = group.dims

        from .dataset import as_dataset
        expected_size = as_dataset(obj).dims[self.group_dim]
        if group.size != expected_size:
            raise ValueError('the group variable\'s length does not '
                             'match the length of this variable along its '
                             'dimension')

        if group.name in obj.dims:
            # assume that group already has sorted, unique values
            if group.dims != (group.name,):
                raise ValueError('`group` is required to be a coordinate if '
                                 '`group.name` is a dimension in `obj`')
            group_indices = np.arange(group.size)
            if not squeeze:
                # group_indices = group_indices.reshape(-1, 1)
                # use slices to do views instead of fancy indexing
                group_indices = [slice(i, i + 1) for i in group_indices]
            unique_coord = group
        else:
            # look through group to find the unique values
            unique_values, group_indices = unique_value_groups(group)
            unique_coord = Coordinate(group.name, unique_values)

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
        return zip(self.unique_coord.values, self._iter_grouped())

    def _iter_grouped(self):
        """Iterate over each element in this group"""
        for indices in self.group_indices:
            yield self.obj.isel(**{self.group_dim: indices})

    def _infer_concat_args(self, applied_example):
        if self.group_dim in applied_example.dims:
            concat_dim = self.group
            indexers = self.group_indices
        else:
            concat_dim = self.unique_coord
            indexers = np.arange(self.unique_coord.size)
        return concat_dim, indexers

    @staticmethod
    def _binary_op(f, reflexive=False):
        @functools.wraps(f)
        def func(self, other):
            g = f if not reflexive else lambda x, y: f(y, x)
            applied = self._yield_binary_applied(g, other)
            return self._concat(applied)
        return func

    def _yield_binary_applied(self, func, other):
        for group_value, obj in self:
            try:
                other_sel = other.sel(**{self.group.name: group_value})
            except AttributeError:
                raise TypeError('GroupBy objects only support arithmetic '
                                'when the other argument is a Dataset or '
                                'DataArray')
            yield func(obj, other_sel)


class ArrayGroupBy(GroupBy, ImplementsArrayReduce):
    """GroupBy object specialized to grouping DataArray objects
    """
    def _iter_grouped_shortcut(self):
        """Fast version of `_iter_grouped` that yields Variables without
        metadata
        """
        from .variable import as_variable
        array = as_variable(self.obj)

        # build the new dimensions
        if isinstance(self.group_indices[0], (int, np.integer)):
            # group_dim is squeezed out
            dims = tuple(d for d in array.dims if d != self.group_dim)
        else:
            dims = array.dims

        # slice the data and build the new Arrays directly
        indexer = [slice(None)] * array.ndim
        group_axis = array.get_axis_num(self.group_dim)
        for indices in self.group_indices:
            indexer[group_axis] = indices
            data = array.values[tuple(indexer)]
            yield Variable(dims, data)

    def _concat_shortcut(self, applied, concat_dim, indexers):
        stacked = Variable.concat(
            applied, concat_dim, indexers, shortcut=True)
        stacked.attrs.update(self.obj.attrs)

        name = self.obj.name
        ds = self.obj._dataset.drop(name)
        ds[concat_dim.name] = concat_dim
        # remove extraneous dimensions
        for dim in ds.dims:
            if dim not in stacked.dims:
                del ds[dim]
        ds[name] = stacked
        return ds[name]

    def _restore_dim_order(self, stacked, concat_dim):
        def lookup_order(dimension):
            if dimension == self.group.name:
                dimension, = concat_dim.dims
            if dimension in self.obj.dims:
                axis = self.obj.get_axis_num(dimension)
            else:
                axis = 1e6  # some arbitrarily high value
            return axis

        new_order = sorted(stacked.dims, key=lookup_order)
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
                metadata (attributes or coordinates) but only on the data and
                dimensions.
            (2) The action of `func` creates arrays with homogeneous metadata,
                that is, with the same dimensions and attributes.
            If these conditions are satisfied `shortcut` provides significant
            speedup. This should be the case for many common groupby operations
            (e.g., applying numpy ufuncs).
        **kwargs
            Used to call `func(ar, **kwargs)` for each array `ar`.

        Returns
        -------
        applied : DataArray
            The result of splitting, applying and combining this array.
        """
        if shortcut:
            grouped = self._iter_grouped_shortcut()
        else:
            grouped = self._iter_grouped()
        applied = (maybe_wrap_array(arr, func(arr, **kwargs)) for arr in grouped)
        return self._concat(applied, shortcut=shortcut)

    def _concat(self, applied, shortcut=False):
        # peek at applied to determine which coordinate to stack over
        applied_example, applied = peek_at(applied)
        concat_dim, indexers = self._infer_concat_args(applied_example)

        if shortcut:
            combined = self._concat_shortcut(applied, concat_dim, indexers)
        else:
            combined = concat(applied, concat_dim, indexers=indexers)

        if type(combined) is type(self.obj):
            combined = self._restore_dim_order(combined, concat_dim)
        return combined

    def reduce(self, func, dim=None, axis=None, keep_attrs=False,
               shortcut=True, **kwargs):
        """Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : function
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of collapsing an
            np.ndarray over an integer valued axis.
        dim : str or sequence of str, optional
            Dimension(s) over which to apply `func`.
        axis : int or sequence of int, optional
            Axis(es) over which to apply `func`. Only one of the 'dimension'
            and 'axis' arguments can be supplied. If neither are supplied, then
            `func` is calculated over all dimension for each group item.
        keep_attrs : bool, optional
            If True, the datasets's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : Array
            Array with summarized data and the indicated dimension(s)
            removed.
        """
        def reduce_array(ar):
            return ar.reduce(func, dim, axis, keep_attrs=keep_attrs, **kwargs)
        return self.apply(reduce_array, shortcut=shortcut)

ops.inject_reduce_methods(ArrayGroupBy)
ops.inject_binary_ops(ArrayGroupBy)


class DatasetGroupBy(GroupBy, ImplementsDatasetReduce):
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
        kwargs.pop('shortcut', None) # ignore shortcut if set (for now)
        applied = (func(ds, **kwargs) for ds in self._iter_grouped())
        return self._concat(applied)

    def _concat(self, applied):
        applied_example, applied = peek_at(applied)
        concat_dim, indexers = self._infer_concat_args(applied_example)
        combined = concat(applied, concat_dim, indexers=indexers)
        return combined

    def reduce(self, func, dim=None, keep_attrs=False, **kwargs):
        """Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : function
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of collapsing an
            np.ndarray over an integer valued axis.
        dim : str or sequence of str, optional
            Dimension(s) over which to apply `func`.
        axis : int or sequence of int, optional
            Axis(es) over which to apply `func`. Only one of the 'dimension'
            and 'axis' arguments can be supplied. If neither are supplied, then
            `func` is calculated over all dimension for each group item.
        keep_attrs : bool, optional
            If True, the datasets's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : Array
            Array with summarized data and the indicated dimension(s)
            removed.
        """
        def reduce_dataset(ds):
            return ds.reduce(func, dim, keep_attrs, **kwargs)
        return self.apply(reduce_dataset)

ops.inject_reduce_methods(DatasetGroupBy)
ops.inject_binary_ops(DatasetGroupBy)
