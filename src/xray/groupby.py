import itertools

from common import ImplementsCollapse
from ops import inject_collapse_methods
import array_
import dataset
import numpy as np


def unique_value_groups(ar):
    """Group an array by its unique values

    Parameters
    ----------
    ar : array_like
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


class GroupBy(ImplementsCollapse):
    """A object that implements the split-apply-combine pattern

    Modeled after `pandas.GroupBy`. The `GroupBy` object can be iterated over
    (unique_value, grouped_array) pairs, but the main way to interact with a
    groupby object are with the `apply` or `collapse` methods. You can also
    directly call numpy methods like `mean` or `std`.

    See Also
    --------
    Array.groupby
    DatasetArray.groupby
    """
    def __init__(self, array, group_name, group_coord, squeeze=True):
        """See Array.groupby and DatasetArray.groupby
        """
        if group_coord.ndim != 1:
            # TODO: remove this limitation?
            raise ValueError('`group_coord` must be 1 dimensional')

        self.array = array
        self.group_coord = group_coord
        self.group_dim, = group_coord.dimensions
        self.group_axis = array.dimensions.index(self.group_dim)

        if group_coord.size != array.shape[self.group_axis]:
            raise ValueError('the group variable\'s length does not '
                             'match the length of this variable along its '
                             'dimension')

        if group_name in array.dimensions:
            # assume that group_coord already has sorted, unique values
            if group_coord.dimensions != (group_name,):
                raise ValueError('`group_coord` is required to be a coordinate '
                                 'variable along the `group_name` dimension '
                                 'if `group_name` is a dimension in `array`')
            group_indices = np.arange(group_coord.size)
            if not squeeze:
                # group_indices = group_indices.reshape(-1, 1)
                # use slices to do views instead of fancy indexing
                group_indices = [slice(i, i + 1) for i in group_indices]
            unique_coord = group_coord
        else:
            # look through group_coord to find the unique values
            unique_values, group_indices = unique_value_groups(group_coord)
            unique_coord = dataset.Dataset(
                {group_name: (group_name, unique_values)})[group_name]

        self.group_indices = group_indices
        self.unique_coord = unique_coord
        self._groups = None

    @property
    def groups(self):
        # provided for compatibility with pandas.groupby
        if self._groups is None:
            self._groups = dict(zip(self.unique_coord, self.group_indices))
        return self._groups

    def __len__(self):
        return self.unique_coord.size

    def __iter__(self):
        return itertools.izip(self.unique_coord, self.iter_arrays())

    def iter_fast(self):
        # extract the underlying Array object
        array = self.array
        if hasattr(self.array, 'variable'):
            array = array.variable

        # build the new dimensions
        index_int = isinstance(self.group_indices[0], int)
        if index_int:
            dims = tuple(d for n, d in enumerate(array.dimensions)
                         if n != self.group_axis)
        else:
            dims = array.dimensions

        # slice the data and build the new Arrays directly
        for indices in self.group_indices:
            indexer = tuple(indices if n == self.group_axis else slice(None)
                            for n in range(array.ndim))
            data = array.data[indexer]
            yield array_.Array(dims, data)

    def iter_arrays(self):
        for indices in self.group_indices:
            yield self.array.indexed_by(**{self.group_dim: indices})

    def apply(self, func, shortcut=True, **kwargs):
        """Apply a function over each array in the group and stack them
        together into a new array

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
            If these conditions are satisfied (and they should be in most
            cases), the `shortcut` provides significant speedup for common
            groupby operations like applying numpy ufuncs.
        **kwargs
            Used to call `func(ar, **kwargs)` for each array `ar.

        Returns
        -------
        applied : Array
            A new Array of the same type from which this grouping was created.
        """
        shortcut = kwargs.pop('shortcut', True)
        applied = (func(ar, **kwargs) for ar in (self.iter_fast() if shortcut
                                                 else self.iter_array()))

        # peek at applied to determine which coordinate to stack over
        applied_example, applied = peek_at(applied)
        if self.group_dim in applied_example.dimensions:
            stack_coord = self.group_coord
            indexers = self.group_indices
        else:
            stack_coord = self.unique_coord
            indexers = np.arange(self.unique_coord.size)

        from_stack_kwargs = {'template': self.array} if shortcut else {}
        stacked = type(self.array).from_stack(applied, stack_coord, indexers,
                                              **from_stack_kwargs)

        # now, reorder the stacked array's dimensions so that those that
        # appeared in the original array appear in the same order they did
        # originally
        stack_dim, = stack_coord.dimensions
        original_dims = [stack_dim if d == self.group_dim else d
                         for d in self.array.dimensions
                         if d in stacked.dimensions or d == self.group_dim]
        iter_original_dims = iter(original_dims)
        new_order = [iter_original_dims.next() if d in original_dims else d
                     for d in stacked.dimensions]
        return stacked.transpose(*new_order)

    def collapse(self, func, dimension=Ellipsis, axis=Ellipsis, shortcut=True,
                 **kwargs):
        # Ellipsis is used as a sentinel value for the altered default
        if axis is Ellipsis and dimension is Ellipsis:
            dimension = self.group_dim
        if dimension is Ellipsis:
            dimension = None
        if axis is Ellipsis:
            axis = None
        def collapse_array(ar):
            return ar.collapse(func, dimension, axis, **kwargs)
        return self.apply(collapse_array, shortcut=shortcut)

    _collapse_method_docstring = \
        """Collapse this {cls}'s data' by applying `{name}` along some
        dimension(s)

        Parameters
        ----------
        dimension : str or sequence of str, optional
            Dimension(s) over which to repeatedly apply `{name}`.
        axis : int or sequence of int, optional
            Axis(es) over which to repeatedly apply `{name}`. Only one of the
            'dimension' and 'axis' arguments can be supplied. If neither are
            supplied, then `{name}` is calculated over the axis of the variable
            over which the group was formed.
        **kwargs : dict
            Additional keyword arguments passed on to `{name}`.

        Note
        ----
        If this method is called with multiple dimensions (or axes, which are
        converted into dimensions), then `{name}` is performed repeatedly along
        each dimension in turn from left to right.

        `Ellipsis` is used as the default dimension and axis for this method to
        indicate that this operation is by default applied along the axis along
        which the grouping variable lies. To instead apply `{name}`
        simultaneously over all grouped values, use `dimension=None` (or
        equivalently `axis=None`).

        Returns
        -------
        collapsed : {cls}
            New {cls} object with `{name}` applied to its data and the
            indicated dimension(s) removed.
        """

    _collapse_dimension_default = Ellipsis
    _collapse_axis_default = Ellipsis


inject_collapse_methods(GroupBy)
