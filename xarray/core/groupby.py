import functools
import numpy as np
import pandas as pd

from . import ops
from .combine import concat
from .common import (
    ImplementsArrayReduce, ImplementsDatasetReduce, _maybe_promote,
)
from .pycompat import zip
from .utils import peek_at, maybe_wrap_array, safe_cast_to_index
from .variable import as_variable, Variable, Coordinate


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


def _get_fill_value(dtype):
    """Return a fill value that appropriately promotes types when used with
    np.concatenate
    """
    dtype, fill_value = _maybe_promote(dtype)
    return fill_value


def _dummy_copy(xarray_obj):
    from .dataset import Dataset
    from .dataarray import DataArray
    if isinstance(xarray_obj, Dataset):
        res = Dataset(dict((k, _get_fill_value(v.dtype))
                           for k, v in xarray_obj.data_vars.items()),
                      dict((k, _get_fill_value(v.dtype))
                           for k, v in xarray_obj.coords.items()
                           if k not in xarray_obj.dims),
                      xarray_obj.attrs)
    elif isinstance(xarray_obj, DataArray):
        res = DataArray(_get_fill_value(xarray_obj.dtype),
                        dict((k, _get_fill_value(v.dtype))
                             for k, v in xarray_obj.coords.items()
                             if k not in xarray_obj.dims),
                        name=xarray_obj.name,
                        attrs=xarray_obj.attrs)
    else:  # pragma: no cover
        raise AssertionError
    return res


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
    def __init__(self, obj, group, squeeze=False, grouper=None):
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
        grouper : pd.Grouper, optional
            Used for grouping values along the `group` array.
        """
        from .dataset import as_dataset

        if group.ndim != 1:
            # TODO: remove this limitation?
            raise ValueError('`group` must be 1 dimensional')
        if getattr(group, 'name', None) is None:
            raise ValueError('`group` must have a name')
        if not hasattr(group, 'dims'):
            raise ValueError("`group` must have a 'dims' attribute")
        group_dim, = group.dims

        try:
            expected_size = obj.dims[group_dim]
        except TypeError:
            expected_size = obj.shape[obj.get_axis_num(group_dim)]
        if group.size != expected_size:
            raise ValueError('the group variable\'s length does not '
                             'match the length of this variable along its '
                             'dimension')
        full_index = None

        if grouper is not None:
            # time-series resampling
            index = safe_cast_to_index(group)
            if not index.is_monotonic:
                # TODO: sort instead of raising an error
                raise ValueError('index must be monotonic for resampling')
            s = pd.Series(np.arange(index.size), index)
            first_items = s.groupby(grouper).first()
            if first_items.isnull().any():
                full_index = first_items.index
                first_items = first_items.dropna()
            bins = first_items.values.astype(np.int64)
            group_indices = ([slice(i, j) for i, j in zip(bins[:-1], bins[1:])] +
                             [slice(bins[-1], None)])
            unique_coord = Coordinate(group.name, first_items.index)
        elif group.name in obj.dims:
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

        self.obj = obj
        self.group = group
        self.group_dim = group_dim
        self.group_indices = group_indices
        self.unique_coord = unique_coord
        self._groups = None
        self._full_index = full_index

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
            positions = self.group_indices
        else:
            concat_dim = self.unique_coord
            positions = None
        return concat_dim, positions

    @staticmethod
    def _binary_op(f, reflexive=False, **ignored_kwargs):
        @functools.wraps(f)
        def func(self, other):
            g = f if not reflexive else lambda x, y: f(y, x)
            applied = self._yield_binary_applied(g, other)
            combined = self._concat(applied)
            return combined
        return func

    def _yield_binary_applied(self, func, other):
        dummy = None

        for group_value, obj in self:
            try:
                other_sel = other.sel(**{self.group.name: group_value})
            except AttributeError:
                raise TypeError('GroupBy objects only support binary ops '
                                'when the other argument is a Dataset or '
                                'DataArray')
            except KeyError:
                if self.group.name not in other.dims:
                    raise ValueError('incompatible dimensions for a grouped '
                                     'binary operation: the group variable %r '
                                     'is not a dimension on the other argument'
                                     % self.group.name)
                if dummy is None:
                    dummy = _dummy_copy(other)
                other_sel = dummy

            result = func(obj, other_sel)
            yield result

    def _maybe_restore_empty_groups(self, combined):
        """Our index contained empty groups (e.g., from a resampling). If we
        reduced on that dimension, we want to restore the full index.
        """
        if (self._full_index is not None and self.group.name in combined.dims):
            indexers = {self.group.name: self._full_index}
            combined = combined.reindex(**indexers)
        return combined

    def fillna(self, value):
        """Fill missing values in this object by group.

        This operation follows the normal broadcasting and alignment rules that
        xarray uses for binary arithmetic, except the result is aligned to this
        object (``join='left'``) instead of aligned to the intersection of
        index coordinates (``join='inner'``).

        Parameters
        ----------
        value : valid type for the grouped object's fillna method
            Used to fill all matching missing values by group.

        Returns
        -------
        same type as the grouped object

        See also
        --------
        Dataset.fillna
        DataArray.fillna
        """
        return self._fillna(value)

    def where(self, cond):
        """Return an object of the same shape with all entries where cond is
        True and all other entries masked.

        This operation follows the normal broadcasting and alignment rules that
        xarray uses for binary arithmetic.

        Parameters
        ----------
        cond : DataArray or Dataset

        Returns
        -------
        same type as the grouped object

        See also
        --------
        Dataset.where
        """
        return self._where(cond)

    def _first_or_last(self, op, skipna, keep_attrs):
        if isinstance(self.group_indices[0], (int, np.integer)):
            # NB. this is currently only used for reductions along an existing
            # dimension
            return self.obj
        return self.reduce(op, self.group_dim, skipna=skipna,
                           keep_attrs=keep_attrs, allow_lazy=True)

    def first(self, skipna=None, keep_attrs=True):
        """Return the first element of each group along the group dimension
        """
        return self._first_or_last(ops.first, skipna, keep_attrs)

    def last(self, skipna=None, keep_attrs=True):
        """Return the last element of each group along the group dimension
        """
        return self._first_or_last(ops.last, skipna, keep_attrs)

    def assign_coords(self, **kwargs):
        """Assign coordinates by group.

        See also
        --------
        Dataset.assign_coords
        """
        return self.apply(lambda ds: ds.assign_coords(**kwargs))


class DataArrayGroupBy(GroupBy, ImplementsArrayReduce):
    """GroupBy object specialized to grouping DataArray objects
    """
    def _iter_grouped_shortcut(self):
        """Fast version of `_iter_grouped` that yields Variables without
        metadata
        """
        var = self.obj.variable
        for indices in self.group_indices:
            yield var[{self.group_dim: indices}]

    def _concat_shortcut(self, applied, concat_dim, positions):
        # nb. don't worry too much about maintaining this method -- it does
        # speed things up, but it's not very interpretable and there are much
        # faster alternatives (e.g., doing the grouped aggregation in a
        # compiled language)
        stacked = Variable.concat(
            applied, concat_dim, positions, shortcut=True)
        stacked.attrs.update(self.obj.attrs)
        result = self.obj._replace_maybe_drop_dims(stacked)
        result._coords[concat_dim.name] = as_variable(concat_dim, copy=True)
        return result

    def _restore_dim_order(self, stacked):
        def lookup_order(dimension):
            if dimension == self.group.name:
                dimension, = self.group.dims
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
        combined = self._concat(applied, shortcut=shortcut)
        result = self._maybe_restore_empty_groups(combined)
        return result

    def _concat(self, applied, shortcut=False):
        # peek at applied to determine which coordinate to stack over
        applied_example, applied = peek_at(applied)
        concat_dim, positions = self._infer_concat_args(applied_example)

        if shortcut:
            combined = self._concat_shortcut(applied, concat_dim, positions)
        else:
            combined = concat(applied, concat_dim, positions=positions)

        if isinstance(combined, type(self.obj)):
            combined = self._restore_dim_order(combined)
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

ops.inject_reduce_methods(DataArrayGroupBy)
ops.inject_binary_ops(DataArrayGroupBy)


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
        kwargs.pop('shortcut', None)  # ignore shortcut if set (for now)
        applied = (func(ds, **kwargs) for ds in self._iter_grouped())
        combined = self._concat(applied)
        result = self._maybe_restore_empty_groups(combined)
        return result

    def _concat(self, applied):
        applied_example, applied = peek_at(applied)
        concat_dim, positions = self._infer_concat_args(applied_example)
        combined = concat(applied, concat_dim, positions=positions)
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

    def assign(self, **kwargs):
        """Assign data variables by group.

        See also
        --------
        Dataset.assign
        """
        return self.apply(lambda ds: ds.assign(**kwargs))

ops.inject_reduce_methods(DatasetGroupBy)
ops.inject_binary_ops(DatasetGroupBy)
