from __future__ import absolute_import, division, print_function

import functools

import numpy as np
import pandas as pd

from . import dtypes, duck_array_ops, nputils, ops
from .arithmetic import SupportsArithmetic
from .combine import concat
from .common import ImplementsArrayReduce, ImplementsDatasetReduce
from .pycompat import integer_types, range, zip
from .utils import hashable, maybe_wrap_array, peek_at, safe_cast_to_index
from .variable import IndexVariable, Variable, as_variable


def unique_value_groups(ar, sort=True):
    """Group an array by its unique values.

    Parameters
    ----------
    ar : array-like
        Input array. This will be flattened if it is not already 1-D.
    sort : boolean, optional
        Whether or not to sort unique values.

    Returns
    -------
    values : np.ndarray
        Sorted, unique values as returned by `np.unique`.
    indices : list of lists of int
        Each element provides the integer indices in `ar` with values given by
        the corresponding value in `unique_values`.
    """
    inverse, values = pd.factorize(ar, sort=sort)
    groups = [[] for _ in range(len(values))]
    for n, g in enumerate(inverse):
        if g >= 0:
            # pandas uses -1 to mark NaN, but doesn't include them in values
            groups[g].append(n)
    return values, groups


def _dummy_copy(xarray_obj):
    from .dataset import Dataset
    from .dataarray import DataArray
    if isinstance(xarray_obj, Dataset):
        res = Dataset(dict((k, dtypes.get_fill_value(v.dtype))
                           for k, v in xarray_obj.data_vars.items()),
                      dict((k, dtypes.get_fill_value(v.dtype))
                           for k, v in xarray_obj.coords.items()
                           if k not in xarray_obj.dims),
                      xarray_obj.attrs)
    elif isinstance(xarray_obj, DataArray):
        res = DataArray(dtypes.get_fill_value(xarray_obj.dtype),
                        dict((k, dtypes.get_fill_value(v.dtype))
                             for k, v in xarray_obj.coords.items()
                             if k not in xarray_obj.dims),
                        dims=[],
                        name=xarray_obj.name,
                        attrs=xarray_obj.attrs)
    else:  # pragma: no cover
        raise AssertionError
    return res


def _is_one_or_none(obj):
    return obj == 1 or obj is None


def _consolidate_slices(slices):
    """Consolidate adjacent slices in a list of slices.
    """
    result = []
    last_slice = slice(None)
    for slice_ in slices:
        if not isinstance(slice_, slice):
            raise ValueError('list element is not a slice: %r' % slice_)
        if (result and last_slice.stop == slice_.start and
                _is_one_or_none(last_slice.step) and
                _is_one_or_none(slice_.step)):
            last_slice = slice(last_slice.start, slice_.stop, slice_.step)
            result[-1] = last_slice
        else:
            result.append(slice_)
            last_slice = slice_
    return result


def _inverse_permutation_indices(positions):
    """Like inverse_permutation, but also handles slices.

    Parameters
    ----------
    positions : list of np.ndarray or slice objects.
        If slice objects, all are assumed to be slices.

    Returns
    -------
    np.ndarray of indices or None, if no permutation is necessary.
    """
    if not positions:
        return None

    if isinstance(positions[0], slice):
        positions = _consolidate_slices(positions)
        if positions == slice(None):
            return None
        positions = [np.arange(sl.start, sl.stop, sl.step) for sl in positions]

    indices = nputils.inverse_permutation(np.concatenate(positions))
    return indices


class _DummyGroup(object):
    """Class for keeping track of grouped dimensions without coordinates.

    Should not be user visible.
    """

    def __init__(self, obj, name, coords):
        self.name = name
        self.coords = coords
        self.dims = (name,)
        self.ndim = 1
        self.size = obj.sizes[name]
        self.values = range(self.size)


def _ensure_1d(group, obj):
    if group.ndim != 1:
        # try to stack the dims of the group into a single dim
        orig_dims = group.dims
        stacked_dim = 'stacked_' + '_'.join(orig_dims)
        # these dimensions get created by the stack operation
        inserted_dims = [dim for dim in group.dims if dim not in group.coords]
        # the copy is necessary here, otherwise read only array raises error
        # in pandas: https://github.com/pydata/pandas/issues/12813
        group = group.stack(**{stacked_dim: orig_dims}).copy()
        obj = obj.stack(**{stacked_dim: orig_dims})
    else:
        stacked_dim = None
        inserted_dims = []
    return group, obj, stacked_dim, inserted_dims


def _unique_and_monotonic(group):
    if isinstance(group, _DummyGroup):
        return True
    else:
        index = safe_cast_to_index(group)
        return index.is_unique and index.is_monotonic


class GroupBy(SupportsArithmetic):
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

    def __init__(self, obj, group, squeeze=False, grouper=None, bins=None,
                 cut_kwargs={}):
        """Create a GroupBy object

        Parameters
        ----------
        obj : Dataset or DataArray
            Object to group.
        group : DataArray
            Array with the group values.
        squeeze : boolean, optional
            If "group" is a coordinate of object, `squeeze` controls whether
            the subarrays have a dimension of length 1 along that coordinate or
            if the dimension is squeezed out.
        grouper : pd.Grouper, optional
            Used for grouping values along the `group` array.
        bins : array-like, optional
            If `bins` is specified, the groups will be discretized into the
            specified bins by `pandas.cut`.
        cut_kwargs : dict, optional
            Extra keyword arguments to pass to `pandas.cut`

        """
        from .dataarray import DataArray

        if grouper is not None and bins is not None:
            raise TypeError("can't specify both `grouper` and `bins`")

        if not isinstance(group, (DataArray, IndexVariable)):
            if not hashable(group):
                raise TypeError('`group` must be an xarray.DataArray or the '
                                'name of an xarray variable or dimension')
            group = obj[group]
            if group.name not in obj.coords and group.name in obj.dims:
                # DummyGroups should not appear on groupby results
                group = _DummyGroup(obj, group.name, group.coords)

        if getattr(group, 'name', None) is None:
            raise ValueError('`group` must have a name')

        group, obj, stacked_dim, inserted_dims = _ensure_1d(group, obj)
        group_dim, = group.dims

        expected_size = obj.sizes[group_dim]
        if group.size != expected_size:
            raise ValueError('the group variable\'s length does not '
                             'match the length of this variable along its '
                             'dimension')

        full_index = None

        if bins is not None:
            binned = pd.cut(group.values, bins, **cut_kwargs)
            new_dim_name = group.name + '_bins'
            group = DataArray(binned, group.coords, name=new_dim_name)
            full_index = binned.categories

        if grouper is not None:
            index = safe_cast_to_index(group)
            if not index.is_monotonic:
                # TODO: sort instead of raising an error
                raise ValueError('index must be monotonic for resampling')
            s = pd.Series(np.arange(index.size), index)
            first_items = s.groupby(grouper).first()
            full_index = first_items.index
            if first_items.isnull().any():
                first_items = first_items.dropna()
            sbins = first_items.values.astype(np.int64)
            group_indices = ([slice(i, j)
                              for i, j in zip(sbins[:-1], sbins[1:])] +
                             [slice(sbins[-1], None)])
            unique_coord = IndexVariable(group.name, first_items.index)
        elif group.dims == (group.name,) and _unique_and_monotonic(group):
            # no need to factorize
            group_indices = np.arange(group.size)
            if not squeeze:
                # use slices to do views instead of fancy indexing
                # equivalent to: group_indices = group_indices.reshape(-1, 1)
                group_indices = [slice(i, i + 1) for i in group_indices]
            unique_coord = group
        else:
            # look through group to find the unique values
            unique_values, group_indices = unique_value_groups(
                safe_cast_to_index(group), sort=(bins is None))
            unique_coord = IndexVariable(group.name, unique_values)

        # specification for the groupby operation
        self._obj = obj
        self._group = group
        self._group_dim = group_dim
        self._group_indices = group_indices
        self._unique_coord = unique_coord
        self._stacked_dim = stacked_dim
        self._inserted_dims = inserted_dims
        self._full_index = full_index

        # cached attributes
        self._groups = None

    @property
    def groups(self):
        # provided to mimic pandas.groupby
        if self._groups is None:
            self._groups = dict(zip(self._unique_coord.values,
                                    self._group_indices))
        return self._groups

    def __len__(self):
        return self._unique_coord.size

    def __iter__(self):
        return zip(self._unique_coord.values, self._iter_grouped())

    def _iter_grouped(self):
        """Iterate over each element in this group"""
        for indices in self._group_indices:
            yield self._obj.isel(**{self._group_dim: indices})

    def _infer_concat_args(self, applied_example):
        if self._group_dim in applied_example.dims:
            coord = self._group
            positions = self._group_indices
        else:
            coord = self._unique_coord
            positions = None
        dim, = coord.dims
        if isinstance(coord, _DummyGroup):
            coord = None
        return coord, dim, positions

    @staticmethod
    def _binary_op(f, reflexive=False, **ignored_kwargs):
        @functools.wraps(f)
        def func(self, other):
            g = f if not reflexive else lambda x, y: f(y, x)
            applied = self._yield_binary_applied(g, other)
            combined = self._combine(applied)
            return combined
        return func

    def _yield_binary_applied(self, func, other):
        dummy = None

        for group_value, obj in self:
            try:
                other_sel = other.sel(**{self._group.name: group_value})
            except AttributeError:
                raise TypeError('GroupBy objects only support binary ops '
                                'when the other argument is a Dataset or '
                                'DataArray')
            except (KeyError, ValueError):
                if self._group.name not in other.dims:
                    raise ValueError('incompatible dimensions for a grouped '
                                     'binary operation: the group variable %r '
                                     'is not a dimension on the other argument'
                                     % self._group.name)
                if dummy is None:
                    dummy = _dummy_copy(other)
                other_sel = dummy

            result = func(obj, other_sel)
            yield result

    def _maybe_restore_empty_groups(self, combined):
        """Our index contained empty groups (e.g., from a resampling). If we
        reduced on that dimension, we want to restore the full index.
        """
        if (self._full_index is not None and
                self._group.name in combined.dims):
            indexers = {self._group.name: self._full_index}
            combined = combined.reindex(**indexers)
        return combined

    def _maybe_unstack(self, obj):
        """This gets called if we are applying on an array with a
        multidimensional group."""
        if self._stacked_dim is not None and self._stacked_dim in obj.dims:
            obj = obj.unstack(self._stacked_dim)
            for dim in self._inserted_dims:
                if dim in obj.coords:
                    del obj.coords[dim]
        return obj

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
        out = ops.fillna(self, value)
        return out

    def where(self, cond, other=dtypes.NA):
        """Return elements from `self` or `other` depending on `cond`.

        Parameters
        ----------
        cond : DataArray or Dataset with boolean dtype
            Locations at which to preserve this objects values.
        other : scalar, DataArray or Dataset, optional
            Value to use for locations in this object where ``cond`` is False.
            By default, inserts missing values.

        Returns
        -------
        same type as the grouped object

        See also
        --------
        Dataset.where
        """
        return ops.where_method(self, cond, other)

    def _first_or_last(self, op, skipna, keep_attrs):
        if isinstance(self._group_indices[0], integer_types):
            # NB. this is currently only used for reductions along an existing
            # dimension
            return self._obj
        return self.reduce(op, self._group_dim, skipna=skipna,
                           keep_attrs=keep_attrs, allow_lazy=True)

    def first(self, skipna=None, keep_attrs=True):
        """Return the first element of each group along the group dimension
        """
        return self._first_or_last(duck_array_ops.first, skipna, keep_attrs)

    def last(self, skipna=None, keep_attrs=True):
        """Return the last element of each group along the group dimension
        """
        return self._first_or_last(duck_array_ops.last, skipna, keep_attrs)

    def assign_coords(self, **kwargs):
        """Assign coordinates by group.

        See also
        --------
        Dataset.assign_coords
        """
        return self.apply(lambda ds: ds.assign_coords(**kwargs))


def _maybe_reorder(xarray_obj, dim, positions):
    order = _inverse_permutation_indices(positions)

    if order is None:
        return xarray_obj
    else:
        return xarray_obj[{dim: order}]


class DataArrayGroupBy(GroupBy, ImplementsArrayReduce):
    """GroupBy object specialized to grouping DataArray objects
    """

    def _iter_grouped_shortcut(self):
        """Fast version of `_iter_grouped` that yields Variables without
        metadata
        """
        var = self._obj.variable
        for indices in self._group_indices:
            yield var[{self._group_dim: indices}]

    def _concat_shortcut(self, applied, dim, positions=None):
        # nb. don't worry too much about maintaining this method -- it does
        # speed things up, but it's not very interpretable and there are much
        # faster alternatives (e.g., doing the grouped aggregation in a
        # compiled language)
        stacked = Variable.concat(applied, dim, shortcut=True)
        reordered = _maybe_reorder(stacked, dim, positions)
        result = self._obj._replace_maybe_drop_dims(reordered)
        return result

    def _restore_dim_order(self, stacked):
        def lookup_order(dimension):
            if dimension == self._group.name:
                dimension, = self._group.dims
            if dimension in self._obj.dims:
                axis = self._obj.get_axis_num(dimension)
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
        applied : DataArray or DataArray
            The result of splitting, applying and combining this array.
        """
        if shortcut:
            grouped = self._iter_grouped_shortcut()
        else:
            grouped = self._iter_grouped()
        applied = (maybe_wrap_array(arr, func(arr, **kwargs))
                   for arr in grouped)
        return self._combine(applied, shortcut=shortcut)

    def _combine(self, applied, shortcut=False):
        """Recombine the applied objects like the original."""
        applied_example, applied = peek_at(applied)
        coord, dim, positions = self._infer_concat_args(applied_example)
        if shortcut:
            combined = self._concat_shortcut(applied, dim, positions)
        else:
            combined = concat(applied, dim)
            combined = _maybe_reorder(combined, dim, positions)

        if isinstance(combined, type(self._obj)):
            # only restore dimension order for arrays
            combined = self._restore_dim_order(combined)
        if coord is not None:
            if shortcut:
                combined._coords[coord.name] = as_variable(coord)
            else:
                combined.coords[coord.name] = coord
        combined = self._maybe_restore_empty_groups(combined)
        combined = self._maybe_unstack(combined)
        return combined

    def reduce(self, func, dim=None, axis=None, keep_attrs=False,
               shortcut=True, **kwargs):
        """Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : function
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of collapsing
            an np.ndarray over an integer valued axis.
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
        applied : Dataset or DataArray
            The result of splitting, applying and combining this dataset.
        """
        kwargs.pop('shortcut', None)  # ignore shortcut if set (for now)
        applied = (func(ds, **kwargs) for ds in self._iter_grouped())
        return self._combine(applied)

    def _combine(self, applied):
        """Recombine the applied objects like the original."""
        applied_example, applied = peek_at(applied)
        coord, dim, positions = self._infer_concat_args(applied_example)
        combined = concat(applied, dim)
        combined = _maybe_reorder(combined, dim, positions)
        if coord is not None:
            combined[coord.name] = coord
        combined = self._maybe_restore_empty_groups(combined)
        combined = self._maybe_unstack(combined)
        return combined

    def reduce(self, func, dim=None, keep_attrs=False, **kwargs):
        """Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : function
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of collapsing
            an np.ndarray over an integer valued axis.
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
