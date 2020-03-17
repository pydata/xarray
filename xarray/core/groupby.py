import datetime
import functools
import warnings

import numpy as np
import pandas as pd

from . import dtypes, duck_array_ops, nputils, ops
from .arithmetic import SupportsArithmetic
from .common import ImplementsArrayReduce, ImplementsDatasetReduce
from .concat import concat
from .formatting import format_array_flat
from .indexes import propagate_indexes
from .options import _get_keep_attrs
from .pycompat import integer_types
from .utils import (
    either_dict_or_kwargs,
    hashable,
    is_scalar,
    maybe_wrap_array,
    peek_at,
    safe_cast_to_index,
)
from .variable import IndexVariable, Variable, as_variable


def check_reduce_dims(reduce_dims, dimensions):

    if reduce_dims is not ...:
        if is_scalar(reduce_dims):
            reduce_dims = [reduce_dims]
        if any([dim not in dimensions for dim in reduce_dims]):
            raise ValueError(
                "cannot reduce over dimensions %r. expected either '...' to reduce over all dimensions or one or more of %r."
                % (reduce_dims, dimensions)
            )


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
        res = Dataset(
            {
                k: dtypes.get_fill_value(v.dtype)
                for k, v in xarray_obj.data_vars.items()
            },
            {
                k: dtypes.get_fill_value(v.dtype)
                for k, v in xarray_obj.coords.items()
                if k not in xarray_obj.dims
            },
            xarray_obj.attrs,
        )
    elif isinstance(xarray_obj, DataArray):
        res = DataArray(
            dtypes.get_fill_value(xarray_obj.dtype),
            {
                k: dtypes.get_fill_value(v.dtype)
                for k, v in xarray_obj.coords.items()
                if k not in xarray_obj.dims
            },
            dims=[],
            name=xarray_obj.name,
            attrs=xarray_obj.attrs,
        )
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
            raise ValueError("list element is not a slice: %r" % slice_)
        if (
            result
            and last_slice.stop == slice_.start
            and _is_one_or_none(last_slice.step)
            and _is_one_or_none(slice_.step)
        ):
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


class _DummyGroup:
    """Class for keeping track of grouped dimensions without coordinates.

    Should not be user visible.
    """

    __slots__ = ("name", "coords", "size")

    def __init__(self, obj, name, coords):
        self.name = name
        self.coords = coords
        self.size = obj.sizes[name]

    @property
    def dims(self):
        return (self.name,)

    @property
    def ndim(self):
        return 1

    @property
    def values(self):
        return range(self.size)

    @property
    def shape(self):
        return (self.size,)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            key = key[0]
        return self.values[key]


def _ensure_1d(group, obj):
    if group.ndim != 1:
        # try to stack the dims of the group into a single dim
        orig_dims = group.dims
        stacked_dim = "stacked_" + "_".join(orig_dims)
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


def _apply_loffset(grouper, result):
    """
    (copied from pandas)
    if loffset is set, offset the result index

    This is NOT an idempotent routine, it will be applied
    exactly once to the result.

    Parameters
    ----------
    result : Series or DataFrame
        the result of resample
    """

    needs_offset = (
        isinstance(grouper.loffset, (pd.DateOffset, datetime.timedelta))
        and isinstance(result.index, pd.DatetimeIndex)
        and len(result.index) > 0
    )

    if needs_offset:
        result.index = result.index + grouper.loffset

    grouper.loffset = None


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

    __slots__ = (
        "_full_index",
        "_inserted_dims",
        "_group",
        "_group_dim",
        "_group_indices",
        "_groups",
        "_obj",
        "_restore_coord_dims",
        "_stacked_dim",
        "_unique_coord",
        "_dims",
    )

    def __init__(
        self,
        obj,
        group,
        squeeze=False,
        grouper=None,
        bins=None,
        restore_coord_dims=None,
        cut_kwargs={},
    ):
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
        restore_coord_dims : bool, optional
            If True, also restore the dimension order of multi-dimensional
            coordinates.
        cut_kwargs : dict, optional
            Extra keyword arguments to pass to `pandas.cut`

        """
        from .dataarray import DataArray

        if grouper is not None and bins is not None:
            raise TypeError("can't specify both `grouper` and `bins`")

        if not isinstance(group, (DataArray, IndexVariable)):
            if not hashable(group):
                raise TypeError(
                    "`group` must be an xarray.DataArray or the "
                    "name of an xarray variable or dimension"
                )
            group = obj[group]
            if len(group) == 0:
                raise ValueError(f"{group.name} must not be empty")

            if group.name not in obj.coords and group.name in obj.dims:
                # DummyGroups should not appear on groupby results
                group = _DummyGroup(obj, group.name, group.coords)

        if getattr(group, "name", None) is None:
            raise ValueError("`group` must have a name")

        group, obj, stacked_dim, inserted_dims = _ensure_1d(group, obj)
        (group_dim,) = group.dims

        expected_size = obj.sizes[group_dim]
        if group.size != expected_size:
            raise ValueError(
                "the group variable's length does not "
                "match the length of this variable along its "
                "dimension"
            )

        full_index = None

        if bins is not None:
            if duck_array_ops.isnull(bins).all():
                raise ValueError("All bin edges are NaN.")
            binned = pd.cut(group.values, bins, **cut_kwargs)
            new_dim_name = group.name + "_bins"
            group = DataArray(binned, group.coords, name=new_dim_name)
            full_index = binned.categories

        if grouper is not None:
            index = safe_cast_to_index(group)
            if not index.is_monotonic:
                # TODO: sort instead of raising an error
                raise ValueError("index must be monotonic for resampling")
            full_index, first_items = self._get_index_and_items(index, grouper)
            sbins = first_items.values.astype(np.int64)
            group_indices = [slice(i, j) for i, j in zip(sbins[:-1], sbins[1:])] + [
                slice(sbins[-1], None)
            ]
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
            if group.isnull().any():
                # drop any NaN valued groups.
                # also drop obj values where group was NaN
                # Use where instead of reindex to account for duplicate coordinate labels.
                obj = obj.where(group.notnull(), drop=True)
                group = group.dropna(group_dim)

            # look through group to find the unique values
            unique_values, group_indices = unique_value_groups(
                safe_cast_to_index(group), sort=(bins is None)
            )
            unique_coord = IndexVariable(group.name, unique_values)

        if len(group_indices) == 0:
            if bins is not None:
                raise ValueError(
                    "None of the data falls within bins with edges %r" % bins
                )
            else:
                raise ValueError(
                    "Failed to group data. Are you grouping by a variable that is all NaN?"
                )

        if (
            isinstance(obj, DataArray)
            and restore_coord_dims is None
            and any(obj[c].ndim > 1 for c in obj.coords)
        ):
            warnings.warn(
                "This DataArray contains multi-dimensional "
                "coordinates. In the future, the dimension order "
                "of these coordinates will be restored as well "
                "unless you specify restore_coord_dims=False.",
                FutureWarning,
                stacklevel=2,
            )
            restore_coord_dims = False

        # specification for the groupby operation
        self._obj = obj
        self._group = group
        self._group_dim = group_dim
        self._group_indices = group_indices
        self._unique_coord = unique_coord
        self._stacked_dim = stacked_dim
        self._inserted_dims = inserted_dims
        self._full_index = full_index
        self._restore_coord_dims = restore_coord_dims

        # cached attributes
        self._groups = None
        self._dims = None

    @property
    def dims(self):
        if self._dims is None:
            self._dims = self._obj.isel(
                **{self._group_dim: self._group_indices[0]}
            ).dims

        return self._dims

    @property
    def groups(self):
        # provided to mimic pandas.groupby
        if self._groups is None:
            self._groups = dict(zip(self._unique_coord.values, self._group_indices))
        return self._groups

    def __len__(self):
        return self._unique_coord.size

    def __iter__(self):
        return zip(self._unique_coord.values, self._iter_grouped())

    def __repr__(self):
        return "{}, grouped over {!r} \n{!r} groups with labels {}.".format(
            self.__class__.__name__,
            self._unique_coord.name,
            self._unique_coord.size,
            ", ".join(format_array_flat(self._unique_coord, 30).split()),
        )

    def _get_index_and_items(self, index, grouper):
        from .resample_cftime import CFTimeGrouper

        s = pd.Series(np.arange(index.size), index)
        if isinstance(grouper, CFTimeGrouper):
            first_items = grouper.first_items(index)
        else:
            first_items = s.groupby(grouper).first()
            _apply_loffset(grouper, first_items)
        full_index = first_items.index
        if first_items.isnull().any():
            first_items = first_items.dropna()
        return full_index, first_items

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
        (dim,) = coord.dims
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
                raise TypeError(
                    "GroupBy objects only support binary ops "
                    "when the other argument is a Dataset or "
                    "DataArray"
                )
            except (KeyError, ValueError):
                if self._group.name not in other.dims:
                    raise ValueError(
                        "incompatible dimensions for a grouped "
                        "binary operation: the group variable %r "
                        "is not a dimension on the other argument" % self._group.name
                    )
                if dummy is None:
                    dummy = _dummy_copy(other)
                other_sel = dummy

            result = func(obj, other_sel)
            yield result

    def _maybe_restore_empty_groups(self, combined):
        """Our index contained empty groups (e.g., from a resampling). If we
        reduced on that dimension, we want to restore the full index.
        """
        if self._full_index is not None and self._group.name in combined.dims:
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
            obj._indexes = propagate_indexes(obj._indexes, exclude=self._inserted_dims)
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

    def quantile(
        self, q, dim=None, interpolation="linear", keep_attrs=None, skipna=True
    ):
        """Compute the qth quantile over each array in the groups and
        concatenate them together into a new array.

        Parameters
        ----------
        q : float in range of [0,1] (or sequence of floats)
            Quantile to compute, which must be between 0 and 1
            inclusive.
        dim : `...`, str or sequence of str, optional
            Dimension(s) over which to apply quantile.
            Defaults to the grouped dimension.
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            This optional parameter specifies the interpolation method to
            use when the desired quantile lies between two data points
            ``i < j``:

                * linear: ``i + (j - i) * fraction``, where ``fraction`` is
                  the fractional part of the index surrounded by ``i`` and
                  ``j``.
                * lower: ``i``.
                * higher: ``j``.
                * nearest: ``i`` or ``j``, whichever is nearest.
                * midpoint: ``(i + j) / 2``.
        skipna : bool, optional
            Whether to skip missing values when aggregating.

        Returns
        -------
        quantiles : Variable
            If `q` is a single quantile, then the result is a
            scalar. If multiple percentiles are given, first axis of
            the result corresponds to the quantile. In either case a
            quantile dimension is added to the return array. The other
            dimensions are the dimensions that remain after the
            reduction of the array.

        See Also
        --------
        numpy.nanquantile, numpy.quantile, pandas.Series.quantile, Dataset.quantile,
        DataArray.quantile

        Examples
        --------

        >>> da = xr.DataArray(
        ...     [[1.3, 8.4, 0.7, 6.9], [0.7, 4.2, 9.4, 1.5], [6.5, 7.3, 2.6, 1.9]],
        ...     coords={"x": [0, 0, 1], "y": [1, 1, 2, 2]},
        ...     dims=("y", "y"),
        ... )
        >>> ds = xr.Dataset({"a": da})
        >>> da.groupby("x").quantile(0)
        <xarray.DataArray (x: 2, y: 4)>
        array([[0.7, 4.2, 0.7, 1.5],
               [6.5, 7.3, 2.6, 1.9]])
        Coordinates:
            quantile  float64 0.0
          * y         (y) int64 1 1 2 2
          * x         (x) int64 0 1
        >>> ds.groupby("y").quantile(0, dim=...)
        <xarray.Dataset>
        Dimensions:   (y: 2)
        Coordinates:
            quantile  float64 0.0
          * y         (y) int64 1 2
        Data variables:
            a         (y) float64 0.7 0.7
        >>> da.groupby("x").quantile([0, 0.5, 1])
        <xarray.DataArray (x: 2, y: 4, quantile: 3)>
        array([[[0.7 , 1.  , 1.3 ],
                [4.2 , 6.3 , 8.4 ],
                [0.7 , 5.05, 9.4 ],
                [1.5 , 4.2 , 6.9 ]],
               [[6.5 , 6.5 , 6.5 ],
                [7.3 , 7.3 , 7.3 ],
                [2.6 , 2.6 , 2.6 ],
                [1.9 , 1.9 , 1.9 ]]])
        Coordinates:
          * y         (y) int64 1 1 2 2
          * quantile  (quantile) float64 0.0 0.5 1.0
          * x         (x) int64 0 1
        >>> ds.groupby("y").quantile([0, 0.5, 1], dim=...)
        <xarray.Dataset>
        Dimensions:   (quantile: 3, y: 2)
        Coordinates:
          * quantile  (quantile) float64 0.0 0.5 1.0
          * y         (y) int64 1 2
        Data variables:
            a         (y, quantile) float64 0.7 5.35 8.4 0.7 2.25 9.4
        """
        if dim is None:
            dim = self._group_dim

        out = self.map(
            self._obj.__class__.quantile,
            shortcut=False,
            q=q,
            dim=dim,
            interpolation=interpolation,
            keep_attrs=keep_attrs,
            skipna=skipna,
        )

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
        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=True)
        return self.reduce(op, self._group_dim, skipna=skipna, keep_attrs=keep_attrs)

    def first(self, skipna=None, keep_attrs=None):
        """Return the first element of each group along the group dimension
        """
        return self._first_or_last(duck_array_ops.first, skipna, keep_attrs)

    def last(self, skipna=None, keep_attrs=None):
        """Return the last element of each group along the group dimension
        """
        return self._first_or_last(duck_array_ops.last, skipna, keep_attrs)

    def assign_coords(self, coords=None, **coords_kwargs):
        """Assign coordinates by group.

        See also
        --------
        Dataset.assign_coords
        Dataset.swap_dims
        """
        coords_kwargs = either_dict_or_kwargs(coords, coords_kwargs, "assign_coords")
        return self.map(lambda ds: ds.assign_coords(**coords_kwargs))


def _maybe_reorder(xarray_obj, dim, positions):
    order = _inverse_permutation_indices(positions)

    if order is None or len(order) != xarray_obj.sizes[dim]:
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
                (dimension,) = self._group.dims
            if dimension in self._obj.dims:
                axis = self._obj.get_axis_num(dimension)
            else:
                axis = 1e6  # some arbitrarily high value
            return axis

        new_order = sorted(stacked.dims, key=lookup_order)
        return stacked.transpose(*new_order, transpose_coords=self._restore_coord_dims)

    def map(self, func, shortcut=False, args=(), **kwargs):
        """Apply a function to each array in the group and concatenate them
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
        ``*args`` : tuple, optional
            Positional arguments passed to `func`.
        ``**kwargs``
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
        applied = (maybe_wrap_array(arr, func(arr, *args, **kwargs)) for arr in grouped)
        return self._combine(applied, shortcut=shortcut)

    def apply(self, func, shortcut=False, args=(), **kwargs):
        """
        Backward compatible implementation of ``map``

        See Also
        --------
        DataArrayGroupBy.map
        """
        warnings.warn(
            "GroupBy.apply may be deprecated in the future. Using GroupBy.map is encouraged",
            PendingDeprecationWarning,
            stacklevel=2,
        )
        return self.map(func, shortcut=shortcut, args=args, **kwargs)

    def _combine(self, applied, restore_coord_dims=False, shortcut=False):
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
        # assign coord when the applied function does not return that coord
        if coord is not None and dim not in applied_example.dims:
            if shortcut:
                coord_var = as_variable(coord)
                combined._coords[coord.name] = coord_var
            else:
                combined.coords[coord.name] = coord
        combined = self._maybe_restore_empty_groups(combined)
        combined = self._maybe_unstack(combined)
        return combined

    def reduce(
        self, func, dim=None, axis=None, keep_attrs=None, shortcut=True, **kwargs
    ):
        """Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : function
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of collapsing
            an np.ndarray over an integer valued axis.
        dim : `...`, str or sequence of str, optional
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
        if dim is None:
            dim = self._group_dim

        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=False)

        def reduce_array(ar):
            return ar.reduce(func, dim, axis, keep_attrs=keep_attrs, **kwargs)

        check_reduce_dims(dim, self.dims)

        return self.map(reduce_array, shortcut=shortcut)


ops.inject_reduce_methods(DataArrayGroupBy)
ops.inject_binary_ops(DataArrayGroupBy)


class DatasetGroupBy(GroupBy, ImplementsDatasetReduce):
    def map(self, func, args=(), shortcut=None, **kwargs):
        """Apply a function to each Dataset in the group and concatenate them
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
        args : tuple, optional
            Positional arguments to pass to `func`.
        **kwargs
            Used to call `func(ds, **kwargs)` for each sub-dataset `ar`.

        Returns
        -------
        applied : Dataset or DataArray
            The result of splitting, applying and combining this dataset.
        """
        # ignore shortcut if set (for now)
        applied = (func(ds, *args, **kwargs) for ds in self._iter_grouped())
        return self._combine(applied)

    def apply(self, func, args=(), shortcut=None, **kwargs):
        """
        Backward compatible implementation of ``map``

        See Also
        --------
        DatasetGroupBy.map
        """

        warnings.warn(
            "GroupBy.apply may be deprecated in the future. Using GroupBy.map is encouraged",
            PendingDeprecationWarning,
            stacklevel=2,
        )
        return self.map(func, shortcut=shortcut, args=args, **kwargs)

    def _combine(self, applied):
        """Recombine the applied objects like the original."""
        applied_example, applied = peek_at(applied)
        coord, dim, positions = self._infer_concat_args(applied_example)
        combined = concat(applied, dim)
        combined = _maybe_reorder(combined, dim, positions)
        # assign coord when the applied function does not return that coord
        if coord is not None and dim not in applied_example.dims:
            combined[coord.name] = coord
        combined = self._maybe_restore_empty_groups(combined)
        combined = self._maybe_unstack(combined)
        return combined

    def reduce(self, func, dim=None, keep_attrs=None, **kwargs):
        """Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : function
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of collapsing
            an np.ndarray over an integer valued axis.
        dim : `...`, str or sequence of str, optional
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
        if dim is None:
            dim = self._group_dim

        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=False)

        def reduce_dataset(ds):
            return ds.reduce(func, dim, keep_attrs, **kwargs)

        check_reduce_dims(dim, self.dims)

        return self.map(reduce_dataset)

    def assign(self, **kwargs):
        """Assign data variables by group.

        See also
        --------
        Dataset.assign
        """
        return self.map(lambda ds: ds.assign(**kwargs))


ops.inject_reduce_methods(DatasetGroupBy)
ops.inject_binary_ops(DatasetGroupBy)
