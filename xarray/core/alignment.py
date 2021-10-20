import functools
import operator
from collections import defaultdict
from contextlib import suppress
from numbers import Number
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    FrozenSet,
    Hashable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import pandas as pd

from . import dtypes
from .indexes import Index, Indexes, PandasIndex, PandasMultiIndex, get_indexer_nd
from .utils import is_dict_like, is_full_slice, safe_cast_to_index
from .variable import Variable, calculate_dimensions

if TYPE_CHECKING:
    from .dataarray import DataArray
    from .dataset import Dataset

    T_Xarray = TypeVar("T_Xarray", bound=Union[Dataset, DataArray])
    DataAlignable = Sequence[Union[Dataset, DataArray]]
    DataAligned = Tuple[Union[Dataset, DataArray], ...]


class Aligner:
    """Implements all the complex logic for the re-indexing and alignment of Xarray
    objects.

    For internal use only, not public API.
    Usage:

    aligned_objects = Alignator(*objects, **kwargs).align()

    """

    CoordNamesAndDims = Tuple[Tuple[Hashable, Tuple[Hashable, ...]], ...]
    MatchingIndexKey = Tuple[CoordNamesAndDims, Type[Index]]
    NormalizedIndexes = Dict[MatchingIndexKey, Index]
    NormalizedIndexVars = Dict[MatchingIndexKey, Dict[Hashable, Variable]]

    objects: "DataAlignable"
    objects_matching_indexes: Tuple[Dict[MatchingIndexKey, Index], ...]
    join: str
    exclude_dims: FrozenSet
    copy: bool
    fill_value: Any
    sparse: bool
    indexes: Dict[MatchingIndexKey, Index]
    index_vars: Dict[MatchingIndexKey, Dict[Hashable, Variable]]
    all_indexes: Dict[MatchingIndexKey, List[Index]]
    all_index_vars: Dict[MatchingIndexKey, List[Dict[Hashable, Variable]]]
    aligned_indexes: Dict[MatchingIndexKey, Index]
    aligned_index_vars: Dict[MatchingIndexKey, Dict[Hashable, Variable]]
    reindex: Dict[MatchingIndexKey, bool]
    reindex_kwargs: Dict[str, Any]
    unindexed_dim_sizes: Dict[Hashable, Set]
    new_indexes: Indexes[Index]

    def __init__(
        self,
        objects: "DataAlignable",
        join: str = "inner",
        indexes: Mapping[Any, Any] = None,
        exclude: Any = frozenset(),
        method: str = None,
        tolerance: Number = None,
        copy: bool = True,
        fill_value: Any = dtypes.NA,
        sparse: bool = False,
    ):
        self.objects = tuple(objects)
        self.objects_matching_indexes = ()

        if join not in ["inner", "outer", "override", "exact", "left", "right"]:
            raise ValueError(f"invalid value for join: {join}")
        self.join = join

        self.copy = copy
        self.fill_value = fill_value
        self.sparse = sparse

        if method is None and tolerance is None:
            self.reindex_kwargs = {}
        else:
            self.reindex_kwargs = {"method": method, "tolerance": tolerance}

        if isinstance(exclude, str):
            exclude = [exclude]
        self.exclude_dims = frozenset(exclude)

        if indexes is None:
            indexes = {}
        self.indexes, self.index_vars = self._normalize_indexes(indexes)

        self.all_indexes = {}
        self.all_index_vars = {}
        self.unindexed_dim_sizes = {}

        self.aligned_indexes = {}
        self.aligned_index_vars = {}
        self.reindex = {}

    def _normalize_indexes(
        self,
        indexes: Mapping[Any, Any],
    ) -> Tuple[NormalizedIndexes, NormalizedIndexVars]:
        """Normalize the indexes/indexers used for re-indexing or alignment.

        Return dictionaries of xarray Index objects and coordinate variables
        such that we can group matching indexes based on the dictionary keys.

        """
        if isinstance(indexes, Indexes):
            xr_variables = dict(indexes.variables)
        else:
            xr_variables = {}

        xr_indexes: Dict[Hashable, Index] = {}
        for k, idx in indexes.items():
            if not isinstance(idx, Index):
                if getattr(idx, "dims", (k,)) != (k,):
                    raise ValueError(
                        f"Indexer has dimensions {idx.dims} that are different "
                        f"from that to be indexed along '{k}'"
                    )
                pd_idx = safe_cast_to_index(idx).copy()
                pd_idx.name = k
                if isinstance(pd_idx, pd.MultiIndex):
                    idx, _ = PandasMultiIndex.from_pandas_index(pd_idx, k)
                else:
                    idx, _ = PandasIndex.from_pandas_index(pd_idx, k)
                xr_variables.update(idx.create_variables())
            xr_indexes[k] = idx

        normalized_indexes = {}
        normalized_index_vars = {}
        for idx, index_vars in Indexes(xr_indexes, xr_variables).group_by_index():
            coord_names_and_dims = []
            all_dims = set()

            for name, var in index_vars.items():
                dims = var.dims
                coord_names_and_dims.append((name, dims))
                all_dims.update(dims)

            exclude_dims = all_dims & self.exclude_dims
            if exclude_dims == all_dims:
                continue
            elif exclude_dims:
                excl_dims_str = ", ".join(str(d) for d in exclude_dims)
                incl_dims_str = ", ".join(str(d) for d in all_dims - exclude_dims)
                raise ValueError(
                    f"cannot exclude dimension(s) {excl_dims_str} from alignment because "
                    "these are used by an index together with non-excluded dimensions "
                    f"{incl_dims_str}"
                )

            key = (tuple(coord_names_and_dims), type(idx))
            normalized_indexes[key] = idx
            normalized_index_vars[key] = index_vars

        return normalized_indexes, normalized_index_vars

    def find_matching_indexes(self):
        all_indexes = defaultdict(list)
        all_index_vars = defaultdict(list)
        all_indexes_dim_sizes = defaultdict(lambda: defaultdict(set))
        objects_matching_indexes = []

        for obj in self.objects:
            obj_indexes, obj_index_vars = self._normalize_indexes(obj.xindexes)
            objects_matching_indexes.append(obj_indexes)
            for key, idx in obj_indexes.items():
                all_indexes[key].append(idx)
            for key, index_vars in obj_index_vars.items():
                all_index_vars[key].append(index_vars)
                for dim, size in calculate_dimensions(index_vars).items():
                    all_indexes_dim_sizes[key][dim].add(size)

        self.objects_matching_indexes = tuple(objects_matching_indexes)
        self.all_indexes = all_indexes
        self.all_index_vars = all_index_vars

        if self.join == "override":
            for dim_sizes in all_indexes_dim_sizes.values():
                for dim, sizes in dim_sizes.items():
                    if len(sizes) > 1:
                        raise ValueError(
                            "cannot align objects with join='override' with matching indexes "
                            f"along dimension {dim!r} that don't have the same size"
                        )

    def find_matching_unindexed_dims(self):
        unindexed_dim_sizes = defaultdict(set)

        for obj in self.objects:
            for dim in obj.dims:
                if dim not in self.exclude_dims and dim not in obj.xindexes.dims:
                    unindexed_dim_sizes[dim].add(obj.sizes[dim])

        self.unindexed_dim_sizes = unindexed_dim_sizes

    def assert_no_index_conflict(self):
        """Check for uniqueness of both coordinate and dimension names accross all sets
        of matching indexes.

        We need to make sure that all indexes used for re-indexing or alignment
        are fully compatible and do not conflict each other.

        Note: perhaps we could choose less restrictive constraints and instead
        check for conflicts among the dimension (position) indexers returned by
        `Index.reindex_like()` for each matching pair of object index / aligned
        index?
        (ref: https://github.com/pydata/xarray/issues/1603#issuecomment-442965602)

        """
        matching_keys = set(self.all_indexes) | set(self.indexes)

        coord_count = defaultdict(int)
        dim_count = defaultdict(int)
        for coord_names_dims, _ in matching_keys:
            dims_set = set()
            for name, dims in coord_names_dims:
                coord_count[name] += 1
                dims_set.update(dims)
            for dim in dims_set:
                dim_count[dim] += 1

        for count, msg in [(coord_count, "coordinates"), (dim_count, "dimensions")]:
            dup = {k: v for k, v in count.items() if v > 1}
            if dup:
                items_msg = ", ".join(
                    f"{k!r} ({v} conflicting indexes)" for k, v in dup.items()
                )
                raise ValueError(
                    "cannot re-index or align objects with conflicting indexes found for "
                    f"the following {msg}: {items_msg}\n"
                    "Conflicting indexes may occur when\n"
                    "- they relate to different sets of coordinate and/or dimension names\n"
                    "- they don't have the same type\n"
                    "- they may be used to reindex data along common dimensions"
                )

    def _need_reindex(self, dims, index, other_indexes, coords, other_coords) -> bool:
        """Whether or not we need to reindex variables for a set of
        matching indexes.

        We don't reindex when all matching indexes are equal for two reasons:
        - It's faster for the usual case (already aligned objects).
        - It ensures it's possible to do operations that don't require alignment
          on indexes with duplicate values (which cannot be reindexed with
          pandas). This is useful, e.g., for overwriting such duplicate indexes.

        """
        try:
            index_not_equal = any(not index.equals(idx) for idx in other_indexes)
        except NotImplementedError:
            # check coordinates equality for indexes that do not support alignment
            index_not_equal = any(
                not coords[k].equals(o_coords[k])
                for o_coords in other_coords
                for k in coords
            )
        has_unindexed_dims = any(dim in self.unindexed_dim_sizes for dim in dims)
        return index_not_equal or has_unindexed_dims

    def _get_index_joiner(self, index_cls) -> Callable:
        if self.join in ["outer", "inner"]:
            return functools.partial(
                functools.reduce,
                functools.partial(index_cls.join, how=self.join),
            )
        elif self.join == "left":
            return operator.itemgetter(0)
        elif self.join == "right":
            return operator.itemgetter(-1)
        elif self.join == "override":
            # We rewrite all indexes and then use join='left'
            return operator.itemgetter(0)
        else:
            # join='exact' return dummy lambda (error is raised)
            return lambda _: None

    def align_indexes(self):
        aligned_indexes = {}
        aligned_index_vars = {}
        reindex = {}
        new_indexes = {}
        new_index_vars = {}

        for key, matching_indexes in self.all_indexes.items():
            matching_index_vars = self.all_index_vars[key]
            dims = set(
                [d for coord in matching_index_vars[0].values() for d in coord.dims]
            )
            index_cls = key[1]

            if self.join == "override":
                joined_index = matching_indexes[0]
                joined_index_vars = matching_index_vars[0]
                need_reindex = False
            elif key in self.indexes:
                joined_index = self.indexes[key]
                joined_index_vars = self.index_vars[key]
                need_reindex = self._need_reindex(
                    dims,
                    joined_index,
                    matching_indexes,
                    joined_index_vars,
                    matching_index_vars,
                )
            else:
                if len(matching_indexes) > 1:
                    need_reindex = self._need_reindex(
                        dims,
                        matching_indexes[0],
                        matching_indexes[1:],
                        matching_index_vars[0],
                        matching_index_vars[1:],
                    )
                else:
                    need_reindex = False
                if need_reindex:
                    if self.join == "exact":
                        raise ValueError(
                            "cannot align objects with join='exact' where "
                            "index/labels/sizes are not equal along "
                            "these coordinates (dimensions): "
                            + ", ".join(f"{name!r} {dims!r}" for name, dims in key[0])
                        )
                    joiner = self._get_index_joiner(index_cls)
                    joined_index = joiner(matching_indexes)
                    if self.join == "left":
                        joined_index_vars = matching_index_vars[0]
                    elif self.join == "right":
                        joined_index_vars = matching_index_vars[-1]
                    else:
                        joined_index_vars = joined_index.create_variables()
                else:
                    joined_index = matching_indexes[0]
                    joined_index_vars = matching_index_vars[0]

            reindex[key] = need_reindex
            aligned_indexes[key] = joined_index
            aligned_index_vars[key] = joined_index_vars

            for name, var in joined_index_vars.items():
                new_indexes[name] = joined_index
                new_index_vars[name] = var

        self.aligned_indexes = aligned_indexes
        self.aligned_index_vars = aligned_index_vars
        self.reindex = reindex
        self.new_indexes = Indexes(new_indexes, new_index_vars)

    def assert_unindexed_dim_sizes_equal(self):
        for dim, sizes in self.unindexed_dim_sizes.items():
            index_size = self.new_indexes.dims.get(dim)
            if index_size is not None:
                sizes.add(index_size)
                add_err_msg = (
                    f" (note: an index is found along that dimension "
                    f"with size={index_size!r})"
                )
            else:
                add_err_msg = ""
            if len(sizes) > 1:
                raise ValueError(
                    f"cannot reindex or align along unlabeled dimension {dim!r} "
                    f"because of conflicting dimension sizes: {sizes!r}" + add_err_msg
                )

    def override_indexes(self) -> "DataAligned":
        objects = list(self.objects)

        for i, obj in enumerate(objects[1:]):
            new_indexes = {}
            new_variables = {}
            matching_indexes = self.objects_matching_indexes[i + 1]

            for key, aligned_idx in self.aligned_indexes.items():
                obj_idx = matching_indexes.get(key)
                if obj_idx is not None:
                    for name, var in self.aligned_index_vars[key].items():
                        new_indexes[name] = aligned_idx
                        new_variables[name] = var

            objects[i + 1] = obj._overwrite_indexes(new_indexes, new_variables)

        return tuple(objects)

    def _reindex_one(self, obj: "T_Xarray", matching_indexes) -> "T_Xarray":
        from .dataarray import DataArray

        new_variables = {}
        new_indexes = {}
        dim_reindexers = {}
        added_new_indexes = False

        for key, aligned_idx in self.aligned_indexes.items():
            index_vars = self.aligned_index_vars[key]
            obj_idx = matching_indexes.get(key)
            if obj_idx is None:
                # add the index if it relates to unindexed dimensions in obj
                index_vars_dims = set(
                    d for var in index_vars.values() for d in var.dims
                )
                if index_vars_dims <= set(obj.dims):
                    obj_idx = aligned_idx
                    added_new_indexes = True
            if obj_idx is not None:
                for name, var in index_vars.items():
                    new_indexes[name] = aligned_idx
                    new_variables[name] = var
                if self.reindex[key]:
                    indexers = obj_idx.reindex_like(aligned_idx, **self.reindex_kwargs)
                    dim_reindexers.update(indexers)

        if not dim_reindexers:
            # fast path for no reindexing necessary
            new_obj = obj.copy(deep=self.copy)
            if added_new_indexes:
                new_obj = new_obj._overwrite_indexes(new_indexes, new_variables)
        else:
            if isinstance(obj, DataArray):
                ds_obj = obj._to_temp_dataset()
            else:
                ds_obj = obj

            # Negative values in dim_indexers mean values missing in the new index
            masked_dims = [
                dim for dim, indxr in dim_reindexers.items() if (indxr < 0).any()
            ]
            unchanged_dims = [dim not in dim_reindexers for dim in obj.dims]

            for name, var in ds_obj.variables.items():
                if name in new_variables:
                    continue

                if isinstance(self.fill_value, dict):
                    fill_value = self.fill_value.get(name, dtypes.NA)
                else:
                    fill_value = self.fill_value

                if self.sparse:
                    var = var._as_sparse(fill_value=fill_value)
                key = tuple(
                    slice(None)
                    if d in unchanged_dims
                    else dim_reindexers.get(d, slice(None))
                    for d in var.dims
                )
                needs_masking = any(d in masked_dims for d in var.dims)

                if needs_masking:
                    new_var = var._getitem_with_mask(key, fill_value=fill_value)
                elif all(is_full_slice(k) for k in key):
                    # no reindexing necessary
                    # here we need to manually deal with copying data, since
                    # we neither created a new ndarray nor used fancy indexing
                    new_var = var.copy(deep=self.copy)
                else:
                    new_var = var[key]

                new_variables[name] = new_var

            new_coord_names = ds_obj._coord_names | set(new_indexes)
            new_ds_obj = ds_obj._replace_with_new_dims(
                new_variables, new_coord_names, indexes=new_indexes
            )

            if isinstance(obj, DataArray):
                new_obj = obj._from_temp_dataset(new_ds_obj)
            else:
                new_obj = new_ds_obj

        new_obj.encoding = obj.encoding
        return new_obj

    def reindex_all(self) -> "DataAligned":
        result = []

        for obj, matching_indexes in zip(self.objects, self.objects_matching_indexes):
            result.append(self._reindex_one(obj, matching_indexes))

        return tuple(result)

    def align(self) -> "DataAligned":
        if not self.indexes and len(self.objects) == 1:
            # fast path for the trivial case
            (obj,) = self.objects
            return (obj.copy(deep=self.copy),)

        self.find_matching_indexes()
        self.find_matching_unindexed_dims()
        self.assert_no_index_conflict()
        self.align_indexes()
        self.assert_unindexed_dim_sizes_equal()

        if self.join == "override":
            return self.override_indexes()
        else:
            return self.reindex_all()


def align(
    *objects: Union["Dataset", "DataArray"],
    join="inner",
    copy=True,
    indexes=None,
    exclude=frozenset(),
    fill_value=dtypes.NA,
) -> "DataAligned":
    """
    Given any number of Dataset and/or DataArray objects, returns new
    objects with aligned indexes and dimension sizes.

    Array from the aligned objects are suitable as input to mathematical
    operators, because along each dimension they have the same index and size.

    Missing values (if ``join != 'inner'``) are filled with ``fill_value``.
    The default fill value is NaN.

    Parameters
    ----------
    *objects : Dataset or DataArray
        Objects to align.
    join : {"outer", "inner", "left", "right", "exact", "override"}, optional
        Method for joining the indexes of the passed objects along each
        dimension:

        - "outer": use the union of object indexes
        - "inner": use the intersection of object indexes
        - "left": use indexes from the first object with each dimension
        - "right": use indexes from the last object with each dimension
        - "exact": instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - "override": if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.
    copy : bool, optional
        If ``copy=True``, data in the return values is always copied. If
        ``copy=False`` and reindexing is unnecessary, or can be performed with
        only slice operations, then the output may share memory with the input.
        In either case, new xarray objects are always returned.
    indexes : dict-like, optional
        Any indexes explicitly provided with the `indexes` argument should be
        used in preference to the aligned indexes.
    exclude : sequence of str, optional
        Dimensions that must be excluded from alignment
    fill_value : scalar or dict-like, optional
        Value to use for newly missing values. If a dict-like, maps
        variable names to fill values. Use a data array's name to
        refer to its values.

    Returns
    -------
    aligned : DataArray or Dataset
        Tuple of objects with the same type as `*objects` with aligned
        coordinates.

    Raises
    ------
    ValueError
        If any dimensions without labels on the arguments have different sizes,
        or a different size than the size of the aligned dimension labels.

    Examples
    --------
    >>> x = xr.DataArray(
    ...     [[25, 35], [10, 24]],
    ...     dims=("lat", "lon"),
    ...     coords={"lat": [35.0, 40.0], "lon": [100.0, 120.0]},
    ... )
    >>> y = xr.DataArray(
    ...     [[20, 5], [7, 13]],
    ...     dims=("lat", "lon"),
    ...     coords={"lat": [35.0, 42.0], "lon": [100.0, 120.0]},
    ... )

    >>> x
    <xarray.DataArray (lat: 2, lon: 2)>
    array([[25, 35],
           [10, 24]])
    Coordinates:
      * lat      (lat) float64 35.0 40.0
      * lon      (lon) float64 100.0 120.0

    >>> y
    <xarray.DataArray (lat: 2, lon: 2)>
    array([[20,  5],
           [ 7, 13]])
    Coordinates:
      * lat      (lat) float64 35.0 42.0
      * lon      (lon) float64 100.0 120.0

    >>> a, b = xr.align(x, y)
    >>> a
    <xarray.DataArray (lat: 1, lon: 2)>
    array([[25, 35]])
    Coordinates:
      * lat      (lat) float64 35.0
      * lon      (lon) float64 100.0 120.0
    >>> b
    <xarray.DataArray (lat: 1, lon: 2)>
    array([[20,  5]])
    Coordinates:
      * lat      (lat) float64 35.0
      * lon      (lon) float64 100.0 120.0

    >>> a, b = xr.align(x, y, join="outer")
    >>> a
    <xarray.DataArray (lat: 3, lon: 2)>
    array([[25., 35.],
           [10., 24.],
           [nan, nan]])
    Coordinates:
      * lat      (lat) float64 35.0 40.0 42.0
      * lon      (lon) float64 100.0 120.0
    >>> b
    <xarray.DataArray (lat: 3, lon: 2)>
    array([[20.,  5.],
           [nan, nan],
           [ 7., 13.]])
    Coordinates:
      * lat      (lat) float64 35.0 40.0 42.0
      * lon      (lon) float64 100.0 120.0

    >>> a, b = xr.align(x, y, join="outer", fill_value=-999)
    >>> a
    <xarray.DataArray (lat: 3, lon: 2)>
    array([[  25,   35],
           [  10,   24],
           [-999, -999]])
    Coordinates:
      * lat      (lat) float64 35.0 40.0 42.0
      * lon      (lon) float64 100.0 120.0
    >>> b
    <xarray.DataArray (lat: 3, lon: 2)>
    array([[  20,    5],
           [-999, -999],
           [   7,   13]])
    Coordinates:
      * lat      (lat) float64 35.0 40.0 42.0
      * lon      (lon) float64 100.0 120.0

    >>> a, b = xr.align(x, y, join="left")
    >>> a
    <xarray.DataArray (lat: 2, lon: 2)>
    array([[25, 35],
           [10, 24]])
    Coordinates:
      * lat      (lat) float64 35.0 40.0
      * lon      (lon) float64 100.0 120.0
    >>> b
    <xarray.DataArray (lat: 2, lon: 2)>
    array([[20.,  5.],
           [nan, nan]])
    Coordinates:
      * lat      (lat) float64 35.0 40.0
      * lon      (lon) float64 100.0 120.0

    >>> a, b = xr.align(x, y, join="right")
    >>> a
    <xarray.DataArray (lat: 2, lon: 2)>
    array([[25., 35.],
           [nan, nan]])
    Coordinates:
      * lat      (lat) float64 35.0 42.0
      * lon      (lon) float64 100.0 120.0
    >>> b
    <xarray.DataArray (lat: 2, lon: 2)>
    array([[20,  5],
           [ 7, 13]])
    Coordinates:
      * lat      (lat) float64 35.0 42.0
      * lon      (lon) float64 100.0 120.0

    >>> a, b = xr.align(x, y, join="exact")
    Traceback (most recent call last):
    ...
        "indexes along dimension {!r} are not equal".format(dim)
    ValueError: indexes along dimension 'lat' are not equal

    >>> a, b = xr.align(x, y, join="override")
    >>> a
    <xarray.DataArray (lat: 2, lon: 2)>
    array([[25, 35],
           [10, 24]])
    Coordinates:
      * lat      (lat) float64 35.0 40.0
      * lon      (lon) float64 100.0 120.0
    >>> b
    <xarray.DataArray (lat: 2, lon: 2)>
    array([[20,  5],
           [ 7, 13]])
    Coordinates:
      * lat      (lat) float64 35.0 40.0
      * lon      (lon) float64 100.0 120.0

    """
    aligner = Aligner(
        objects,
        join=join,
        copy=copy,
        indexes=indexes,
        exclude=exclude,
        fill_value=fill_value,
    )

    return aligner.align()


def deep_align(
    objects,
    join="inner",
    copy=True,
    indexes=None,
    exclude=frozenset(),
    raise_on_invalid=True,
    fill_value=dtypes.NA,
):
    """Align objects for merging, recursing into dictionary values.

    This function is not public API.
    """
    from .dataarray import DataArray
    from .dataset import Dataset

    if indexes is None:
        indexes = {}

    def is_alignable(obj):
        return isinstance(obj, (DataArray, Dataset))

    positions = []
    keys = []
    out = []
    targets = []
    no_key = object()
    not_replaced = object()
    for position, variables in enumerate(objects):
        if is_alignable(variables):
            positions.append(position)
            keys.append(no_key)
            targets.append(variables)
            out.append(not_replaced)
        elif is_dict_like(variables):
            current_out = {}
            for k, v in variables.items():
                if is_alignable(v) and k not in indexes:
                    # Skip variables in indexes for alignment, because these
                    # should to be overwritten instead:
                    # https://github.com/pydata/xarray/issues/725
                    # https://github.com/pydata/xarray/issues/3377
                    # TODO(shoyer): doing this here feels super-hacky -- can we
                    # move it explicitly into merge instead?
                    positions.append(position)
                    keys.append(k)
                    targets.append(v)
                    current_out[k] = not_replaced
                else:
                    current_out[k] = v
            out.append(current_out)
        elif raise_on_invalid:
            raise ValueError(
                "object to align is neither an xarray.Dataset, "
                "an xarray.DataArray nor a dictionary: {!r}".format(variables)
            )
        else:
            out.append(variables)

    aligned = align(
        *targets,
        join=join,
        copy=copy,
        indexes=indexes,
        exclude=exclude,
        fill_value=fill_value,
    )

    for position, key, aligned_obj in zip(positions, keys, aligned):
        if key is no_key:
            out[position] = aligned_obj
        else:
            out[position][key] = aligned_obj

    # something went wrong: we should have replaced all sentinel values
    for arg in out:
        assert arg is not not_replaced
        if is_dict_like(arg):
            assert all(value is not not_replaced for value in arg.values())

    return out


def reindex(
    obj: "T_Xarray",
    indexers: Mapping[Any, Any],
    method: str = None,
    tolerance: Number = None,
    copy: bool = True,
    fill_value: Any = dtypes.NA,
    sparse: bool = False,
) -> "T_Xarray":
    """Re-index either a Dataset or a DataArray.

    Not public API.

    """
    bad_keys = [k for k in indexers if k not in obj.xindexes and k not in obj.dims]
    if bad_keys:
        raise ValueError(
            f"indexer keys {bad_keys} do not correspond to any indexed coordinate "
            "or unindexed dimension in the object to reindex"
        )

    aligner = Aligner(
        (obj,),
        indexes=indexers,
        method=method,
        tolerance=tolerance,
        copy=copy,
        fill_value=fill_value,
        sparse=sparse,
    )

    return cast("T_Xarray", aligner.align()[0])


def reindex_like_indexers(
    target: "Union[DataArray, Dataset]", other: "Union[DataArray, Dataset]"
) -> Dict[Hashable, pd.Index]:
    """Extract indexers to align target with other.

    Not public API.

    Parameters
    ----------
    target : Dataset or DataArray
        Object to be aligned.
    other : Dataset or DataArray
        Object to be aligned with.

    Returns
    -------
    Dict[Hashable, pandas.Index] providing indexes for reindex keyword
    arguments.

    Raises
    ------
    ValueError
        If any dimensions without labels have different sizes.
    """
    # TODO: benbovy - flexible indexes: https://github.com/pydata/xarray/issues/5647
    # this doesn't support yet indexes other than pd.Index
    indexers = {
        k: v.to_pandas_index() for k, v in other.xindexes.items() if k in target.dims
    }

    for dim in other.dims:
        if dim not in indexers and dim in target.dims:
            other_size = other.sizes[dim]
            target_size = target.sizes[dim]
            if other_size != target_size:
                raise ValueError(
                    "different size for unlabeled "
                    f"dimension on argument {dim!r}: {other_size!r} vs {target_size!r}"
                )
    return indexers


def reindex_variables(
    variables: Mapping[Any, Variable],
    sizes: Mapping[Any, int],
    indexes: Mapping[Any, Index],
    indexers: Mapping,
    method: Optional[str] = None,
    tolerance: Any = None,
    copy: bool = True,
    fill_value: Optional[Any] = dtypes.NA,
    sparse: bool = False,
) -> Tuple[Dict[Hashable, Variable], Dict[Hashable, Index]]:
    """Conform a dictionary of aligned variables onto a new set of variables,
    filling in missing values with NaN.

    Not public API.

    Parameters
    ----------
    variables : dict-like
        Dictionary of xarray.Variable objects.
    sizes : dict-like
        Dictionary from dimension names to integer sizes.
    indexes : dict-like
        Dictionary of indexes associated with variables.
    indexers : dict
        Dictionary with keys given by dimension names and values given by
        arrays of coordinates tick labels. Any mis-matched coordinate values
        will be filled in with NaN, and any mis-matched dimension names will
        simply be ignored.
    method : {None, 'nearest', 'pad'/'ffill', 'backfill'/'bfill'}, optional
        Method to use for filling index values in ``indexers`` not found in
        this dataset:
          * None (default): don't fill gaps
          * pad / ffill: propagate last valid index value forward
          * backfill / bfill: propagate next valid index value backward
          * nearest: use nearest valid index value
    tolerance : optional
        Maximum distance between original and new labels for inexact matches.
        The values of the index at the matching locations must satisfy the
        equation ``abs(index[indexer] - target) <= tolerance``.
    copy : bool, optional
        If ``copy=True``, data in the return values is always copied. If
        ``copy=False`` and reindexing is unnecessary, or can be performed
        with only slice operations, then the output may share memory with
        the input. In either case, new xarray objects are always returned.
    fill_value : scalar, optional
        Value to use for newly missing values
    sparse : bool, optional
        Use an sparse-array

    Returns
    -------
    reindexed : dict
        Dict of reindexed variables.
    new_indexes : dict
        Dict of indexes associated with the reindexed variables.
    """
    from .dataarray import DataArray

    # create variables for the new dataset
    reindexed: Dict[Hashable, Variable] = {}

    # build up indexers for assignment along each dimension
    int_indexers = {}
    new_indexes = dict(indexes)
    masked_dims = set()
    unchanged_dims = set()

    for dim, indexer in indexers.items():
        if isinstance(indexer, DataArray) and indexer.dims != (dim,):
            raise ValueError(
                "Indexer has dimensions {:s} that are different "
                "from that to be indexed along {:s}".format(str(indexer.dims), dim)
            )

        var_meta = {dim: {"dtype": getattr(indexer, "dtype", None)}}
        if dim in variables:
            var = variables[dim]
            var_meta[dim].update({"attrs": var.attrs, "encoding": var.encoding})

        target = safe_cast_to_index(indexers[dim]).rename(dim)
        idx, idx_vars = PandasIndex.from_pandas_index(target, dim, var_meta=var_meta)
        new_indexes[dim] = idx
        reindexed.update(idx_vars)

        if dim in indexes:
            # TODO (benbovy - flexible indexes): support other indexes than pd.Index?
            index = indexes[dim].to_pandas_index()

            if not index.is_unique:
                raise ValueError(
                    f"cannot reindex or align along dimension {dim!r} because the "
                    "index has duplicate values"
                )

            int_indexer = get_indexer_nd(index, target, method, tolerance)

            # We uses negative values from get_indexer_nd to signify
            # values that are missing in the index.
            if (int_indexer < 0).any():
                masked_dims.add(dim)
            elif np.array_equal(int_indexer, np.arange(len(index))):
                unchanged_dims.add(dim)

            int_indexers[dim] = int_indexer

    for dim in sizes:
        if dim not in indexes and dim in indexers:
            existing_size = sizes[dim]
            new_size = indexers[dim].size
            if existing_size != new_size:
                raise ValueError(
                    f"cannot reindex or align along dimension {dim!r} without an "
                    f"index because its size {existing_size!r} is different from the size of "
                    f"the new index {new_size!r}"
                )

    for name, var in variables.items():
        if name not in indexers:
            if isinstance(fill_value, dict):
                fill_value_ = fill_value.get(name, dtypes.NA)
            else:
                fill_value_ = fill_value

            if sparse:
                var = var._as_sparse(fill_value=fill_value_)
            key = tuple(
                slice(None) if d in unchanged_dims else int_indexers.get(d, slice(None))
                for d in var.dims
            )
            needs_masking = any(d in masked_dims for d in var.dims)

            if needs_masking:
                new_var = var._getitem_with_mask(key, fill_value=fill_value_)
            elif all(is_full_slice(k) for k in key):
                # no reindexing necessary
                # here we need to manually deal with copying data, since
                # we neither created a new ndarray nor used fancy indexing
                new_var = var.copy(deep=copy)
            else:
                new_var = var[key]

            reindexed[name] = new_var

    return reindexed, new_indexes


def _get_broadcast_dims_map_common_coords(args, exclude):

    common_coords = {}
    dims_map = {}
    for arg in args:
        for dim in arg.dims:
            if dim not in common_coords and dim not in exclude:
                dims_map[dim] = arg.sizes[dim]
                if dim in arg.coords:
                    common_coords[dim] = arg.coords[dim].variable

    return dims_map, common_coords


def _broadcast_helper(arg, exclude, dims_map, common_coords):

    from .dataarray import DataArray
    from .dataset import Dataset

    def _set_dims(var):
        # Add excluded dims to a copy of dims_map
        var_dims_map = dims_map.copy()
        for dim in exclude:
            with suppress(ValueError):
                # ignore dim not in var.dims
                var_dims_map[dim] = var.shape[var.dims.index(dim)]

        return var.set_dims(var_dims_map)

    def _broadcast_array(array):
        data = _set_dims(array.variable)
        coords = dict(array.coords)
        coords.update(common_coords)
        return DataArray(data, coords, data.dims, name=array.name, attrs=array.attrs)

    def _broadcast_dataset(ds):
        data_vars = {k: _set_dims(ds.variables[k]) for k in ds.data_vars}
        coords = dict(ds.coords)
        coords.update(common_coords)
        return Dataset(data_vars, coords, ds.attrs)

    if isinstance(arg, DataArray):
        return _broadcast_array(arg)
    elif isinstance(arg, Dataset):
        return _broadcast_dataset(arg)
    else:
        raise ValueError("all input must be Dataset or DataArray objects")


def broadcast(*args, exclude=None):
    """Explicitly broadcast any number of DataArray or Dataset objects against
    one another.

    xarray objects automatically broadcast against each other in arithmetic
    operations, so this function should not be necessary for normal use.

    If no change is needed, the input data is returned to the output without
    being copied.

    Parameters
    ----------
    *args : DataArray or Dataset
        Arrays to broadcast against each other.
    exclude : sequence of str, optional
        Dimensions that must not be broadcasted

    Returns
    -------
    broadcast : tuple of DataArray or tuple of Dataset
        The same data as the input arrays, but with additional dimensions
        inserted so that all data arrays have the same dimensions and shape.

    Examples
    --------
    Broadcast two data arrays against one another to fill out their dimensions:

    >>> a = xr.DataArray([1, 2, 3], dims="x")
    >>> b = xr.DataArray([5, 6], dims="y")
    >>> a
    <xarray.DataArray (x: 3)>
    array([1, 2, 3])
    Dimensions without coordinates: x
    >>> b
    <xarray.DataArray (y: 2)>
    array([5, 6])
    Dimensions without coordinates: y
    >>> a2, b2 = xr.broadcast(a, b)
    >>> a2
    <xarray.DataArray (x: 3, y: 2)>
    array([[1, 1],
           [2, 2],
           [3, 3]])
    Dimensions without coordinates: x, y
    >>> b2
    <xarray.DataArray (x: 3, y: 2)>
    array([[5, 6],
           [5, 6],
           [5, 6]])
    Dimensions without coordinates: x, y

    Fill out the dimensions of all data variables in a dataset:

    >>> ds = xr.Dataset({"a": a, "b": b})
    >>> (ds2,) = xr.broadcast(ds)  # use tuple unpacking to extract one dataset
    >>> ds2
    <xarray.Dataset>
    Dimensions:  (x: 3, y: 2)
    Dimensions without coordinates: x, y
    Data variables:
        a        (x, y) int64 1 1 2 2 3 3
        b        (x, y) int64 5 6 5 6 5 6
    """

    if exclude is None:
        exclude = set()
    args = align(*args, join="outer", copy=False, exclude=exclude)

    dims_map, common_coords = _get_broadcast_dims_map_common_coords(args, exclude)
    result = [_broadcast_helper(arg, exclude, dims_map, common_coords) for arg in args]

    return tuple(result)
