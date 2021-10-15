import functools
import operator
from collections import defaultdict
from contextlib import suppress
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
)

import numpy as np
import pandas as pd

from . import dtypes
from .indexes import Index, Indexes, PandasIndex, get_indexer_nd
from .utils import is_dict_like, is_full_slice, maybe_coerce_to_str, safe_cast_to_index
from .variable import Variable

if TYPE_CHECKING:
    from .common import DataWithCoords
    from .dataarray import DataArray
    from .dataset import Dataset

    DataAlignable = TypeVar("DataAlignable", bound=DataWithCoords)


def _get_joiner(join, index_cls):
    if join == "outer":
        return functools.partial(functools.reduce, index_cls.union)
    elif join == "inner":
        return functools.partial(functools.reduce, index_cls.intersection)
    elif join == "left":
        return operator.itemgetter(0)
    elif join == "right":
        return operator.itemgetter(-1)
    elif join == "exact":
        # We cannot return a function to "align" in this case, because it needs
        # access to the dimension name to give a good error message.
        return None
    elif join == "override":
        # We rewrite all indexes and then use join='left'
        return operator.itemgetter(0)
    else:
        raise ValueError(f"invalid value for join: {join}")


def _override_indexes(objects, all_indexes, exclude):
    for dim, dim_indexes in all_indexes.items():
        if dim not in exclude:
            lengths = {
                getattr(index, "size", index.to_pandas_index().size)
                for index in dim_indexes
            }
            if len(lengths) != 1:
                raise ValueError(
                    f"Indexes along dimension {dim!r} don't have the same length."
                    " Cannot use join='override'."
                )

    objects = list(objects)
    for idx, obj in enumerate(objects[1:]):
        new_indexes = {
            dim: all_indexes[dim][0] for dim in obj.xindexes if dim not in exclude
        }

        # TODO: benbovy - explicit indexes: not refactored yet!
        objects[idx + 1] = obj._overwrite_indexes(new_indexes)

    return objects


class Alignator:
    """Implements all the complex logic for the alignment of Xarray objects.

    For internal use only, not public API.

    """

    CoordNamesAndDims = FrozenSet[Tuple[Hashable, Tuple[Hashable, ...]]]
    MatchingIndexKey = Tuple[CoordNamesAndDims, Type[Index]]
    NormalizedIndexes = Dict[MatchingIndexKey, Index]
    NormalizedIndexVars = Dict[MatchingIndexKey, Dict[Hashable, Variable]]
    AlignedObjects = Tuple[Union["Dataset", "DataArray"], ...]

    objects: List[Union["Dataset", "DataArray"]]
    join: str
    exclude_dims: FrozenSet
    reindex_dims: Set
    indexes: Dict[MatchingIndexKey, Index]
    index_vars: Dict[MatchingIndexKey, Dict[Hashable, Variable]]
    all_indexes: Dict[MatchingIndexKey, List[Index]]
    all_index_vars: Dict[MatchingIndexKey, List[Dict[Hashable, Variable]]]
    unindexed_dim_sizes: Dict[Hashable, Set]
    aligned_indexes: Indexes[Index]

    def __init__(
        self,
        objects: List[Union["Dataset", "DataArray"]],
        join: str,
        indexes: Union[Mapping[Any, Any], None],
        exclude: Union[str, Set, Sequence],
    ):
        self.objects = objects

        if join not in ["inner", "outer", "overwrite", "exact", "left", "right"]:
            raise ValueError(f"invalid value for join: {join}")
        self.join = join

        if isinstance(exclude, str):
            exclude = [exclude]
        self.exclude_dims = frozenset(exclude)

        self.reindex_dims = set()

        if indexes is None:
            indexes = {}
        self.indexes, self.index_vars = self._normalize_indexes(indexes)

        self.all_indexes = defaultdict(list)
        self.all_index_vars = defaultdict(list)
        self.unindexed_dim_sizes = defaultdict(set)

    def _normalize_indexes(
        self,
        indexes: Mapping[Any, Any],
    ) -> Tuple[NormalizedIndexes, NormalizedIndexVars]:
        """Normalize the indexes used for alignment.

        Return dictionaries of xarray Index objects and coordinate variables
        such that we can group matching indexes based on the dictionary keys.

        """
        if isinstance(indexes, Indexes):
            variables = dict(indexes.variables)
        else:
            variables = {}

        xr_indexes = {}
        for k, idx in indexes.items():
            if not isinstance(idx, Index):
                pd_idx = safe_cast_to_index(idx).copy()
                pd_idx.name = k
                idx, _ = PandasIndex.from_pandas_index(pd_idx, k)
                variables.update(idx.create_variables())
            xr_indexes[k] = idx

        normalized_indexes = {}
        normalized_index_vars = {}
        for idx, index_vars in Indexes(xr_indexes, variables).group_by_index():
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

            key = (frozenset(coord_names_and_dims), type(idx))
            normalized_indexes[key] = idx
            normalized_index_vars[key] = index_vars

        return normalized_indexes, normalized_index_vars

    def find_matching_indexes(self):
        for obj in self.objects:
            obj_indexes, obj_index_vars = self._normalize_indexes(obj.xindexes)
            for key, idx in obj_indexes.items():
                self.all_indexes[key].append(idx)
                self.all_index_vars[key].append(obj_index_vars[key])

    def find_matching_unindexed_dims(self):
        for obj in self.objects:
            for dim in obj.dims:
                if dim not in self.exclude_dims and dim not in obj.xindexes.dims:
                    self.unindexed_dim_sizes[dim].add(obj.sizes[dim])

    def assert_no_index_conflict(self):
        """Check for uniqueness of both coordinate and dimension names accross all sets
        of matching indexes.

        We need to make sure that all indexes used for alignment are fully compatible
        and do not conflict each other.

        """
        matching_keys = set(self.all_indexes) | set(self.indexes)

        coord_count = defaultdict(int)
        dim_count = defaultdict(int)
        for coord_names_dims, _ in matching_keys:
            dims_set = set()
            for name, dims in coord_names_dims:
                coord_count[name] += 1
                dims_set |= dims
            for dim in dims_set:
                dim_count[dim] += 1

        for count, msg in [(coord_count, "coordinates"), (dim_count, "dimensions")]:
            dup = {k: v for k, v in count.items() if v > 1}
            if dup:
                items_msg = ", ".join(
                    f"{k} ({v} conflicting indexes)" for k, v in dup.items()
                )
                raise ValueError(
                    "cannot align objects with conflicting indexes found for "
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
            return functools.partial(functools.reduce, index_cls.join, how=self.join)
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
        reindex_dims = set()

        for key, matching_indexes in self.all_indexes.items():
            matching_index_vars = self.all_index_vars[key]
            dims = set(
                [d for coord in matching_index_vars[0].values() for d in coord.dims]
            )
            index_cls = key[1]

            if key in self.indexes:
                joined_index = self.indexes[key]
                joined_index_vars = self.index_vars[key]
                reindex = self._need_reindex(
                    dims,
                    joined_index,
                    matching_indexes,
                    joined_index_vars,
                    matching_index_vars,
                )
            else:
                reindex = self._need_reindex(
                    dims,
                    matching_indexes[0],
                    matching_indexes[1:],
                    matching_index_vars[0],
                    matching_index_vars[1:],
                )
                if reindex:
                    if self.join == "exact":
                        # TODO: more informative error message
                        raise ValueError(
                            "cannot align objects with join='exact' where "
                            "index/labels/sizes are not equal along "
                            "these coordinates (dimensions): "
                            + ", ".join(f"{name!r} {dims!r}" for name, dims in key[0])
                        )
                    joiner = self._get_index_joiner(index_cls)
                    try:
                        joined_index = joiner(matching_indexes)
                        if self.join == "left":
                            joined_index_vars = matching_index_vars[0]
                        elif self.join == "right":
                            joined_index_vars = matching_index_vars[-1]
                        else:
                            joined_index_vars = joined_index.create_variables()
                    except NotImplementedError:
                        raise TypeError(
                            f"{index_cls.__qualname__} doesn't support alignment "
                            "with inner/outer join method"
                        )
                else:
                    joined_index = matching_indexes[0]
                    joined_index_vars = matching_index_vars[0]

            for name, var in joined_index_vars.items():
                aligned_indexes[name] = joined_index
                aligned_index_vars[name] = var

            if reindex:
                reindex_dims |= dims

        self.aligned_indexes = Indexes(aligned_indexes, aligned_index_vars)
        self.reindex_dims = reindex_dims

    def assert_unindexed_dim_sizes_equal(self):
        for dim, sizes in self.unindexed_dim_sizes.items():
            index_size = self.aligned_indexes.dims.get(dim)
            if index_size is not None:
                sizes.add(index_size)
                add_err_msg = (
                    f" (note: indexed labels also found for dimension {dim!r} "
                    f"with size {index_size!r})"
                )
            else:
                add_err_msg = ""
            if len(sizes) > 1:
                raise ValueError(
                    f"arguments without labels along dimension {dim!r} cannot be "
                    f"aligned because they have different dimension sizes: {sizes!r}"
                    + add_err_msg
                )

    def reindex(self, copy: bool, fill_value: Any) -> AlignedObjects:
        result = []

        for obj in self.objects:
            valid_indexers = {}
            for dim in self.aligned_indexes.dims:
                if (
                    dim in obj.dims
                    and dim in self.reindex_dims
                    # TODO: default dim var instead?
                    and dim in self.aligned_indexes.variables
                ):
                    valid_indexers[dim] = self.aligned_indexes.variables[dim]
            if not valid_indexers:
                # fast path for no reindexing necessary
                new_obj = obj.copy(deep=copy)
            else:
                # TODO: propagate aligned indexes and index vars
                new_obj = obj.reindex(
                    copy=copy, fill_value=fill_value, indexers=valid_indexers
                )
            new_obj.encoding = obj.encoding
            result.append(new_obj)

        return tuple(result)

    def align(self, copy: bool = True, fill_value: Any = dtypes.NA) -> AlignedObjects:

        if not self.indexes and len(self.objects) == 1:
            # fast path for the trivial case
            (obj,) = self.objects
            return (obj.copy(deep=copy),)

        self.find_matching_indexes()
        self.find_matching_unindexed_dims()
        self.assert_no_index_conflict()
        self.align_indexes()
        self.assert_unindexed_dim_sizes_equal()
        return self.reindex(copy, fill_value)


def align(
    *objects: "DataAlignable",
    join="inner",
    copy=True,
    indexes=None,
    exclude=frozenset(),
    fill_value=dtypes.NA,
) -> Tuple["DataAlignable", ...]:
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
    if indexes is None:
        indexes = {}

    if not indexes and len(objects) == 1:
        # fast path for the trivial case
        (obj,) = objects
        return (obj.copy(deep=copy),)

    all_indexes = defaultdict(list)
    all_coords = defaultdict(list)
    unlabeled_dim_sizes = defaultdict(set)
    for obj in objects:
        for dim in obj.dims:
            if dim not in exclude:
                all_coords[dim].append(obj.coords[dim])
                try:
                    index = obj.xindexes[dim]
                except KeyError:
                    unlabeled_dim_sizes[dim].add(obj.sizes[dim])
                else:
                    all_indexes[dim].append(index)

    if join == "override":
        objects = _override_indexes(objects, all_indexes, exclude)

    # We don't reindex over dimensions with all equal indexes for two reasons:
    # - It's faster for the usual case (already aligned objects).
    # - It ensures it's possible to do operations that don't require alignment
    #   on indexes with duplicate values (which cannot be reindexed with
    #   pandas). This is useful, e.g., for overwriting such duplicate indexes.
    joined_indexes = {}
    for dim, matching_indexes in all_indexes.items():
        if dim in indexes:
            index, _ = PandasIndex.from_pandas_index(
                safe_cast_to_index(indexes[dim]), dim
            )
            if (
                any(not index.equals(other) for other in matching_indexes)
                or dim in unlabeled_dim_sizes
            ):
                joined_indexes[dim] = indexes[dim]
        else:
            if (
                any(
                    not matching_indexes[0].equals(other)
                    for other in matching_indexes[1:]
                )
                or dim in unlabeled_dim_sizes
            ):
                if join == "exact":
                    raise ValueError(f"indexes along dimension {dim!r} are not equal")
                joiner = _get_joiner(join, type(matching_indexes[0]))
                index = joiner(matching_indexes)
                # make sure str coords are not cast to object
                index = maybe_coerce_to_str(index.to_pandas_index(), all_coords[dim])
                joined_indexes[dim] = index
            else:
                index = all_coords[dim][0]

        if dim in unlabeled_dim_sizes:
            unlabeled_sizes = unlabeled_dim_sizes[dim]
            # TODO: benbovy - flexible indexes: https://github.com/pydata/xarray/issues/5647
            if isinstance(index, PandasIndex):
                labeled_size = index.to_pandas_index().size
            else:
                labeled_size = index.size
            if len(unlabeled_sizes | {labeled_size}) > 1:
                raise ValueError(
                    f"arguments without labels along dimension {dim!r} cannot be "
                    f"aligned because they have different dimension size(s) {unlabeled_sizes!r} "
                    f"than the size of the aligned dimension labels: {labeled_size!r}"
                )

    for dim, sizes in unlabeled_dim_sizes.items():
        if dim not in all_indexes and len(sizes) > 1:
            raise ValueError(
                f"arguments without labels along dimension {dim!r} cannot be "
                f"aligned because they have different dimension sizes: {sizes!r}"
            )

    result = []
    for obj in objects:
        # TODO: benbovy - flexible indexes: https://github.com/pydata/xarray/issues/5647
        valid_indexers = {}
        for k, index in joined_indexes.items():
            if k in obj.dims:
                if isinstance(index, Index):
                    valid_indexers[k] = index.to_pandas_index()
                else:
                    valid_indexers[k] = index
        if not valid_indexers:
            # fast path for no reindexing necessary
            new_obj = obj.copy(deep=copy)
        else:
            new_obj = obj.reindex(
                copy=copy, fill_value=fill_value, indexers=valid_indexers
            )
        new_obj.encoding = obj.encoding
        result.append(new_obj)

    return tuple(result)


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
