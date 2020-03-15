import itertools
import warnings
from collections import Counter
from textwrap import dedent

import pandas as pd

from . import dtypes
from .concat import concat
from .dataarray import DataArray
from .dataset import Dataset
from .merge import merge


def _infer_concat_order_from_positions(datasets):
    combined_ids = dict(_infer_tile_ids_from_nested_list(datasets, ()))
    return combined_ids


def _infer_tile_ids_from_nested_list(entry, current_pos):
    """
    Given a list of lists (of lists...) of objects, returns a iterator
    which returns a tuple containing the index of each object in the nested
    list structure as the key, and the object. This can then be called by the
    dict constructor to create a dictionary of the objects organised by their
    position in the original nested list.

    Recursively traverses the given structure, while keeping track of the
    current position. Should work for any type of object which isn't a list.

    Parameters
    ----------
    entry : list[list[obj, obj, ...], ...]
        List of lists of arbitrary depth, containing objects in the order
        they are to be concatenated.

    Returns
    -------
    combined_tile_ids : dict[tuple(int, ...), obj]
    """

    if isinstance(entry, list):
        for i, item in enumerate(entry):
            yield from _infer_tile_ids_from_nested_list(item, current_pos + (i,))
    else:
        yield current_pos, entry


def _infer_concat_order_from_coords(datasets):

    concat_dims = []
    tile_ids = [() for ds in datasets]

    # All datasets have same variables because they've been grouped as such
    ds0 = datasets[0]
    for dim in ds0.dims:

        # Check if dim is a coordinate dimension
        if dim in ds0:

            # Need to read coordinate values to do ordering
            indexes = [ds.indexes.get(dim) for ds in datasets]
            if any(index is None for index in indexes):
                raise ValueError(
                    "Every dimension needs a coordinate for "
                    "inferring concatenation order"
                )

            # If dimension coordinate values are same on every dataset then
            # should be leaving this dimension alone (it's just a "bystander")
            if not all(index.equals(indexes[0]) for index in indexes[1:]):

                # Infer order datasets should be arranged in along this dim
                concat_dims.append(dim)

                if all(index.is_monotonic_increasing for index in indexes):
                    ascending = True
                elif all(index.is_monotonic_decreasing for index in indexes):
                    ascending = False
                else:
                    raise ValueError(
                        "Coordinate variable {} is neither "
                        "monotonically increasing nor "
                        "monotonically decreasing on all datasets".format(dim)
                    )

                # Assume that any two datasets whose coord along dim starts
                # with the same value have the same coord values throughout.
                if any(index.size == 0 for index in indexes):
                    raise ValueError("Cannot handle size zero dimensions")
                first_items = pd.Index([index[0] for index in indexes])

                # Sort datasets along dim
                # We want rank but with identical elements given identical
                # position indices - they should be concatenated along another
                # dimension, not along this one
                series = first_items.to_series()
                rank = series.rank(method="dense", ascending=ascending)
                order = rank.astype(int).values - 1

                # Append positions along extra dimension to structure which
                # encodes the multi-dimensional concatentation order
                tile_ids = [
                    tile_id + (position,) for tile_id, position in zip(tile_ids, order)
                ]

    if len(datasets) > 1 and not concat_dims:
        raise ValueError(
            "Could not find any dimension coordinates to use to "
            "order the datasets for concatenation"
        )

    combined_ids = dict(zip(tile_ids, datasets))

    return combined_ids, concat_dims


def _check_dimension_depth_tile_ids(combined_tile_ids):
    """
    Check all tuples are the same length, i.e. check that all lists are
    nested to the same depth.
    """
    tile_ids = combined_tile_ids.keys()
    nesting_depths = [len(tile_id) for tile_id in tile_ids]
    if not nesting_depths:
        nesting_depths = [0]
    if not set(nesting_depths) == {nesting_depths[0]}:
        raise ValueError(
            "The supplied objects do not form a hypercube because"
            " sub-lists do not have consistent depths"
        )
    # return these just to be reused in _check_shape_tile_ids
    return tile_ids, nesting_depths


def _check_shape_tile_ids(combined_tile_ids):
    """Check all lists along one dimension are same length."""
    tile_ids, nesting_depths = _check_dimension_depth_tile_ids(combined_tile_ids)
    for dim in range(nesting_depths[0]):
        indices_along_dim = [tile_id[dim] for tile_id in tile_ids]
        occurrences = Counter(indices_along_dim)
        if len(set(occurrences.values())) != 1:
            raise ValueError(
                "The supplied objects do not form a hypercube "
                "because sub-lists do not have consistent "
                "lengths along dimension" + str(dim)
            )


def _combine_nd(
    combined_ids,
    concat_dims,
    data_vars="all",
    coords="different",
    compat="no_conflicts",
    fill_value=dtypes.NA,
    join="outer",
):
    """
    Combines an N-dimensional structure of datasets into one by applying a
    series of either concat and merge operations along each dimension.

    No checks are performed on the consistency of the datasets, concat_dims or
    tile_IDs, because it is assumed that this has already been done.

    Parameters
    ----------
    combined_ids : Dict[Tuple[int, ...]], xarray.Dataset]
        Structure containing all datasets to be concatenated with "tile_IDs" as
        keys, which specify position within the desired final combined result.
    concat_dims : sequence of str
        The dimensions along which the datasets should be concatenated. Must be
        in order, and the length must match the length of the tuples used as
        keys in combined_ids. If the string is a dimension name then concat
        along that dimension, if it is None then merge.

    Returns
    -------
    combined_ds : xarray.Dataset
    """

    example_tile_id = next(iter(combined_ids.keys()))

    n_dims = len(example_tile_id)
    if len(concat_dims) != n_dims:
        raise ValueError(
            "concat_dims has length {} but the datasets "
            "passed are nested in a {}-dimensional structure".format(
                len(concat_dims), n_dims
            )
        )

    # Each iteration of this loop reduces the length of the tile_ids tuples
    # by one. It always combines along the first dimension, removing the first
    # element of the tuple
    for concat_dim in concat_dims:
        combined_ids = _combine_all_along_first_dim(
            combined_ids,
            dim=concat_dim,
            data_vars=data_vars,
            coords=coords,
            compat=compat,
            fill_value=fill_value,
            join=join,
        )
    (combined_ds,) = combined_ids.values()
    return combined_ds


def _combine_all_along_first_dim(
    combined_ids, dim, data_vars, coords, compat, fill_value=dtypes.NA, join="outer"
):

    # Group into lines of datasets which must be combined along dim
    # need to sort by _new_tile_id first for groupby to work
    # TODO: is the sorted need?
    combined_ids = dict(sorted(combined_ids.items(), key=_new_tile_id))
    grouped = itertools.groupby(combined_ids.items(), key=_new_tile_id)

    # Combine all of these datasets along dim
    new_combined_ids = {}
    for new_id, group in grouped:
        combined_ids = dict(sorted(group))
        datasets = combined_ids.values()
        new_combined_ids[new_id] = _combine_1d(
            datasets, dim, compat, data_vars, coords, fill_value, join
        )
    return new_combined_ids


def _combine_1d(
    datasets,
    concat_dim,
    compat="no_conflicts",
    data_vars="all",
    coords="different",
    fill_value=dtypes.NA,
    join="outer",
):
    """
    Applies either concat or merge to 1D list of datasets depending on value
    of concat_dim
    """

    if concat_dim is not None:
        try:
            combined = concat(
                datasets,
                dim=concat_dim,
                data_vars=data_vars,
                coords=coords,
                compat=compat,
                fill_value=fill_value,
                join=join,
            )
        except ValueError as err:
            if "encountered unexpected variable" in str(err):
                raise ValueError(
                    "These objects cannot be combined using only "
                    "xarray.combine_nested, instead either use "
                    "xarray.combine_by_coords, or do it manually "
                    "with xarray.concat, xarray.merge and "
                    "xarray.align"
                )
            else:
                raise
    else:
        combined = merge(datasets, compat=compat, fill_value=fill_value, join=join)

    return combined


def _new_tile_id(single_id_ds_pair):
    tile_id, ds = single_id_ds_pair
    return tile_id[1:]


def _nested_combine(
    datasets,
    concat_dims,
    compat,
    data_vars,
    coords,
    ids,
    fill_value=dtypes.NA,
    join="outer",
):

    if len(datasets) == 0:
        return Dataset()

    # Arrange datasets for concatenation
    # Use information from the shape of the user input
    if not ids:
        # Determine tile_IDs by structure of input in N-D
        # (i.e. ordering in list-of-lists)
        combined_ids = _infer_concat_order_from_positions(datasets)
    else:
        # Already sorted so just use the ids already passed
        combined_ids = dict(zip(ids, datasets))

    # Check that the inferred shape is combinable
    _check_shape_tile_ids(combined_ids)

    # Apply series of concatenate or merge operations along each dimension
    combined = _combine_nd(
        combined_ids,
        concat_dims,
        compat=compat,
        data_vars=data_vars,
        coords=coords,
        fill_value=fill_value,
        join=join,
    )
    return combined


def combine_nested(
    datasets,
    concat_dim,
    compat="no_conflicts",
    data_vars="all",
    coords="different",
    fill_value=dtypes.NA,
    join="outer",
):
    """
    Explicitly combine an N-dimensional grid of datasets into one by using a
    succession of concat and merge operations along each dimension of the grid.

    Does not sort the supplied datasets under any circumstances, so the
    datasets must be passed in the order you wish them to be concatenated. It
    does align coordinates, but different variables on datasets can cause it to
    fail under some scenarios. In complex cases, you may need to clean up your
    data and use concat/merge explicitly.

    To concatenate along multiple dimensions the datasets must be passed as a
    nested list-of-lists, with a depth equal to the length of ``concat_dims``.
    ``manual_combine`` will concatenate along the top-level list first.

    Useful for combining datasets from a set of nested directories, or for
    collecting the output of a simulation parallelized along multiple
    dimensions.

    Parameters
    ----------
    datasets : list or nested list of xarray.Dataset objects.
        Dataset objects to combine.
        If concatenation or merging along more than one dimension is desired,
        then datasets must be supplied in a nested list-of-lists.
    concat_dim : str, or list of str, DataArray, Index or None
        Dimensions along which to concatenate variables, as used by
        :py:func:`xarray.concat`.
        Set ``concat_dim=[..., None, ...]`` explicitly to disable concatenation
        and merge instead along a particular dimension.
        The position of ``None`` in the list specifies the dimension of the
        nested-list input along which to merge.
        Must be the same length as the depth of the list passed to
        ``datasets``.
    compat : {'identical', 'equals', 'broadcast_equals',
              'no_conflicts', 'override'}, optional
        String indicating how to compare variables of the same name for
        potential merge conflicts:

        - 'broadcast_equals': all values must be equal when variables are
          broadcast against each other to ensure common dimensions.
        - 'equals': all values and dimensions must be the same.
        - 'identical': all values, dimensions and attributes must be the
          same.
        - 'no_conflicts': only values which are not null in both datasets
          must be equal. The returned dataset then contains the combination
          of all non-null values.
        - 'override': skip comparing and pick variable from first dataset
    data_vars : {'minimal', 'different', 'all' or list of str}, optional
        Details are in the documentation of concat
    coords : {'minimal', 'different', 'all' or list of str}, optional
        Details are in the documentation of concat
    fill_value : scalar, optional
        Value to use for newly missing values
    join : {'outer', 'inner', 'left', 'right', 'exact'}, optional
        String indicating how to combine differing indexes
        (excluding concat_dim) in objects

        - 'outer': use the union of object indexes
        - 'inner': use the intersection of object indexes
        - 'left': use indexes from the first object with each dimension
        - 'right': use indexes from the last object with each dimension
        - 'exact': instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - 'override': if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.

    Returns
    -------
    combined : xarray.Dataset

    Examples
    --------

    A common task is collecting data from a parallelized simulation in which
    each process wrote out to a separate file. A domain which was decomposed
    into 4 parts, 2 each along both the x and y axes, requires organising the
    datasets into a doubly-nested list, e.g:

    >>> x1y1
    <xarray.Dataset>
    Dimensions:         (x: 2, y: 2)
    Dimensions without coordinates: x, y
    Data variables:
      temperature       (x, y) float64 11.04 23.57 20.77 ...
      precipitation     (x, y) float64 5.904 2.453 3.404 ...

    >>> ds_grid = [[x1y1, x1y2], [x2y1, x2y2]]
    >>> combined = xr.combine_nested(ds_grid, concat_dim=["x", "y"])
    <xarray.Dataset>
    Dimensions:         (x: 4, y: 4)
    Dimensions without coordinates: x, y
    Data variables:
      temperature       (x, y) float64 11.04 23.57 20.77 ...
      precipitation     (x, y) float64 5.904 2.453 3.404 ...

    ``manual_combine`` can also be used to explicitly merge datasets with
    different variables. For example if we have 4 datasets, which are divided
    along two times, and contain two different variables, we can pass ``None``
    to ``concat_dim`` to specify the dimension of the nested list over which
    we wish to use ``merge`` instead of ``concat``:

    >>> t1temp
    <xarray.Dataset>
    Dimensions:         (t: 5)
    Dimensions without coordinates: t
    Data variables:
      temperature       (t) float64 11.04 23.57 20.77 ...

    >>> t1precip
    <xarray.Dataset>
    Dimensions:         (t: 5)
    Dimensions without coordinates: t
    Data variables:
      precipitation     (t) float64 5.904 2.453 3.404 ...

    >>> ds_grid = [[t1temp, t1precip], [t2temp, t2precip]]
    >>> combined = xr.combine_nested(ds_grid, concat_dim=["t", None])
    <xarray.Dataset>
    Dimensions:         (t: 10)
    Dimensions without coordinates: t
    Data variables:
      temperature       (t) float64 11.04 23.57 20.77 ...
      precipitation     (t) float64 5.904 2.453 3.404 ...

    See also
    --------
    concat
    merge
    auto_combine
    """
    if isinstance(concat_dim, (str, DataArray)) or concat_dim is None:
        concat_dim = [concat_dim]

    # The IDs argument tells _manual_combine that datasets aren't yet sorted
    return _nested_combine(
        datasets,
        concat_dims=concat_dim,
        compat=compat,
        data_vars=data_vars,
        coords=coords,
        ids=False,
        fill_value=fill_value,
        join=join,
    )


def vars_as_keys(ds):
    return tuple(sorted(ds))


def combine_by_coords(
    datasets,
    compat="no_conflicts",
    data_vars="all",
    coords="different",
    fill_value=dtypes.NA,
    join="outer",
):
    """
    Attempt to auto-magically combine the given datasets into one by using
    dimension coordinates.

    This method attempts to combine a group of datasets along any number of
    dimensions into a single entity by inspecting coords and metadata and using
    a combination of concat and merge.

    Will attempt to order the datasets such that the values in their dimension
    coordinates are monotonic along all dimensions. If it cannot determine the
    order in which to concatenate the datasets, it will raise a ValueError.
    Non-coordinate dimensions will be ignored, as will any coordinate
    dimensions which do not vary between each dataset.

    Aligns coordinates, but different variables on datasets can cause it
    to fail under some scenarios. In complex cases, you may need to clean up
    your data and use concat/merge explicitly (also see `manual_combine`).

    Works well if, for example, you have N years of data and M data variables,
    and each combination of a distinct time period and set of data variables is
    saved as its own dataset. Also useful for if you have a simulation which is
    parallelized in multiple dimensions, but has global coordinates saved in
    each file specifying the positions of points within the global domain.

    Parameters
    ----------
    datasets : sequence of xarray.Dataset
        Dataset objects to combine.
    compat : {'identical', 'equals', 'broadcast_equals', 'no_conflicts', 'override'}, optional
        String indicating how to compare variables of the same name for
        potential conflicts:

        - 'broadcast_equals': all values must be equal when variables are
          broadcast against each other to ensure common dimensions.
        - 'equals': all values and dimensions must be the same.
        - 'identical': all values, dimensions and attributes must be the
          same.
        - 'no_conflicts': only values which are not null in both datasets
          must be equal. The returned dataset then contains the combination
          of all non-null values.
        - 'override': skip comparing and pick variable from first dataset
    data_vars : {'minimal', 'different', 'all' or list of str}, optional
        These data variables will be concatenated together:

        * 'minimal': Only data variables in which the dimension already
          appears are included.
        * 'different': Data variables which are not equal (ignoring
          attributes) across all datasets are also concatenated (as well as
          all for which dimension already appears). Beware: this option may
          load the data payload of data variables into memory if they are not
          already loaded.
        * 'all': All data variables will be concatenated.
        * list of str: The listed data variables will be concatenated, in
          addition to the 'minimal' data variables.

        If objects are DataArrays, `data_vars` must be 'all'.
    coords : {'minimal', 'different', 'all' or list of str}, optional
        As per the 'data_vars' kwarg, but for coordinate variables.
    fill_value : scalar, optional
        Value to use for newly missing values. If None, raises a ValueError if
        the passed Datasets do not create a complete hypercube.
    join : {'outer', 'inner', 'left', 'right', 'exact'}, optional
        String indicating how to combine differing indexes
        (excluding concat_dim) in objects

        - 'outer': use the union of object indexes
        - 'inner': use the intersection of object indexes
        - 'left': use indexes from the first object with each dimension
        - 'right': use indexes from the last object with each dimension
        - 'exact': instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - 'override': if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.

    Returns
    -------
    combined : xarray.Dataset

    See also
    --------
    concat
    merge
    combine_nested

    Examples
    --------

    Combining two datasets using their common dimension coordinates. Notice
    they are concatenated based on the values in their dimension coordinates,
    not on their position in the list passed to `combine_by_coords`.

    >>> import numpy as np
    >>> import xarray as xr

    >>> x1 = xr.Dataset(
    ...     {
    ...         "temperature": (("y", "x"), 20 * np.random.rand(6).reshape(2, 3)),
    ...         "precipitation": (("y", "x"), np.random.rand(6).reshape(2, 3)),
    ...     },
    ...     coords={"y": [0, 1], "x": [10, 20, 30]},
    ... )
    >>> x2 = xr.Dataset(
    ...     {
    ...         "temperature": (("y", "x"), 20 * np.random.rand(6).reshape(2, 3)),
    ...         "precipitation": (("y", "x"), np.random.rand(6).reshape(2, 3)),
    ...     },
    ...     coords={"y": [2, 3], "x": [10, 20, 30]},
    ... )
    >>> x3 = xr.Dataset(
    ...     {
    ...         "temperature": (("y", "x"), 20 * np.random.rand(6).reshape(2, 3)),
    ...         "precipitation": (("y", "x"), np.random.rand(6).reshape(2, 3)),
    ...     },
    ...     coords={"y": [2, 3], "x": [40, 50, 60]},
    ... )

    >>> x1
    <xarray.Dataset>
    Dimensions:        (x: 3, y: 2)
    Coordinates:
    * y              (y) int64 0 1
    * x              (x) int64 10 20 30
    Data variables:
        temperature    (y, x) float64 1.654 10.63 7.015 2.543 13.93 9.436
        precipitation  (y, x) float64 0.2136 0.9974 0.7603 0.4679 0.3115 0.945

    >>> x2
    <xarray.Dataset>
    Dimensions:        (x: 3, y: 2)
    Coordinates:
    * y              (y) int64 2 3
    * x              (x) int64 10 20 30
    Data variables:
        temperature    (y, x) float64 9.341 0.1251 6.269 7.709 8.82 2.316
        precipitation  (y, x) float64 0.1728 0.1178 0.03018 0.6509 0.06938 0.3792

    >>> x3
    <xarray.Dataset>
    Dimensions:        (x: 3, y: 2)
    Coordinates:
    * y              (y) int64 2 3
    * x              (x) int64 40 50 60
    Data variables:
        temperature    (y, x) float64 2.789 2.446 6.551 12.46 2.22 15.96
        precipitation  (y, x) float64 0.4804 0.1902 0.2457 0.6125 0.4654 0.5953

    >>> xr.combine_by_coords([x2, x1])
    <xarray.Dataset>
    Dimensions:        (x: 3, y: 4)
    Coordinates:
    * x              (x) int64 10 20 30
    * y              (y) int64 0 1 2 3
    Data variables:
        temperature    (y, x) float64 1.654 10.63 7.015 2.543 ... 7.709 8.82 2.316
        precipitation  (y, x) float64 0.2136 0.9974 0.7603 ... 0.6509 0.06938 0.3792

    >>> xr.combine_by_coords([x3, x1])
    <xarray.Dataset>
    Dimensions:        (x: 6, y: 4)
    Coordinates:
    * x              (x) int64 10 20 30 40 50 60
    * y              (y) int64 0 1 2 3
    Data variables:
        temperature    (y, x) float64 1.654 10.63 7.015 nan ... nan 12.46 2.22 15.96
        precipitation  (y, x) float64 0.2136 0.9974 0.7603 ... 0.6125 0.4654 0.5953

    >>> xr.combine_by_coords([x3, x1], join="override")
    <xarray.Dataset>
    Dimensions:        (x: 3, y: 4)
    Coordinates:
    * x              (x) int64 10 20 30
    * y              (y) int64 0 1 2 3
    Data variables:
    temperature    (y, x) float64 1.654 10.63 7.015 2.543 ... 12.46 2.22 15.96
    precipitation  (y, x) float64 0.2136 0.9974 0.7603 ... 0.6125 0.4654 0.5953

    >>> xr.combine_by_coords([x1, x2, x3])
    <xarray.Dataset>
    Dimensions:        (x: 6, y: 4)
    Coordinates:
    * x              (x) int64 10 20 30 40 50 60
    * y              (y) int64 0 1 2 3
    Data variables:
    temperature    (y, x) float64 1.654 10.63 7.015 nan ... 12.46 2.22 15.96
    precipitation  (y, x) float64 0.2136 0.9974 0.7603 ... 0.6125 0.4654 0.5953
    """

    # Group by data vars
    sorted_datasets = sorted(datasets, key=vars_as_keys)
    grouped_by_vars = itertools.groupby(sorted_datasets, key=vars_as_keys)

    # Perform the multidimensional combine on each group of data variables
    # before merging back together
    concatenated_grouped_by_data_vars = []
    for vars, datasets_with_same_vars in grouped_by_vars:
        combined_ids, concat_dims = _infer_concat_order_from_coords(
            list(datasets_with_same_vars)
        )

        if fill_value is None:
            # check that datasets form complete hypercube
            _check_shape_tile_ids(combined_ids)
        else:
            # check only that all datasets have same dimension depth for these
            # vars
            _check_dimension_depth_tile_ids(combined_ids)

        # Concatenate along all of concat_dims one by one to create single ds
        concatenated = _combine_nd(
            combined_ids,
            concat_dims=concat_dims,
            data_vars=data_vars,
            coords=coords,
            compat=compat,
            fill_value=fill_value,
            join=join,
        )

        # Check the overall coordinates are monotonically increasing
        for dim in concat_dims:
            indexes = concatenated.indexes.get(dim)
            if not (indexes.is_monotonic_increasing or indexes.is_monotonic_decreasing):
                raise ValueError(
                    "Resulting object does not have monotonic"
                    " global indexes along dimension {}".format(dim)
                )
        concatenated_grouped_by_data_vars.append(concatenated)

    return merge(
        concatenated_grouped_by_data_vars,
        compat=compat,
        fill_value=fill_value,
        join=join,
    )


# Everything beyond here is only needed until the deprecation cycle in #2616
# is completed


_CONCAT_DIM_DEFAULT = "__infer_concat_dim__"


def auto_combine(
    datasets,
    concat_dim="_not_supplied",
    compat="no_conflicts",
    data_vars="all",
    coords="different",
    fill_value=dtypes.NA,
    join="outer",
    from_openmfds=False,
):
    """
    Attempt to auto-magically combine the given datasets into one.

    This entire function is deprecated in favour of ``combine_nested`` and
    ``combine_by_coords``.

    This method attempts to combine a list of datasets into a single entity by
    inspecting metadata and using a combination of concat and merge.
    It does not concatenate along more than one dimension or sort data under
    any circumstances. It does align coordinates, but different variables on
    datasets can cause it to fail under some scenarios. In complex cases, you
    may need to clean up your data and use ``concat``/``merge`` explicitly.
    ``auto_combine`` works well if you have N years of data and M data
    variables, and each combination of a distinct time period and set of data
    variables is saved its own dataset.

    Parameters
    ----------
    datasets : sequence of xarray.Dataset
        Dataset objects to merge.
    concat_dim : str or DataArray or Index, optional
        Dimension along which to concatenate variables, as used by
        :py:func:`xarray.concat`. You only need to provide this argument if
        the dimension along which you want to concatenate is not a dimension
        in the original datasets, e.g., if you want to stack a collection of
        2D arrays along a third dimension.
        By default, xarray attempts to infer this argument by examining
        component files. Set ``concat_dim=None`` explicitly to disable
        concatenation.
    compat : {'identical', 'equals', 'broadcast_equals',
             'no_conflicts', 'override'}, optional
        String indicating how to compare variables of the same name for
        potential conflicts:

        - 'broadcast_equals': all values must be equal when variables are
          broadcast against each other to ensure common dimensions.
        - 'equals': all values and dimensions must be the same.
        - 'identical': all values, dimensions and attributes must be the
          same.
        - 'no_conflicts': only values which are not null in both datasets
          must be equal. The returned dataset then contains the combination
          of all non-null values.
        - 'override': skip comparing and pick variable from first dataset
    data_vars : {'minimal', 'different', 'all' or list of str}, optional
        Details are in the documentation of concat
    coords : {'minimal', 'different', 'all' o list of str}, optional
        Details are in the documentation of concat
    fill_value : scalar, optional
        Value to use for newly missing values
    join : {'outer', 'inner', 'left', 'right', 'exact'}, optional
        String indicating how to combine differing indexes
        (excluding concat_dim) in objects

        - 'outer': use the union of object indexes
        - 'inner': use the intersection of object indexes
        - 'left': use indexes from the first object with each dimension
        - 'right': use indexes from the last object with each dimension
        - 'exact': instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - 'override': if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.

    Returns
    -------
    combined : xarray.Dataset

    See also
    --------
    concat
    Dataset.merge
    """

    if not from_openmfds:
        basic_msg = dedent(
            """\
        In xarray version 0.15 `auto_combine` will be deprecated. See
        http://xarray.pydata.org/en/stable/combining.html#combining-multi"""
        )
        warnings.warn(basic_msg, FutureWarning, stacklevel=2)

    if concat_dim == "_not_supplied":
        concat_dim = _CONCAT_DIM_DEFAULT
        message = ""
    else:
        message = dedent(
            """\
        Also `open_mfdataset` will no longer accept a `concat_dim` argument.
        To get equivalent behaviour from now on please use the new
        `combine_nested` function instead (or the `combine='nested'` option to
        `open_mfdataset`)."""
        )

    if _dimension_coords_exist(datasets):
        message += dedent(
            """\
        The datasets supplied have global dimension coordinates. You may want
        to use the new `combine_by_coords` function (or the
        `combine='by_coords'` option to `open_mfdataset`) to order the datasets
        before concatenation. Alternatively, to continue concatenating based
        on the order the datasets are supplied in future, please use the new
        `combine_nested` function (or the `combine='nested'` option to
        open_mfdataset)."""
        )
    else:
        message += dedent(
            """\
        The datasets supplied do not have global dimension coordinates. In
        future, to continue concatenating without supplying dimension
        coordinates, please use the new `combine_nested` function (or the
        `combine='nested'` option to open_mfdataset."""
        )

    if _requires_concat_and_merge(datasets):
        manual_dims = [concat_dim].append(None)
        message += dedent(
            """\
        The datasets supplied require both concatenation and merging. From
        xarray version 0.15 this will operation will require either using the
        new `combine_nested` function (or the `combine='nested'` option to
        open_mfdataset), with a nested list structure such that you can combine
        along the dimensions {}. Alternatively if your datasets have global
        dimension coordinates then you can use the new `combine_by_coords`
        function.""".format(
                manual_dims
            )
        )

    warnings.warn(message, FutureWarning, stacklevel=2)

    return _old_auto_combine(
        datasets,
        concat_dim=concat_dim,
        compat=compat,
        data_vars=data_vars,
        coords=coords,
        fill_value=fill_value,
        join=join,
    )


def _dimension_coords_exist(datasets):
    """
    Check if the datasets have consistent global dimension coordinates
    which would in future be used by `auto_combine` for concatenation ordering.
    """

    # Group by data vars
    sorted_datasets = sorted(datasets, key=vars_as_keys)
    grouped_by_vars = itertools.groupby(sorted_datasets, key=vars_as_keys)

    # Simulates performing the multidimensional combine on each group of data
    # variables before merging back together
    try:
        for vars, datasets_with_same_vars in grouped_by_vars:
            _infer_concat_order_from_coords(list(datasets_with_same_vars))
        return True
    except ValueError:
        # ValueError means datasets don't have global dimension coordinates
        # Or something else went wrong in trying to determine them
        return False


def _requires_concat_and_merge(datasets):
    """
    Check if the datasets require the use of both xarray.concat and
    xarray.merge, which in future might require the user to use
    `manual_combine` instead.
    """
    # Group by data vars
    sorted_datasets = sorted(datasets, key=vars_as_keys)
    grouped_by_vars = itertools.groupby(sorted_datasets, key=vars_as_keys)

    return len(list(grouped_by_vars)) > 1


def _old_auto_combine(
    datasets,
    concat_dim=_CONCAT_DIM_DEFAULT,
    compat="no_conflicts",
    data_vars="all",
    coords="different",
    fill_value=dtypes.NA,
    join="outer",
):
    if concat_dim is not None:
        dim = None if concat_dim is _CONCAT_DIM_DEFAULT else concat_dim

        sorted_datasets = sorted(datasets, key=vars_as_keys)
        grouped = itertools.groupby(sorted_datasets, key=vars_as_keys)

        concatenated = [
            _auto_concat(
                list(datasets),
                dim=dim,
                data_vars=data_vars,
                coords=coords,
                compat=compat,
                fill_value=fill_value,
                join=join,
            )
            for vars, datasets in grouped
        ]
    else:
        concatenated = datasets
    merged = merge(concatenated, compat=compat, fill_value=fill_value, join=join)
    return merged


def _auto_concat(
    datasets,
    dim=None,
    data_vars="all",
    coords="different",
    fill_value=dtypes.NA,
    join="outer",
    compat="no_conflicts",
):
    if len(datasets) == 1 and dim is None:
        # There is nothing more to combine, so kick out early.
        return datasets[0]
    else:
        if dim is None:
            ds0 = datasets[0]
            ds1 = datasets[1]
            concat_dims = set(ds0.dims)
            if ds0.dims != ds1.dims:
                dim_tuples = set(ds0.dims.items()) - set(ds1.dims.items())
                concat_dims = {i for i, _ in dim_tuples}
            if len(concat_dims) > 1:
                concat_dims = {d for d in concat_dims if not ds0[d].equals(ds1[d])}
            if len(concat_dims) > 1:
                raise ValueError(
                    "too many different dimensions to " "concatenate: %s" % concat_dims
                )
            elif len(concat_dims) == 0:
                raise ValueError(
                    "cannot infer dimension to concatenate: "
                    "supply the ``concat_dim`` argument "
                    "explicitly"
                )
            (dim,) = concat_dims
        return concat(
            datasets,
            dim=dim,
            data_vars=data_vars,
            coords=coords,
            fill_value=fill_value,
            compat=compat,
        )
