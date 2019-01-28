from __future__ import absolute_import, division, print_function

import itertools
from collections import Counter

import pandas as pd

from .dataarray import DataArray
from . import utils
from .merge import merge
from .concat import _auto_concat
from .pycompat import OrderedDict


_CONCAT_DIM_DEFAULT = utils.ReprObject('<inferred>')


def _infer_concat_order_from_positions(datasets, concat_dims):

    combined_ids = OrderedDict(_infer_tile_ids_from_nested_list(datasets, ()))

    tile_id, ds = list(combined_ids.items())[0]
    n_dims = len(tile_id)

    if concat_dims is _CONCAT_DIM_DEFAULT:
        concat_dims = [_CONCAT_DIM_DEFAULT] * n_dims
    else:
        if len(concat_dims) != n_dims:
            raise ValueError("concat_dims has length {} but the datasets "
                             "passed are nested in a {}-dimensional structure"
                             .format(str(len(concat_dims)), str(n_dims)))

    return combined_ids, concat_dims


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
    entry : list[list[obj, obj, ...]]
        List of lists of arbitrary depth, containing objects in the order
        they are to be concatenated.

    Returns
    -------
    combined_tile_ids : dict[tuple(int, ...), obj]
    """

    if isinstance(entry, list):
        for i, item in enumerate(entry):
            for result in _infer_tile_ids_from_nested_list(item,
                                                           current_pos + (i,)):
                yield result
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
                raise ValueError("Every dimension needs a coordinate for "
                                 "inferring concatenation order")

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
                    raise ValueError("Coordinate variable {} is neither "
                                     "monotonically increasing nor "
                                     "monotonically decreasing on all datasets"
                                     .format(dim))

                # Assume that any two datasets whose coord along dim starts
                # with the same value have the same coord values throughout.
                try:
                    first_items = pd.Index([index.take([0])
                                            for index in indexes])
                except IndexError:
                    raise ValueError('Cannot handle size zero dimensions')

                # Sort datasets along dim
                # We want rank but with identical elements given identical
                # position indices - they should be concatenated along another
                # dimension, not along this one
                series = first_items.to_series()
                rank = series.rank(method='dense', ascending=ascending)
                order = rank.astype(int).values - 1

                # TODO check that resulting global coordinate is monotonic

                # Append positions along extra dimension to structure which
                # encodes the multi-dimensional concatentation order
                tile_ids = [tile_id + (position,) for tile_id, position
                            in zip(tile_ids, order)]

    if len(datasets) > 1 and not concat_dims:
        raise ValueError("Could not find any dimension coordinates to use to "
                         "order the datasets for concatenation")

    combined_ids = OrderedDict(zip(tile_ids, datasets))

    return combined_ids, concat_dims


def _check_shape_tile_ids(combined_tile_ids):
    tile_ids = combined_tile_ids.keys()

    # Check all tuples are the same length
    # i.e. check that all lists are nested to the same depth
    nesting_depths = [len(tile_id) for tile_id in tile_ids]
    if not set(nesting_depths) == {nesting_depths[0]}:
        raise ValueError("The supplied objects do not form a hypercube because"
                         " sub-lists do not have consistent depths")

    # Check all lists along one dimension are same length
    for dim in range(nesting_depths[0]):
        indices_along_dim = [tile_id[dim] for tile_id in tile_ids]
        occurrences = Counter(indices_along_dim)
        if len(set(occurrences.values())) != 1:
            raise ValueError("The supplied objects do not form a hypercube "
                             "because sub-lists do not have consistent "
                             "lengths along dimension" + str(dim))


def _combine_nd(combined_ids, concat_dims, data_vars='all',
                coords='different', compat='no_conflicts'):
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

    # Each iteration of this loop reduces the length of the tile_ids tuples
    # by one. It always combines along the first dimension, removing the first
    # element of the tuple
    for concat_dim in concat_dims:
        combined_ids = _combine_all_along_first_dim(combined_ids,
                                                    dim=concat_dim,
                                                    data_vars=data_vars,
                                                    coords=coords,
                                                    compat=compat)
    combined_ds = list(combined_ids.values())[0]
    return combined_ds


def _combine_all_along_first_dim(combined_ids, dim, data_vars, coords, compat):

    # Group into lines of datasets which must be combined along dim
    # need to sort by _new_tile_id first for groupby to work
    # TODO remove all these sorted OrderedDicts once python >= 3.6 only
    combined_ids = OrderedDict(sorted(combined_ids.items(), key=_new_tile_id))
    grouped = itertools.groupby(combined_ids.items(), key=_new_tile_id)

    # Combine all of these datasets along dim
    new_combined_ids = {}
    for new_id, group in grouped:
        combined_ids = OrderedDict(sorted(group))
        datasets = combined_ids.values()
        new_combined_ids[new_id] = _combine_1d(datasets, dim, compat,
                                               data_vars, coords)
    return new_combined_ids


def _combine_1d(datasets, concat_dim=_CONCAT_DIM_DEFAULT,
                compat='no_conflicts', data_vars='all', coords='different'):
    """
    Applies either concat or merge to 1D list of datasets depending on value
    of concat_dim
    """

    # TODO this logic is taken from old 1D auto_combine - check if it's right
    # Should it just use concat directly instead?
    if concat_dim is not None:
        dim = None if concat_dim is _CONCAT_DIM_DEFAULT else concat_dim
        combined = _auto_concat(datasets, dim=dim, data_vars=data_vars,
                                coords=coords)
    else:
        combined = merge(datasets, compat=compat)

    return combined


def _new_tile_id(single_id_ds_pair):
    tile_id, ds = single_id_ds_pair
    return tile_id[1:]


def _manual_combine(datasets, concat_dims, compat, data_vars, coords, ids):

    # Arrange datasets for concatenation
    # Use information from the shape of the user input
    if not ids:
        # Determine tile_IDs by structure of input in N-D
        # (i.e. ordering in list-of-lists)
        combined_ids, concat_dims = _infer_concat_order_from_positions(
            datasets, concat_dims)
    else:
        # Already sorted so just use the ids already passed
        combined_ids = OrderedDict(zip(ids, datasets))

    # Check that the inferred shape is combinable
    _check_shape_tile_ids(combined_ids)

    # Apply series of concatenate or merge operations along each dimension
    combined = _combine_nd(combined_ids, concat_dims, compat=compat,
                           data_vars=data_vars, coords=coords)
    return combined


def manual_combine(datasets, concat_dim=_CONCAT_DIM_DEFAULT,
                   compat='no_conflicts', data_vars='all', coords='different'):
    """
    Explicitly combine an N-dimensional grid of datasets into one by using a
    succession of concat and merge operations along each dimension of the grid.

    Does not sort data under any circumstances, so the datasets must be passed
    in the order you wish them to be concatenated. It does align coordinates,
    but different variables on datasets can cause it to fail under some
    scenarios. In complex cases, you may need to clean up your data and use
    concat/merge explicitly.

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
    concat_dim : str, or list of str, DataArray, Index or None, optional
        Dimensions along which to concatenate variables, as used by
        :py:func:`xarray.concat`.
        By default, xarray attempts to infer this argument by examining
        component files. Set ``concat_dim=[..., None, ...]`` explicitly to
        disable concatenation and merge instead along a particular dimension.
        The position of ``None`` in the list specifies the dimension of the
        nested-list input along which to merge.
        Must be the same length as the depth of the list passed to
        ``datasets``.
    compat : {'identical', 'equals', 'broadcast_equals',
              'no_conflicts'}, optional
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
    data_vars : {'minimal', 'different', 'all' or list of str}, optional
        Details are in the documentation of concat
    coords : {'minimal', 'different', 'all' or list of str}, optional
        Details are in the documentation of concat

    Returns
    -------
    combined : xarray.Dataset

    Examples
    --------

    A common task is collecting data from a parallelized simulation in which
    each processor wrote out to a separate file. A domain which was decomposed
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
    >>> combined = xr.manual_combine(ds_grid, concat_dim=['x', 'y'])
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
    >>> combined = xr.manual_combine(ds_grid, concat_dim=['t', None])
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

    if concat_dim is not _CONCAT_DIM_DEFAULT:
        if isinstance(concat_dim, (str, DataArray)) or concat_dim is None:
            concat_dim = [concat_dim]

    # The IDs argument tells _manual_combine that datasets aren't yet sorted
    return _manual_combine(datasets, concat_dims=concat_dim, compat=compat,
                           data_vars=data_vars, coords=coords, ids=False)


def vars_as_keys(ds):
    return tuple(sorted(ds))


def auto_combine(datasets, compat='no_conflicts', data_vars='all',
                 coords='different'):
    """
    Attempt to auto-magically combine the given datasets into one by using
    dimension coordinates.

    This method attempts to combine a group of datasets along any number of
    dimensions into a single entity by inspecting coords and metadata and using
    a combination of concat and merge.

    Will attempt to order the datasets such that the values in their dimension
    coordinates are monotonically increasing along all dimensions. If it cannot
    determine the order in which to concatenate the datasets, it will raise an
    error.
    Non-coordinate dimensions will be ignored, as will any coordinate
    dimensions which do not vary between each dataset.

    Aligns coordinates, but different variables on datasets can cause it
    to fail under some scenarios. In complex cases, you may need to clean up
    your data and use concat/merge explicitly (also see `manual_combine`).

    Works well if, for example, you have N years of data and M data variables,
    and each combination of a distinct time period and set of data variables is
    saved as its own dataset. Also useful for if you have a simulation which is
    parallelized in multiple dimensions, but has global coordinates saved in
    each file specifying it's position within the domain.

    Parameters
    ----------
    datasets : sequence of xarray.Dataset
        Dataset objects to combine.
    compat : {'identical', 'equals', 'broadcast_equals',
              'no_conflicts'}, optional
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
    data_vars : {'minimal', 'different', 'all' or list of str}, optional
        Details are in the documentation of concat
    coords : {'minimal', 'different', 'all' or list of str}, optional
        Details are in the documentation of concat

    Returns
    -------
    combined : xarray.Dataset

    See also
    --------
    concat
    merge
    manual_combine
    """

    # Group by data vars
    sorted_datasets = sorted(datasets, key=vars_as_keys)
    grouped_by_vars = itertools.groupby(sorted_datasets, key=vars_as_keys)

    # Perform the multidimensional combine on each group of data variables
    # before merging back together
    concatenated_grouped_by_data_vars = []
    for vars, datasets_with_same_vars in grouped_by_vars:
        combined_ids, concat_dims = _infer_concat_order_from_coords(
            list(datasets_with_same_vars))

        # TODO checking the shape of the combined ids appropriate here?
        _check_shape_tile_ids(combined_ids)

        # Concatenate along all of concat_dims one by one to create single ds
        concatenated = _combine_nd(combined_ids, concat_dims=concat_dims,
                                   data_vars=data_vars, coords=coords)

        # TODO check the overall coordinates are monotonically increasing?
        concatenated_grouped_by_data_vars.append(concatenated)

    return merge(concatenated_grouped_by_data_vars, compat=compat)
