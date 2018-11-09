from __future__ import absolute_import, division, print_function

import warnings
import toolz.itertoolz as itertoolz

import pandas as pd

from . import utils
from .alignment import align
from .merge import merge
from .pycompat import OrderedDict, basestring, iteritems
from .variable import IndexVariable, Variable, as_variable
from .variable import concat as concat_vars


def concat(objs, dim=None, data_vars='all', coords='different',
           compat='equals', positions=None, indexers=None, mode=None,
           concat_over=None):
    """Concatenate xarray objects along a new or existing dimension.

    Parameters
    ----------
    objs : sequence of Dataset and DataArray objects
        xarray objects to concatenate together. Each object is expected to
        consist of variables and coordinates with matching shapes except for
        along the concatenated dimension.
    dim : str or DataArray or pandas.Index
        Name of the dimension to concatenate along. This can either be a new
        dimension name, in which case it is added along axis=0, or an existing
        dimension name, in which case the location of the dimension is
        unchanged. If dimension is provided as a DataArray or Index, its name
        is used as the dimension to concatenate along and the values are added
        as a coordinate.
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
        If objects are DataArrays, data_vars must be 'all'.
    coords : {'minimal', 'different', 'all' o list of str}, optional
        These coordinate variables will be concatenated together:
          * 'minimal': Only coordinates in which the dimension already appears
            are included.
          * 'different': Coordinates which are not equal (ignoring attributes)
            across all datasets are also concatenated (as well as all for which
            dimension already appears). Beware: this option may load the data
            payload of coordinate variables into memory if they are not already
            loaded.
          * 'all': All coordinate variables will be concatenated, except
            those corresponding to other dimensions.
          * list of str: The listed coordinate variables will be concatenated,
            in addition the 'minimal' coordinates.
    compat : {'equals', 'identical'}, optional
        String indicating how to compare non-concatenated variables and
        dataset global attributes for potential conflicts. 'equals' means
        that all variable values and dimensions must be the same;
        'identical' means that variable attributes and global attributes
        must also be equal.
    positions : None or list of integer arrays, optional
        List of integer arrays which specifies the integer positions to which
        to assign each dataset along the concatenated dimension. If not
        supplied, objects are concatenated in the provided order.
    indexers, mode, concat_over : deprecated

    Returns
    -------
    concatenated : type of objs

    See also
    --------
    merge
    auto_combine
    """
    # TODO: add join and ignore_index arguments copied from pandas.concat
    # TODO: support concatenating scalar coordinates even if the concatenated
    # dimension already exists
    from .dataset import Dataset
    from .dataarray import DataArray

    try:
        first_obj, objs = utils.peek_at(objs)
    except StopIteration:
        raise ValueError('must supply at least one object to concatenate')

    if dim is None:
        warnings.warn('the `dim` argument to `concat` will be required '
                      'in a future version of xarray; for now, setting it to '
                      "the old default of 'concat_dim'",
                      FutureWarning, stacklevel=2)
        dim = 'concat_dims'

    if indexers is not None:  # pragma: nocover
        warnings.warn('indexers has been renamed to positions; the alias '
                      'will be removed in a future version of xarray',
                      FutureWarning, stacklevel=2)
        positions = indexers

    if mode is not None:
        raise ValueError('`mode` is no longer a valid argument to '
                         'xarray.concat; it has been split into the '
                         '`data_vars` and `coords` arguments')
    if concat_over is not None:
        raise ValueError('`concat_over` is no longer a valid argument to '
                         'xarray.concat; it has been split into the '
                         '`data_vars` and `coords` arguments')

    if isinstance(first_obj, DataArray):
        f = _dataarray_concat
    elif isinstance(first_obj, Dataset):
        f = _dataset_concat
    else:
        raise TypeError('can only concatenate xarray Dataset and DataArray '
                        'objects, got %s' % type(first_obj))
    return f(objs, dim, data_vars, coords, compat, positions)


def _calc_concat_dim_coord(dim):
    """
    Infer the dimension name and 1d coordinate variable (if appropriate)
    for concatenating along the new dimension.
    """
    from .dataarray import DataArray

    if isinstance(dim, basestring):
        coord = None
    elif not isinstance(dim, (DataArray, Variable)):
        dim_name = getattr(dim, 'name', None)
        if dim_name is None:
            dim_name = 'concat_dim'
        coord = IndexVariable(dim_name, dim)
        dim = dim_name
    elif not isinstance(dim, DataArray):
        coord = as_variable(dim).to_index_variable()
        dim, = coord.dims
    else:
        coord = dim
        dim, = coord.dims
    return dim, coord


def _calc_concat_over(datasets, dim, data_vars, coords):
    """
    Determine which dataset variables need to be concatenated in the result,
    and which can simply be taken from the first dataset.
    """
    # Return values
    concat_over = set()
    equals = {}

    if dim in datasets[0]:
        concat_over.add(dim)
    for ds in datasets:
        concat_over.update(k for k, v in ds.variables.items()
                           if dim in v.dims)

    def process_subset_opt(opt, subset):
        if isinstance(opt, basestring):
            if opt == 'different':
                # all nonindexes that are not the same in each dataset
                for k in getattr(datasets[0], subset):
                    if k not in concat_over:
                        # Compare the variable of all datasets vs. the one
                        # of the first dataset. Perform the minimum amount of
                        # loads in order to avoid multiple loads from disk
                        # while keeping the RAM footprint low.
                        v_lhs = datasets[0].variables[k].load()
                        # We'll need to know later on if variables are equal.
                        computed = []
                        for ds_rhs in datasets[1:]:
                            v_rhs = ds_rhs.variables[k].compute()
                            computed.append(v_rhs)
                            if not v_lhs.equals(v_rhs):
                                concat_over.add(k)
                                equals[k] = False
                                # computed variables are not to be re-computed
                                # again in the future
                                for ds, v in zip(datasets[1:], computed):
                                    ds.variables[k].data = v.data
                                break
                        else:
                            equals[k] = True

            elif opt == 'all':
                concat_over.update(set(getattr(datasets[0], subset)) -
                                   set(datasets[0].dims))
            elif opt == 'minimal':
                pass
            else:
                raise ValueError("unexpected value for %s: %s" % (subset, opt))
        else:
            invalid_vars = [k for k in opt
                            if k not in getattr(datasets[0], subset)]
            if invalid_vars:
                if subset == 'coords':
                    raise ValueError(
                        'some variables in coords are not coordinates on '
                        'the first dataset: %s' % (invalid_vars,))
                else:
                    raise ValueError(
                        'some variables in data_vars are not data variables '
                        'on the first dataset: %s' % (invalid_vars,))
            concat_over.update(opt)

    process_subset_opt(data_vars, 'data_vars')
    process_subset_opt(coords, 'coords')
    return concat_over, equals


def _dataset_concat(datasets, dim, data_vars, coords, compat, positions):
    """
    Concatenate a sequence of datasets along a new or existing dimension
    """
    from .dataset import Dataset

    if compat not in ['equals', 'identical']:
        raise ValueError("compat=%r invalid: must be 'equals' "
                         "or 'identical'" % compat)

    dim, coord = _calc_concat_dim_coord(dim)
    # Make sure we're working on a copy (we'll be loading variables)
    datasets = [ds.copy() for ds in datasets]
    datasets = align(*datasets, join='outer', copy=False, exclude=[dim])

    concat_over, equals = _calc_concat_over(datasets, dim, data_vars, coords)

    def insert_result_variable(k, v):
        assert isinstance(v, Variable)
        if k in datasets[0].coords:
            result_coord_names.add(k)
        result_vars[k] = v

    # create the new dataset and add constant variables
    result_vars = OrderedDict()
    result_coord_names = set(datasets[0].coords)
    result_attrs = datasets[0].attrs
    result_encoding = datasets[0].encoding

    for k, v in datasets[0].variables.items():
        if k not in concat_over:
            insert_result_variable(k, v)

    # check that global attributes and non-concatenated variables are fixed
    # across all datasets
    for ds in datasets[1:]:
        if (compat == 'identical' and
                not utils.dict_equiv(ds.attrs, result_attrs)):
            raise ValueError('dataset global attributes not equal')
        for k, v in iteritems(ds.variables):
            if k not in result_vars and k not in concat_over:
                raise ValueError('encountered unexpected variable %r' % k)
            elif (k in result_coord_names) != (k in ds.coords):
                raise ValueError('%r is a coordinate in some datasets but not '
                                 'others' % k)
            elif k in result_vars and k != dim:
                # Don't use Variable.identical as it internally invokes
                # Variable.equals, and we may already know the answer
                if compat == 'identical' and not utils.dict_equiv(
                        v.attrs, result_vars[k].attrs):
                    raise ValueError(
                        'variable %s not identical across datasets' % k)

                # Proceed with equals()
                try:
                    # May be populated when using the "different" method
                    is_equal = equals[k]
                except KeyError:
                    result_vars[k].load()
                    is_equal = v.equals(result_vars[k])
                if not is_equal:
                    raise ValueError(
                        'variable %s not equal across datasets' % k)

    # we've already verified everything is consistent; now, calculate
    # shared dimension sizes so we can expand the necessary variables
    dim_lengths = [ds.dims.get(dim, 1) for ds in datasets]
    non_concat_dims = {}
    for ds in datasets:
        non_concat_dims.update(ds.dims)
    non_concat_dims.pop(dim, None)

    def ensure_common_dims(vars):
        # ensure each variable with the given name shares the same
        # dimensions and the same shape for all of them except along the
        # concat dimension
        common_dims = tuple(pd.unique([d for v in vars for d in v.dims]))
        if dim not in common_dims:
            common_dims = (dim,) + common_dims
        for var, dim_len in zip(vars, dim_lengths):
            if var.dims != common_dims:
                common_shape = tuple(non_concat_dims.get(d, dim_len)
                                     for d in common_dims)
                var = var.set_dims(common_dims, common_shape)
            yield var

    # stack up each variable to fill-out the dataset (in order)
    for k in datasets[0].variables:
        if k in concat_over:
            vars = ensure_common_dims([ds.variables[k] for ds in datasets])
            combined = concat_vars(vars, dim, positions)
            insert_result_variable(k, combined)

    result = Dataset(result_vars, attrs=result_attrs)
    result = result.set_coords(result_coord_names)
    result.encoding = result_encoding

    if coord is not None:
        # add concat dimension last to ensure that its in the final Dataset
        result[coord.name] = coord

    return result


def _dataarray_concat(arrays, dim, data_vars, coords, compat,
                      positions):
    arrays = list(arrays)

    if data_vars != 'all':
        raise ValueError('data_vars is not a valid argument when '
                         'concatenating DataArray objects')

    datasets = []
    for n, arr in enumerate(arrays):
        if n == 0:
            name = arr.name
        elif name != arr.name:
            if compat == 'identical':
                raise ValueError('array names not identical')
            else:
                arr = arr.rename(name)
        datasets.append(arr._to_temp_dataset())

    ds = _dataset_concat(datasets, dim, data_vars, coords, compat,
                         positions)
    return arrays[0]._from_temp_dataset(ds, name)


def _auto_concat(datasets, dim=None, data_vars='all', coords='different'):
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
                concat_dims = set(i for i, _ in dim_tuples)
            if len(concat_dims) > 1:
                concat_dims = set(d for d in concat_dims
                                  if not ds0[d].equals(ds1[d]))
            if len(concat_dims) > 1:
                raise ValueError('too many different dimensions to '
                                 'concatenate: %s' % concat_dims)
            elif len(concat_dims) == 0:
                raise ValueError('cannot infer dimension to concatenate: '
                                 'supply the ``concat_dim`` argument '
                                 'explicitly')
            dim, = concat_dims
        return concat(datasets, dim=dim, data_vars=data_vars, coords=coords)


_CONCAT_DIM_DEFAULT = '__infer_concat_dim__'


def _infer_concat_order_from_nested_list(datasets, concat_dims):

    # TODO check that datasets is a list containing multiple elements

    combined_ids = _infer_tile_ids_from_nested_list(datasets, [], {})

    # Currently if concat_dims is not supplied then _auto_concat attempts to deduce it on every call
    # TODO would be faster in this case to just work out the concat_dims once here
    tile_id, ds = combined_ids[0]
    n_dims = len(tile_id)
    if concat_dims is None:
        concat_dims = [_CONCAT_DIM_DEFAULT]*n_dims
    else:
        if len(concat_dims) != n_dims:
            raise ValueError("concat_dims is of length " + str(len(concat_dims))
                             + " but the datasets passed are nested in a " +
                             str(n_dims) + "-dimensional structure")

    return concat_dims, combined_ids


def _infer_tile_ids_from_nested_list(entry, current_pos, combined_tile_ids):
    """
    Given a list of lists (of lists...) of datasets, returns a dictionary
    with the index of each dataset in the nested list structure as the key.

    Recursively traverses the given structure, while keeping track of the current
    position.

    Parameters
    ----------
    entry : list[list[xarray.Dataset, xarray.Dataset, ...]]
        List of lists of arbitrary depth, containing datasets in the order they
        are to be concatenated.

    Returns
    -------
    combined_tile_ids : dict[tuple(int, ...), xarray.Dataset]
    """

    from .dataset import Dataset

    if isinstance(entry, list):
        # Check if list is redundant
        if len(entry) == 1:
            raise TypeError('Redundant list nesting at '
                            'position ' + str(tuple(current_pos)))

        # Dive down tree
        current_pos.append(0)
        for i, item in enumerate(entry):
            current_pos[-1] = i
            combined_tile_ids = _infer_tile_ids_from_nested_list(item, current_pos,
                                                                 combined_tile_ids)
        # Move back up tree
        del current_pos[-1]
        return combined_tile_ids

    elif isinstance(entry, Dataset):
        # Termination condition
        combined_tile_ids[tuple(current_pos)] = entry
        return combined_tile_ids

    else:
        raise TypeError("Element at position " + str(tuple(current_pos)) +
                        " is neither a list nor an xarray.Dataset")


def _check_shape_tile_ids(combined_tile_ids):
    # TODO create custom exception class instead of using asserts?

    tile_ids = combined_tile_ids.keys()

    # Check all tuples are the same length
    lengths = [len(id) for id in tile_ids]
    assert set(lengths) == {lengths[0]}

    # Check each dimension has indices 0 to n represented with no gaps
    for dim in range(lengths[0]):
        indices = [id[dim] for id in tile_ids]
        assert len(indices) > 1
        assert sorted(indices) == range(max(indices))

    # Check only datasets are contained
    from .dataset import Dataset
    for v in combined_tile_ids.values():
        assert isinstance(v, Dataset)


def _data_vars(combined_id):
    id, ds = combined_id
    return tuple(sorted(ds.data_vars))


def _combine_nd(combined_IDs, concat_dims, data_vars='all',
                coords='different', compat='no_conflicts'):
    """
    Concatenates and merges an N-dimensional structure of datasets.

    No checks are performed on the consistency of the datasets, concat_dims or
    tile_IDs, because it is assumed that this has already been done.

    Parameters
    ----------
    combined_IDs : Dict[Tuple[int, ...]], xarray.Dataset]
        Structure containing all datasets to be concatenated with "tile_IDs" as
        keys, which specify position within the desired final combined result.
    concat_dims : sequence of str
        The dimensions along which the datasets should be concatenated. Must be
        in order, and the length must match

    Returns
    -------

    """

    # Organise by data variables
    grouped_by_data_vars = itertoolz.groupby(_data_vars,
                                             combined_IDs.items()).values()
    concatenated_datasets = []
    for tiled_datasets in grouped_by_data_vars:
        concatenated_ids = tiled_datasets

        # Perform N-D dimensional concatenation
        for concat_dim in concat_dims:
            dim = None if concat_dim is _CONCAT_DIM_DEFAULT else concat_dim

            concatenated_ids = _concat_along_first_dim(concatenated_ids,
                                                       dim=dim,
                                                       data_vars=data_vars,
                                                       coords=coords)
        concatenated_datasets.append(concatenated_ids.values())

    return merge(concatenated_datasets, compat=compat)


def _new_tile_id(single_id_ds_pair):
    # probably replace with something like lambda x: x[0][1:]
    tile_id, ds = single_id_ds_pair
    return tile_id[1:]


def _concat_along_first_dim(combined_IDs, dim, data_vars='all',
                                coords='different'):
    grouped = itertoolz.groupby(_new_tile_id, combined_IDs.items())
    new_combined_IDs = {}

    # TODO Would there be any point in parallelizing this concatenation step?
    for new_ID, group in grouped.items():
        to_concat = [ds for old_ID, ds in group]
        new_combined_IDs[new_ID] = _auto_concat(to_concat, dim=dim,
                                                data_vars=data_vars,
                                                coords=coords)
    return new_combined_IDs


def auto_combine(datasets,
                 concat_dims=_CONCAT_DIM_DEFAULT,
                 compat='no_conflicts',
                 data_vars='all', coords='different',
                 infer_order_from_coords=True):
    """Attempt to auto-magically combine the given datasets into one.

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
    concat_dims : list of str or DataArray or Index, optional
        Dimensions along which to concatenate variables, as used by
        :py:func:`xarray.concat`. You only need to provide this argument if
        the dimensions along which you want to concatenate is not a dimension
        in the original datasets, e.g., if you want to stack a collection of
        2D arrays along a third dimension.
        By default, xarray attempts to infer this argument by examining
        component files. Set ``concat_dim=None`` explicitly to disable
        concatenation.
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
    infer_order_from_coords : bool, optional
        If true attempt to deduce the order in which the datasets should be
        concatenated from their coordinates. To do this the coordinates should
        be monotonic along the dimension to be concatenated.
        If false instead read the order from the structure the datasets are
        supplied in. This structure should be a nested list of lists.

    Returns
    -------
    combined : xarray.Dataset

    See also
    --------
    concat
    Dataset.merge
    """
    if concat_dims is not None:

        # TODO this could be where we would optionally check alignment, as in #2039

        # Organise datasets in concatentation order in N-D
        if infer_order_from_coords:
            # TODO Use coordinates to determine tile_ID for each dataset in N-D
            # i.e. (shoyer's (1) from discussion in #2159)
            raise NotImplementedError
        else:
            # Determine tile_IDs by structure of input in N-D (i.e. ordering in list-of-lists)
            concat_dims, combined_ids = _infer_concat_order_from_nested_list(datasets, concat_dims)

        # Check that the combined_ids are sensible
        _check_shape_tile_ids(combined_ids)

        # Repeatedly concatenate then merge along each dimension
        combined = _combine_nd(combined_ids, concat_dims, compat=compat,
                               data_vars=data_vars, coords=coords)
    else:
        # Case of no concatenation wanted
        concatenated = datasets
        combined = merge(concatenated, compat=compat)
    return combined
