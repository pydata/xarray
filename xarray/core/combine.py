import itertools
import warnings
from collections import Counter, OrderedDict

import pandas as pd

from . import utils
from .alignment import align
from .merge import merge
from .variable import IndexVariable, Variable, as_variable
from .variable import concat as concat_vars
from .computation import result_name


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
    coords : {'minimal', 'different', 'all' or list of str}, optional
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

    if isinstance(dim, str):
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
        if isinstance(opt, str):
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
        for k, v in ds.variables.items():
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
    result = arrays[0]._from_temp_dataset(ds, name)

    result.name = result_name(arrays)
    return result


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


_CONCAT_DIM_DEFAULT = utils.ReprObject('<inferred>')


def _infer_concat_order_from_positions(datasets, concat_dims):

    combined_ids = OrderedDict(_infer_tile_ids_from_nested_list(datasets, ()))

    tile_id, ds = list(combined_ids.items())[0]
    n_dims = len(tile_id)
    if concat_dims == _CONCAT_DIM_DEFAULT or concat_dims is None:
        concat_dims = [concat_dims] * n_dims
    else:
        if len(concat_dims) != n_dims:
            raise ValueError("concat_dims has length {} but the datasets "
                             "passed are nested in a {}-dimensional "
                             "structure".format(str(len(concat_dims)),
                                                str(n_dims)))

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
    Concatenates and merges an N-dimensional structure of datasets.

    No checks are performed on the consistency of the datasets, concat_dims or
    tile_IDs, because it is assumed that this has already been done.

    Parameters
    ----------
    combined_ids : Dict[Tuple[int, ...]], xarray.Dataset]
        Structure containing all datasets to be concatenated with "tile_IDs" as
        keys, which specify position within the desired final combined result.
    concat_dims : sequence of str
        The dimensions along which the datasets should be concatenated. Must be
        in order, and the length must match

    Returns
    -------
    combined_ds : xarray.Dataset
    """

    # Perform N-D dimensional concatenation
    # Each iteration of this loop reduces the length of the tile_ids tuples
    # by one. It always combines along the first dimension, removing the first
    # element of the tuple
    for concat_dim in concat_dims:
        combined_ids = _auto_combine_all_along_first_dim(combined_ids,
                                                         dim=concat_dim,
                                                         data_vars=data_vars,
                                                         coords=coords,
                                                         compat=compat)
    combined_ds = list(combined_ids.values())[0]
    return combined_ds


def _auto_combine_all_along_first_dim(combined_ids, dim, data_vars,
                                      coords, compat):
    # Group into lines of datasets which must be combined along dim
    # need to sort by _new_tile_id first for groupby to work
    # TODO remove all these sorted OrderedDicts once python >= 3.6 only
    combined_ids = OrderedDict(sorted(combined_ids.items(), key=_new_tile_id))
    grouped = itertools.groupby(combined_ids.items(), key=_new_tile_id)

    new_combined_ids = {}
    for new_id, group in grouped:
        combined_ids = OrderedDict(sorted(group))
        datasets = combined_ids.values()
        new_combined_ids[new_id] = _auto_combine_1d(datasets, dim, compat,
                                                    data_vars, coords)
    return new_combined_ids


def vars_as_keys(ds):
    return tuple(sorted(ds))


def _auto_combine_1d(datasets, concat_dim=_CONCAT_DIM_DEFAULT,
                     compat='no_conflicts',
                     data_vars='all', coords='different'):
    # This is just the old auto_combine function (which only worked along 1D)
    if concat_dim is not None:
        dim = None if concat_dim is _CONCAT_DIM_DEFAULT else concat_dim
        sorted_datasets = sorted(datasets, key=vars_as_keys)
        grouped_by_vars = itertools.groupby(sorted_datasets, key=vars_as_keys)
        concatenated = [_auto_concat(list(ds_group), dim=dim,
                                     data_vars=data_vars, coords=coords)
                        for id, ds_group in grouped_by_vars]
    else:
        concatenated = datasets
    merged = merge(concatenated, compat=compat)
    return merged


def _new_tile_id(single_id_ds_pair):
    tile_id, ds = single_id_ds_pair
    return tile_id[1:]


def _auto_combine(datasets, concat_dims, compat, data_vars, coords,
                  infer_order_from_coords, ids):
    """
    Calls logic to decide concatenation order before concatenating.
    """

    # Arrange datasets for concatenation
    if infer_order_from_coords:
        raise NotImplementedError
        # TODO Use coordinates to determine tile_ID for each dataset in N-D
        # Ignore how they were ordered previously
        # Should look like:
        # combined_ids, concat_dims = _infer_tile_ids_from_coords(datasets,
        # concat_dims)
    else:
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

    # Repeatedly concatenate then merge along each dimension
    combined = _combine_nd(combined_ids, concat_dims, compat=compat,
                           data_vars=data_vars, coords=coords)
    return combined


def auto_combine(datasets, concat_dim=_CONCAT_DIM_DEFAULT,
                 compat='no_conflicts', data_vars='all', coords='different'):
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
    concat_dim : str or DataArray or Index, optional
        Dimension along which to concatenate variables, as used by
        :py:func:`xarray.concat`. You only need to provide this argument if
        the dimension along which you want to concatenate is not a dimension
        in the original datasets, e.g., if you want to stack a collection of
        2D arrays along a third dimension.
        By default, xarray attempts to infer this argument by examining
        component files. Set ``concat_dim=None`` explicitly to disable
        concatenation.
    compat : {'identical', 'equals', 'broadcast_equals', 'no_conflicts'}, optional
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
        Details are in the documentation of conca

    Returns
    -------
    combined : xarray.Dataset

    See also
    --------
    concat
    Dataset.merge
    """  # noqa

    # Coerce 1D input into ND to maintain backwards-compatible API until API
    # for N-D combine decided
    # (see https://github.com/pydata/xarray/pull/2553/#issuecomment-445892746)
    if concat_dim is None or concat_dim == _CONCAT_DIM_DEFAULT:
        concat_dims = concat_dim
    elif not isinstance(concat_dim, list):
        concat_dims = [concat_dim]
    else:
        concat_dims = concat_dim
    infer_order_from_coords = False

    # The IDs argument tells _auto_combine that the datasets are not yet sorted
    return _auto_combine(datasets, concat_dims=concat_dims, compat=compat,
                         data_vars=data_vars, coords=coords,
                         infer_order_from_coords=infer_order_from_coords,
                         ids=False)
