import pandas as pd
from collections import OrderedDict

from . import dtypes, utils
from .alignment import align
from .common import full_like
from .duck_array_ops import lazy_array_equiv
from .merge import _VALID_COMPAT, unique_variable
from .variable import IndexVariable, Variable, as_variable
from .variable import concat as concat_vars


def concat(
    objs,
    dim,
    data_vars="all",
    coords="different",
    compat="equals",
    positions=None,
    fill_value=dtypes.NA,
    join="outer",
):
    """Concatenate xarray objects along a new or existing dimension.

    Parameters
    ----------
    objs : sequence of Dataset and DataArray objects
        xarray objects to concatenate together. Each object is expected to
        consist of variables and coordinates with matching shapes except for
        along the concatenated dimension.
    dim : str, DataArray, Variable, or pandas.Index
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
            in addition to the 'minimal' coordinates.
    compat : {'identical', 'equals', 'broadcast_equals', 'no_conflicts', 'override'}, optional
        String indicating how to compare non-concatenated variables of the same name for
        potential conflicts. This is passed down to merge.

        - 'broadcast_equals': all values must be equal when variables are
          broadcast against each other to ensure common dimensions.
        - 'equals': all values and dimensions must be the same.
        - 'identical': all values, dimensions and attributes must be the
          same.
        - 'no_conflicts': only values which are not null in both datasets
          must be equal. The returned dataset then contains the combination
          of all non-null values.
        - 'override': skip comparing and pick variable from first dataset
    positions : None or list of integer arrays, optional
        List of integer arrays which specifies the integer positions to which
        to assign each dataset along the concatenated dimension. If not
        supplied, objects are concatenated in the provided order.
    fill_value : scalar, optional
        Value to use for newly missing values as well as to fill values where the
        variable is not present in all datasets.
    join : {'outer', 'inner', 'left', 'right', 'exact'}, optional
        String indicating how to combine differing indexes
        (excluding dim) in objects

        - 'outer': use the union of object indexes
        - 'inner': use the intersection of object indexes
        - 'left': use indexes from the first object with each dimension
        - 'right': use indexes from the last object with each dimension
        - 'exact': instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - 'override': if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.

    indexers, mode, concat_over : deprecated

    Returns
    -------
    concatenated : type of objs

    See also
    --------
    merge
    auto_combine
    """
    # TODO: add ignore_index arguments copied from pandas.concat
    # TODO: support concatenating scalar coordinates even if the concatenated
    # dimension already exists
    from .dataset import Dataset
    from .dataarray import DataArray

    try:
        first_obj, objs = utils.peek_at(objs)
    except StopIteration:
        raise ValueError("must supply at least one object to concatenate")

    if compat not in _VALID_COMPAT:
        raise ValueError(
            "compat=%r invalid: must be 'broadcast_equals', 'equals', 'identical', 'no_conflicts' or 'override'"
            % compat
        )

    if isinstance(first_obj, DataArray):
        f = _dataarray_concat
    elif isinstance(first_obj, Dataset):
        f = _dataset_concat
    else:
        raise TypeError(
            "can only concatenate xarray Dataset and DataArray "
            "objects, got %s" % type(first_obj)
        )

    return f(objs, dim, data_vars, coords, compat, positions, fill_value, join)


def _calc_concat_dim_coord(dim):
    """
    Infer the dimension name and 1d coordinate variable (if appropriate)
    for concatenating along the new dimension.
    """
    from .dataarray import DataArray

    if isinstance(dim, str):
        coord = None
    elif not isinstance(dim, (DataArray, Variable)):
        dim_name = getattr(dim, "name", None)
        if dim_name is None:
            dim_name = "concat_dim"
        coord = IndexVariable(dim_name, dim)
        dim = dim_name
    elif not isinstance(dim, DataArray):
        coord = as_variable(dim).to_index_variable()
        (dim,) = coord.dims
    else:
        coord = dim
        (dim,) = coord.dims
    return dim, coord


def _calc_concat_over(datasets, dim, dim_names, data_vars, coords, compat):
    """
    Determine which dataset variables need to be concatenated in the result,
    """
    # Return values
    concat_over = set()
    equals = {}

    if dim in dim_names:
        concat_over_existing_dim = True
        concat_over.add(dim)
    else:
        concat_over_existing_dim = False

    concat_dim_lengths = []
    for ds in datasets:
        if concat_over_existing_dim:
            if dim not in ds.dims:
                if dim in ds:
                    ds = ds.set_coords(dim)
        concat_over.update(k for k, v in ds.variables.items() if dim in v.dims)
        concat_dim_lengths.append(ds.dims.get(dim, 1))

    def process_subset_opt(opt, subset):
        if isinstance(opt, str):
            if opt == "different":
                if compat == "override":
                    raise ValueError(
                        "Cannot specify both %s='different' and compat='override'."
                        % subset
                    )
                # all nonindexes that are not the same in each dataset
                for k in getattr(datasets[0], subset):
                    if k not in concat_over:
                        equals[k] = None
                        variables = [ds.variables[k] for ds in datasets]
                        # first check without comparing values i.e. no computes
                        for var in variables[1:]:
                            equals[k] = getattr(variables[0], compat)(
                                var, equiv=lazy_array_equiv
                            )
                            if equals[k] is not True:
                                # exit early if we know these are not equal or that
                                # equality cannot be determined i.e. one or all of
                                # the variables wraps a numpy array
                                break

                        if equals[k] is False:
                            concat_over.add(k)

                        elif equals[k] is None:
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
                                if not getattr(v_lhs, compat)(v_rhs):
                                    concat_over.add(k)
                                    equals[k] = False
                                    # computed variables are not to be re-computed
                                    # again in the future
                                    for ds, v in zip(datasets[1:], computed):
                                        ds.variables[k].data = v.data
                                    break
                            else:
                                equals[k] = True

            elif opt == "all":
                concat_over.update(
                    set(getattr(datasets[0], subset)) - set(datasets[0].dims)
                )
            elif opt == "minimal":
                pass
            else:
                raise ValueError(f"unexpected value for {subset}: {opt}")
        else:
            invalid_vars = [k for k in opt if k not in getattr(datasets[0], subset)]
            if invalid_vars:
                if subset == "coords":
                    raise ValueError(
                        "some variables in coords are not coordinates on "
                        "the first dataset: %s" % (invalid_vars,)
                    )
                else:
                    raise ValueError(
                        "some variables in data_vars are not data variables "
                        "on the first dataset: %s" % (invalid_vars,)
                    )
            concat_over.update(opt)

    process_subset_opt(data_vars, "data_vars")
    process_subset_opt(coords, "coords")
    return concat_over, equals, concat_dim_lengths


# determine dimensional coordinate names and a dict mapping name to DataArray
def _parse_datasets(datasets):

    dims = set()
    all_coord_names = set()
    data_vars = set()  # list of data_vars
    dim_coords = {}  # maps dim name to variable
    dims_sizes = {}  # shared dimension sizes to expand variables

    for ds in datasets:
        dims_sizes.update(ds.dims)
        all_coord_names.update(ds.coords)
        data_vars.update(ds.data_vars)

        for dim in set(ds.dims) - dims:
            if dim not in dim_coords:
                dim_coords[dim] = ds.coords[dim].variable
        dims = dims | set(ds.dims)

    return dim_coords, dims_sizes, all_coord_names, data_vars


def _dataset_concat(
    datasets,
    dim,
    data_vars,
    coords,
    compat,
    positions,
    fill_value=dtypes.NA,
    join="outer",
):
    """
    Concatenate a sequence of datasets along a new or existing dimension
    """
    from .dataset import Dataset

    dim, coord = _calc_concat_dim_coord(dim)
    # Make sure we're working on a copy (we'll be loading variables)
    datasets = [ds.copy() for ds in datasets]
    datasets = align(
        *datasets, join=join, copy=False, exclude=[dim], fill_value=fill_value
    )

    dim_coords, dims_sizes, coord_names, data_names = _parse_datasets(datasets)
    dim_names = set(dim_coords)
    unlabeled_dims = dim_names - coord_names

    both_data_and_coords = coord_names & data_names
    if both_data_and_coords:
        raise ValueError(
            "%r is a coordinate in some datasets but not others." % both_data_and_coords
        )
    # we don't want the concat dimension in the result dataset yet
    dim_coords.pop(dim, None)
    dims_sizes.pop(dim, None)

    # case where concat dimension is a coordinate or data_var but not a dimension
    if (dim in coord_names or dim in data_names) and dim not in dim_names:
        datasets = [ds.expand_dims(dim) for ds in datasets]

    # determine which variables to concatentate
    concat_over, equals, concat_dim_lengths = _calc_concat_over(
        datasets, dim, dim_names, data_vars, coords, compat
    )

    # determine which variables to merge, and then merge them according to compat
    variables_to_merge = (coord_names | data_names) - concat_over - dim_names

    result_vars = {}
    if variables_to_merge:
        to_merge = {var: [] for var in variables_to_merge}

        for ds in datasets:
            for var in variables_to_merge:
                if var in ds:
                    to_merge[var].append(ds.variables[var])

        for var in variables_to_merge:
            result_vars[var] = unique_variable(
                var, to_merge[var], compat=compat, equals=equals.get(var, None)
            )
    else:
        result_vars = {}
    result_vars.update(dim_coords)

    # assign attrs and encoding from first dataset
    result_attrs = datasets[0].attrs
    result_encoding = datasets[0].encoding

    # check that global attributes are fixed across all datasets if necessary
    for ds in datasets[1:]:
        if compat == "identical" and not utils.dict_equiv(ds.attrs, result_attrs):
            raise ValueError("Dataset global attributes not equal.")

    # we've already verified everything is consistent; now, calculate
    # shared dimension sizes so we can expand the necessary variables
    def ensure_common_dims(vars):
        # ensure each variable with the given name shares the same
        # dimensions and the same shape for all of them except along the
        # concat dimension
        common_dims = tuple(pd.unique([d for v in vars for d in v.dims]))
        if dim not in common_dims:
            common_dims = (dim,) + common_dims
        for var, dim_len in zip(vars, concat_dim_lengths):
            if var.dims != common_dims:
                common_shape = tuple(dims_sizes.get(d, dim_len) for d in common_dims)
                var = var.set_dims(common_dims, common_shape)
            yield var

    # Find union of all data variables (preserving order)
    # assumes all datasets are relatively in the same order
    # and missing variables are inserted in the correct position
    # if datasets have variables in drastically different orders
    # the resulting order will be dependent on the order they are in the list
    # passed to concat
    union_of_variables = OrderedDict()
    union_of_coordinates = OrderedDict()
    for ds in datasets:
        var_list = list(ds.variables.keys())

        _find_ordering_inplace(var_list, union_of_variables)

        # check that all datasets have the same coordinate set
        if len(union_of_coordinates) > 0:
            coord_set_diff = (
                union_of_coordinates.keys() ^ ds.coords.keys()
            ) & concat_over
            if len(coord_set_diff) > 0:
                raise ValueError(
                    "Variables %r are coordinates in some datasets but not others."
                    % coord_set_diff
                )

        union_of_coordinates = OrderedDict(
            union_of_coordinates.items() | OrderedDict.fromkeys(ds.coords).items()
        )

    # we don't want to fill coordinate variables so remove them
    for k in union_of_coordinates.keys():
        union_of_variables.pop(k, None)

    # Cache a filled tmp variable with correct dims for filling missing variables
    # doing this here allows us to concat with variables missing from any dataset
    # only will run until it finds one protype for each variable in concat list
    # we will also only fill defaults for data_vars not coordinates

    # optimization to allow us to break when filling variable
    def find_fill_variable_from_ds(variable_key, union_of_variables, datasets):
        for ds in datasets:
            if union_of_variables[variable_key] is not None:
                continue

            if variable_key not in ds.variables:
                continue

            v_fill_value = fill_value
            dtype, v_fill_value = dtypes.get_fill_value_for_variable(
                ds[variable_key], fill_value
            )

            union_of_variables[variable_key] = full_like(
                ds[variable_key], fill_value=v_fill_value, dtype=dtype
            )
            return

    for v in union_of_variables.keys():
        find_fill_variable_from_ds(v, union_of_variables, datasets)

    # create the concat list filling in missing variables
    while len(union_of_variables) > 0 or len(union_of_coordinates) > 0:
        k = None
        # get the variables in order
        if len(union_of_variables) > 0:
            k = union_of_variables.popitem(last=False)
        elif len(union_of_coordinates) > 0:
            k = union_of_coordinates.popitem()

        if k[0] in concat_over:
            variables = []
            for ds in datasets:
                if k[0] in ds.variables:
                    variables.append(ds.variables[k[0]])
                else:
                    # var is missing, fill with cached value
                    variables.append(k[1])

            vars = ensure_common_dims(variables)
            combined = concat_vars(vars, dim, positions)
            assert isinstance(combined, Variable)
            result_vars[k[0]] = combined

    result = Dataset(result_vars, attrs=result_attrs)
    result = result.set_coords(coord_names)
    result.encoding = result_encoding

    result = result.drop_vars(unlabeled_dims, errors="ignore")

    if coord is not None:
        # add concat dimension last to ensure that its in the final Dataset
        result[coord.name] = coord

    return result


def _find_ordering_inplace(l, union):
    # this logic maintains the order of the variable list and runs in
    # O(n^2) where n is number of variables in the uncommon worst case
    # where there are no missing variables this will be O(n)
    # could potentially be refactored to a more generic function to determine
    # a consistent ordering of variables if proper consideration were
    # given both to the runtime as well as to the user scenarios
    for i in range(0, len(l)):
        if l[i] not in union:
            # need to determine the correct place
            # first add the new item which will be at the end
            union[l[i]] = None
            union.move_to_end(l[i])
            # move any items after this in the variables list to the end
            # this will only happen for missing variables
            for j in range(i + 1, len(l)):
                if l[j] in union:
                    union.move_to_end(l[j])


def _dataarray_concat(
    arrays,
    dim,
    data_vars,
    coords,
    compat,
    positions,
    fill_value=dtypes.NA,
    join="outer",
):
    arrays = list(arrays)

    if data_vars != "all":
        raise ValueError(
            "data_vars is not a valid argument when concatenating DataArray objects"
        )

    datasets = []
    for n, arr in enumerate(arrays):
        if n == 0:
            name = arr.name
        elif name != arr.name:
            if compat == "identical":
                raise ValueError("array names not identical")
            else:
                arr = arr.rename(name)
        datasets.append(arr._to_temp_dataset())

    ds = _dataset_concat(
        datasets,
        dim,
        data_vars,
        coords,
        compat,
        positions,
        fill_value=fill_value,
        join=join,
    )
    return arrays[0]._from_temp_dataset(ds, name)
