import warnings
from collections import OrderedDict

import pandas as pd

from . import dtypes, utils
from .alignment import align
from .merge import (
    determine_coords,
    merge_variables,
    expand_variable_dicts,
    _VALID_COMPAT,
)
from .variable import IndexVariable, Variable, as_variable
from .variable import concat as concat_vars


def concat(
    objs,
    dim=None,
    data_vars="all",
    coords="different",
    compat="equals",
    positions=None,
    indexers=None,
    mode=None,
    concat_over=None,
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
            in addition to the 'minimal' coordinates.
    compat : {'equals', 'identical', 'override'}, optional
        String indicating how to compare non-concatenated variables and
        dataset global attributes for potential conflicts. This is passed down to merge.
          * 'equals' means that all variable values and dimensions must be the same;
          * 'identical' means that variable attributes and global attributes
             must also be equal.
          * 'override' means that checks are skipped and values from the first dataset
            are used. This cannot be used with coords='different' or data_vars='different'.
    positions : None or list of integer arrays, optional
        List of integer arrays which specifies the integer positions to which
        to assign each dataset along the concatenated dimension. If not
        supplied, objects are concatenated in the provided order.
    fill_value : scalar, optional
        Value to use for newly missing values
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

    if dim is None:
        warnings.warn(
            "the `dim` argument to `concat` will be required "
            "in a future version of xarray; for now, setting it to "
            "the old default of 'concat_dim'",
            FutureWarning,
            stacklevel=2,
        )
        dim = "concat_dims"

    if indexers is not None:  # pragma: no cover
        warnings.warn(
            "indexers has been renamed to positions; the alias "
            "will be removed in a future version of xarray",
            FutureWarning,
            stacklevel=2,
        )
        positions = indexers

    if mode is not None:
        raise ValueError(
            "`mode` is no longer a valid argument to "
            "xarray.concat; it has been split into the "
            "`data_vars` and `coords` arguments"
        )
    if concat_over is not None:
        raise ValueError(
            "`concat_over` is no longer a valid argument to "
            "xarray.concat; it has been split into the "
            "`data_vars` and `coords` arguments"
        )

    if compat not in _VALID_COMPAT:
        raise ValueError(
            "compat=%r invalid: must be 'equals', 'identical or 'override'" % compat
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
        dim, = coord.dims
    else:
        coord = dim
        dim, = coord.dims
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

    for ds in datasets:
        if concat_over_existing_dim:
            if dim not in ds.dims:
                # TODO: why did I do this
                if dim in ds:
                    ds = ds.set_coords(dim)
                else:
                    raise ValueError("%r is not present in all datasets" % dim)
        concat_over.update(k for k, v in ds.variables.items() if dim in v.dims)

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
            elif opt == "sensible":
                if not concat_over_existing_dim:
                    concat_over.update(
                        set(getattr(datasets[0], subset)) - set(datasets[0].dims)
                    )
            else:
                raise ValueError("unexpected value for %s: %s" % (subset, opt))
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
    return concat_over, equals


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

    # determine dimensional coordinate names and a dict mapping name to DataArray
    def determine_dims(datasets, result_coord_names):
        dims = set()
        coords = dict()
        for ds in datasets:
            for dim in set(ds.dims) - dims:
                if dim not in coords:
                    coords[dim] = ds.coords[dim].variable
            dims = dims | set(ds.dims)
        return dims, coords

    result_coord_names, noncoord_names = determine_coords(datasets)
    both_data_and_coords = result_coord_names & noncoord_names
    if both_data_and_coords:
        raise ValueError(
            "%r is a coordinate in some datasets but not others."
            % list(both_data_and_coords)[0]  # preserve format of error message
        )
    dim_names, result_coords = determine_dims(datasets, result_coord_names)
    # we don't want the concat dimension in the result dataset yet
    result_coords.pop(dim, None)

    # case where concat dimension is a coordinate but not a dimension
    if dim in result_coord_names and dim not in dim_names:
        datasets = [ds.expand_dims(dim) for ds in datasets]

    # determine which variables to concatentate
    concat_over, equals = _calc_concat_over(
        datasets, dim, dim_names, data_vars, coords, compat
    )

    # determine which variables to merge
    variables_to_merge = (result_coord_names | noncoord_names) - concat_over - dim_names
    if variables_to_merge:
        to_merge = []
        for ds in datasets:
            if variables_to_merge - set(ds.variables):
                raise ValueError(
                    "Encountered unexpected variables %r" % list(variables_to_merge)[0]
                )
            to_merge.append(ds.reset_coords()[list(variables_to_merge)])

        merge_equals = {k: equals.get(k, None) for k in variables_to_merge}

        # TODO: Provide equals as an argument and thread that down to merge.unique_variable
        result_vars = merge_variables(
            expand_variable_dicts(to_merge),
            priority_vars=None,
            compat=compat,
            equals=merge_equals,
        )
    else:
        result_vars = OrderedDict()
    result_vars.update(result_coords)

    # assign attrs and encoding from first dataset
    result_attrs = datasets[0].attrs
    result_encoding = datasets[0].encoding

    def insert_result_variable(k, v):
        assert isinstance(v, Variable)
        result_vars[k] = v

    # check that global attributes are fixed across all datasets if necessary
    for ds in datasets[1:]:
        if compat == "identical" and not utils.dict_equiv(ds.attrs, result_attrs):
            raise ValueError("Dataset global attributes are not equal.")

    ##############
    # TODO: do this stuff earlier so we loop over datasets only once
    #############
    # we've already verified everything is consistent; now, calculate
    # shared dimension sizes so we can expand the necessary variables
    dim_lengths = [ds.dims.get(dim, 1) for ds in datasets]
    # non_concat_dims = dim_names - concat_over
    non_concat_dims = {}
    for ds in datasets:
        non_concat_dims.update(ds.dims)
    non_concat_dims.pop(dim, None)

    # seems like there should be a helper function for this. We would need to add
    # an exclude kwarg to exclude comparing along concat_dim
    def ensure_common_dims(vars):
        # ensure each variable with the given name shares the same
        # dimensions and the same shape for all of them except along the
        # concat dimension
        common_dims = tuple(pd.unique([d for v in vars for d in v.dims]))
        if dim not in common_dims:
            common_dims = (dim,) + common_dims
        for var, dim_len in zip(vars, dim_lengths):
            if var.dims != common_dims:
                common_shape = tuple(
                    non_concat_dims.get(d, dim_len) for d in common_dims
                )
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

    # TODO: avoid this
    unlabeled_dims = dim_names - result_coord_names
    result = result.drop(unlabeled_dims, errors="ignore")

    # need to pass test when provided dim is a DataArray
    if coord is not None:
        # add concat dimension last to ensure that its in the final Dataset
        result[coord.name] = coord

    return result


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
