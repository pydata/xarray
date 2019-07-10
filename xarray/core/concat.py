import warnings
from collections import OrderedDict

import pandas as pd

from . import utils, dtypes
from .alignment import align
from .variable import IndexVariable, Variable, as_variable
from .variable import concat as concat_vars


def concat(objs, dim=None, data_vars='all', coords='different',
           compat='equals', positions=None, indexers=None, mode=None,
           concat_over=None, fill_value=dtypes.NA):
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
    fill_value : scalar, optional
        Value to use for newly missing values
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

    if indexers is not None:  # pragma: no cover
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
    return f(objs, dim, data_vars, coords, compat, positions, fill_value)


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


def _dataset_concat(datasets, dim, data_vars, coords, compat, positions,
                    fill_value=dtypes.NA):
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
    datasets = align(*datasets, join='outer', copy=False, exclude=[dim],
                     fill_value=fill_value)

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
                      positions, fill_value=dtypes.NA):
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
                         positions, fill_value=fill_value)
    return arrays[0]._from_temp_dataset(ds, name)
