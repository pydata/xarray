from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings

import pandas as pd

from . import utils
from .alignment import align
from .merge import merge
from .pycompat import iteritems, OrderedDict, basestring
from .variable import Variable, as_variable, IndexVariable, concat as concat_vars


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
                         'xarray.concat; it has been split into the `data_vars` '
                         'and `coords` arguments')
    if concat_over is not None:
        raise ValueError('`concat_over` is no longer a valid argument to '
                         'xarray.concat; it has been split into the `data_vars` '
                         'and `coords` arguments')

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
    if isinstance(dim, basestring):
        coord = None
    elif not hasattr(dim, 'dims'):
        # dim is not a DataArray or IndexVariable
        dim_name = getattr(dim, 'name', None)
        if dim_name is None:
            dim_name = 'concat_dim'
        coord = IndexVariable(dim_name, dim)
        dim = dim_name
    elif not hasattr(dim, 'name'):
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
    def process_subset_opt(opt, subset):
        if subset == 'coords':
            subset_long_name = 'coordinates'
        else:
            subset_long_name = 'data variables'

        if isinstance(opt, basestring):
            if opt == 'different':
                def differs(vname):
                    # simple helper function which compares a variable
                    # across all datasets and indicates whether that
                    # variable differs or not.
                    v = datasets[0].variables[vname]
                    return any(not ds.variables[vname].equals(v)
                               for ds in datasets[1:])
                # all nonindexes that are not the same in each dataset
                concat_new = set(k for k in getattr(datasets[0], subset)
                                 if k not in concat_over and differs(k))
            elif opt == 'all':
                concat_new = (set(getattr(datasets[0], subset)) -
                              set(datasets[0].dims))
            elif opt == 'minimal':
                concat_new = set()
            else:
                raise ValueError("unexpected value for concat_%s: %s"
                                 % (subset, opt))
        else:
            invalid_vars = [k for k in opt
                            if k not in getattr(datasets[0], subset)]
            if invalid_vars:
                raise ValueError('some variables in %s are not '
                                 '%s on the first dataset: %s'
                                 % (subset, subset_long_name, invalid_vars))
            concat_new = set(opt)
        return concat_new

    concat_over = set()
    for ds in datasets:
        concat_over.update(k for k, v in ds.variables.items()
                           if dim in v.dims)
    concat_over.update(process_subset_opt(data_vars, 'data_vars'))
    concat_over.update(process_subset_opt(coords, 'coords'))
    if dim in datasets[0]:
        concat_over.add(dim)
    return concat_over


def _dataset_concat(datasets, dim, data_vars, coords, compat, positions):
    """
    Concatenate a sequence of datasets along a new or existing dimension
    """
    from .dataset import Dataset, as_dataset

    if compat not in ['equals', 'identical']:
        raise ValueError("compat=%r invalid: must be 'equals' "
                         "or 'identical'" % compat)

    dim, coord = _calc_concat_dim_coord(dim)
    datasets = [as_dataset(ds) for ds in datasets]
    datasets = align(*datasets, join='outer', copy=False, exclude=[dim])

    concat_over = _calc_concat_over(datasets, dim, data_vars, coords)

    def insert_result_variable(k, v):
        assert isinstance(v, Variable)
        if k in datasets[0].coords:
            result_coord_names.add(k)
        result_vars[k] = v

    # create the new dataset and add constant variables
    result_vars = OrderedDict()
    result_coord_names = set(datasets[0].coords)
    result_attrs = datasets[0].attrs

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
            elif (k in result_vars and k != dim and
                  not getattr(v, compat)(result_vars[k])):
                verb = 'equal' if compat == 'equals' else compat
                raise ValueError(
                    'variable %r not %s across datasets' % (k, verb))

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
                var = var.expand_dims(common_dims, common_shape)
            yield var

    # stack up each variable to fill-out the dataset (in order)
    for k in datasets[0].variables:
        if k in concat_over:
            vars = ensure_common_dims([ds.variables[k] for ds in datasets])
            combined = concat_vars(vars, dim, positions)
            insert_result_variable(k, combined)

    result = Dataset(result_vars, attrs=result_attrs)
    result = result.set_coords(result_coord_names)

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


def _auto_concat(datasets, dim=None):
    if len(datasets) == 1:
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
        return concat(datasets, dim=dim)


_CONCAT_DIM_DEFAULT = '__infer_concat_dim__'


def auto_combine(datasets,
                 concat_dim=_CONCAT_DIM_DEFAULT,
                 compat='no_conflicts'):
    """Attempt to auto-magically combine the given datasets into one.

    This method attempts to combine a list of datasets into a single entity by
    inspecting metadata and using a combination of concat and merge.

    It does not concatenate along more than one dimension or sort data under any
    circumstances. It does align coordinates, but different variables on
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
        :py:func:`xarray.concat`. You only need to provide this argument if the
        dimension along which you want to concatenate is not a dimension in
        the original datasets, e.g., if you want to stack a collection of
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

    Returns
    -------
    combined : xarray.Dataset

    See also
    --------
    concat
    Dataset.merge
    """
    from toolz import itertoolz
    if concat_dim is not None:
        dim = None if concat_dim is _CONCAT_DIM_DEFAULT else concat_dim
        grouped = itertoolz.groupby(lambda ds: tuple(sorted(ds.data_vars)),
                                    datasets).values()
        concatenated = [_auto_concat(ds, dim=dim) for ds in grouped]
    else:
        concatenated = datasets
    merged = merge(concatenated, compat=compat)
    return merged
