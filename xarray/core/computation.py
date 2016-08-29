import itertools
import re
from collections import namedtuple

from .core.pycompat import dask_array_type


# see http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html
DIMENSION_NAME = r'\w+'
CORE_DIMENSION_LIST = '(?:' + DIMENSION_NAME + '(?:,' + DIMENSION_NAME + ')*)?'
ARGUMENT = r'\(' + CORE_DIMENSION_LIST + r'\)'
ARGUMENT_LIST = ARGUMENT + '(?:,'+ ARGUMENT + ')*'
SIGNATURE = '^' + ARGUMENT_LIST + '->' + ARGUMENT_LIST + '$'


class Signature(object):
    """Core dimensions signature for a given function

    Based on the signature provided by generalized ufuncs in NumPy.

    Attributes
    ----------
    input_core_dims : list of tuples
        A list of tuples of core dimension names on each input variable.
    output_core_dims : list of tuples
        A list of tuples of core dimension names on each output variable.
    """
    def __init__(self, input_core_dims, output_core_dims=((),)):
        if dtypes is None:
            dtypes = {}
        self.input_core_dims = input_core_dims
        self.output_core_dims = output_core_dims
        self._all_input_core_dims = None
        self._all_output_core_dims = None

    @property
    def all_input_core_dims(self):
        if self._all_input_core_dims is None:
            self._all_input_core_dims = frozenset(
                dim for dims in self.input_core_dims for dim in dims)
        return self._all_input_core_dims

    @property
    def all_output_core_dims(self):
        if self._all_output_core_dims is None:
            self._all_output_core_dims = frozenset(
                dim for dims in self.output_core_dims for dim in dims)
        return self._all_output_core_dims

    @classmethod
    def from_string(cls, string):
        """Create a Signature object from a NumPy gufunc signature.

        Parameters
        ----------
        string : str
            Signature string, e.g., (m,n),(n,p)->(m,p).
        """
        if not re.match(SIGNATURE, string):
            raise ValueError('not a valid gufunc signature: {}'.format(string))
        return cls(*[[re.findall(DIMENSION_NAME, arg)
                      for arg in re.findall(ARGUMENT, arg_list)]
                     for arg_list in string.split('->')])

    @classmethod
    def from_ufunc(cls, ufunc):
        raise NotImplementedError


def _default_signature(n_inputs):
    return Signature([()] * n_inputs, [()])


def _result_name(objects):
    # use the same naming heuristics as pandas:
    # https://github.com/blaze/blaze/issues/458#issuecomment-51936356
    names = set(getattr(obj, 'name', None) for obj in objects)
    names.discard(None)
    if len(names) == 1:
        name, = names
    else:
        name = None
    return name


def _default_result_attrs(attrs, func, signature):
    return [{}] * len(signature.outputs)


def _build_output_coords(args, signature, new_coords=None):

    def get_coord_variables(arg):
        return getattr(getattr(arg, 'coords', {}), 'variables')

    coord_variables = [get_coord_variables(a) for a in args]
    if new_coords is not None:
        coord_variables.append(get_coord_variables(new_coords))

    merged = merge_coords_without_align(coord_variables)

    output = []
    for output_dims in signature.output_core_dims:
        dropped_dims = signature.all_input_core_dims - set(output_dims)
        coords = OrderedDict((k, v) for k, v in merged.items()
                             if set(v.dims).isdisjoint(dropped_dims))
        output.append(coords)

    return output


def apply_dataarray_ufunc(args, func, signature=None, join='inner',
                          new_coords=None):
    if signature is None:
        signature = _default_signature(len(args))

    args = deep_align(*args, join=join, copy=False, raise_on_invalid=False)

    name = _result_name(args)
    list_of_coords = _build_output_coords(args, signature, new_coords)

    data_vars = [getattr(a, 'variable') for a in args]
    variable_or_variables = func(*data_vars)

    if len(signature.output_dims) > 1:
        return tuple(DataArray(variable, coords, name=name, fastpath=True)
                     for variable, coords, name in zip(
                         variable_or_variables, list_of_coords))
    else:
        variable = variable_or_variables
        coords, = list_of_coords
        return DataArray(variable, coords, name=name, fastpath=True)


def join_dict_keys(objects, how='inner')
    joiner = _get_joiner(how)
    all_keys = (obj.keys() for obj in objects if hasattr(obj, 'keys'))
    result_keys = list(joiner(pd.Index(keys) for keys in all_keys))


def collect_dict_values(objects, keys, fill_value=None)
    result_values = []
    for key in keys:
        values = []
        for obj in objects:
            if hasattr(obj, 'keys'):
                values.append(obj.get(key, fill_value))
            else:
                values = obj
        result_values.append(values)
    return result_values


def apply_dataset_ufunc(args, func, signature=None, join='inner',
                        fill_value=None, new_coords=None):
    if signature is None:
        signature = _default_signature(len(args))

    args = deep_align(*args, join=join, copy=False, raise_on_invalid=False)

    list_of_coords = _build_output_coords(args, signature, new_coords)

    list_of_data_vars = [getattr(a, 'data_vars', {}) for a in args]
    names = join_dict_keys(list_of_data_vars, how=join)

    list_of_variables = [getattr(a, 'variables', {}) for a in args]
    lists_of_args = reindex_dict_values(list_of_variables, names, fill_value)

    result_vars = OrderedDict()
    for name, variable_args in zip(names, lists_of_args):
        result_vars[name] = func(*variable_args)

    def make_dataset(data_vars, coord_vars):
        # Normally, we would copy data_vars to be safe, but we created the
        # OrderedDict in this function and don't use it for anything else.
        variables = data_vars
        variables.update(coord_vars)
        coord_names = set(coord_vars)
        return Dataset._from_vars_and_coord_names(variables, coord_names)

    n_outputs = len(signature.output_dims)
    if n_outputs > 1:
        # we need to unpack result_vars from Dict[object, Tuple[Variable]] ->
        # Tuple[Dict[object, Variable]].
        list_of_result_vars = [OrderedDict() for _ in n_outputs]
        for name, values in result_vars.items():
            for value, results_dict in zip(values, list_of_result_vars):
                list_of_result_vars[name] = value

        return tuple(make_dataset(*args)
                     for args in zip(list_of_result_vars, list_of_coords))
    else:
        data_vars = result_vars
        coords_vars, = list_of_coords
        return make_dataset(data_vars, coord_vars)




def _iter_over_selections(obj, dim, values):
    """Iterate over selections of an xarray object in the provided order.
    """
    dummy = None
    for value in values:
        try:
            obj_sel = obj.sel(**{dim: values})
        except KeyError:
            if dim not in obj.dims:
                raise ValueError('incompatible dimensions for a grouped '
                                 'binary operation: the group variable %r '
                                 'is not a dimension on the other argument'
                                 % dim)
            if dummy is None:
                dummy = _dummy_copy(obj)
            obj_sel = dummy
        yield obj_sel


def apply_groupby_ufunc(args, func):
    groupbys = [arg for arg in args if isinstance(GroupBy)]
    if not groupbys:
        raise ValueError('must have at least one groupby to iterate over')
    first_groupby = groups[0]
    if any(not first_groupby.unique_coord.equals(gb.unique_coord)
           for gb in groupbys[1:]):
        raise ValueError('can only perform operations over multiple groupbys '
                         'at once if they have all the same unique coordinate')

    grouped_dim = first_groupby.group.name
    unique_values = first_groupby.unique_coord.values

    iterators = []
    for arg in args:
        if isinstance(arg, GroupBy):
            iterator = (value for _, value in arg)
        elif hasattr(arg, 'dims') and group_name in arg.dims:
            if isinstance(arg, Variable):
                raise ValueError(
                    'groupby operations cannot be performed with '
                    'xarray.Variable objects that share a dimension with '
                    'the grouped dimension')
            iterator = _iter_over_selections(arg, grouped_dim, unique_vlaues)
        else:
            iterator = itertools.repeat(arg)
        iterators.append(iterator)

    applied = (func(*zipped_args) for zipped_args in zip(iterators))
    applied_example, applied = peek_at(applied)
    combine = first_groupby._combined
    if isinstance(applied_example, tuple):
        combined = tuple(combine(output) for output in zip(*applied))
    else:
        combined = combine(applied)
    return combined


def _calculate_unified_dim_sizes(variables):
    dim_sizes = OrderedDict()

    for var in variables:
        try:
            var_dims = var.dims
        except AttributeError:
            continue

        if len(set(var_dims)) < len(var_dims):
            raise ValueError('broadcasting cannot handle duplicate '
                             'dimensions: %r' % list(var_dims))
        for dim, size in zip(var_dims, var.shape):
            if dim not in dim_sizes:
                dim_sizes[dim] = size
            elif dim_sizes[dim] != size:
                raise ValueError('operands cannot be broadcast together '
                                 'with mismatched lengths for dimension '
                                 '%r: %s vs %s'
                                 % (dim, dim_sizes[dim], size))
    return dim_sizes


def _broadcast_variable_data_to(variable, broadcast_dims):

    data = variable.data

    old_dims = variable.dims
    if broadcast_dims == old_dims:
        return data

    assert set(broadcast_dims) <= set(old_dims)

    # for consistency with numpy, keep broadcast dimensions to the left
    reordered_dims = (tuple(d for d in broadcast_dims if d in old_dims) +
                      tuple(d for d in old_dims if d not in broadcast_dims))
    if reordered_dims != old_dims:
        order = tuple(old_dims.index(d) for d in reordered_dims)
        data = ops.transpose(data, order)

    new_dims = tuple(d for d in broadcast_dims if d not in old_dims)
    if new_dims:
        data = data[(np.newaxis,) * len(new_dims) + (Ellipsis,)]

    return data


def _deep_unpack_list(arg):
    if isinstance(arg, list):
        arg, = arg
        return _deep_unpack_list(arg)
    return arg


def _apply_with_dask_atop(func, list_of_input_data, signature, kwargs, dtype):
    import toolz  # required dependency of dask.array

    if len(signature.output_core_dims) > 1:
        raise ValueError('cannot create use dask.array.atop for '
                         'multiple outputs')
    if signature.all_output_core_dims - signature.all_input_core_dims:
        raise ValueError('cannot create new dimensions in dask.array.atop')

    input_dims = [broadcast_dims + inp for inp in signature.input_core_dims]
    dropped = signature.all_input_core_dims - signature.all_output_core_dims
    for data, dims in zip(list_of_input_data, input_dims):
        if isinstance(data, dask_array_type):
            for dropped_dim in dropped:
                if (dropped_dim in dims and
                    len(data.chunks[dims.index(dropped_dim)]) != 1):
                raise ValueError('dimension %r dropped in the output does not '
                                 'consist of exactly one chunk on all arrays '
                                 'in the inputs' % dropped_dim)

    out_ind, = output_dims
    atop_args = [ai for a in (list_of_input_data, input_dims) for ai in a]
    func2 = toolz.functools.compose(func, _deep_unpack_list)
    result_data = da.atop(func2, out_ind, *atop_args, dtype=dtype, **kwargs)


def apply_variable_ufunc(args, func, signature=None, dask_array='forbidden',
                         kwargs=None, dtype=None):

    if signature is None:
        signature = _default_signature(len(args))
    if dask_array not in {'forbidden', 'allowed', 'auto'}:
        raise ValueError('unknown setting for dask array handling')
    if kwargs is None:
        kwargs = {}

    dim_sizes = _calculate_unified_dim_sizes(variables)
    core_dims = signature.input_core_dims | signature.output_core_dims
    broadcast_dims = tuple(d for d in dim_sizes if d not in core_dims)
    output_dims = [broadcast_dims + out for out in signature.output_core_dims]

    list_of_input_data = []
    for arg in args:
        if isinstance(arg, Variable):
            data = _broadcast_variable_data_to(arg, broadcast_dims)
        else:
            data = arg
        list_of_input_data.append(data)

    contains_dask = any(isinstance(d, dask_array_type)
                        for d in list_of_input_data)

    if dask_array == 'forbidden' and contains_dask:
        raise ValueError('encountered dask array')
    elif dask_array == 'auto' and contains_dask:
        result_data = _apply_with_dask_atop(func, list_of_input_data, signature,
                                            kwargs, dtype)
    else:
        result_data = func(*list_of_input_data, **kwargs)

    if len(output_dims) > 1:
        output = []
        for dims, data in zip(output_dims, result_data):
            output.append(Variable(dims, data))
        return tuple(output)
    else:
        dims, = output_dims
        data = result_data
        return Variable(dims, data)


def apply_ufunc(args, func=None, kwargs=None, signature=None, join='inner',
                dask_array='forbidden', dtype=None, new_coords=None):

    if signature is None:
        signature = _default_signature(len(args))

    variables_ufunc = functools.partial(
        apply_variable_ufunc, func=func, dask_array=dask_array, kwargs=kwargs)

    if any(isinstance(a, GroupBy) for a in args):
        partial_apply_ufunc = functools.partial(
            apply_ufunc, func=func, kwargs=kwargs, signature=signature,
            join=join, dask_array=dask_array, dtype=dtype,
            new_coords=new_coords)
        return apply_groupby_ufunc(args, partial_apply_ufunc)
    elif any(is_dict_like(a) for a in args):
        return apply_dataset_ufunc(args, variables_ufunc, signature=signature,
                                   join=join, new_coords=new_coords)
    elif any(isinstance(a, DataArray) for a in args):
        return apply_dataarray_ufunc(args, variables_ufunc, signature=signature,
                                     join=join, new_coords=new_coords)
    elif any(isinstance(a, Variable) for a in args):
        return variables_ufunc(args)
    elif dask_array == 'auto' and any(
            isinstance(arg, dask_array_type) for arg in args):
        import dask.array as da
        if signature.all_input_core_dims or signature.all_output_core_dims:
            raise ValueError("cannot use dask_array='auto' on unlabeled dask "
                             'arrays with a function signature that uses core '
                             'dimensions')
        return da.elemwise(func, *args, dtype=dtype)
    else:
        return func(args)


# def mean(xarray_object, dim=None):
#     if dim is None:
#         signature = Signature([(dim,)])
#         kwargs = {'axis': -1}
#     else:
#         signature = Signature([xarray_object.dims])
#         kwargs = {}
#     return apply_ufunc([xarray_object], ops.mean, signature,
#                        dask_array='allowed', kwargs=kwargs)
