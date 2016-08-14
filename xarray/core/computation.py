import itertools
from collections import namedtuple


def result_name(objects):
    # use the same naming heuristics as pandas:
    # https://github.com/blaze/blaze/issues/458#issuecomment-51936356
    names = set(getattr(obj, 'name', None) for obj in objects)
    names.discard(None)
    if len(names) == 1:
        name, = names
    else:
        name = None
    return name


def apply_dataarray(func, args, join='inner', gufunc_signature=None,
                    kwargs=None, combine_names=None):
    if kwargs is None:
        kwargs = {}

    if combine_names is None:
        combine_names = result_name

    args = deep_align(*args, join=join, copy=False, raise_on_invalid=False)

    coord_variables = [getattr(getattr(a, 'coords', {}), 'variables')
                       for a in args]
    coords = merge_coords_without_align(coord_variables)
    name = combine_names(args)

    data_vars = [getattr(a, 'variable') for a in args]
    variables = func(*data_vars, **kwargs)

    # TODO handle gufunc_signature

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


def apply_dataset(func, args, join='inner', fill_value=None, kwargs=None,
                  combine_attrs=None):
    if kwargs is None:
        kwargs = {}

    if combine_attrs is None:
        combine_attrs = lambda func, attrs: None

    attrs = combine_attrs(getattr(func, 'func', func),
                          [getattr(a, 'attrs') for a in args])

    args = deep_align(*args, join=join, copy=False, raise_on_invalid=False)

    coord_variables = [getattr(getattr(a, 'coords', {}), 'variables')
                       for a in args]
    coords = merge_coords_without_align(coord_variables)

    list_of_data_vars = [getattr(a, 'data_vars', {}) for a in args]
    names = join_dict_keys(list_of_data_vars, how=join)

    list_of_variables = [getattr(a, 'variables', {}) for a in args]
    lists_of_args = reindex_dict_values(list_of_variables, names, fill_value)

    result_vars = OrderedDict()
    for name, variable_args in zip(names, lists_of_args):
        result[name] = func(*variable_args, **kwargs)
    result_vars.update(coords)

    return Dataset._from_vars_and_coord_names(result_vars, coords, attrs)


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



def _as_sequence(arg, cls):
    if is_scalar(arg):
        return cls([arg])
    else:
        return cls(arg)


_ElemwiseSignature = namedtuple(
    '_ElemwiseSignature', 'broadcast_dims, output_dims')

class GUFuncSignature(object):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    @classmethod
    def from_string(cls, string):
        raise NotImplementedError


def _build_and_check_signature(variables, gufunc_signature):
    # core_dims are not broadcast over, and moved to the right with order
    # preserved.

    dim_sizes = _calculate_unified_dim_sizes(variables)

    if gufunc_signature is None:
        # broadcast everything, one output
        dims = tuple(size_dims)
        return _ElemwiseSignature(dims, [dims])

    core_dims = set(itertools.chain.from_iterable(
        itertools.chain(gufunc_signature.inputs, gufunc_signature.outputs)))
    broadcast_dims = tuple(d for d in dim_sizes if d not in core_dims)
    output_dims = [broadcast_dims + out for out in gufunc_signature.outputs]
    return _ElemwiseSignature(broadcast_dims, output_dims)


def _broadcast_variable_data_to(variable, broadcast_dims, allow_dask=True):

    data_attr = 'data' if allow_dask else 'values'
    data = getattr(variable, data_attr)

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


def apply_variable_ufunc(func, args, allow_dask=True, gufunc_signature=None,
                         combine_attrs=None, kwargs=None):

    if kwargs is None:
        kwargs = {}

    if combine_attrs is None:
        combine_attrs = lambda func, attrs: None

    sig = _build_and_check_signature(variables, gufunc_signature)

    n_out = len(sig.output_dims)
    input_attrs = [getattr(a, 'attrs', {}) for a in args]
    result_attrs = [combine_attrs(input_attrs, func, n) for n in range(n_out)]

    list_of_data = []
    for arg in args:
        if isinstance(arg, Variable):
            data = _broadcast_variable_data_to(arg, sig.broadcast_dims,
                                               allow_dask=allow_dask)
        else:
            data = arg
        list_of_data.append(data)

    result_data = func(*list_of_data, **kwargs)

    if n_out > 1:
        output = []
        for dims, data, attrs in zip(
                sig.output_dims, result_data, result_attrs):
            output.append(Variable(dims, data, attrs))
        return tuple(output)
    else:
        dims, = sig.output_dims
        data, = result_data
        attrs = result_attrs
        return Variable(dims, data, attrs)


def apply_ufunc(func, args, join='inner', allow_dask=True, kwargs=None,
                combine_dataset_attrs=None, combine_variable_attrs=None):

    variables_ufunc = functools.partial(
        apply_variable_ufunc, func=func, allow_dask=allow_dask,
        combine_attrs=combine_variable_attrs, kwargs=kwargs)

    if any(is_dict_like(a) for a in args):
        return apply_dataset(variables_ufunc, args, join=join,
                             combine_attrs=combine_dataset_attrs)
    elif any(isinstance(a, DataArray) for a in args):
        return apply_dataarray(variables_ufunc, args, join=join)
    elif any(isinstance(a, Variable) for a in args):
        return variables_ufunc(args=args)
    else:
        return func(args)
