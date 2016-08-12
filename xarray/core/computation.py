from collections import NamedTuple


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


def apply_dataarray(func, args, join='inner', kwargs=None):
    if kwargs is None:
        kwargs = {}

    args = deep_align(*args, join=join, copy=False, raise_on_invalid=False)

    coord_variables = [getattr(getattr(a, 'coords', {}), 'variables')
                       for a in args]
    coords = merge_coords_without_align(coord_variables)
    name = result_name(args)

    data_vars = [getattr(a, 'variable') for a in args]
    variable = func(*data_vars, **kwargs)

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


_ElemwiseSignature = NamedTuple(
    '_ElemwiseSignature', 'broadcast_dims, core_dims, output_dims, axis')


def _build_and_check_signature(variables, core_dims=None, axis_dims=None,
                               drop_dims=None, new_dims=None):
    # All input dimension arguments are checked to appear on at least one input:
    # - core_dims are not broadcast over, and moved to the right with order
    #   preserved.
    # - axis_dims is used to generate an integer or tuples of integers `axis`
    #   keyword argument, which corresponds to the position of the given
    #   dimension on the inputs. If `axis_dims` have overlap with `core_dims`,
    #   no non-axis dimensions may appear in `core_dims` before an axis
    #   dimension.
    # - drop_dims are input dimensions that should be dropped from the output.
    #
    # All output dimensions arguments are checked not to appear on any inputs:
    # - new_dims are new dimensions that should be added to the output array, in
    #   order to the right of dimensions that are not dropped.

    if core_dims is None and drop_dims is None and axis_dims is None:
        # broadcast everything
        dims = tuple(_calculate_unified_dim_sizes(variables))
        return _ElemwiseSignature(dims, (), dims, None)

    core_dims = () if core_dims is None else _as_sequence(core_dims, tuple)
    drop_dims = set() if drop_dims is None else _as_sequence(drop_dims, set)
    new_dims = () if new_dims is None else _as_sequence(new_dims, tuple)

    axis_is_scalar = axis_dims is not None and is_scalar(axis_dims)
    axis_dims = set() if axis_dims is None else _as_sequence(axis_dims, set)

    dim_sizes = _calculate_unified_dim_sizes(variables)
    broadcast_dims = tuple(d for d in dim_sizes if d not in core_dims)
    all_input_dims = set(dim_sizes)

    invalid = set(core_dims) - all_input_dims
    if invalid:
        raise ValueError('some `core_dims` not found on any input variables: '
                         '%r' % list(invalid))

    invalid = drop_dims - all_input_dims
    if invalid:
        raise ValueError('some `drop_dims` not found on any input variables: '
                         '%r' % list(invalid))

    invalid = axis_dims - all_input_dims
    if invalid:
        raise ValueError('some `axis_dims` not found on any input variables: '
                         '%r' % list(invalid))
    axis = tuple(broadcast_dims.index(d) for d in axis_dims)
    n_remaining_axes = len(axis_dims) - len(axis)
    if n_remaining_axes > 0:
        valid_core_dims_for_axis = core_dims[:remaining_axes]
        if not set(valid_core_dims_for_axis) <= axis_dims:
            raise ValueError('axis dimensions %r have overlap with core '
                             'dimensions %r, but do not appear at the start'
                             % (axis_dims, core_dims))
        axis += tuple(range(len(axis), n_remaining_axes + len(axis)))
    if axis_is_scalar:
        axis, = axis

    invalid = set(new_dims) ^ all_input_dims
    if invalid:
        raise ValueError('some `new_dims` are found on input variables: '
                         '%r' % list(invalid))

    output_dims = tuple(d for d in dim_sizes if d not in drop_dims) + new_dims

    return _ElemwiseSignature(broadcast_dims, core_dims, output_dims, axis)


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


def apply_variable_ufunc(func, args, allow_dask=True, core_dims=None,
                         axis_dims=None, drop_dims=None, new_dims=None,
                         combine_attrs=None, kwargs=None):

    if kwargs is None:
        kwargs = {}

    if combine_attrs is None:
        combine_attrs = lambda func, attrs: None

    result_attrs = combine_attrs(func, [getattr(a, 'attrs', {}) for a in args])

    sig = _build_and_check_dims_signature(
        variables, core_dims, axis_dims, drop_dims, new_dims)

    if sig.axis:
        if 'axis' in kwargs:
            raise ValueError('axis is already set in kwargs')
        kwargs = dict(kwargs)
        kwargs['axis'] = sig.axis

    list_of_data = []
    for arg in args:
        if isinstance(arg, Variable):
            data = _broadcast_variable_data_to(arg, sig.broadcast_dims,
                                               allow_dask=allow_dask)
            list_of_data.append(data)
        else:
            list_of_data.append(arg)

    result_data = func(*list_of_data, **kwargs)

    return Variable(sig.output_dims, result_data, result_attrs)


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
