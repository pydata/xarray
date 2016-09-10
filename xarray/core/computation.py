import collections
import functools
import itertools
import re
from collections import namedtuple

import pandas as pd

from . import ops
from .alignment import _get_joiner
from .merge import _align_for_merge as deep_align
from .merge import merge_coords_without_align
from .utils import is_dict_like
from .pycompat import dask_array_type, OrderedDict, basestring, suppress


SLICE_NONE = slice(None)

# see http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html
DIMENSION_NAME = r'\w+'
CORE_DIMENSION_LIST = '(?:' + DIMENSION_NAME + '(?:,' + DIMENSION_NAME + ')*)?'
ARGUMENT = r'\(' + CORE_DIMENSION_LIST + r'\)'
ARGUMENT_LIST = ARGUMENT + '(?:,'+ ARGUMENT + ')*'
SIGNATURE = '^' + ARGUMENT_LIST + '->' + ARGUMENT_LIST + '$'


def safe_tuple(x):
    if isinstance(x, basestring):
        raise ValueError('cannot safely convert %r to a tuple')
    return tuple(x)


class _Signature(object):
    """Core dimensions signature for a given function.

    Based on the signature provided by generalized ufuncs in NumPy.

    Attributes
    ----------
    input_core_dims : list of tuples
        A list of tuples of core dimension names on each input variable.
    output_core_dims : list of tuples
        A list of tuples of core dimension names on each output variable.
    """
    def __init__(self, input_core_dims, output_core_dims=((),)):
        self.input_core_dims = tuple(safe_tuple(a) for a in input_core_dims)
        self.output_core_dims = tuple(safe_tuple(a) for a in output_core_dims)
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

    @property
    def n_inputs(self):
        return len(self.input_core_dims)

    @property
    def n_outputs(self):
        return len(self.output_core_dims)

    @classmethod
    def default(cls, n_inputs):
        return cls([()] * n_inputs, [()])

    @classmethod
    def from_sequence(cls, nested):
        if (not isinstance(nested, collections.Sequence) or
                not len(nested) == 2 or
                any(not isinstance(arg_list, collections.Sequence)
                    for arg_list in nested) or
                any(isinstance(arg, basestring) or
                    not isinstance(arg, collections.Sequence)
                    for arg_list in nested for arg in arg_list)):
            raise TypeError('functions signatures not provided as a string '
                            'must be a triply nested sequence providing the '
                            'list of core dimensions for each variable, for '
                            'both input and output.')
        return cls(*nested)

    @classmethod
    def from_string(cls, string):
        """Create a _Signature object from a NumPy gufunc signature.

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

    def __eq__(self, other):
        try:
            return (self.input_core_dims == other.input_core_dims and
                    self.output_core_dims == other.output_core_dims)
        except AttributeError:
            return False

    def __ne__(self, other):
        return self != other

    def __repr__(self):
        return ('%s(%r, %r)'
                % (type(self).__name__,
                   list(self.input_core_dims),
                   list(self.output_core_dims)))


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


def _default_result_attrs(attrs, func, signature):
    return [{}] * signature.n_outputs


def build_output_coords(args, signature, new_coords=None):
    coord_variables = []
    for arg in args:
        try:
            coords = arg.coords
        except AttributeError:
            pass  # skip this argument
        else:
            coord_vars = getattr(coords, 'variables', coords)
            coord_variables.append(coord_vars)

    if new_coords is not None:
        coord_variables.append(getattr(new_coords, 'variables', new_coords))

    if len(args) == 1 and new_coords is None:
        # we can skip the expensive merge
        coord_vars, = coord_variables
        merged = OrderedDict(coord_vars)
    else:
        merged = merge_coords_without_align(coord_variables)

    missing_dims = signature.all_output_core_dims - set(merged)
    if missing_dims:
        raise ValueError('new output dimensions must have matching entries in '
                         '`new_coords`: %r' % missing_dims)

    output_coords = []
    for output_dims in signature.output_core_dims:
        dropped_dims = signature.all_input_core_dims - set(output_dims)
        if dropped_dims:
            coords = OrderedDict((k, v) for k, v in merged.items()
                                 if set(v.dims).isdisjoint(dropped_dims))
        else:
            coords = merged
        output_coords.append(coords)

    return output_coords


def apply_dataarray_ufunc(func, *args, **kwargs):
    """apply_dataarray_ufunc(func, *args, signature=None, join='inner',
                             new_coords=None)
    """
    from .dataarray import DataArray

    signature = kwargs.pop('signature')
    join = kwargs.pop('join', 'inner')
    new_coords = kwargs.pop('new_coords', None)
    if kwargs:
        raise TypeError('apply_dataarray_ufunc() got unexpected keyword arguments: %s'
                        % list(kwargs))

    if signature is None:
        signature = _default_signature(len(args))

    if len(args) > 1:
        args = deep_align(args, join=join, copy=False, raise_on_invalid=False)

    name = result_name(args)
    result_coords = build_output_coords(args, signature, new_coords)

    data_vars = [getattr(a, 'variable', a) for a in args]
    result_var = func(*data_vars)

    if signature.n_outputs > 1:
        return tuple(DataArray(variable, coords, name=name, fastpath=True)
                     for variable, coords in zip(result_var, result_coords))
    else:
        coords, = result_coords
        return DataArray(result_var, coords, name=name, fastpath=True)


def join_dict_keys(objects, how='inner'):
    all_keys = [obj.keys() for obj in objects if hasattr(obj, 'keys')]

    if len(all_keys) == 1:
        # shortcut
        result_keys, = all_keys
        return result_keys

    joiner = _get_joiner(how)
    # TODO: use a faster ordered set than a pandas.Index
    result_keys = joiner([pd.Index(keys) for keys in all_keys])
    return result_keys


def collect_dict_values(objects, keys, fill_value=None):
    return [[obj.get(key, fill_value)
             if is_dict_like(obj)
             else obj
             for obj in objects]
            for key in keys]


def _fast_dataset(variables, coord_variables):
    """Create a dataset as quickly as possible.

    Variables are modified *inplace*.
    """
    from .dataset import Dataset
    variables.update(coord_variables)
    coord_names = set(coord_variables)
    return Dataset._from_vars_and_coord_names(variables, coord_names)


def apply_dataset_ufunc(func, *args, **kwargs):
    """
    def apply_dataset_ufunc(func, args, signature=None, join='inner',
                            fill_value=None, new_coords=None):
    """
    from .dataset import Dataset

    signature = kwargs.pop('signature')
    join = kwargs.pop('join', 'inner')
    fill_value = kwargs.pop('fill_value', None)
    new_coords = kwargs.pop('new_coords', None)
    if kwargs:
        raise TypeError('apply_dataarray_ufunc() got unexpected keyword '
                        'arguments: %s' % list(kwargs))

    if signature is None:
        signature = _default_signature(len(args))

    if len(args) > 1:
        args = deep_align(args, join=join, copy=False, raise_on_invalid=False)

    list_of_coords = build_output_coords(args, signature, new_coords)

    list_of_data_vars = [getattr(a, 'data_vars', a) for a in args]
    names = join_dict_keys(list_of_data_vars, how=join)

    list_of_variables = [getattr(a, 'variables', a) for a in args]
    lists_of_args = collect_dict_values(list_of_variables, names, fill_value)

    result_vars = OrderedDict()
    for name, variable_args in zip(names, lists_of_args):
        result_vars[name] = func(*variable_args)

    if signature.n_outputs > 1:
        # we need to unpack result_vars from Dict[object, Tuple[Variable]] ->
        # Tuple[Dict[object, Variable]].
        result_dict_list = [OrderedDict() for _ in range(signature.n_outputs)]
        for name, values in result_vars.items():
            for value, results_dict in zip(values, result_dict_list):
                results_dict[name] = value

        return tuple(_fast_dataset(*args)
                     for args in zip(result_dict_list, list_of_coords))
    else:
        coord_vars, = list_of_coords
        return _fast_dataset(result_vars, coord_vars)


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


def apply_groupby_ufunc(func, *args):
    from .groupby import GroupBy, peek_at

    groupbys = [arg for arg in args if isinstance(arg, GroupBy)]
    if not groupbys:
        raise ValueError('must have at least one groupby to iterate over')
    first_groupby = groupbys[0]
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

    applied = (func(*zipped_args) for zipped_args in zip(*iterators))
    applied_example, applied = peek_at(applied)
    combine = first_groupby._concat
    if isinstance(applied_example, tuple):
        combined = tuple(combine(output) for output in applied)
    else:
        combined = combine(applied)
    return combined


def _calculate_unified_dim_sizes(variables):
    dim_sizes = OrderedDict()

    for var in variables:
        if len(set(var.dims)) < len(var.dims):
            raise ValueError('broadcasting cannot handle duplicate '
                             'dimensions: %r' % list(var.dims))
        for dim, size in zip(var.dims, var.shape):
            if dim not in dim_sizes:
                dim_sizes[dim] = size
            elif dim_sizes[dim] != size:
                raise ValueError('operands cannot be broadcast together '
                                 'with mismatched lengths for dimension '
                                 '%r: %s vs %s'
                                 % (dim, dim_sizes[dim], size))
    return dim_sizes


def broadcast_compat_data(variable, broadcast_dims, core_dims):

    data = variable.data

    old_dims = variable.dims
    new_dims = broadcast_dims + core_dims

    if new_dims == old_dims:
        # optimize for the typical case
        return data

    set_old_dims = set(old_dims)
    missing_core_dims = [d for d in core_dims if d not in set_old_dims]
    if missing_core_dims:
        raise ValueError('operation requires dimensions missing on input '
                         'variable: %r' % missing_core_dims)

    set_new_dims = set(new_dims)
    unexpected_dims = [d for d in old_dims if d not in set_new_dims]
    if unexpected_dims:
        raise ValueError('operation encountered unexpected dimensions %r '
                         'on input variable: these are core dimensions on '
                         'other input or output variables' % unexpected_dims)

    # for consistency with numpy, keep broadcast dimensions to the left
    old_broadcast_dims = tuple(d for d in broadcast_dims if d in set_old_dims)
    reordered_dims = old_broadcast_dims + core_dims
    if reordered_dims != old_dims:
        order = tuple(old_dims.index(d) for d in reordered_dims)
        data = ops.transpose(data, order)

    if new_dims != reordered_dims:
        key = tuple(SLICE_NONE if dim in set_old_dims else None
                    for dim in new_dims)
        data = data[key]

    return data


def apply_variable_ufunc(func, *args, **kwargs):
    """
    def apply_variable_ufunc(func, args, signature=None, dask_array='forbidden',
                             kwargs=None, dtype=None)
    """
    from .variable import Variable

    signature = kwargs.pop('signature')
    dask_array = kwargs.pop('dask_array', 'forbidden')
    kwargs_ = kwargs.pop('kwargs', None)
    dask_dtype = kwargs.pop('dask_dtype', None)
    if kwargs:
        raise TypeError('apply_ufunc() got unexpected keyword arguments: %s'
                        % list(kwargs))

    if dask_array not in {'forbidden', 'allowed', 'auto'}:
        raise ValueError('unknown setting for dask array handling')
    if kwargs_ is None:
        kwargs_ = {}

    dim_sizes = _calculate_unified_dim_sizes([a for a in args
                                              if hasattr(a, 'dims')])
    all_core_dims = (signature.all_input_core_dims
                     | signature.all_output_core_dims)
    broadcast_dims = tuple(d for d in dim_sizes if d not in all_core_dims)
    output_dims = [broadcast_dims + out for out in signature.output_core_dims]

    list_of_input_data = []
    for arg, core_dims in zip(args, signature.input_core_dims):
        if isinstance(arg, Variable):
            data = broadcast_compat_data(arg, broadcast_dims, core_dims)
        else:
            data = arg
        list_of_input_data.append(data)

    contains_dask = any(isinstance(d, dask_array_type)
                        for d in list_of_input_data)

    if dask_array == 'forbidden' and contains_dask:
        raise ValueError('encountered dask array')
    elif dask_array == 'auto' and contains_dask:
        result_data = apply_dask_array(func, *args, signature=signature,
                                       kwargs=kwargs, dtype=dask_dtype)
    else:
        result_data = func(*list_of_input_data, **kwargs_)

    if len(output_dims) > 1:
        output = []
        for dims, data in zip(output_dims, result_data):
            output.append(Variable(dims, data))
        return tuple(output)
    else:
        dims, = output_dims
        data = result_data
        return Variable(dims, data)


def apply_dask_ufunc(func, *args, **kwargs):
    import dask.array as da

    signature = kwargs.pop('signature')
    kwargs_ = kwargs.pop('kwargs', None)
    dtype = kwargs.pop('dtype', None)

    if signature.n_outputs != 1:
        raise ValueError("cannot use dask_array='auto' with functions that "
                         'return multiple values')

    if signature.all_input_core_dims or signature.all_output_core_dims:
        raise ValueError("cannot use dask_array='auto' on unlabeled dask "
                         'arrays with a function signature that uses core '
                         'dimensions')

    f = functools.partial(func, **kwargs_)
    return da.elemwise(f, *args, dtype=dtype)


def apply_ufunc(func, *args, **kwargs):
    """apply_ufunc(func, *args, signature=None, join='inner', new_coords=None,
                   kwargs=None, dask_array='forbidden', dask_dtype=None)

    Apply a broadcasting function for unlabeled arrays to xarray objects.

    The input arguments will be handled using xarray's standard rules for
    labeled computation, including alignment, broadcasting and merging of
    coordinates.

    Parameters
    ----------
    func : callable
        Function to call like ``func(*args, **kwargs)`` on unlabeled arrays.
    *args : Dataset, DataArray, GroupBy, Variable, numpy/dask arrays or scalars
        Mix of labeled and/or unlabeled arrays to which to apply the function.
    signature : string or triply nested sequence, optional
        Object indicating any core dimensions that should not be broadcast on
        the input arguments, new dimensions in the output, and/or multiple
        outputs. Two forms of signatures are accepted:
        (a) A signature string of the form used by NumPy's generalized universal
            functions [1], e.g., '(),(time)->()' indicating a function that
            accepts two arguments and returns a single argument, on which all
            dimensions should be broadcast except 'time' on the second argument.
        (a) A triply nested sequence providing lists of core dimensions for each
            variable, for both input and output, e.g., ([(), ('time',)], [()]).

        Unlike the NumPy gufunc signature spec, the names of all dimensions
        provided in signatures must be the names of actual dimensions on the
        xarray objects.

    [1] http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html
    """
    from .groupby import GroupBy
    from .dataarray import DataArray
    from .dataset import Dataset
    from .variable import Variable

    signature = kwargs.pop('signature', None)
    join = kwargs.pop('join', 'inner')
    new_coords = kwargs.pop('new_coords', None)
    kwargs_ = kwargs.pop('kwargs', None)
    dask_array = kwargs.pop('dask_array', 'forbidden')
    dask_dtype = kwargs.pop('dask_dtype', None)
    if kwargs:
        raise TypeError('apply_ufunc() got unexpected keyword arguments: %s'
                        % list(kwargs))

    if signature is None:
        signature = _Signature.default(len(args))
    elif isinstance(signature, basestring):
        signature = _Signature.from_string(signature)
    elif not isinstance(signature, _Signature):
        signature = _Signature.from_sequence(signature)

    variables_ufunc = functools.partial(
        apply_variable_ufunc, func, signature=signature, dask_array=dask_array,
        kwargs=kwargs_, dask_dtype=dask_dtype)

    if any(isinstance(a, GroupBy) for a in args):
        partial_apply_ufunc = functools.partial(
            apply_ufunc, func, kwargs=kwargs_, signature=signature,
            join=join, dask_array=dask_array, dask_dtype=dask_dtype,
            new_coords=new_coords)
        return apply_groupby_ufunc(partial_apply_ufunc, *args)
    elif any(is_dict_like(a) for a in args):
        return apply_dataset_ufunc(variables_ufunc, *args, signature=signature,
                                   join=join, new_coords=new_coords)
    elif any(isinstance(a, DataArray) for a in args):
        return apply_dataarray_ufunc(variables_ufunc, *args, signature=signature,
                                     join=join, new_coords=new_coords)
    elif any(isinstance(a, Variable) for a in args):
        return variables_ufunc(*args)
    elif dask_array == 'auto' and any(
            isinstance(arg, dask_array_type) for arg in args):
        return apply_dask_array(func, *args, signature=signature, kwargs=kwargs,
                                dtype=dask_dtype)
    else:
        return func(*args)
