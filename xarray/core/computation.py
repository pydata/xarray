"""Functions for applying functions that act on arrays to xarray's labeled data.

NOT PUBLIC API.
"""
import collections
import functools
import itertools
import operator
import re

from . import ops
from .alignment import deep_align
from .merge import expand_and_merge_variables
from .pycompat import OrderedDict, basestring, dask_array_type
from .utils import is_dict_like


_DEFAULT_FROZEN_SET = frozenset()

# see http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html
DIMENSION_NAME = r'\w+'
CORE_DIMENSION_LIST = '(?:' + DIMENSION_NAME + '(?:,' + DIMENSION_NAME + ')*)?'
ARGUMENT = r'\(' + CORE_DIMENSION_LIST + r'\)'
ARGUMENT_LIST = ARGUMENT + '(?:,' + ARGUMENT + ')*'
SIGNATURE = '^' + ARGUMENT_LIST + '->' + ARGUMENT_LIST + '$'


def safe_tuple(x):
    # type: Iterable -> tuple
    if isinstance(x, basestring):
        raise ValueError('cannot safely convert %r to a tuple')
    return tuple(x)


class UFuncSignature(object):
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
        self._all_core_dims = None

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
    def all_core_dims(self):
        if self._all_core_dims is None:
            self._all_core_dims = (self.all_input_core_dims |
                                   self.all_output_core_dims)
        return self._all_core_dims

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
        """Create a UFuncSignature object from a NumPy gufunc signature.

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
        return not self == other

    def __repr__(self):
        return ('%s(%r, %r)'
                % (type(self).__name__,
                   list(self.input_core_dims),
                   list(self.output_core_dims)))


def result_name(objects):
    # type: List[object] -> Any
    # use the same naming heuristics as pandas:
    # https://github.com/blaze/blaze/issues/458#issuecomment-51936356
    names = {getattr(obj, 'name', None) for obj in objects}
    names.discard(None)
    if len(names) == 1:
        name, = names
    else:
        name = None
    return name


_REPEAT_NONE = itertools.repeat(None)


def _get_coord_variables(args):
    input_coords = []
    for arg in args:
        try:
            coords = arg.coords
        except AttributeError:
            pass  # skip this argument
        else:
            coord_vars = getattr(coords, 'variables', coords)
            input_coords.append(coord_vars)
    return input_coords


def build_output_coords(
        args,                      # type: list
        signature,                 # type: UFuncSignature
        exclude_dims=frozenset(),  # type: set
):
    # type: (...) -> List[OrderedDict[Any, Variable]]
    input_coords = _get_coord_variables(args)

    if exclude_dims:
        input_coords = [OrderedDict((k, v) for k, v in coord_vars.items()
                                    if exclude_dims.isdisjoint(v.dims))
                        for coord_vars in input_coords]

    if len(input_coords) == 1:
        # we can skip the expensive merge
        unpacked_input_coords, = input_coords
        merged = OrderedDict(unpacked_input_coords)
    else:
        merged = expand_and_merge_variables(input_coords)

    output_coords = []
    for output_dims in signature.output_core_dims:
        dropped_dims = signature.all_input_core_dims - set(output_dims)
        if dropped_dims:
            filtered = OrderedDict((k, v) for k, v in merged.items()
                                   if dropped_dims.isdisjoint(v.dims))
        else:
            filtered = merged
        output_coords.append(filtered)

    return output_coords


def apply_dataarray_ufunc(func, *args, **kwargs):
    """apply_dataarray_ufunc(func, *args, signature, join='inner',
                             exclude_dims=frozenset())
    """
    from .dataarray import DataArray

    signature = kwargs.pop('signature')
    join = kwargs.pop('join', 'inner')
    exclude_dims = kwargs.pop('exclude_dims', _DEFAULT_FROZEN_SET)
    if kwargs:
        raise TypeError('apply_dataarray_ufunc() got unexpected keyword '
                        'arguments: %s' % list(kwargs))

    if len(args) > 1:
        args = deep_align(args, join=join, copy=False, exclude=exclude_dims,
                          raise_on_invalid=False)

    name = result_name(args)
    result_coords = build_output_coords(args, signature, exclude_dims)

    data_vars = [getattr(a, 'variable', a) for a in args]
    result_var = func(*data_vars)

    if signature.n_outputs > 1:
        return tuple(DataArray(variable, coords, name=name, fastpath=True)
                     for variable, coords in zip(result_var, result_coords))
    else:
        coords, = result_coords
        return DataArray(result_var, coords, name=name, fastpath=True)


def ordered_set_union(all_keys):
    # type: List[Iterable] -> Iterable
    result_dict = OrderedDict()
    for keys in all_keys:
        for key in keys:
            result_dict[key] = None
    return result_dict.keys()


def ordered_set_intersection(all_keys):
    # type: List[Iterable] -> Iterable
    intersection = set(all_keys[0])
    for keys in all_keys[1:]:
        intersection.intersection_update(keys)
    return [key for key in all_keys[0] if key in intersection]


_JOINERS = {
    'inner': ordered_set_intersection,
    'outer': ordered_set_union,
    'left': operator.itemgetter(0),
    'right': operator.itemgetter(-1),
}


def join_dict_keys(objects, how='inner'):
    # type: (Iterable[Union[Mapping, Any]], str) -> Iterable
    joiner = _JOINERS[how]
    all_keys = [obj.keys() for obj in objects if hasattr(obj, 'keys')]
    return joiner(all_keys)


def collect_dict_values(objects, keys, fill_value=None):
    # type: (Iterable[Union[Mapping, Any]], Iterable, Any) -> List[list]
    return [[obj.get(key, fill_value)
             if is_dict_like(obj)
             else obj
             for obj in objects]
            for key in keys]


def _as_variables_or_variable(arg):
    try:
        return arg.variables
    except AttributeError:
        try:
            return arg.variable
        except AttributeError:
            return arg


def _unpack_dict_tuples(
        result_vars,  # type: Mapping[Any, Tuple[Variable]]
        n_outputs,    # type: int
):
    # type: (...) -> Tuple[Dict[Any, Variable]]
    out = tuple(OrderedDict() for _ in range(n_outputs))
    for name, values in result_vars.items():
        for value, results_dict in zip(values, out):
            results_dict[name] = value
    return out


def apply_dict_of_variables_ufunc(func, *args, **kwargs):
    """apply_dict_of_variables_ufunc(func, *args, signature, join='inner',
                                     fill_value=None):
    """
    signature = kwargs.pop('signature')
    join = kwargs.pop('join', 'inner')
    fill_value = kwargs.pop('fill_value', None)
    if kwargs:
        raise TypeError('apply_dict_of_variables_ufunc() got unexpected '
                        'keyword arguments: %s' % list(kwargs))

    args = [_as_variables_or_variable(arg) for arg in args]
    names = join_dict_keys(args, how=join)
    grouped_by_name = collect_dict_values(args, names, fill_value)

    result_vars = OrderedDict()
    for name, variable_args in zip(names, grouped_by_name):
        result_vars[name] = func(*variable_args)

    if signature.n_outputs > 1:
        return _unpack_dict_tuples(result_vars, signature.n_outputs)
    else:
        return result_vars


def _fast_dataset(variables, coord_variables):
    # type: (OrderedDict[Any, Variable], Mapping[Any, Variable]) -> Dataset
    """Create a dataset as quickly as possible.

    Beware: the `variables` OrderedDict is modified INPLACE.
    """
    from .dataset import Dataset
    variables.update(coord_variables)
    coord_names = set(coord_variables)
    return Dataset._from_vars_and_coord_names(variables, coord_names)


def apply_dataset_ufunc(func, *args, **kwargs):
    """apply_dataset_ufunc(func, *args, signature, join='inner',
                           fill_value=None, exclude_dims=frozenset()):
    """
    signature = kwargs.pop('signature')
    join = kwargs.pop('join', 'inner')
    fill_value = kwargs.pop('fill_value', None)
    exclude_dims = kwargs.pop('exclude_dims', _DEFAULT_FROZEN_SET)
    if kwargs:
        raise TypeError('apply_dataset_ufunc() got unexpected keyword '
                        'arguments: %s' % list(kwargs))

    if len(args) > 1:
        args = deep_align(args, join=join, copy=False, exclude=exclude_dims,
                          raise_on_invalid=False)

    list_of_coords = build_output_coords(args, signature, exclude_dims)

    args = [getattr(arg, 'data_vars', arg) for arg in args]
    result_vars = apply_dict_of_variables_ufunc(
        func, *args, signature=signature, join=join, fill_value=fill_value)

    if signature.n_outputs > 1:
        return tuple(_fast_dataset(*args)
                     for args in zip(result_vars, list_of_coords))
    else:
        coord_vars, = list_of_coords
        return _fast_dataset(result_vars, coord_vars)


def _iter_over_selections(obj, dim, values):
    """Iterate over selections of an xarray object in the provided order."""
    from .groupby import _dummy_copy

    dummy = None
    for value in values:
        try:
            obj_sel = obj.sel(**{dim: value})
        except (KeyError, IndexError):
            if dummy is None:
                dummy = _dummy_copy(obj)
            obj_sel = dummy
        yield obj_sel


def apply_groupby_ufunc(func, *args):
    from .groupby import GroupBy, peek_at
    from .variable import Variable

    groupbys = [arg for arg in args if isinstance(arg, GroupBy)]
    if not groupbys:
        raise ValueError('must have at least one groupby to iterate over')
    first_groupby = groupbys[0]
    if any(not first_groupby._group.equals(gb._group) for gb in groupbys[1:]):
        raise ValueError('can only perform operations over multiple groupbys '
                         'at once if they are all grouped the same way')

    grouped_dim = first_groupby._group.name
    unique_values = first_groupby._unique_coord.values

    iterators = []
    for arg in args:
        if isinstance(arg, GroupBy):
            iterator = (value for _, value in arg)
        elif hasattr(arg, 'dims') and grouped_dim in arg.dims:
            if isinstance(arg, Variable):
                raise ValueError(
                    'groupby operations cannot be performed with '
                    'xarray.Variable objects that share a dimension with '
                    'the grouped dimension')
            iterator = _iter_over_selections(arg, grouped_dim, unique_values)
        else:
            iterator = itertools.repeat(arg)
        iterators.append(iterator)

    applied = (func(*zipped_args) for zipped_args in zip(*iterators))
    applied_example, applied = peek_at(applied)
    combine = first_groupby._combine
    if isinstance(applied_example, tuple):
        combined = tuple(combine(output) for output in zip(*applied))
    else:
        combined = combine(applied)
    return combined


def unified_dim_sizes(variables, exclude_dims=frozenset()):
    # type: Iterable[Variable] -> OrderedDict[Any, int]
    dim_sizes = OrderedDict()

    for var in variables:
        if len(set(var.dims)) < len(var.dims):
            raise ValueError('broadcasting cannot handle duplicate '
                             'dimensions: %r' % list(var.dims))
        for dim, size in zip(var.dims, var.shape):
            if dim not in exclude_dims:
                if dim not in dim_sizes:
                    dim_sizes[dim] = size
                elif dim_sizes[dim] != size:
                    raise ValueError('operands cannot be broadcast together '
                                     'with mismatched lengths for dimension '
                                     '%r: %s vs %s'
                                     % (dim, dim_sizes[dim], size))
    return dim_sizes


SLICE_NONE = slice(None)

# A = TypeVar('A', numpy.ndarray, dask.array.Array)


def broadcast_compat_data(variable, broadcast_dims, core_dims):
    # type: (Variable[A], tuple, tuple) -> A
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
    """apply_variable_ufunc(func, *args, signature, exclude_dims=frozenset())
    """
    from .variable import Variable

    signature = kwargs.pop('signature')
    exclude_dims = kwargs.pop('exclude_dims', _DEFAULT_FROZEN_SET)
    if kwargs:
        raise TypeError('apply_variable_ufunc() got unexpected keyword '
                        'arguments: %s' % list(kwargs))

    dim_sizes = unified_dim_sizes((a for a in args if hasattr(a, 'dims')),
                                  exclude_dims=exclude_dims)
    broadcast_dims = tuple(dim for dim in dim_sizes
                           if dim not in signature.all_core_dims)
    output_dims = [broadcast_dims + out for out in signature.output_core_dims]

    input_data = [broadcast_compat_data(arg, broadcast_dims, core_dims)
                  if isinstance(arg, Variable)
                  else arg
                  for arg, core_dims in zip(args, signature.input_core_dims)]

    result_data = func(*input_data)

    if signature.n_outputs > 1:
        output = []
        for dims, data in zip(output_dims, result_data):
            output.append(Variable(dims, data))
        return tuple(output)
    else:
        dims, = output_dims
        return Variable(dims, result_data)


def apply_array_ufunc(func, *args, **kwargs):
    """apply_variable_ufunc(func, *args, dask_array='forbidden')
    """
    dask_array = kwargs.pop('dask_array', 'forbidden')
    if kwargs:
        raise TypeError('apply_array_ufunc() got unexpected keyword '
                        'arguments: %s' % list(kwargs))

    if any(isinstance(arg, dask_array_type) for arg in args):
        # TODO: add a mode dask_array='auto' when dask.array gets a function
        # for applying arbitrary gufuncs
        if dask_array == 'forbidden':
            raise ValueError('encountered dask array, but did not set '
                             "dask_array='allowed'")
        elif dask_array != 'allowed':
            raise ValueError('unknown setting for dask array handling: %r'
                             % dask_array)
        # fall through
    return func(*args)


def apply_ufunc(func, *args, **kwargs):
    """apply_ufunc(func, *args, signature=None, join='inner',
                   exclude_dims=frozenset(), dataset_fill_value=None,
                   kwargs=None, dask_array='forbidden')

    Apply a vectorized function for unlabeled arrays to xarray objects.

    The input arguments will be handled using xarray's standard rules for
    labeled computation, including alignment, broadcasting, looping over
    GroupBy/Dataset variables, and merging of coordinates.

    Parameters
    ----------
    func : callable
        Function to call like ``func(*args, **kwargs)`` on unlabeled arrays
        (``.data``). If multiple arguments with non-matching dimensions are
        supplied, this function is expected to vectorize (broadcast) over
        axes of positional arguments in the style of NumPy universal
        functions [1]_.
    *args : Dataset, DataArray, GroupBy, Variable, numpy/dask arrays or scalars
        Mix of labeled and/or unlabeled arrays to which to apply the function.
    signature : string or triply nested sequence, optional
        Object indicating core dimensions that should not be broadcast on
        the input and outputs arguments. If omitted, inputs will be broadcast
        to share all dimensions in common before calling ``func`` on their
        values, and the output of ``func`` will be assumed to be a single array
        with the same shape as the inputs.

        Two forms of signatures are accepted:
        (a) A signature string of the form used by NumPy's generalized
            universal functions [2]_, e.g., '(),(time)->()' indicating a
            function that accepts two arguments and returns a single argument,
            on which all dimensions should be broadcast except 'time' on the
            second argument.
        (a) A triply nested sequence providing lists of core dimensions for
            each variable, for both input and output, e.g.,
            ``([(), ('time',)], [()])``.

        Core dimensions are automatically moved to the last axes of any input
        variables, which facilitates using NumPy style generalized ufuncs (see
        the examples below).

        Unlike the NumPy gufunc signature spec, the names of all dimensions
        provided in signatures must be the names of actual dimensions on the
        xarray objects.
    join : {'outer', 'inner', 'left', 'right'}, optional
        Method for joining the indexes of the passed objects along each
        dimension, and the variables of Dataset objects with mismatched
        data variables:
        - 'outer': use the union of object indexes
        - 'inner': use the intersection of object indexes
        - 'left': use indexes from the first object with each dimension
        - 'right': use indexes from the last object with each dimension
    exclude_dims : set, optional
        Dimensions to exclude from alignment and broadcasting. Any inputs
        coordinates along these dimensions will be dropped. Each excluded
        dimension must be a core dimension in the function signature.
    dataset_fill_value : optional
        Value used in place of missing variables on Dataset inputs when the
        datasets do not share the exact same ``data_vars``. Only relevant if
        ``join != 'inner'``.
    kwargs: dict, optional
        Optional keyword arguments passed directly on to call ``func``.
    dask_array: 'forbidden' or 'allowed', optional
        Whether or not to allow applying the ufunc to objects containing lazy
        data in the form of dask arrays. By default, this is forbidden, to
        avoid implicitly converting lazy data.

    Returns
    -------
    Single value or tuple of Dataset, DataArray, Variable, dask.array.Array or
    numpy.ndarray, the first type on that list to appear on an input.

    Examples
    --------
    For illustrative purposes only, here are examples of how you could use
    ``apply_ufunc`` to write functions to (very nearly) replicate existing
    xarray functionality:

    Calculate the vector magnitude of two arguments:

        def magnitude(a, b):
            func = lambda x, y: np.sqrt(x ** 2 + y ** 2)
            return xr.apply_func(func, a, b)

    Compute the mean (``.mean``)::

        def mean(obj, dim):
            # note: apply always moves core dimensions to the end
            sig = ([(dim,)], [()])
            kwargs = {'axis': -1}
            return apply_ufunc(np.mean, obj, signature=sig, kwargs=kwargs)

    Inner product over a specific dimension::

        def _inner(x, y):
            result = np.matmul(x[..., np.newaxis, :], y[..., :, np.newaxis])
            return result[..., 0, 0]

        def inner_product(a, b, dim):
            sig = ([(dim,), (dim,)], [()])
            return apply_ufunc(_inner, a, b, signature=sig)

    Stack objects along a new dimension (like ``xr.concat``)::

        def stack(objects, dim, new_coord):
            sig = ([()] * len(objects), [(dim,)])
            func = lambda *x: np.stack(x, axis=-1)
            result = apply_ufunc(func, *objects, signature=sig,
                                 join='outer', dataset_fill_value=np.nan)
            result[dim] = new_coord
            return result

    Most of NumPy's builtin functions already broadcast their inputs
    appropriately for use in `apply`. You may find helper functions such as
    numpy.broadcast_arrays or numpy.vectorize helpful in writing your function.
    `apply_ufunc` also works well with numba's vectorize and guvectorize.

    See also
    --------
    numpy.broadcast_arrays
    numpy.vectorize
    numba.vectorize
    numba.guvectorize

    References
    ----------
    .. [1] http://docs.scipy.org/doc/numpy/reference/ufuncs.html
    .. [2] http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html
    """
    from .groupby import GroupBy
    from .dataarray import DataArray
    from .variable import Variable

    signature = kwargs.pop('signature', None)
    join = kwargs.pop('join', 'inner')
    exclude_dims = kwargs.pop('exclude_dims', frozenset())
    dataset_fill_value = kwargs.pop('dataset_fill_value', None)
    kwargs_ = kwargs.pop('kwargs', None)
    dask_array = kwargs.pop('dask_array', 'forbidden')
    if kwargs:
        raise TypeError('apply_ufunc() got unexpected keyword arguments: %s'
                        % list(kwargs))

    if signature is None:
        signature = UFuncSignature.default(len(args))
    elif isinstance(signature, basestring):
        signature = UFuncSignature.from_string(signature)
    elif not isinstance(signature, UFuncSignature):
        signature = UFuncSignature.from_sequence(signature)

    if exclude_dims and not exclude_dims <= signature.all_core_dims:
        raise ValueError('each dimension in `exclude_dims` must also be a '
                         'core dimension in the function signature')

    if kwargs_:
        func = functools.partial(func, **kwargs_)

    array_ufunc = functools.partial(
        apply_array_ufunc, func, dask_array=dask_array)

    variables_ufunc = functools.partial(
        apply_variable_ufunc, array_ufunc, signature=signature,
        exclude_dims=exclude_dims)

    if any(isinstance(a, GroupBy) for a in args):
        this_apply = functools.partial(
            apply_ufunc, func, signature=signature, join=join,
            dask_array=dask_array, exclude_dims=exclude_dims,
            dataset_fill_value=dataset_fill_value)
        return apply_groupby_ufunc(this_apply, *args)
    elif any(is_dict_like(a) for a in args):
        return apply_dataset_ufunc(variables_ufunc, *args, signature=signature,
                                   join=join, exclude_dims=exclude_dims,
                                   fill_value=dataset_fill_value)
    elif any(isinstance(a, DataArray) for a in args):
        return apply_dataarray_ufunc(variables_ufunc, *args,
                                     signature=signature,
                                     join=join, exclude_dims=exclude_dims)
    elif any(isinstance(a, Variable) for a in args):
        return variables_ufunc(*args)
    else:
        return array_ufunc(*args)
