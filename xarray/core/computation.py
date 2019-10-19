"""
Functions for applying functions that act on arrays to xarray's labeled data.
"""
import functools
import itertools
import operator
from collections import Counter
from typing import (
    TYPE_CHECKING,
    AbstractSet,
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from . import duck_array_ops, utils
from .alignment import deep_align
from .merge import merge_coordinates_without_align
from .pycompat import dask_array_type
from .utils import is_dict_like
from .variable import Variable

if TYPE_CHECKING:
    from .coordinates import Coordinates  # noqa
    from .dataset import Dataset

_NO_FILL_VALUE = utils.ReprObject("<no-fill-value>")
_DEFAULT_NAME = utils.ReprObject("<default-name>")
_JOINS_WITHOUT_FILL_VALUES = frozenset({"inner", "exact"})


class _UFuncSignature:
    """Core dimensions signature for a given function.

    Based on the signature provided by generalized ufuncs in NumPy.

    Attributes
    ----------
    input_core_dims : tuple[tuple]
        Core dimension names on each input variable.
    output_core_dims : tuple[tuple]
        Core dimension names on each output variable.
    """

    __slots__ = (
        "input_core_dims",
        "output_core_dims",
        "_all_input_core_dims",
        "_all_output_core_dims",
        "_all_core_dims",
    )

    def __init__(self, input_core_dims, output_core_dims=((),)):
        self.input_core_dims = tuple(tuple(a) for a in input_core_dims)
        self.output_core_dims = tuple(tuple(a) for a in output_core_dims)
        self._all_input_core_dims = None
        self._all_output_core_dims = None
        self._all_core_dims = None

    @property
    def all_input_core_dims(self):
        if self._all_input_core_dims is None:
            self._all_input_core_dims = frozenset(
                dim for dims in self.input_core_dims for dim in dims
            )
        return self._all_input_core_dims

    @property
    def all_output_core_dims(self):
        if self._all_output_core_dims is None:
            self._all_output_core_dims = frozenset(
                dim for dims in self.output_core_dims for dim in dims
            )
        return self._all_output_core_dims

    @property
    def all_core_dims(self):
        if self._all_core_dims is None:
            self._all_core_dims = self.all_input_core_dims | self.all_output_core_dims
        return self._all_core_dims

    @property
    def num_inputs(self):
        return len(self.input_core_dims)

    @property
    def num_outputs(self):
        return len(self.output_core_dims)

    def __eq__(self, other):
        try:
            return (
                self.input_core_dims == other.input_core_dims
                and self.output_core_dims == other.output_core_dims
            )
        except AttributeError:
            return False

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return "{}({!r}, {!r})".format(
            type(self).__name__, list(self.input_core_dims), list(self.output_core_dims)
        )

    def __str__(self):
        lhs = ",".join("({})".format(",".join(dims)) for dims in self.input_core_dims)
        rhs = ",".join("({})".format(",".join(dims)) for dims in self.output_core_dims)
        return f"{lhs}->{rhs}"

    def to_gufunc_string(self):
        """Create an equivalent signature string for a NumPy gufunc.

        Unlike __str__, handles dimensions that don't map to Python
        identifiers.
        """
        all_dims = self.all_core_dims
        dims_map = dict(zip(sorted(all_dims), range(len(all_dims))))
        input_core_dims = [
            ["dim%d" % dims_map[dim] for dim in core_dims]
            for core_dims in self.input_core_dims
        ]
        output_core_dims = [
            ["dim%d" % dims_map[dim] for dim in core_dims]
            for core_dims in self.output_core_dims
        ]
        alt_signature = type(self)(input_core_dims, output_core_dims)
        return str(alt_signature)


def result_name(objects: list) -> Any:
    # use the same naming heuristics as pandas:
    # https://github.com/blaze/blaze/issues/458#issuecomment-51936356
    names = {getattr(obj, "name", _DEFAULT_NAME) for obj in objects}
    names.discard(_DEFAULT_NAME)
    if len(names) == 1:
        name, = names
    else:
        name = None
    return name


def _get_coords_list(args) -> List["Coordinates"]:
    coords_list = []
    for arg in args:
        try:
            coords = arg.coords
        except AttributeError:
            pass  # skip this argument
        else:
            coords_list.append(coords)
    return coords_list


def build_output_coords(
    args: list, signature: _UFuncSignature, exclude_dims: AbstractSet = frozenset()
) -> "List[Dict[Any, Variable]]":
    """Build output coordinates for an operation.

    Parameters
    ----------
    args : list
        List of raw operation arguments. Any valid types for xarray operations
        are OK, e.g., scalars, Variable, DataArray, Dataset.
    signature : _UfuncSignature
        Core dimensions signature for the operation.
    exclude_dims : optional set
        Dimensions excluded from the operation. Coordinates along these
        dimensions are dropped.

    Returns
    -------
    Dictionary of Variable objects with merged coordinates.
    """
    coords_list = _get_coords_list(args)

    if len(coords_list) == 1 and not exclude_dims:
        # we can skip the expensive merge
        unpacked_coords, = coords_list
        merged_vars = dict(unpacked_coords.variables)
    else:
        # TODO: save these merged indexes, instead of re-computing them later
        merged_vars, unused_indexes = merge_coordinates_without_align(
            coords_list, exclude_dims=exclude_dims
        )

    output_coords = []
    for output_dims in signature.output_core_dims:
        dropped_dims = signature.all_input_core_dims - set(output_dims)
        if dropped_dims:
            filtered = {
                k: v for k, v in merged_vars.items() if dropped_dims.isdisjoint(v.dims)
            }
        else:
            filtered = merged_vars
        output_coords.append(filtered)

    return output_coords


def apply_dataarray_vfunc(
    func, *args, signature, join="inner", exclude_dims=frozenset(), keep_attrs=False
):
    """Apply a variable level function over DataArray, Variable and/or ndarray
    objects.
    """
    from .dataarray import DataArray

    if len(args) > 1:
        args = deep_align(
            args, join=join, copy=False, exclude=exclude_dims, raise_on_invalid=False
        )

    if keep_attrs and hasattr(args[0], "name"):
        name = args[0].name
    else:
        name = result_name(args)
    result_coords = build_output_coords(args, signature, exclude_dims)

    data_vars = [getattr(a, "variable", a) for a in args]
    result_var = func(*data_vars)

    if signature.num_outputs > 1:
        out = tuple(
            DataArray(variable, coords, name=name, fastpath=True)
            for variable, coords in zip(result_var, result_coords)
        )
    else:
        coords, = result_coords
        out = DataArray(result_var, coords, name=name, fastpath=True)

    return out


def ordered_set_union(all_keys: List[Iterable]) -> Iterable:
    return {key: None for keys in all_keys for key in keys}.keys()


def ordered_set_intersection(all_keys: List[Iterable]) -> Iterable:
    intersection = set(all_keys[0])
    for keys in all_keys[1:]:
        intersection.intersection_update(keys)
    return [key for key in all_keys[0] if key in intersection]


def assert_and_return_exact_match(all_keys):
    first_keys = all_keys[0]
    for keys in all_keys[1:]:
        if keys != first_keys:
            raise ValueError(
                "exact match required for all data variable names, "
                "but %r != %r" % (keys, first_keys)
            )
    return first_keys


_JOINERS = {
    "inner": ordered_set_intersection,
    "outer": ordered_set_union,
    "left": operator.itemgetter(0),
    "right": operator.itemgetter(-1),
    "exact": assert_and_return_exact_match,
}


def join_dict_keys(
    objects: Iterable[Union[Mapping, Any]], how: str = "inner"
) -> Iterable:
    joiner = _JOINERS[how]
    all_keys = [obj.keys() for obj in objects if hasattr(obj, "keys")]
    return joiner(all_keys)


def collect_dict_values(
    objects: Iterable[Union[Mapping, Any]], keys: Iterable, fill_value: object = None
) -> List[list]:
    return [
        [obj.get(key, fill_value) if is_dict_like(obj) else obj for obj in objects]
        for key in keys
    ]


def _as_variables_or_variable(arg):
    try:
        return arg.variables
    except AttributeError:
        try:
            return arg.variable
        except AttributeError:
            return arg


def _unpack_dict_tuples(
    result_vars: Mapping[Hashable, Tuple[Variable, ...]], num_outputs: int
) -> Tuple[Dict[Hashable, Variable], ...]:
    out = tuple({} for _ in range(num_outputs))  # type: ignore
    for name, values in result_vars.items():
        for value, results_dict in zip(values, out):
            results_dict[name] = value
    return out


def apply_dict_of_variables_vfunc(
    func, *args, signature, join="inner", fill_value=None
):
    """Apply a variable level function over dicts of DataArray, DataArray,
    Variable and ndarray objects.
    """
    args = [_as_variables_or_variable(arg) for arg in args]
    names = join_dict_keys(args, how=join)
    grouped_by_name = collect_dict_values(args, names, fill_value)

    result_vars = {}
    for name, variable_args in zip(names, grouped_by_name):
        result_vars[name] = func(*variable_args)

    if signature.num_outputs > 1:
        return _unpack_dict_tuples(result_vars, signature.num_outputs)
    else:
        return result_vars


def _fast_dataset(
    variables: Dict[Hashable, Variable], coord_variables: Mapping[Hashable, Variable]
) -> "Dataset":
    """Create a dataset as quickly as possible.

    Beware: the `variables` dict is modified INPLACE.
    """
    from .dataset import Dataset

    variables.update(coord_variables)
    coord_names = set(coord_variables)
    return Dataset._from_vars_and_coord_names(variables, coord_names)


def apply_dataset_vfunc(
    func,
    *args,
    signature,
    join="inner",
    dataset_join="exact",
    fill_value=_NO_FILL_VALUE,
    exclude_dims=frozenset(),
    keep_attrs=False,
):
    """Apply a variable level function over Dataset, dict of DataArray,
    DataArray, Variable and/or ndarray objects.
    """
    from .dataset import Dataset

    first_obj = args[0]  # we'll copy attrs from this in case keep_attrs=True

    if dataset_join not in _JOINS_WITHOUT_FILL_VALUES and fill_value is _NO_FILL_VALUE:
        raise TypeError(
            "to apply an operation to datasets with different "
            "data variables with apply_ufunc, you must supply the "
            "dataset_fill_value argument."
        )

    if len(args) > 1:
        args = deep_align(
            args, join=join, copy=False, exclude=exclude_dims, raise_on_invalid=False
        )

    list_of_coords = build_output_coords(args, signature, exclude_dims)
    args = [getattr(arg, "data_vars", arg) for arg in args]

    result_vars = apply_dict_of_variables_vfunc(
        func, *args, signature=signature, join=dataset_join, fill_value=fill_value
    )

    if signature.num_outputs > 1:
        out = tuple(_fast_dataset(*args) for args in zip(result_vars, list_of_coords))
    else:
        coord_vars, = list_of_coords
        out = _fast_dataset(result_vars, coord_vars)

    if keep_attrs and isinstance(first_obj, Dataset):
        if isinstance(out, tuple):
            out = tuple(ds._copy_attrs_from(first_obj) for ds in out)
        else:
            out._copy_attrs_from(first_obj)
    return out


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


def apply_groupby_func(func, *args):
    """Apply a dataset or datarray level function over GroupBy, Dataset,
    DataArray, Variable and/or ndarray objects.
    """
    from .groupby import GroupBy, peek_at
    from .variable import Variable

    groupbys = [arg for arg in args if isinstance(arg, GroupBy)]
    assert groupbys, "must have at least one groupby to iterate over"
    first_groupby = groupbys[0]
    if any(not first_groupby._group.equals(gb._group) for gb in groupbys[1:]):
        raise ValueError(
            "apply_ufunc can only perform operations over "
            "multiple GroupBy objets at once if they are all "
            "grouped the same way"
        )

    grouped_dim = first_groupby._group.name
    unique_values = first_groupby._unique_coord.values

    iterators = []
    for arg in args:
        if isinstance(arg, GroupBy):
            iterator = (value for _, value in arg)
        elif hasattr(arg, "dims") and grouped_dim in arg.dims:
            if isinstance(arg, Variable):
                raise ValueError(
                    "groupby operations cannot be performed with "
                    "xarray.Variable objects that share a dimension with "
                    "the grouped dimension"
                )
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


def unified_dim_sizes(
    variables: Iterable[Variable], exclude_dims: AbstractSet = frozenset()
) -> Dict[Hashable, int]:

    dim_sizes: Dict[Hashable, int] = {}

    for var in variables:
        if len(set(var.dims)) < len(var.dims):
            raise ValueError(
                "broadcasting cannot handle duplicate "
                "dimensions on a variable: %r" % list(var.dims)
            )
        for dim, size in zip(var.dims, var.shape):
            if dim not in exclude_dims:
                if dim not in dim_sizes:
                    dim_sizes[dim] = size
                elif dim_sizes[dim] != size:
                    raise ValueError(
                        "operands cannot be broadcast together "
                        "with mismatched lengths for dimension "
                        "%r: %s vs %s" % (dim, dim_sizes[dim], size)
                    )
    return dim_sizes


SLICE_NONE = slice(None)


def broadcast_compat_data(
    variable: Variable,
    broadcast_dims: Tuple[Hashable, ...],
    core_dims: Tuple[Hashable, ...],
) -> Any:
    data = variable.data

    old_dims = variable.dims
    new_dims = broadcast_dims + core_dims

    if new_dims == old_dims:
        # optimize for the typical case
        return data

    set_old_dims = set(old_dims)
    missing_core_dims = [d for d in core_dims if d not in set_old_dims]
    if missing_core_dims:
        raise ValueError(
            "operand to apply_ufunc has required core dimensions {}, but "
            "some of these dimensions are absent on an input variable: {}".format(
                list(core_dims), missing_core_dims
            )
        )

    set_new_dims = set(new_dims)
    unexpected_dims = [d for d in old_dims if d not in set_new_dims]
    if unexpected_dims:
        raise ValueError(
            "operand to apply_ufunc encountered unexpected "
            "dimensions %r on an input variable: these are core "
            "dimensions on other input or output variables" % unexpected_dims
        )

    # for consistency with numpy, keep broadcast dimensions to the left
    old_broadcast_dims = tuple(d for d in broadcast_dims if d in set_old_dims)
    reordered_dims = old_broadcast_dims + core_dims
    if reordered_dims != old_dims:
        order = tuple(old_dims.index(d) for d in reordered_dims)
        data = duck_array_ops.transpose(data, order)

    if new_dims != reordered_dims:
        key_parts = []
        for dim in new_dims:
            if dim in set_old_dims:
                key_parts.append(SLICE_NONE)
            elif key_parts:
                # no need to insert new axes at the beginning that are already
                # handled by broadcasting
                key_parts.append(np.newaxis)
        data = data[tuple(key_parts)]

    return data


def apply_variable_ufunc(
    func,
    *args,
    signature,
    exclude_dims=frozenset(),
    dask="forbidden",
    output_dtypes=None,
    output_sizes=None,
    keep_attrs=False,
):
    """Apply a ndarray level function over Variable and/or ndarray objects.
    """
    from .variable import Variable, as_compatible_data

    dim_sizes = unified_dim_sizes(
        (a for a in args if hasattr(a, "dims")), exclude_dims=exclude_dims
    )
    broadcast_dims = tuple(
        dim for dim in dim_sizes if dim not in signature.all_core_dims
    )
    output_dims = [broadcast_dims + out for out in signature.output_core_dims]

    input_data = [
        broadcast_compat_data(arg, broadcast_dims, core_dims)
        if isinstance(arg, Variable)
        else arg
        for arg, core_dims in zip(args, signature.input_core_dims)
    ]

    if any(isinstance(array, dask_array_type) for array in input_data):
        if dask == "forbidden":
            raise ValueError(
                "apply_ufunc encountered a dask array on an "
                "argument, but handling for dask arrays has not "
                "been enabled. Either set the ``dask`` argument "
                "or load your data into memory first with "
                "``.load()`` or ``.compute()``"
            )
        elif dask == "parallelized":
            input_dims = [broadcast_dims + dims for dims in signature.input_core_dims]
            numpy_func = func

            def func(*arrays):
                return _apply_blockwise(
                    numpy_func,
                    arrays,
                    input_dims,
                    output_dims,
                    signature,
                    output_dtypes,
                    output_sizes,
                )

        elif dask == "allowed":
            pass
        else:
            raise ValueError(
                "unknown setting for dask array handling in "
                "apply_ufunc: {}".format(dask)
            )
    result_data = func(*input_data)

    if signature.num_outputs == 1:
        result_data = (result_data,)
    elif (
        not isinstance(result_data, tuple) or len(result_data) != signature.num_outputs
    ):
        raise ValueError(
            "applied function does not have the number of "
            "outputs specified in the ufunc signature. "
            "Result is not a tuple of {} elements: {!r}".format(
                signature.num_outputs, result_data
            )
        )

    output = []
    for dims, data in zip(output_dims, result_data):
        data = as_compatible_data(data)
        if data.ndim != len(dims):
            raise ValueError(
                "applied function returned data with unexpected "
                "number of dimensions: {} vs {}, for dimensions {}".format(
                    data.ndim, len(dims), dims
                )
            )

        var = Variable(dims, data, fastpath=True)
        for dim, new_size in var.sizes.items():
            if dim in dim_sizes and new_size != dim_sizes[dim]:
                raise ValueError(
                    "size of dimension {!r} on inputs was unexpectedly "
                    "changed by applied function from {} to {}. Only "
                    "dimensions specified in ``exclude_dims`` with "
                    "xarray.apply_ufunc are allowed to change size.".format(
                        dim, dim_sizes[dim], new_size
                    )
                )

        if keep_attrs and isinstance(args[0], Variable):
            var.attrs.update(args[0].attrs)
        output.append(var)

    if signature.num_outputs == 1:
        return output[0]
    else:
        return tuple(output)


def _apply_blockwise(
    func, args, input_dims, output_dims, signature, output_dtypes, output_sizes=None
):
    import dask.array

    if signature.num_outputs > 1:
        raise NotImplementedError(
            "multiple outputs from apply_ufunc not yet "
            "supported with dask='parallelized'"
        )

    if output_dtypes is None:
        raise ValueError(
            "output dtypes (output_dtypes) must be supplied to "
            "apply_func when using dask='parallelized'"
        )
    if not isinstance(output_dtypes, list):
        raise TypeError(
            "output_dtypes must be a list of objects coercible to "
            "numpy dtypes, got {}".format(output_dtypes)
        )
    if len(output_dtypes) != signature.num_outputs:
        raise ValueError(
            "apply_ufunc arguments output_dtypes and "
            "output_core_dims must have the same length: {} vs {}".format(
                len(output_dtypes), signature.num_outputs
            )
        )
    (dtype,) = output_dtypes

    if output_sizes is None:
        output_sizes = {}

    new_dims = signature.all_output_core_dims - signature.all_input_core_dims
    if any(dim not in output_sizes for dim in new_dims):
        raise ValueError(
            "when using dask='parallelized' with apply_ufunc, "
            "output core dimensions not found on inputs must "
            "have explicitly set sizes with ``output_sizes``: {}".format(new_dims)
        )

    for n, (data, core_dims) in enumerate(zip(args, signature.input_core_dims)):
        if isinstance(data, dask_array_type):
            # core dimensions cannot span multiple chunks
            for axis, dim in enumerate(core_dims, start=-len(core_dims)):
                if len(data.chunks[axis]) != 1:
                    raise ValueError(
                        "dimension {!r} on {}th function argument to "
                        "apply_ufunc with dask='parallelized' consists of "
                        "multiple chunks, but is also a core dimension. To "
                        "fix, rechunk into a single dask array chunk along "
                        "this dimension, i.e., ``.chunk({})``, but beware "
                        "that this may significantly increase memory usage.".format(
                            dim, n, {dim: -1}
                        )
                    )

    (out_ind,) = output_dims

    blockwise_args = []
    for arg, dims in zip(args, input_dims):
        # skip leading dimensions that are implicitly added by broadcasting
        ndim = getattr(arg, "ndim", 0)
        trimmed_dims = dims[-ndim:] if ndim else ()
        blockwise_args.extend([arg, trimmed_dims])

    return dask.array.blockwise(
        func,
        out_ind,
        *blockwise_args,
        dtype=dtype,
        concatenate=True,
        new_axes=output_sizes,
    )


def apply_array_ufunc(func, *args, dask="forbidden"):
    """Apply a ndarray level function over ndarray objects."""
    if any(isinstance(arg, dask_array_type) for arg in args):
        if dask == "forbidden":
            raise ValueError(
                "apply_ufunc encountered a dask array on an "
                "argument, but handling for dask arrays has not "
                "been enabled. Either set the ``dask`` argument "
                "or load your data into memory first with "
                "``.load()`` or ``.compute()``"
            )
        elif dask == "parallelized":
            raise ValueError(
                "cannot use dask='parallelized' for apply_ufunc "
                "unless at least one input is an xarray object"
            )
        elif dask == "allowed":
            pass
        else:
            raise ValueError(f"unknown setting for dask array handling: {dask}")
    return func(*args)


def apply_ufunc(
    func: Callable,
    *args: Any,
    input_core_dims: Sequence[Sequence] = None,
    output_core_dims: Optional[Sequence[Sequence]] = ((),),
    exclude_dims: AbstractSet = frozenset(),
    vectorize: bool = False,
    join: str = "exact",
    dataset_join: str = "exact",
    dataset_fill_value: object = _NO_FILL_VALUE,
    keep_attrs: bool = False,
    kwargs: Mapping = None,
    dask: str = "forbidden",
    output_dtypes: Sequence = None,
    output_sizes: Mapping[Any, int] = None,
) -> Any:
    """Apply a vectorized function for unlabeled arrays on xarray objects.

    The function will be mapped over the data variable(s) of the input
    arguments using xarray's standard rules for labeled computation, including
    alignment, broadcasting, looping over GroupBy/Dataset variables, and
    merging of coordinates.

    Parameters
    ----------
    func : callable
        Function to call like ``func(*args, **kwargs)`` on unlabeled arrays
        (``.data``) that returns an array or tuple of arrays. If multiple
        arguments with non-matching dimensions are supplied, this function is
        expected to vectorize (broadcast) over axes of positional arguments in
        the style of NumPy universal functions [1]_ (if this is not the case,
        set ``vectorize=True``). If this function returns multiple outputs, you
        must set ``output_core_dims`` as well.
    *args : Dataset, DataArray, GroupBy, Variable, numpy/dask arrays or scalars
        Mix of labeled and/or unlabeled arrays to which to apply the function.
    input_core_dims : Sequence[Sequence], optional
        List of the same length as ``args`` giving the list of core dimensions
        on each input argument that should not be broadcast. By default, we
        assume there are no core dimensions on any input arguments.

        For example, ``input_core_dims=[[], ['time']]`` indicates that all
        dimensions on the first argument and all dimensions other than 'time'
        on the second argument should be broadcast.

        Core dimensions are automatically moved to the last axes of input
        variables before applying ``func``, which facilitates using NumPy style
        generalized ufuncs [2]_.
    output_core_dims : List[tuple], optional
        List of the same length as the number of output arguments from
        ``func``, giving the list of core dimensions on each output that were
        not broadcast on the inputs. By default, we assume that ``func``
        outputs exactly one array, with axes corresponding to each broadcast
        dimension.

        Core dimensions are assumed to appear as the last dimensions of each
        output in the provided order.
    exclude_dims : set, optional
        Core dimensions on the inputs to exclude from alignment and
        broadcasting entirely. Any input coordinates along these dimensions
        will be dropped. Each excluded dimension must also appear in
        ``input_core_dims`` for at least one argument. Only dimensions listed
        here are allowed to change size between input and output objects.
    vectorize : bool, optional
        If True, then assume ``func`` only takes arrays defined over core
        dimensions as input and vectorize it automatically with
        :py:func:`numpy.vectorize`. This option exists for convenience, but is
        almost always slower than supplying a pre-vectorized function.
        Using this option requires NumPy version 1.12 or newer.
    join : {'outer', 'inner', 'left', 'right', 'exact'}, optional
        Method for joining the indexes of the passed objects along each
        dimension, and the variables of Dataset objects with mismatched
        data variables:

        - 'outer': use the union of object indexes
        - 'inner': use the intersection of object indexes
        - 'left': use indexes from the first object with each dimension
        - 'right': use indexes from the last object with each dimension
        - 'exact': raise `ValueError` instead of aligning when indexes to be
          aligned are not equal
    dataset_join : {'outer', 'inner', 'left', 'right', 'exact'}, optional
        Method for joining variables of Dataset objects with mismatched
        data variables.

        - 'outer': take variables from both Dataset objects
        - 'inner': take only overlapped variables
        - 'left': take only variables from the first object
        - 'right': take only variables from the last object
        - 'exact': data variables on all Dataset objects must match exactly
    dataset_fill_value : optional
        Value used in place of missing variables on Dataset inputs when the
        datasets do not share the exact same ``data_vars``. Required if
        ``dataset_join not in {'inner', 'exact'}``, otherwise ignored.
    keep_attrs: boolean, Optional
        Whether to copy attributes from the first argument to the output.
    kwargs: dict, optional
        Optional keyword arguments passed directly on to call ``func``.
    dask: 'forbidden', 'allowed' or 'parallelized', optional
        How to handle applying to objects containing lazy data in the form of
        dask arrays:

        - 'forbidden' (default): raise an error if a dask array is encountered.
        - 'allowed': pass dask arrays directly on to ``func``.
        - 'parallelized': automatically parallelize ``func`` if any of the
          inputs are a dask array. If used, the ``output_dtypes`` argument must
          also be provided. Multiple output arguments are not yet supported.
    output_dtypes : list of dtypes, optional
        Optional list of output dtypes. Only used if dask='parallelized'.
    output_sizes : dict, optional
        Optional mapping from dimension names to sizes for outputs. Only used
        if dask='parallelized' and new dimensions (not found on inputs) appear
        on outputs.

    Returns
    -------
    Single value or tuple of Dataset, DataArray, Variable, dask.array.Array or
    numpy.ndarray, the first type on that list to appear on an input.

    Examples
    --------

    Calculate the vector magnitude of two arguments:

    >>> def magnitude(a, b):
    ...     func = lambda x, y: np.sqrt(x ** 2 + y ** 2)
    ...     return xr.apply_ufunc(func, a, b)

    You can now apply ``magnitude()`` to ``xr.DataArray`` and ``xr.Dataset``
    objects, with automatically preserved dimensions and coordinates, e.g.,

    >>> array = xr.DataArray([1, 2, 3], coords=[('x', [0.1, 0.2, 0.3])])
    >>> magnitude(array, -array)
    <xarray.DataArray (x: 3)>
    array([1.414214, 2.828427, 4.242641])
    Coordinates:
      * x        (x) float64 0.1 0.2 0.3

    Plain scalars, numpy arrays and a mix of these with xarray objects is also
    supported:

    >>> magnitude(4, 5)
    5.0
    >>> magnitude(3, np.array([0, 4]))
    array([3., 5.])
    >>> magnitude(array, 0)
    <xarray.DataArray (x: 3)>
    array([1., 2., 3.])
    Coordinates:
      * x        (x) float64 0.1 0.2 0.3

    Other examples of how you could use ``apply_ufunc`` to write functions to
    (very nearly) replicate existing xarray functionality:

    Compute the mean (``.mean``) over one dimension::

        def mean(obj, dim):
            # note: apply always moves core dimensions to the end
            return apply_ufunc(np.mean, obj,
                               input_core_dims=[[dim]],
                               kwargs={'axis': -1})

    Inner product over a specific dimension (like ``xr.dot``)::

        def _inner(x, y):
            result = np.matmul(x[..., np.newaxis, :], y[..., :, np.newaxis])
            return result[..., 0, 0]

        def inner_product(a, b, dim):
            return apply_ufunc(_inner, a, b, input_core_dims=[[dim], [dim]])

    Stack objects along a new dimension (like ``xr.concat``)::

        def stack(objects, dim, new_coord):
            # note: this version does not stack coordinates
            func = lambda *x: np.stack(x, axis=-1)
            result = apply_ufunc(func, *objects,
                                 output_core_dims=[[dim]],
                                 join='outer',
                                 dataset_fill_value=np.nan)
            result[dim] = new_coord
            return result

    If your function is not vectorized but can be applied only to core
    dimensions, you can use ``vectorize=True`` to turn into a vectorized
    function. This wraps :py:func:`numpy.vectorize`, so the operation isn't
    terribly fast. Here we'll use it to calculate the distance between
    empirical samples from two probability distributions, using a scipy
    function that needs to be applied to vectors::

        import scipy.stats

        def earth_mover_distance(first_samples,
                                 second_samples,
                                 dim='ensemble'):
            return apply_ufunc(scipy.stats.wasserstein_distance,
                               first_samples, second_samples,
                               input_core_dims=[[dim], [dim]],
                               vectorize=True)

    Most of NumPy's builtin functions already broadcast their inputs
    appropriately for use in `apply`. You may find helper functions such as
    numpy.broadcast_arrays helpful in writing your function. `apply_ufunc` also
    works well with numba's vectorize and guvectorize. Further explanation with
    examples are provided in the xarray documentation [3].

    See also
    --------
    numpy.broadcast_arrays
    numba.vectorize
    numba.guvectorize

    References
    ----------
    .. [1] http://docs.scipy.org/doc/numpy/reference/ufuncs.html
    .. [2] http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html
    .. [3] http://xarray.pydata.org/en/stable/computation.html#wrapping-custom-computation
    """
    from .groupby import GroupBy
    from .dataarray import DataArray
    from .variable import Variable

    if input_core_dims is None:
        input_core_dims = ((),) * (len(args))
    elif len(input_core_dims) != len(args):
        raise ValueError(
            "input_core_dims must be None or a tuple with the length same to "
            "the number of arguments. Given input_core_dims: {}, "
            "number of args: {}.".format(input_core_dims, len(args))
        )

    if kwargs is None:
        kwargs = {}

    signature = _UFuncSignature(input_core_dims, output_core_dims)

    if exclude_dims and not exclude_dims <= signature.all_core_dims:
        raise ValueError(
            "each dimension in `exclude_dims` must also be a "
            "core dimension in the function signature"
        )

    if kwargs:
        func = functools.partial(func, **kwargs)

    if vectorize:
        if signature.all_core_dims:
            func = np.vectorize(
                func, otypes=output_dtypes, signature=signature.to_gufunc_string()
            )
        else:
            func = np.vectorize(func, otypes=output_dtypes)

    variables_vfunc = functools.partial(
        apply_variable_ufunc,
        func,
        signature=signature,
        exclude_dims=exclude_dims,
        keep_attrs=keep_attrs,
        dask=dask,
        output_dtypes=output_dtypes,
        output_sizes=output_sizes,
    )

    if any(isinstance(a, GroupBy) for a in args):
        this_apply = functools.partial(
            apply_ufunc,
            func,
            input_core_dims=input_core_dims,
            output_core_dims=output_core_dims,
            exclude_dims=exclude_dims,
            join=join,
            dataset_join=dataset_join,
            dataset_fill_value=dataset_fill_value,
            keep_attrs=keep_attrs,
            dask=dask,
        )
        return apply_groupby_func(this_apply, *args)
    elif any(is_dict_like(a) for a in args):
        return apply_dataset_vfunc(
            variables_vfunc,
            *args,
            signature=signature,
            join=join,
            exclude_dims=exclude_dims,
            dataset_join=dataset_join,
            fill_value=dataset_fill_value,
            keep_attrs=keep_attrs,
        )
    elif any(isinstance(a, DataArray) for a in args):
        return apply_dataarray_vfunc(
            variables_vfunc,
            *args,
            signature=signature,
            join=join,
            exclude_dims=exclude_dims,
            keep_attrs=keep_attrs,
        )
    elif any(isinstance(a, Variable) for a in args):
        return variables_vfunc(*args)
    else:
        return apply_array_ufunc(func, *args, dask=dask)


def dot(*arrays, dims=None, **kwargs):
    """Generalized dot product for xarray objects. Like np.einsum, but
    provides a simpler interface based on array dimensions.

    Parameters
    ----------
    arrays: DataArray (or Variable) objects
        Arrays to compute.
    dims: str or tuple of strings, optional
        Which dimensions to sum over.
        If not speciified, then all the common dimensions are summed over.
    **kwargs: dict
        Additional keyword arguments passed to numpy.einsum or
        dask.array.einsum

    Returns
    -------
    dot: DataArray

    Examples
    --------

    >>> import numpy as np
    >>> import xarray as xp
    >>> da_a = xr.DataArray(np.arange(3 * 2).reshape(3, 2), dims=['a', 'b'])
    >>> da_b = xr.DataArray(np.arange(3 * 2 * 2).reshape(3, 2, 2),
    ...                     dims=['a', 'b', 'c'])
    >>> da_c = xr.DataArray(np.arange(2 * 3).reshape(2, 3), dims=['c', 'd'])

    >>> da_a
    <xarray.DataArray (a: 3, b: 2)>
    array([[0, 1],
           [2, 3],
           [4, 5]])
    Dimensions without coordinates: a, b

    >>> da_b
    <xarray.DataArray (a: 3, b: 2, c: 2)>
    array([[[ 0,  1],
            [ 2,  3]],
           [[ 4,  5],
            [ 6,  7]],
           [[ 8,  9],
            [10, 11]]])
    Dimensions without coordinates: a, b, c

    >>> da_c
    <xarray.DataArray (c: 2, d: 3)>
    array([[0, 1, 2],
           [3, 4, 5]])
    Dimensions without coordinates: c, d

    >>> xr.dot(da_a, da_b, dims=['a', 'b'])
    <xarray.DataArray (c: 2)>
    array([110, 125])
    Dimensions without coordinates: c

    >>> xr.dot(da_a, da_b, dims=['a'])
    <xarray.DataArray (b: 2, c: 2)>
    array([[40, 46],
           [70, 79]])
    Dimensions without coordinates: b, c

    >>> xr.dot(da_a, da_b, da_c, dims=['b', 'c'])
    <xarray.DataArray (a: 3, d: 3)>
    array([[  9,  14,  19],
           [ 93, 150, 207],
           [273, 446, 619]])
    Dimensions without coordinates: a, d

    """
    from .dataarray import DataArray
    from .variable import Variable

    if any(not isinstance(arr, (Variable, DataArray)) for arr in arrays):
        raise TypeError(
            "Only xr.DataArray and xr.Variable are supported."
            "Given {}.".format([type(arr) for arr in arrays])
        )

    if len(arrays) == 0:
        raise TypeError("At least one array should be given.")

    if isinstance(dims, str):
        dims = (dims,)

    common_dims = set.intersection(*[set(arr.dims) for arr in arrays])
    all_dims = []
    for arr in arrays:
        all_dims += [d for d in arr.dims if d not in all_dims]

    einsum_axes = "abcdefghijklmnopqrstuvwxyz"
    dim_map = {d: einsum_axes[i] for i, d in enumerate(all_dims)}

    if dims is None:
        # find dimensions that occur more than one times
        dim_counts = Counter()
        for arr in arrays:
            dim_counts.update(arr.dims)
        dims = tuple(d for d, c in dim_counts.items() if c > 1)

    dims = tuple(dims)  # make dims a tuple

    # dimensions to be parallelized
    broadcast_dims = tuple(d for d in all_dims if d in common_dims and d not in dims)
    input_core_dims = [
        [d for d in arr.dims if d not in broadcast_dims] for arr in arrays
    ]
    output_core_dims = [tuple(d for d in all_dims if d not in dims + broadcast_dims)]

    # construct einsum subscripts, such as '...abc,...ab->...c'
    # Note: input_core_dims are always moved to the last position
    subscripts_list = [
        "..." + "".join([dim_map[d] for d in ds]) for ds in input_core_dims
    ]
    subscripts = ",".join(subscripts_list)
    subscripts += "->..." + "".join([dim_map[d] for d in output_core_dims[0]])

    # subscripts should be passed to np.einsum as arg, not as kwargs. We need
    # to construct a partial function for apply_ufunc to work.
    func = functools.partial(duck_array_ops.einsum, subscripts, **kwargs)
    result = apply_ufunc(
        func,
        *arrays,
        input_core_dims=input_core_dims,
        output_core_dims=output_core_dims,
        dask="allowed",
    )
    return result.transpose(*[d for d in all_dims if d in result.dims])


def where(cond, x, y):
    """Return elements from `x` or `y` depending on `cond`.

    Performs xarray-like broadcasting across input arguments.

    Parameters
    ----------
    cond : scalar, array, Variable, DataArray or Dataset with boolean dtype
        When True, return values from `x`, otherwise returns values from `y`.
    x, y : scalar, array, Variable, DataArray or Dataset
        Values from which to choose. All dimension coordinates on these objects
        must be aligned with each other and with `cond`.

    Returns
    -------
    In priority order: Dataset, DataArray, Variable or array, whichever
    type appears as an input argument.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> x = xr.DataArray(0.1 * np.arange(10), dims=['lat'],
    ...                  coords={'lat': np.arange(10)}, name='sst')
    >>> x
    <xarray.DataArray 'sst' (lat: 10)>
    array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    Coordinates:
    * lat      (lat) int64 0 1 2 3 4 5 6 7 8 9

    >>> xr.where(x < 0.5, x,  100*x)
    <xarray.DataArray 'sst' (lat: 10)>
    array([ 0. ,  0.1,  0.2,  0.3,  0.4, 50. , 60. , 70. , 80. , 90. ])
    Coordinates:
    * lat      (lat) int64 0 1 2 3 4 5 6 7 8 9

    >>> >>> y = xr.DataArray(
    ...     0.1 * np.arange(9).reshape(3, 3),
    ...     dims=["lat", "lon"],
    ...     coords={"lat": np.arange(3), "lon": 10 + np.arange(3)},
    ...     name="sst",
    ... )
    >>> y
    <xarray.DataArray 'sst' (lat: 3, lon: 3)>
    array([[0. , 0.1, 0.2],
           [0.3, 0.4, 0.5],
           [0.6, 0.7, 0.8]])
    Coordinates:
    * lat      (lat) int64 0 1 2
    * lon      (lon) int64 10 11 12

    >>> xr.where(y.lat < 1, y, -1)
    <xarray.DataArray (lat: 3, lon: 3)>
    array([[ 0. ,  0.1,  0.2],
           [-1. , -1. , -1. ],
           [-1. , -1. , -1. ]])
    Coordinates:
    * lat      (lat) int64 0 1 2
    * lon      (lon) int64 10 11 12

    >>> cond = xr.DataArray([True, False], dims=['x'])
    >>> x = xr.DataArray([1, 2], dims=['y'])
    >>> xr.where(cond, x, 0)
    <xarray.DataArray (x: 2, y: 2)>
    array([[1, 2],
           [0, 0]])
    Dimensions without coordinates: x, y

    See also
    --------
    numpy.where : corresponding numpy function
    Dataset.where, DataArray.where : equivalent methods
    """
    # alignment for three arguments is complicated, so don't support it yet
    return apply_ufunc(
        duck_array_ops.where,
        cond,
        x,
        y,
        join="exact",
        dataset_join="exact",
        dask="allowed",
    )
