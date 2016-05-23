from collections import defaultdict, deque

import pandas as pd

from .alignment import align, deep_align
from .utils import ChainMap, Frozen
from .variable import as_variable, Coordinate, Variable, default_index_coordinate
from .pycompat import (basestring, iteritems, OrderedDict)


def broadcast_dimension_size(variables):
    # type: (List[Variable],) -> Variable
    """Extract common dimension sizes from a dictionary of variables.
    """
    dims = OrderedDict()
    for var in variables:
        for dim, size in zip(var.dims, var.shape):
            if dim in dims and size != dims[dim]:
                raise ValueError('index %r not aligned' % dim)
            dims[dim] = size
    return dims


class MergeError(ValueError):
    """Error class for merge failures due to incompatible arguments.
    """
    # inherits from ValueError for backward compatibility
    # TODO: move this to an xarray.exceptions module?


def unique_variable(name, variables, compat='broadcast_equals'):
    # type: (Any, List[Variable], str) -> Variable
    """Return the unique variable from a list of variables or raise MergeError.
    """
    out = variables[0]
    if len(variables) > 1:
        if compat == 'minimal':
            compat = 'broadcast_equals'

        if compat == 'broadcast_equals':
            dim_lengths = broadcast_dimension_size(variables)
            out = out.expand_dims(dim_lengths)

        for var in variables[1:]:
            if not getattr(out, compat)(var):
                raise MergeError('conflicting value for variable %s:\n'
                                 'first value: %r\nsecond value: %r'
                                 % (name, out, var))
    return out


_VALID_COMPAT = Frozen({'identical': 0,
                        'equals': 1,
                        'broadcast_equals': 2,
                        'minimal': 3})


class OrderedDefaultDict(OrderedDict):
    # minimal version of an ordered defaultdict
    # beware: does not pickle or copy properly
    def __init__(self, default_factory):
        self.default_factory = default_factory
        super(OrderedDefaultDict, self).__init__()

    def __missing__(self, key):
        self[key] = default = self.default_factory()
        return default


def get_top_priority_variable(priority_arg=None):
    if priority_arg is not None:
        # one of these arguments (e.g., the first for in-place
        # arithmetic or the second for Dataset.update) takes priority
        priority_variables = list_of_variables_dicts[priority_arg]
    else:
        priority_variables = {}



def merge_variables(
        list_of_variables_dicts,  # type: List[Mapping[Any, Variable]]
        priority_vars=None,       # type: Optional[int]
        compat='minimal',         # type: str
        ):
    # type: (...) -> Dict[Any, Variable]
    """Merge dicts of variables, while resolving conflicts appropriately.
    """
    if priority_vars is None:
        # one of these arguments (e.g., the first for in-place
        # arithmetic or the second for Dataset.update) takes priority
        priority_vars = {}

    assert compat in _VALID_COMPAT
    dim_compat = min(compat, 'equals', key=_VALID_COMPAT.get)

    lookup = OrderedDefaultDict(list)
    for variables in list_of_variables_dicts:
        for name, var in variables.items():
            lookup[name].append(var)

    # n.b. it's important to fill up merged in the original order in which
    # variables appear
    merged = OrderedDict()

    for name, variables in lookup.items():
        if name in priority_vars:
            merged[name] = priority_vars[name]
        else:
            dim_variables = [var for var in variables if (name,) == var.dims]
            if dim_variables:
                # if there are dimension coordinates, these must be equal (or
                # identical), and they take priority over non-dimension
                # coordinates
                merged[name] = unique_variable(name, dim_variables, dim_compat)
            else:
                try:
                    merged[name] = unique_variable(name, variables, compat)
                except MergeError:
                    if compat != 'minimal':
                        # we need more than "minimal" compatibility (for which
                        # we drop conflicting coordinates)
                        raise

    return merged


def expand_variable_dicts(list_of_variable_dicts):
    # type: (List[Dict]) -> List[Dict[Any, Variable]]
    var_dicts = []

    for variables in list_of_variable_dicts:
        if hasattr(variables, 'variables'):  # duck-type Dataset
            sanitized_vars = variables.variables
        else:
            # append sanitized_vars before filling it up because we want coords
            # to appear afterwards
            sanitized_vars = OrderedDict()

            for name, var in variables.items():
                if hasattr(var, '_coords'):  # duck-type DataArray
                    # use private API for speed
                    coords = var._coords.copy()
                    # explicitly overwritten variables should take precedence
                    coords.pop(name, None)
                    var_dicts.append(coords)

                var = as_variable(var, name=name)
                sanitized_vars[name] = var

        var_dicts.append(sanitized_vars)

    return var_dicts


def determine_coords(list_of_variable_dicts):
    # type: (List[Dict]) -> Tuple[Set, Set]
    coord_names = set()
    noncoord_names = set()

    for variables in list_of_variable_dicts:
        if hasattr(variables, 'coords') and hasattr(variables, 'data_vars'):
            # duck-type Dataset
            coord_names.update(variables.coords)
            noncoord_names.update(variables.data_vars)
        else:
            for name, var in variables.items():
                if hasattr(var, '_coords'):  # duck-type DataArray
                    coords = set(var._coords)  # use private API for speed
                    # explicitly overwritten variables should take precedence
                    coords.discard(name)
                    coord_names.update(coords)

    return coord_names, noncoord_names



PANDAS_TYPES = (pd.Series, pd.DataFrame, pd.Panel)


def coerce_pandas_values(objects):
    from .dataset import Dataset
    from .dataarray import DataArray

    out = []
    for obj in objects:
        if isinstance(obj, Dataset):
            variables = obj
        else:
            variables = OrderedDict()
            if isinstance(obj, PANDAS_TYPES):
                obj = OrderedDict(obj.iteritems())
            for k, v in obj.items():
                if isinstance(v, PANDAS_TYPES):
                    v = DataArray(v)
                variables[k] = v
        out.append(variables)
    return out


def merge_coords_only(objs, compat='minimal', join='outer', priority_vars=None):
    if compat not in _VALID_COMPAT:
        raise ValueError("compat=%r invalid: must be %s"
                         % (compat, set(_VALID_COMPAT)))

    expanded = expand_variable_dicts(objs)
    variables = merge_variables(expanded, priority_vars, compat=compat)

    return variables


def _get_priority_vars(objects, priority_arg, compat='equals'):
    if priority_arg is None:
        priority_vars = None
    else:
        expanded = expand_variable_dicts([objects[priority_arg]])
        priority_vars = merge_variables(expanded, compat=compat)
    return priority_vars


def align_and_merge_coords(objs, compat='minimal', join='outer',
                           priority_arg=None, indexes=None):
    if compat not in _VALID_COMPAT:
        raise ValueError("compat=%r invalid: must be %s"
                         % (compat, set(_VALID_COMPAT)))

    coerced = coerce_pandas_values(objs)
    aligned = deep_align(coerced, join=join, copy=False, indexes=indexes)
    expanded = expand_variable_dicts(aligned)
    priority_vars = _get_priority_vars(aligned, priority_arg, compat=compat)
    variables = merge_variables(expanded, priority_vars, compat=compat)

    return variables


def merge_data_and_coords(
        data, coords, compat='broadcast_equals', join='outer'):
    objs = [data, coords]
    explicit_coords = coords.keys()
    return merge_core(objs, compat, join, explicit_coords=explicit_coords)


def merge_core(objs, compat='broadcast_equals', join='outer', priority_arg=None,
               explicit_coords=None, indexes=None):
    from .dataset import calculate_dimensions

    if compat not in _VALID_COMPAT:
        raise ValueError("compat=%r invalid: must be %s"
                         % (compat, set(_VALID_COMPAT)))

    coerced = coerce_pandas_values(objs)
    aligned = deep_align(coerced, join=join, copy=False, indexes=indexes)
    expanded = expand_variable_dicts(aligned)

    coord_names, noncoord_names = determine_coords(coerced)

    if explicit_coords is not None:
        coord_names.update(explicit_coords)

    priority_vars = _get_priority_vars(aligned, priority_arg, compat=compat)
    variables = merge_variables(expanded, priority_vars, compat=compat)

    dims = calculate_dimensions(variables)

    for dim, size in dims.items():
        if dim not in variables:
            variables[dim] = default_index_coordinate(dim, size)

    coord_names.update(dims)

    ambiguous_coords = coord_names.intersection(noncoord_names)
    if ambiguous_coords:
        raise MergeError('unable to determine if these variables should be '
                         'coordinates or not in the merged result: %s'
                         % ambiguous_coords)

    return variables, coord_names, dims


def merge(objects, compat='broadcast_equals', join='outer'):
    """Merge any number of xarray objects into a single Dataset as variables.

    Parameters
    ----------
    objects : Iterable[Union[xarray.Dataset, xarray.DataArray, dict]]
        Merge together all variables from these objects. If any of them are
        DataArray objects, they must have a name.
    compat : 'broadcast_equals', 'equals' or 'identical', optional
        Compatibility checks to use when merging dataset variables.
    join : 'outer', 'inner', 'left' or 'right', optional
        How to combine objects with different indexes.

    Returns
    -------
    Dataset
        Dataset with combined variables from each object.

    Examples
    --------
    >>> arrays = [xr.DataArray(n, name='var%d' % n) for n in range(5)]
    >>> xr.merge(arrays)
    <xarray.Dataset>
    Dimensions:  ()
    Coordinates:
        *empty*
    Data variables:
        var0     int64 0
        var1     int64 1
        var2     int64 2
        var3     int64 3
        var4     int64 4

    Raises
    ------
    xarray.MergeError
        If any variables with the same name have conflicting values.

    See also
    --------
    concat
    """
    from .dataarray import DataArray
    from .dataset import Dataset

    dict_like_objects = [obj.to_dataset() if isinstance(obj, DataArray) else obj
                         for obj in objects]

    variables, coord_names, dims = merge_core(dict_like_objects, compat, join)
    merged = Dataset._construct_direct(variables, coord_names, dims)

    return merged


def dataset_merge_method(dataset, other, overwrite_vars=frozenset(),
                         compat='broadcast_equals', join='outer'):

    # we are locked into supporting overwrite_vars for the Dataset.merge
    # method due for backwards compatibility
    # TODO: consider deprecating it?

    if isinstance(overwrite_vars, basestring):
        overwrite_vars = set([overwrite_vars])
    overwrite_vars = set(overwrite_vars)

    if not overwrite_vars:
        objs = [dataset, other]
        priority_arg = None
    elif overwrite_vars == set(other):
        objs = [dataset, other]
        priority_arg = 1
    else:
        other_overwrite = OrderedDict()
        other_no_overwrite = OrderedDict()
        for k, v in other.items():
            if k in overwrite_vars:
                other_overwrite[k] = v
            else:
                other_no_overwrite[k] = v
        objs = [dataset, other_no_overwrite, other_overwrite]
        priority_arg = 2

    return merge_core(objs, compat, join, priority_arg=priority_arg)


def dataset_update_method(dataset, other):
    objs = [dataset, other]
    priority_arg = 1
    indexes = dataset.indexes
    return merge_core(objs, priority_arg=priority_arg, indexes=indexes)
