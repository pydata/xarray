import pandas as pd

from .alignment import align
from .utils import Frozen, is_dict_like
from .variable import as_variable, default_index_coordinate
from .pycompat import (basestring, OrderedDict)


PANDAS_TYPES = (pd.Series, pd.DataFrame, pd.Panel)

_VALID_COMPAT = Frozen({'identical': 0,
                        'equals': 1,
                        'broadcast_equals': 2,
                        'minimal': 3})


def broadcast_dimension_size(variables):
    # type: (List[Variable],) -> Variable
    """Extract dimension sizes from a dictionary of variables.

    Raises ValueError if any dimensions have different sizes.
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

    Parameters
    ----------
    name : hashable
        Name for this variable.
    variables : list of xarray.Variable
        List of Variable objects, all of which go by the same name in different
        inputs.
    compat : {'identical', 'equals', 'broadcast_equals'}, optional
        Type of equality check to use.

    Returns
    -------
    Variable to use in the result.

    Raises
    ------
    MergeError: if any of the variables are not equal.
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
                raise MergeError('conflicting values for variable %r on '
                                 'objects to be combined:\n'
                                 'first value: %r\nsecond value: %r'
                                 % (name, out, var))
    return out


def _assert_compat_valid(compat):
    if compat not in _VALID_COMPAT:
        raise ValueError("compat=%r invalid: must be %s"
                         % (compat, set(_VALID_COMPAT)))


class OrderedDefaultDict(OrderedDict):
    # minimal version of an ordered defaultdict
    # beware: does not pickle or copy properly
    def __init__(self, default_factory):
        self.default_factory = default_factory
        super(OrderedDefaultDict, self).__init__()

    def __missing__(self, key):
        self[key] = default = self.default_factory()
        return default


def merge_variables(
        list_of_variables_dicts,  # type: List[Mapping[Any, Variable]]
        priority_vars=None,       # type: Optional[Mapping[Any, Variable]]
        compat='minimal',         # type: str
        ):
    # type: (...) -> OrderedDict[Any, Variable]
    """Merge dicts of variables, while resolving conflicts appropriately.

    Parameters
    ----------
    lists_of_variables_dicts : list of mappings with Variable values
        List of mappings for which each value is a xarray.Variable object.
    priority_vars : mapping with Variable values, optional
        If provided, variables are always taken from this dict in preference to
        the input variable dictionaries, without checking for conflicts.
    compat : {'identical', 'equals', 'broadcast_equals', 'minimal'}, optional
        Type of equality check to use wben checking for conflicts.

    Returns
    -------
    OrderedDict keys given by the union of keys on list_of_variable_dicts
    (unless compat=='minimal', in which case some variables with conflicting
    values can be dropped), and Variable values corresponding to those that
    should be found in the result.
    """
    if priority_vars is None:
        # one of these arguments (e.g., the first for in-place
        # arithmetic or the second for Dataset.update) takes priority
        priority_vars = {}

    _assert_compat_valid(compat)
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
    # type: (List[Union[Dataset, Dict]]) -> List[Dict[Any, Variable]]
    """Given a list of dicts with xarray object values, expand the values.

    Parameters
    ----------
    list_of_variable_dicts : list of dict or Dataset objects
        The each value for the mappings must be of the following types:
        - an xarray.Variable
        - a tuple `(dims, data[, attrs[, encoding]])` that can be converted in
          an xarray.Variable
        - or an xarray.DataArray

    Returns
    -------
    A list of ordered dictionaries corresponding to inputs, or coordinates from
    an input's values. The values of each ordered dictionary are all
    xarray.Variable objects.
    """
    var_dicts = []

    for variables in list_of_variable_dicts:
        if hasattr(variables, 'variables'):  # duck-type Dataset
            sanitized_vars = variables.variables
        else:
            # append coords to var_dicts before appending sanitized_vars,
            # because we want coords to appear first
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
    """Given a list of dicts with xarray object values, identify coordinates.

    Parameters
    ----------
    list_of_variable_dicts : list of dict or Dataset objects
        Of the same form as the arguments to expand_variable_dicts.

    Returns
    -------
    coord_names : set of variable names
    noncoord_names : set of variable names
        All variable found in the input should appear in either the set of
        coordinate or non-coordinate names.
    """
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


def coerce_pandas_values(objects):
    """Convert pandas values found in a list of labeled objects.

    Parameters
    ----------
    objects : list of Dataset or mappings
        The mappings may contain any sort of objects coercible to
        xarray.Variables as keys, including pandas objects.

    Returns
    -------
    List of Dataset or OrderedDict objects. Any inputs or values in the inputs
    that were pandas objects have been converted into native xarray objects.
    """
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


def merge_coords_without_align(objs, priority_vars=None):
    """Merge coordinate variables without worrying about alignment.

    This function is used for merging variables in coordinates.py.
    """
    expanded = expand_variable_dicts(objs)
    variables = merge_variables(expanded, priority_vars)
    return variables


def _align_for_merge(input_objects, join, copy, indexes=None):
    """Align objects for merging, recursing into dictionary values.
    """
    if indexes is None:
        indexes = {}

    def is_alignable(obj):
        return hasattr(obj, 'indexes') and hasattr(obj, 'reindex')

    positions = []
    keys = []
    out = []
    targets = []
    no_key = object()
    not_replaced = object()
    for n, variables in enumerate(input_objects):
        if is_alignable(variables):
            positions.append(n)
            keys.append(no_key)
            targets.append(variables)
            out.append(not_replaced)
        else:
            for k, v in variables.items():
                if is_alignable(v) and k not in indexes:
                    # Skip variables in indexes for alignment, because these
                    # should to be overwritten instead:
                    # https://github.com/pydata/xarray/issues/725
                    positions.append(n)
                    keys.append(k)
                    targets.append(v)
            out.append(OrderedDict(variables))

    aligned = align(*targets, join=join, copy=copy, indexes=indexes)

    for position, key, aligned_obj in zip(positions, keys, aligned):
        if key is no_key:
            out[position] = aligned_obj
        else:
            out[position][key] = aligned_obj

    # something went wrong: we should have replaced all sentinel values
    assert all(arg is not not_replaced for arg in out)

    return out


def _get_priority_vars(objects, priority_arg, compat='equals'):
    """Extract the priority variable from a list of mappings.

    We need this method because in some cases the priority argument itself might
    have conflicting values (e.g., if it is a dict with two DataArray values
    with conflicting coordinate values).

    Parameters
    ----------
    objects : list of dictionaries of variables
        Dictionaries in which to find the priority variables.
    priority_arg : int or None
        Integer object whose variable should take priority.
    compat : 'broadcast_equals', 'equals' or 'identical', optional
        Compatibility checks to use when merging variables.

    Returns
    -------
    None, if priority_arg is None, or an OrderedDict with Variable objects as
    values indicating priority variables.
    """
    if priority_arg is None:
        priority_vars = None
    else:
        expanded = expand_variable_dicts([objects[priority_arg]])
        priority_vars = merge_variables(expanded, compat=compat)
    return priority_vars


def merge_coords(objs, compat='minimal', join='outer', priority_arg=None,
                 indexes=None):
    """Merge coordinate variables.

    See merge_core below for argument descriptions. This works similarly to
    merge_core, except everything we don't worry about whether variables are
    coordinates or not.
    """
    _assert_compat_valid(compat)
    coerced = coerce_pandas_values(objs)
    aligned = _align_for_merge(coerced, join=join, copy=False, indexes=indexes)
    expanded = expand_variable_dicts(aligned)
    priority_vars = _get_priority_vars(aligned, priority_arg, compat=compat)
    variables = merge_variables(expanded, priority_vars, compat=compat)

    return variables


def merge_data_and_coords(data, coords, compat='broadcast_equals',
                          join='outer'):
    """Used in Dataset.__init__."""
    objs = [data, coords]
    explicit_coords = coords.keys()
    return merge_core(objs, compat, join, explicit_coords=explicit_coords)


def merge_core(objs, compat='broadcast_equals', join='outer', priority_arg=None,
               explicit_coords=None, indexes=None):
    """Core logic for merging labeled objects.

    This is not public API.

    Parameters
    ----------
    objs : list of mappings
        All values must be convertable to labeled arrays.
    compat : 'broadcast_equals', 'equals' or 'identical', optional
        Compatibility checks to use when merging variables.
    join : 'outer', 'inner', 'left' or 'right', optional
        How to combine objects with different indexes.
    priority_arg : integer, optional
        Optional argument in `objs` that takes precedence over the others.
    explicit_coords : set, optional
        An explicit list of variables from `objs` that are coordinates.
    indexes : dict, optional
        Dictionary with values given by pandas.Index objects.

    Returns
    -------
    variables : OrderedDict
        Ordered dictionary of Variable objects.
    coord_names : set
        Set of coordinate names.
    dims : dict
        Dictionary mapping from dimension names to sizes.

    Raises
    ------
    MergeError if the merge cannot be done successfully.
    """
    from .dataset import calculate_dimensions

    _assert_compat_valid(compat)

    coerced = coerce_pandas_values(objs)
    aligned = _align_for_merge(coerced, join=join, copy=False, indexes=indexes)
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

    return variables, coord_names, dict(dims)


def merge(objects, compat='broadcast_equals', join='outer'):
    """Merge any number of xarray objects into a single Dataset as variables.

    Parameters
    ----------
    objects : Iterable[Union[xarray.Dataset, xarray.DataArray, dict]]
        Merge together all variables from these objects. If any of them are
        DataArray objects, they must have a name.
    compat : 'broadcast_equals', 'equals' or 'identical', optional
        Compatibility checks to use when merging variables.
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
    """Guts of the Dataset.merge method."""

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
    """Guts of the Dataset.update method"""
    return merge_core([dataset, other], priority_arg=1, indexes=dataset.indexes)
