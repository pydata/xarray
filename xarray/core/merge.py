from collections import OrderedDict
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Sequence,
    Set,
    Tuple,
    Union,
)

import pandas as pd

from . import dtypes, pdcompat
from .alignment import deep_align
from .utils import Frozen
from .variable import Variable, as_variable, assert_unique_multiindex_level_names

if TYPE_CHECKING:
    from .dataarray import DataArray
    from .dataset import Dataset

    DatasetLikeValue = Union[
        DataArray, Variable, Tuple[Hashable, Any], Tuple[Sequence[Hashable], Any]
    ]
    DatasetLike = Union[Dataset, Mapping[Hashable, DatasetLikeValue]]
    """Any object type that can be used on the rhs of Dataset.update,
    Dataset.merge, etc.
    """
    MutableDatasetLike = Union[Dataset, MutableMapping[Hashable, DatasetLikeValue]]


PANDAS_TYPES = (pd.Series, pd.DataFrame, pdcompat.Panel)

_VALID_COMPAT = Frozen(
    {
        "identical": 0,
        "equals": 1,
        "broadcast_equals": 2,
        "minimal": 3,
        "no_conflicts": 4,
        "override": 5,
    }
)


def broadcast_dimension_size(variables: List[Variable],) -> "OrderedDict[Any, int]":
    """Extract dimension sizes from a dictionary of variables.

    Raises ValueError if any dimensions have different sizes.
    """
    dims = OrderedDict()  # type: OrderedDict[Any, int]
    for var in variables:
        for dim, size in zip(var.dims, var.shape):
            if dim in dims and size != dims[dim]:
                raise ValueError("index %r not aligned" % dim)
            dims[dim] = size
    return dims


class MergeError(ValueError):
    """Error class for merge failures due to incompatible arguments.
    """

    # inherits from ValueError for backward compatibility
    # TODO: move this to an xarray.exceptions module?


def unique_variable(name, variables, compat="broadcast_equals", equals=None):
    # type: (Any, List[Variable], str, bool) -> Variable
    """Return the unique variable from a list of variables or raise MergeError.

    Parameters
    ----------
    name : hashable
        Name for this variable.
    variables : list of xarray.Variable
        List of Variable objects, all of which go by the same name in different
        inputs.
    compat : {'identical', 'equals', 'broadcast_equals', 'no_conflicts', 'override'}, optional
        Type of equality check to use.
    equals: None or bool,
        corresponding to result of compat test

    Returns
    -------
    Variable to use in the result.

    Raises
    ------
    MergeError: if any of the variables are not equal.
    """  # noqa
    out = variables[0]

    if len(variables) == 1 or compat == "override":
        return out

    combine_method = None

    if compat == "minimal":
        compat = "broadcast_equals"

    if compat == "broadcast_equals":
        dim_lengths = broadcast_dimension_size(variables)
        out = out.set_dims(dim_lengths)

    if compat == "no_conflicts":
        combine_method = "fillna"

    if equals is None:
        out = out.compute()
        for var in variables[1:]:
            equals = getattr(out, compat)(var)
            if not equals:
                break

    if not equals:
        raise MergeError(
            "conflicting values for variable %r on objects to be combined. You can skip this check by specifying compat='override'."
            % (name)
        )

    if combine_method:
        for var in variables[1:]:
            out = getattr(out, combine_method)(var)

    return out


def _assert_compat_valid(compat):
    if compat not in _VALID_COMPAT:
        raise ValueError("compat=%r invalid: must be %s" % (compat, set(_VALID_COMPAT)))


class OrderedDefaultDict(OrderedDict):
    # minimal version of an ordered defaultdict
    # beware: does not pickle or copy properly
    def __init__(self, default_factory):
        self.default_factory = default_factory
        super().__init__()

    def __missing__(self, key):
        self[key] = default = self.default_factory()
        return default


def merge_variables(
    list_of_variables_dicts: List[Mapping[Any, Variable]],
    priority_vars: Mapping[Any, Variable] = None,
    compat: str = "minimal",
) -> "OrderedDict[Any, Variable]":
    """Merge dicts of variables, while resolving conflicts appropriately.

    Parameters
    ----------
    lists_of_variables_dicts : list of mappings with Variable values
        List of mappings for which each value is a xarray.Variable object.
    priority_vars : mapping with Variable or None values, optional
        If provided, variables are always taken from this dict in preference to
        the input variable dictionaries, without checking for conflicts.
    compat : {'identical', 'equals', 'broadcast_equals', 'minimal', 'no_conflicts', 'override'}, optional
        Type of equality check to use when checking for conflicts.

    Returns
    -------
    OrderedDict with keys taken by the union of keys on list_of_variable_dicts,
    and Variable values corresponding to those that should be found on the
    merged result.
    """  # noqa
    if priority_vars is None:
        priority_vars = {}

    _assert_compat_valid(compat)
    dim_compat = min(compat, "equals", key=_VALID_COMPAT.get)

    lookup = OrderedDefaultDict(list)
    for variables in list_of_variables_dicts:
        for name, var in variables.items():
            lookup[name].append(var)

    # n.b. it's important to fill up merged in the original order in which
    # variables appear
    merged = OrderedDict()  # type: OrderedDict[Any, Variable]

    for name, var_list in lookup.items():
        if name in priority_vars:
            # one of these arguments (e.g., the first for in-place arithmetic
            # or the second for Dataset.update) takes priority
            merged[name] = priority_vars[name]
        else:
            dim_variables = [var for var in var_list if (name,) == var.dims]
            if dim_variables:
                # if there are dimension coordinates, these must be equal (or
                # identical), and they take priority over non-dimension
                # coordinates
                merged[name] = unique_variable(name, dim_variables, dim_compat)
            else:
                try:
                    merged[name] = unique_variable(name, var_list, compat)
                except MergeError:
                    if compat != "minimal":
                        # we need more than "minimal" compatibility (for which
                        # we drop conflicting coordinates)
                        raise

    return merged


def expand_variable_dicts(
    list_of_variable_dicts: "List[Union[Dataset, OrderedDict]]",
) -> "List[Mapping[Any, Variable]]":
    """Given a list of dicts with xarray object values, expand the values.

    Parameters
    ----------
    list_of_variable_dicts : list of dict or Dataset objects
        Each value for the mappings must be of the following types:
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
    from .dataarray import DataArray  # noqa: F811
    from .dataset import Dataset

    var_dicts = []

    for variables in list_of_variable_dicts:
        if isinstance(variables, Dataset):
            var_dicts.append(variables.variables)
            continue

        # append coords to var_dicts before appending sanitized_vars,
        # because we want coords to appear first
        sanitized_vars = OrderedDict()  # type: OrderedDict[Any, Variable]

        for name, var in variables.items():
            if isinstance(var, DataArray):
                # use private API for speed
                coords = var._coords.copy()
                # explicitly overwritten variables should take precedence
                coords.pop(name, None)
                var_dicts.append(coords)

            var = as_variable(var, name=name)
            sanitized_vars[name] = var

        var_dicts.append(sanitized_vars)

    return var_dicts


def determine_coords(
    list_of_variable_dicts: Iterable["DatasetLike"]
) -> Tuple[Set[Hashable], Set[Hashable]]:
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
    from .dataarray import DataArray  # noqa: F811
    from .dataset import Dataset

    coord_names = set()  # type: set
    noncoord_names = set()  # type: set

    for variables in list_of_variable_dicts:
        if isinstance(variables, Dataset):
            coord_names.update(variables.coords)
            noncoord_names.update(variables.data_vars)
        else:
            for name, var in variables.items():
                if isinstance(var, DataArray):
                    coords = set(var._coords)  # use private API for speed
                    # explicitly overwritten variables should take precedence
                    coords.discard(name)
                    coord_names.update(coords)

    return coord_names, noncoord_names


def coerce_pandas_values(objects: Iterable["DatasetLike"]) -> List["DatasetLike"]:
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
    from .dataarray import DataArray  # noqa: F811
    from .dataset import Dataset

    out = []
    for obj in objects:
        if isinstance(obj, Dataset):
            variables = obj  # type: DatasetLike
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


def merge_coords_for_inplace_math(objs, priority_vars=None):
    """Merge coordinate variables without worrying about alignment.

    This function is used for merging variables in coordinates.py.
    """
    expanded = expand_variable_dicts(objs)
    variables = merge_variables(expanded, priority_vars)
    assert_unique_multiindex_level_names(variables)
    return variables


def _get_priority_vars(objects, priority_arg, compat="equals"):
    """Extract the priority variable from a list of mappings.

    We need this method because in some cases the priority argument itself
    might have conflicting values (e.g., if it is a dict with two DataArray
    values with conflicting coordinate values).

    Parameters
    ----------
    objects : list of dictionaries of variables
        Dictionaries in which to find the priority variables.
    priority_arg : int or None
        Integer object whose variable should take priority.
    compat : {'identical', 'equals', 'broadcast_equals', 'no_conflicts'}, optional
        Compatibility checks to use when merging variables.

    Returns
    -------
    None, if priority_arg is None, or an OrderedDict with Variable objects as
    values indicating priority variables.
    """  # noqa
    if priority_arg is None:
        priority_vars = {}
    else:
        expanded = expand_variable_dicts([objects[priority_arg]])
        priority_vars = merge_variables(expanded, compat=compat)
    return priority_vars


def expand_and_merge_variables(objs, priority_arg=None):
    """Merge coordinate variables without worrying about alignment.

    This function is used for merging variables in computation.py.
    """
    expanded = expand_variable_dicts(objs)
    priority_vars = _get_priority_vars(objs, priority_arg)
    variables = merge_variables(expanded, priority_vars)
    return variables


def merge_coords(
    objs,
    compat="minimal",
    join="outer",
    priority_arg=None,
    indexes=None,
    fill_value=dtypes.NA,
):
    """Merge coordinate variables.

    See merge_core below for argument descriptions. This works similarly to
    merge_core, except everything we don't worry about whether variables are
    coordinates or not.
    """
    _assert_compat_valid(compat)
    coerced = coerce_pandas_values(objs)
    aligned = deep_align(
        coerced, join=join, copy=False, indexes=indexes, fill_value=fill_value
    )
    expanded = expand_variable_dicts(aligned)
    priority_vars = _get_priority_vars(aligned, priority_arg, compat=compat)
    variables = merge_variables(expanded, priority_vars, compat=compat)
    assert_unique_multiindex_level_names(variables)

    return variables


def merge_data_and_coords(data, coords, compat="broadcast_equals", join="outer"):
    """Used in Dataset.__init__."""
    objs = [data, coords]
    explicit_coords = coords.keys()
    indexes = dict(extract_indexes(coords))
    return merge_core(
        objs, compat, join, explicit_coords=explicit_coords, indexes=indexes
    )


def extract_indexes(coords):
    """Yields the name & index of valid indexes from a mapping of coords"""
    for name, variable in coords.items():
        variable = as_variable(variable, name=name)
        if variable.dims == (name,):
            yield name, variable.to_index()


def assert_valid_explicit_coords(variables, dims, explicit_coords):
    """Validate explicit coordinate names/dims.

    Raise a MergeError if an explicit coord shares a name with a dimension
    but is comprised of arbitrary dimensions.
    """
    for coord_name in explicit_coords:
        if coord_name in dims and variables[coord_name].dims != (coord_name,):
            raise MergeError(
                "coordinate %s shares a name with a dataset dimension, but is "
                "not a 1D variable along that dimension. This is disallowed "
                "by the xarray data model." % coord_name
            )


def merge_core(
    objs,
    compat="broadcast_equals",
    join="outer",
    priority_arg=None,
    explicit_coords=None,
    indexes=None,
    fill_value=dtypes.NA,
) -> Tuple["OrderedDict[Hashable, Variable]", Set[Hashable], Dict[Hashable, int]]:
    """Core logic for merging labeled objects.

    This is not public API.

    Parameters
    ----------
    objs : list of mappings
        All values must be convertable to labeled arrays.
    compat : {'identical', 'equals', 'broadcast_equals', 'no_conflicts', 'override'}, optional
        Compatibility checks to use when merging variables.
    join : {'outer', 'inner', 'left', 'right'}, optional
        How to combine objects with different indexes.
    priority_arg : integer, optional
        Optional argument in `objs` that takes precedence over the others.
    explicit_coords : set, optional
        An explicit list of variables from `objs` that are coordinates.
    indexes : dict, optional
        Dictionary with values given by pandas.Index objects.
    fill_value : scalar, optional
        Value to use for newly missing values

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
    """  # noqa
    from .dataset import calculate_dimensions

    _assert_compat_valid(compat)

    coerced = coerce_pandas_values(objs)
    aligned = deep_align(
        coerced, join=join, copy=False, indexes=indexes, fill_value=fill_value
    )
    expanded = expand_variable_dicts(aligned)

    coord_names, noncoord_names = determine_coords(coerced)

    priority_vars = _get_priority_vars(aligned, priority_arg, compat=compat)
    variables = merge_variables(expanded, priority_vars, compat=compat)
    assert_unique_multiindex_level_names(variables)

    dims = calculate_dimensions(variables)

    if explicit_coords is not None:
        assert_valid_explicit_coords(variables, dims, explicit_coords)
        coord_names.update(explicit_coords)

    for dim, size in dims.items():
        if dim in variables:
            coord_names.add(dim)

    ambiguous_coords = coord_names.intersection(noncoord_names)
    if ambiguous_coords:
        raise MergeError(
            "unable to determine if these variables should be "
            "coordinates or not in the merged result: %s" % ambiguous_coords
        )

    return variables, coord_names, dims


def merge(objects, compat="no_conflicts", join="outer", fill_value=dtypes.NA):
    """Merge any number of xarray objects into a single Dataset as variables.

    Parameters
    ----------
    objects : Iterable[Union[xarray.Dataset, xarray.DataArray, dict]]
        Merge together all variables from these objects. If any of them are
        DataArray objects, they must have a name.
    compat : {'identical', 'equals', 'broadcast_equals', 'no_conflicts', 'override'}, optional
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
        - 'override': skip comparing and pick variable from first dataset
    join : {'outer', 'inner', 'left', 'right', 'exact'}, optional
        String indicating how to combine differing indexes in objects.

        - 'outer': use the union of object indexes
        - 'inner': use the intersection of object indexes
        - 'left': use indexes from the first object with each dimension
        - 'right': use indexes from the last object with each dimension
        - 'exact': instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - 'override': if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.
    fill_value : scalar, optional
        Value to use for newly missing values

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
    """  # noqa
    from .dataarray import DataArray  # noqa: F811
    from .dataset import Dataset

    dict_like_objects = list()
    for obj in objects:
        if not (isinstance(obj, (DataArray, Dataset, dict))):
            raise TypeError(
                "objects must be an iterable containing only "
                "Dataset(s), DataArray(s), and dictionaries."
            )

        obj = obj.to_dataset() if isinstance(obj, DataArray) else obj
        dict_like_objects.append(obj)

    variables, coord_names, dims = merge_core(
        dict_like_objects, compat, join, fill_value=fill_value
    )
    # TODO: don't always recompute indexes
    merged = Dataset._construct_direct(variables, coord_names, dims, indexes=None)

    return merged


def dataset_merge_method(
    dataset: "Dataset",
    other: "DatasetLike",
    overwrite_vars: Union[Hashable, Iterable[Hashable]],
    compat: str,
    join: str,
    fill_value: Any,
) -> Tuple["OrderedDict[Hashable, Variable]", Set[Hashable], Dict[Hashable, int]]:
    """Guts of the Dataset.merge method.
    """

    # we are locked into supporting overwrite_vars for the Dataset.merge
    # method due for backwards compatibility
    # TODO: consider deprecating it?

    if isinstance(overwrite_vars, Iterable) and not isinstance(overwrite_vars, str):
        overwrite_vars = set(overwrite_vars)
    else:
        overwrite_vars = {overwrite_vars}

    if not overwrite_vars:
        objs = [dataset, other]
        priority_arg = None
    elif overwrite_vars == set(other):
        objs = [dataset, other]
        priority_arg = 1
    else:
        other_overwrite = OrderedDict()  # type: MutableDatasetLike
        other_no_overwrite = OrderedDict()  # type: MutableDatasetLike
        for k, v in other.items():
            if k in overwrite_vars:
                other_overwrite[k] = v
            else:
                other_no_overwrite[k] = v
        objs = [dataset, other_no_overwrite, other_overwrite]
        priority_arg = 2

    return merge_core(
        objs, compat, join, priority_arg=priority_arg, fill_value=fill_value
    )


def dataset_update_method(
    dataset: "Dataset", other: "DatasetLike"
) -> Tuple["OrderedDict[Hashable, Variable]", Set[Hashable], Dict[Hashable, int]]:
    """Guts of the Dataset.update method.

    This drops a duplicated coordinates from `other` if `other` is not an
    `xarray.Dataset`, e.g., if it's a dict with DataArray values (GH2068,
    GH2180).
    """
    from .dataarray import DataArray  # noqa: F811
    from .dataset import Dataset

    if not isinstance(other, Dataset):
        other = OrderedDict(other)
        for key, value in other.items():
            if isinstance(value, DataArray):
                # drop conflicting coordinates
                coord_names = [
                    c
                    for c in value.coords
                    if c not in value.dims and c in dataset.coords
                ]
                if coord_names:
                    other[key] = value.drop(coord_names)

    return merge_core([dataset, other], priority_arg=1, indexes=dataset.indexes)
