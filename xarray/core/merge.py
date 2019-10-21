from typing import (
    TYPE_CHECKING,
    AbstractSet,
    Any,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import pandas as pd

from . import dtypes, pdcompat
from .alignment import deep_align
from .utils import Frozen, dict_equiv
from .variable import Variable, as_variable, assert_unique_multiindex_level_names

if TYPE_CHECKING:
    from .coordinates import Coordinates
    from .dataarray import DataArray
    from .dataset import Dataset

    DimsLike = Union[Hashable, Sequence[Hashable]]
    ArrayLike = Any
    VariableLike = Union[
        ArrayLike,
        Tuple[DimsLike, ArrayLike],
        Tuple[DimsLike, ArrayLike, Mapping],
        Tuple[DimsLike, ArrayLike, Mapping, Mapping],
    ]
    XarrayValue = Union[DataArray, Variable, VariableLike]
    DatasetLike = Union[Dataset, Mapping[Hashable, XarrayValue]]
    CoercibleValue = Union[XarrayValue, pd.Series, pd.DataFrame]
    CoercibleMapping = Union[Dataset, Mapping[Hashable, CoercibleValue]]


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


def broadcast_dimension_size(variables: List[Variable],) -> Dict[Hashable, int]:
    """Extract dimension sizes from a dictionary of variables.

    Raises ValueError if any dimensions have different sizes.
    """
    dims: Dict[Hashable, int] = {}
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


def unique_variable(
    name: Hashable,
    variables: List[Variable],
    compat: str = "broadcast_equals",
    equals: bool = None,
) -> Variable:
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
    """
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
            "conflicting values for variable {!r} on objects to be combined. "
            "You can skip this check by specifying compat='override'.".format(name)
        )

    if combine_method:
        for var in variables[1:]:
            out = getattr(out, combine_method)(var)

    return out


def _assert_compat_valid(compat):
    if compat not in _VALID_COMPAT:
        raise ValueError(
            "compat={!r} invalid: must be {}".format(compat, set(_VALID_COMPAT))
        )


MergeElement = Tuple[Variable, Optional[pd.Index]]


def merge_collected(
    grouped: Dict[Hashable, List[MergeElement]],
    prioritized: Mapping[Hashable, MergeElement] = None,
    compat: str = "minimal",
) -> Tuple[Dict[Hashable, Variable], Dict[Hashable, pd.Index]]:
    """Merge dicts of variables, while resolving conflicts appropriately.

    Parameters
    ----------

        Type of equality check to use when checking for conflicts.

    Returns
    -------
    Dict with keys taken by the union of keys on list_of_mappings,
    and Variable values corresponding to those that should be found on the
    merged result.
    """
    if prioritized is None:
        prioritized = {}

    _assert_compat_valid(compat)

    merged_vars: Dict[Hashable, Variable] = {}
    merged_indexes: Dict[Hashable, pd.Index] = {}

    for name, elements_list in grouped.items():
        if name in prioritized:
            variable, index = prioritized[name]
            merged_vars[name] = variable
            if index is not None:
                merged_indexes[name] = index
        else:
            indexed_elements = [
                (variable, index)
                for variable, index in elements_list
                if index is not None
            ]

            if indexed_elements:
                # TODO(shoyer): consider adjusting this logic. Are we really
                # OK throwing away variable without an index in favor of
                # indexed variables, without even checking if values match?
                variable, index = indexed_elements[0]
                for _, other_index in indexed_elements[1:]:
                    if not index.equals(other_index):
                        raise MergeError(
                            "conflicting values for index %r on objects to be "
                            "combined:\nfirst value: %r\nsecond value: %r"
                            % (name, index, other_index)
                        )
                if compat == "identical":
                    for other_variable, _ in indexed_elements[1:]:
                        if not dict_equiv(variable.attrs, other_variable.attrs):
                            raise MergeError(
                                "conflicting attribute values on combined "
                                "variable %r:\nfirst value: %r\nsecond value: %r"
                                % (name, variable.attrs, other_variable.attrs)
                            )
                merged_vars[name] = variable
                merged_indexes[name] = index
            else:
                variables = [variable for variable, _ in elements_list]
                try:
                    merged_vars[name] = unique_variable(name, variables, compat)
                except MergeError:
                    if compat != "minimal":
                        # we need more than "minimal" compatibility (for which
                        # we drop conflicting coordinates)
                        raise

    return merged_vars, merged_indexes


def collect_variables_and_indexes(
    list_of_mappings: "List[DatasetLike]",
) -> Dict[Hashable, List[MergeElement]]:
    """Collect variables and indexes from list of mappings of xarray objects.

    Mappings must either be Dataset objects, or have values of one of the
    following types:
    - an xarray.Variable
    - a tuple `(dims, data[, attrs[, encoding]])` that can be converted in
      an xarray.Variable
    - or an xarray.DataArray
    """
    from .dataarray import DataArray
    from .dataset import Dataset

    grouped: Dict[Hashable, List[Tuple[Variable, pd.Index]]] = {}

    def append(name, variable, index):
        values = grouped.setdefault(name, [])
        values.append((variable, index))

    def append_all(variables, indexes):
        for name, variable in variables.items():
            append(name, variable, indexes.get(name))

    for mapping in list_of_mappings:
        if isinstance(mapping, Dataset):
            append_all(mapping.variables, mapping.indexes)
            continue

        for name, variable in mapping.items():
            if isinstance(variable, DataArray):
                coords = variable._coords.copy()  # use private API for speed
                indexes = dict(variable.indexes)
                # explicitly overwritten variables should take precedence
                coords.pop(name, None)
                indexes.pop(name, None)
                append_all(coords, indexes)

            variable = as_variable(variable, name=name)
            if variable.dims == (name,):
                variable = variable.to_index_variable()
                index = variable.to_index()
            else:
                index = None
            append(name, variable, index)

    return grouped


def collect_from_coordinates(
    list_of_coords: "List[Coordinates]"
) -> Dict[Hashable, List[MergeElement]]:
    """Collect variables and indexes to be merged from Coordinate objects."""
    grouped: Dict[Hashable, List[Tuple[Variable, pd.Index]]] = {}

    for coords in list_of_coords:
        variables = coords.variables
        indexes = coords.indexes
        for name, variable in variables.items():
            value = grouped.setdefault(name, [])
            value.append((variable, indexes.get(name)))
    return grouped


def merge_coordinates_without_align(
    objects: "List[Coordinates]",
    prioritized: Mapping[Hashable, MergeElement] = None,
    exclude_dims: AbstractSet = frozenset(),
) -> Tuple[Dict[Hashable, Variable], Dict[Hashable, pd.Index]]:
    """Merge variables/indexes from coordinates without automatic alignments.

    This function is used for merging coordinate from pre-existing xarray
    objects.
    """
    collected = collect_from_coordinates(objects)

    if exclude_dims:
        filtered: Dict[Hashable, List[MergeElement]] = {}
        for name, elements in collected.items():
            new_elements = [
                (variable, index)
                for variable, index in elements
                if exclude_dims.isdisjoint(variable.dims)
            ]
            if new_elements:
                filtered[name] = new_elements
    else:
        filtered = collected

    return merge_collected(filtered, prioritized)


def determine_coords(
    list_of_mappings: Iterable["DatasetLike"]
) -> Tuple[Set[Hashable], Set[Hashable]]:
    """Given a list of dicts with xarray object values, identify coordinates.

    Parameters
    ----------
    list_of_mappings : list of dict or Dataset objects
        Of the same form as the arguments to expand_variable_dicts.

    Returns
    -------
    coord_names : set of variable names
    noncoord_names : set of variable names
        All variable found in the input should appear in either the set of
        coordinate or non-coordinate names.
    """
    from .dataarray import DataArray
    from .dataset import Dataset

    coord_names: Set[Hashable] = set()
    noncoord_names: Set[Hashable] = set()

    for mapping in list_of_mappings:
        if isinstance(mapping, Dataset):
            coord_names.update(mapping.coords)
            noncoord_names.update(mapping.data_vars)
        else:
            for name, var in mapping.items():
                if isinstance(var, DataArray):
                    coords = set(var._coords)  # use private API for speed
                    # explicitly overwritten variables should take precedence
                    coords.discard(name)
                    coord_names.update(coords)

    return coord_names, noncoord_names


def coerce_pandas_values(objects: Iterable["CoercibleMapping"]) -> List["DatasetLike"]:
    """Convert pandas values found in a list of labeled objects.

    Parameters
    ----------
    objects : list of Dataset or mappings
        The mappings may contain any sort of objects coercible to
        xarray.Variables as keys, including pandas objects.

    Returns
    -------
    List of Dataset or dictionary objects. Any inputs or values in the inputs
    that were pandas objects have been converted into native xarray objects.
    """
    from .dataarray import DataArray
    from .dataset import Dataset

    out = []
    for obj in objects:
        if isinstance(obj, Dataset):
            variables: "DatasetLike" = obj
        else:
            variables = {}
            if isinstance(obj, PANDAS_TYPES):
                obj = dict(obj.iteritems())
            for k, v in obj.items():
                if isinstance(v, PANDAS_TYPES):
                    v = DataArray(v)
                variables[k] = v
        out.append(variables)
    return out


def _get_priority_vars_and_indexes(
    objects: List["DatasetLike"], priority_arg: Optional[int], compat: str = "equals"
) -> Dict[Hashable, MergeElement]:
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
    A dictionary of variables and associated indexes (if any) to prioritize.
    """
    if priority_arg is None:
        return {}

    collected = collect_variables_and_indexes([objects[priority_arg]])
    variables, indexes = merge_collected(collected, compat=compat)
    grouped: Dict[Hashable, MergeElement] = {}
    for name, variable in variables.items():
        grouped[name] = (variable, indexes.get(name))
    return grouped


def merge_coords(
    objects: Iterable["CoercibleMapping"],
    compat: str = "minimal",
    join: str = "outer",
    priority_arg: Optional[int] = None,
    indexes: Optional[Mapping[Hashable, pd.Index]] = None,
    fill_value: object = dtypes.NA,
) -> Tuple[Dict[Hashable, Variable], Dict[Hashable, pd.Index]]:
    """Merge coordinate variables.

    See merge_core below for argument descriptions. This works similarly to
    merge_core, except everything we don't worry about whether variables are
    coordinates or not.
    """
    _assert_compat_valid(compat)
    coerced = coerce_pandas_values(objects)
    aligned = deep_align(
        coerced, join=join, copy=False, indexes=indexes, fill_value=fill_value
    )
    collected = collect_variables_and_indexes(aligned)
    prioritized = _get_priority_vars_and_indexes(aligned, priority_arg, compat=compat)
    variables, out_indexes = merge_collected(collected, prioritized, compat=compat)
    assert_unique_multiindex_level_names(variables)
    return variables, out_indexes


def merge_data_and_coords(data, coords, compat="broadcast_equals", join="outer"):
    """Used in Dataset.__init__."""
    objects = [data, coords]
    explicit_coords = coords.keys()
    indexes = dict(_extract_indexes_from_coords(coords))
    return merge_core(
        objects, compat, join, explicit_coords=explicit_coords, indexes=indexes
    )


def _extract_indexes_from_coords(coords):
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


class _MergeResult(NamedTuple):
    variables: Dict[Hashable, Variable]
    coord_names: Set[Hashable]
    dims: Dict[Hashable, int]
    indexes: Dict[Hashable, pd.Index]


def merge_core(
    objects: Iterable["CoercibleMapping"],
    compat: str = "broadcast_equals",
    join: str = "outer",
    priority_arg: Optional[int] = None,
    explicit_coords: Optional[Sequence] = None,
    indexes: Optional[Mapping[Hashable, pd.Index]] = None,
    fill_value: object = dtypes.NA,
) -> _MergeResult:
    """Core logic for merging labeled objects.

    This is not public API.

    Parameters
    ----------
    objects : list of mappings
        All values must be convertable to labeled arrays.
    compat : {'identical', 'equals', 'broadcast_equals', 'no_conflicts', 'override'}, optional
        Compatibility checks to use when merging variables.
    join : {'outer', 'inner', 'left', 'right'}, optional
        How to combine objects with different indexes.
    priority_arg : integer, optional
        Optional argument in `objects` that takes precedence over the others.
    explicit_coords : set, optional
        An explicit list of variables from `objects` that are coordinates.
    indexes : dict, optional
        Dictionary with values given by pandas.Index objects.
    fill_value : scalar, optional
        Value to use for newly missing values

    Returns
    -------
    variables : dict
        Dictionary of Variable objects.
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

    coerced = coerce_pandas_values(objects)
    aligned = deep_align(
        coerced, join=join, copy=False, indexes=indexes, fill_value=fill_value
    )
    collected = collect_variables_and_indexes(aligned)

    prioritized = _get_priority_vars_and_indexes(aligned, priority_arg, compat=compat)
    variables, out_indexes = merge_collected(collected, prioritized, compat=compat)
    assert_unique_multiindex_level_names(variables)

    dims = calculate_dimensions(variables)

    coord_names, noncoord_names = determine_coords(coerced)
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

    return _MergeResult(variables, coord_names, dims, out_indexes)


def merge(
    objects: Iterable[Union["DataArray", "CoercibleMapping"]],
    compat: str = "no_conflicts",
    join: str = "outer",
    fill_value: object = dtypes.NA,
) -> "Dataset":
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
    >>> import xarray as xr
    >>> x = xr.DataArray(
    ...     [[1.0, 2.0], [3.0, 5.0]],
    ...     dims=("lat", "lon"),
    ...     coords={"lat": [35.0, 40.0], "lon": [100.0, 120.0]},
    ...     name="var1",
    ... )
    >>> y = xr.DataArray(
    ...     [[5.0, 6.0], [7.0, 8.0]],
    ...     dims=("lat", "lon"),
    ...     coords={"lat": [35.0, 42.0], "lon": [100.0, 150.0]},
    ...     name="var2",
    ... )
    >>> z = xr.DataArray(
    ...     [[0.0, 3.0], [4.0, 9.0]],
    ...     dims=("time", "lon"),
    ...     coords={"time": [30.0, 60.0], "lon": [100.0, 150.0]},
    ...     name="var3",
    ... )

    >>> x
    <xarray.DataArray 'var1' (lat: 2, lon: 2)>
    array([[1., 2.],
           [3., 5.]])
    Coordinates:
    * lat      (lat) float64 35.0 40.0
    * lon      (lon) float64 100.0 120.0

    >>> y
    <xarray.DataArray 'var2' (lat: 2, lon: 2)>
    array([[5., 6.],
           [7., 8.]])
    Coordinates:
    * lat      (lat) float64 35.0 42.0
    * lon      (lon) float64 100.0 150.0

    >>> z
    <xarray.DataArray 'var3' (time: 2, lon: 2)>
    array([[0., 3.],
           [4., 9.]])
    Coordinates:
    * time     (time) float64 30.0 60.0
    * lon      (lon) float64 100.0 150.0

    >>> xr.merge([x, y, z])
    <xarray.Dataset>
    Dimensions:  (lat: 3, lon: 3, time: 2)
    Coordinates:
    * lat      (lat) float64 35.0 40.0 42.0
    * lon      (lon) float64 100.0 120.0 150.0
    * time     (time) float64 30.0 60.0
    Data variables:
        var1     (lat, lon) float64 1.0 2.0 nan 3.0 5.0 nan nan nan nan
        var2     (lat, lon) float64 5.0 nan 6.0 nan nan nan 7.0 nan 8.0
        var3     (time, lon) float64 0.0 nan 3.0 4.0 nan 9.0

    >>> xr.merge([x, y, z], compat='identical')
    <xarray.Dataset>
    Dimensions:  (lat: 3, lon: 3, time: 2)
    Coordinates:
    * lat      (lat) float64 35.0 40.0 42.0
    * lon      (lon) float64 100.0 120.0 150.0
    * time     (time) float64 30.0 60.0
    Data variables:
        var1     (lat, lon) float64 1.0 2.0 nan 3.0 5.0 nan nan nan nan
        var2     (lat, lon) float64 5.0 nan 6.0 nan nan nan 7.0 nan 8.0
        var3     (time, lon) float64 0.0 nan 3.0 4.0 nan 9.0

    >>> xr.merge([x, y, z], compat='equals')
    <xarray.Dataset>
    Dimensions:  (lat: 3, lon: 3, time: 2)
    Coordinates:
    * lat      (lat) float64 35.0 40.0 42.0
    * lon      (lon) float64 100.0 120.0 150.0
    * time     (time) float64 30.0 60.0
    Data variables:
        var1     (lat, lon) float64 1.0 2.0 nan 3.0 5.0 nan nan nan nan
        var2     (lat, lon) float64 5.0 nan 6.0 nan nan nan 7.0 nan 8.0
        var3     (time, lon) float64 0.0 nan 3.0 4.0 nan 9.0

    >>> xr.merge([x, y, z], compat='equals', fill_value=-999.)
    <xarray.Dataset>
    Dimensions:  (lat: 3, lon: 3, time: 2)
    Coordinates:
    * lat      (lat) float64 35.0 40.0 42.0
    * lon      (lon) float64 100.0 120.0 150.0
    * time     (time) float64 30.0 60.0
    Data variables:
        var1     (lat, lon) float64 1.0 2.0 -999.0 3.0 ... -999.0 -999.0 -999.0
        var2     (lat, lon) float64 5.0 -999.0 6.0 -999.0 ... -999.0 7.0 -999.0 8.0
        var3     (time, lon) float64 0.0 -999.0 3.0 4.0 -999.0 9.0

    >>> xr.merge([x, y, z], join='override')
    <xarray.Dataset>
    Dimensions:  (lat: 2, lon: 2, time: 2)
    Coordinates:
    * lat      (lat) float64 35.0 40.0
    * lon      (lon) float64 100.0 120.0
    * time     (time) float64 30.0 60.0
    Data variables:
        var1     (lat, lon) float64 1.0 2.0 3.0 5.0
        var2     (lat, lon) float64 5.0 6.0 7.0 8.0
        var3     (time, lon) float64 0.0 3.0 4.0 9.0

    >>> xr.merge([x, y, z], join='inner')
    <xarray.Dataset>
    Dimensions:  (lat: 1, lon: 1, time: 2)
    Coordinates:
    * lat      (lat) float64 35.0
    * lon      (lon) float64 100.0
    * time     (time) float64 30.0 60.0
    Data variables:
        var1     (lat, lon) float64 1.0
        var2     (lat, lon) float64 5.0
        var3     (time, lon) float64 0.0 4.0

    >>> xr.merge([x, y, z], compat='identical', join='inner')
    <xarray.Dataset>
    Dimensions:  (lat: 1, lon: 1, time: 2)
    Coordinates:
    * lat      (lat) float64 35.0
    * lon      (lon) float64 100.0
    * time     (time) float64 30.0 60.0
    Data variables:
        var1     (lat, lon) float64 1.0
        var2     (lat, lon) float64 5.0
        var3     (time, lon) float64 0.0 4.0

    >>> xr.merge([x, y, z], compat='broadcast_equals', join='outer')
    <xarray.Dataset>
    Dimensions:  (lat: 3, lon: 3, time: 2)
    Coordinates:
    * lat      (lat) float64 35.0 40.0 42.0
    * lon      (lon) float64 100.0 120.0 150.0
    * time     (time) float64 30.0 60.0
    Data variables:
        var1     (lat, lon) float64 1.0 2.0 nan 3.0 5.0 nan nan nan nan
        var2     (lat, lon) float64 5.0 nan 6.0 nan nan nan 7.0 nan 8.0
        var3     (time, lon) float64 0.0 nan 3.0 4.0 nan 9.0

    >>> xr.merge([x, y, z], join='exact')
    Traceback (most recent call last):
    ...
    ValueError: indexes along dimension 'lat' are not equal

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

    dict_like_objects = list()
    for obj in objects:
        if not isinstance(obj, (DataArray, Dataset, dict)):
            raise TypeError(
                "objects must be an iterable containing only "
                "Dataset(s), DataArray(s), and dictionaries."
            )

        obj = obj.to_dataset() if isinstance(obj, DataArray) else obj
        dict_like_objects.append(obj)

    merge_result = merge_core(dict_like_objects, compat, join, fill_value=fill_value)
    merged = Dataset._construct_direct(**merge_result._asdict())
    return merged


def dataset_merge_method(
    dataset: "Dataset",
    other: "CoercibleMapping",
    overwrite_vars: Union[Hashable, Iterable[Hashable]],
    compat: str,
    join: str,
    fill_value: Any,
) -> _MergeResult:
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
        other_overwrite: Dict[Hashable, CoercibleValue] = {}
        other_no_overwrite: Dict[Hashable, CoercibleValue] = {}
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
    dataset: "Dataset", other: "CoercibleMapping"
) -> _MergeResult:
    """Guts of the Dataset.update method.

    This drops a duplicated coordinates from `other` if `other` is not an
    `xarray.Dataset`, e.g., if it's a dict with DataArray values (GH2068,
    GH2180).
    """
    from .dataarray import DataArray
    from .dataset import Dataset

    if not isinstance(other, Dataset):
        other = dict(other)
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
