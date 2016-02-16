from .alignment import align, partial_align, align_variables
from .utils import ChainMap
from .variable import as_variable
from .pycompat import (basestring, iteritems, OrderedDict)


def _as_dataset_variable(name, var):
    """Prepare a variable for adding it to a Dataset
    """
    try:
        var = as_variable(var, key=name)
    except TypeError:
        raise TypeError('variables must be given by arrays or a tuple of '
                        'the form (dims, data[, attrs, encoding])')
    if name in var.dims:
        # convert the into an Index
        if var.ndim != 1:
            raise ValueError('the variable %r has the same name as one of its '
                             'dimensions %r, but it is not 1-dimensional and '
                             'thus it is not a valid index' % (name, var.dims))
        var = var.to_coord()
    return var


def expand_variables(raw_variables, old_variables=None, compat='identical'):
    """Expand a dictionary of variables.

    Returns a dictionary of Variable objects suitable for inserting into a
    Dataset._variables dictionary.

    This includes converting tuples (dims, data) into Variable objects,
    converting coordinate variables into Coordinate objects and expanding
    DataArray objects into Variables plus coordinates.

    Raises ValueError if any conflicting values are found, between any of the
    new or old variables.
    """
    if old_variables is None:
        old_variables = {}
    new_variables = OrderedDict()
    new_coord_names = set()
    variables = ChainMap(new_variables, old_variables)

    def maybe_promote_or_replace(name, var):
        existing_var = variables[name]
        if name not in existing_var.dims:
            if name in var.dims:
                variables[name] = var
            else:
                common_dims = OrderedDict(zip(existing_var.dims,
                                              existing_var.shape))
                common_dims.update(zip(var.dims, var.shape))
                variables[name] = existing_var.expand_dims(common_dims)
                new_coord_names.update(var.dims)

    def add_variable(name, var):
        var = _as_dataset_variable(name, var)
        if name not in variables:
            variables[name] = var
            new_coord_names.update(variables[name].dims)
        else:
            if not getattr(variables[name], compat)(var):
                raise ValueError('conflicting value for variable %s:\n'
                                 'first value: %r\nsecond value: %r'
                                 % (name, variables[name], var))
            if compat == 'broadcast_equals':
                maybe_promote_or_replace(name, var)

    for name, var in iteritems(raw_variables):
        if hasattr(var, 'coords'):
            # it's a DataArray
            new_coord_names.update(var.coords)
            for dim, coord in iteritems(var.coords):
                if dim != name:
                    add_variable(dim, coord.variable)
            var = var.variable
        add_variable(name, var)

    return new_variables, new_coord_names


def _merge_expand(variables, other, overwrite_vars, compat):
    possible_conflicts = dict((k, v) for k, v in variables.items()
                              if k not in overwrite_vars)
    new_vars, new_coord_names = expand_variables(other, possible_conflicts, compat)
    replace_vars = variables.copy()
    replace_vars.update(new_vars)
    return replace_vars, new_vars, new_coord_names


def _merge_dataset_with_dataset(self, other, overwrite_vars, compat, join):
    aligned_self, other = align(self, other, join=join, copy=False)

    replace_vars, new_vars, new_coord_names = _merge_expand(
        aligned_self._variables, other._variables, overwrite_vars, compat)
    new_coord_names.update(other._coord_names)

    return replace_vars, new_vars, new_coord_names


def _merge_dataset_with_dict(self, other, overwrite_vars, compat, join):
    other = align_variables(other, join='outer', copy=False)

    alignable = [k for k, v in other.items() if hasattr(v, 'indexes')]
    aligned = partial_align(self, *[other[a] for a in alignable],
                            join=join, copy=False, exclude=overwrite_vars)

    aligned_self = aligned[0]

    other = OrderedDict(other)
    other.update(zip(alignable, aligned[1:]))

    return _merge_expand(aligned_self._variables, other, overwrite_vars, compat)


def merge_datasets(dataset, other, overwrite_vars=set(),
                   compat='broadcast_equals', join='outer'):
    """
    Guts of Dataset.merge
    """
    from .dataset import Dataset

    if compat not in ['broadcast_equals', 'equals', 'identical']:
        raise ValueError("compat=%r invalid: must be 'broadcast_equals', "
                         "'equals' or 'identical'" % compat)

    if isinstance(overwrite_vars, basestring):
        overwrite_vars = [overwrite_vars]
    overwrite_vars = set(overwrite_vars)

    if isinstance(other, Dataset):
        merge_func = _merge_dataset_with_dataset
    else:
        merge_func = _merge_dataset_with_dict

    replace_vars, new_vars, new_coord_names = merge_func(
        dataset, other, overwrite_vars, compat=compat, join=join)

    newly_coords = new_coord_names & set(dataset.data_vars)
    no_longer_coords = set(dataset.coords) & (set(new_vars) - new_coord_names)
    ambiguous_coords = (newly_coords | no_longer_coords) - overwrite_vars
    if ambiguous_coords:
        raise ValueError('cannot merge: the following variables are '
                         'coordinates on one dataset but not the other: %s'
                         % list(ambiguous_coords))

    return replace_vars, new_coord_names


def _reindex_variables_against(variables, indexes, copy=False):
    """Reindex all DataArrays in the provided dict, leaving other values alone.
    """
    alignable = [k for k, v in variables.items() if hasattr(v, 'indexes')]
    aligned = []
    for a in alignable:
        valid_indexes = dict((k, v) for k, v in indexes.items()
                             if k in variables[a].dims and k != a)
        aligned.append(variables[a].reindex(copy=copy, **valid_indexes))
    new_variables = OrderedDict(variables)
    new_variables.update(zip(alignable, aligned))
    return new_variables


def merge_dataarray_coords(indexes, variables, other):
    """
    Return the new dictionary of coordinate variables given by merging in
    ``other`` to to these variables.
    """
    other = align_variables(other, join='outer', copy=False)
    other = _reindex_variables_against(other, indexes, copy=False)
    replace_vars, _, __ = _merge_expand(
        variables, other, other, compat='broadcast_equals')
    return replace_vars
