"""Functions for converting to and from xarray objects
"""
import numpy as np

from .core.dataarray import DataArray
from .core.pycompat import OrderedDict
from .conventions import (
    maybe_encode_timedelta, maybe_encode_datetime, decode_cf)

cdms2_ignored_attrs = set(['name', 'tileIndex'])
iris_forbidden_keys = set(
    ['standard_name', 'long_name', 'units', 'bounds', 'axis',
     'calendar', 'leap_month', 'leap_year', 'month_lengths',
     'coordinates', 'grid_mapping', 'climatology',
     'cell_methods', 'formula_terms', 'compress',
     'missing_value', 'add_offset', 'scale_factor',
     'valid_max', 'valid_min', 'valid_range', '_FillValue'])
cell_methods_strings = set(['point', 'sum', 'maximum', 'median', 'mid_range',
                            'minimum', 'mean', 'mode', 'standard_deviation',
                            'variance'])


def encode(var):
    return maybe_encode_timedelta(maybe_encode_datetime(var.variable))


def filter_attrs(_attrs, ignored_attrs):
    return dict((k, v) for k, v in _attrs.items() if k not in ignored_attrs)


def from_cdms2(variable):
    """Convert a cdms2 variable into an DataArray
    """
    values = np.asarray(variable)
    name = variable.id
    coords = [(v.id, np.asarray(v),
               filter_attrs(v.attributes, cdms2_ignored_attrs))
              for v in variable.getAxisList()]
    attrs = filter_attrs(variable.attributes, cdms2_ignored_attrs)
    dataarray = DataArray(values, coords=coords, name=name, attrs=attrs)
    return decode_cf(dataarray.to_dataset())[dataarray.name]


def to_cdms2(dataarray):
    """Convert a DataArray into a cdms2 variable
    """
    # we don't want cdms2 to be a hard dependency
    import cdms2

    def set_cdms2_attrs(_var, attrs):
        for k, v in attrs.items():
            setattr(_var, k, v)

    axes = []
    for dim in dataarray.dims:
        coord = encode(dataarray.coords[dim])
        axis = cdms2.createAxis(coord.values, id=dim)
        set_cdms2_attrs(axis, coord.attrs)
        axes.append(axis)

    var = encode(dataarray)
    cdms2_var = cdms2.createVariable(var.values, axes=axes, id=dataarray.name)
    set_cdms2_attrs(cdms2_var, var.attrs)
    return cdms2_var


# TODO: Add converting bounds from xarray to Iris and back
def to_iris(dataarray):
    """Convert a DataArray into a Iris Cube
    """
    # Iris not a hard dependency
    import iris
    # iris.unit is deprecated in Iris v1.9
    import cf_units

    def check_attrs(attrs, keys):
        return dict((k, v) for k, v in attrs.items() if k in keys)

    def get_args(attrs):
        _args = {'attributes': filter_attrs(attrs, iris_forbidden_keys)}
        _args.update(check_attrs(attrs, ('standard_name', 'long_name',)))
        _unit_args = check_attrs(coord.attrs, ('calendar',))
        if 'units' in attrs:
            _args['units'] = cf_units.Unit(attrs['units'], **_unit_args)
        return _args

    def get_cell_methods(cell_methods_str):
        """Converts string to iris cell method objects"""
        cell_methods = []
        _cell_method_words = [w.strip() for w in cell_methods_str.split(':')]
        cm = {'coords': [], 'method': '', 'interval': [], 'comment': []}
        skip = False
        for i, word in enumerate(_cell_method_words):
            # If this value is a comment or an interval don't read
            if skip:
                skip = False
                continue
            # If this word is an axis
            if word not in cell_methods_strings | set(['interval', 'comment']):
                # If we already have a method this must be the next cell_method
                if cm['method']:
                    cell_methods.append(
                        iris.coords.CellMethod(cm['method'],
                                               coords=cm['coords'],
                                               intervals=cm['interval'],
                                               comments=cm['comment']))
                    cm = {'coords': [], 'method': '', 'interval': [],
                          'comment': []}
                    cm['coords'].append(word)
                    continue
                else:
                    cm['coords'].append(word)
            elif word in ['interval', 'comment']:
                cm[word].append(_cell_method_words[i + 1])
                skip = True
                continue
            else:
                cm['method'] = word
        else:
            cell_methods.append(
                iris.coords.CellMethod(cm['method'], coords=cm['coords'],
                                       intervals=cm['interval'],
                                       comments=cm['comment']))
        return cell_methods

    dim_coords = []
    aux_coords = []

    for coord_name in dataarray.coords:
        coord = encode(dataarray.coords[coord_name])
        coord_args = get_args(coord.attrs)
        coord_args['var_name'] = coord_name
        axis = None
        if coord.dims:
            axis = dataarray.get_axis_num(coord.dims)
        if coord_name in dataarray.dims:
            iris_coord = iris.coords.DimCoord(coord.values, **coord_args)
            dim_coords.append((iris_coord, axis))
        else:
            iris_coord = iris.coords.AuxCoord(coord.values, **coord_args)
            aux_coords.append((iris_coord, axis))

    args = get_args(dataarray.attrs)
    args['var_name'] = dataarray.name
    args['dim_coords_and_dims'] = dim_coords
    args['aux_coords_and_dims'] = aux_coords
    if 'cell_methods' in dataarray.attrs:
        args['cell_methods'] = get_cell_methods(dataarray.attrs['cell_methods'])

    cube = iris.cube.Cube(dataarray.to_masked_array(), **args)
    return cube


def from_iris(cube):
    """Convert a Iris cube into an DataArray
    """

    def get_attr(_obj):
        attrs = {'standard_name': _obj.standard_name,
                 'long_name': _obj.long_name}
        if _obj.units.calendar:
            attrs['calendar'] = _obj.units.calendar
        if _obj.units.origin != '1':
            attrs['units'] = _obj.units.origin
        attrs.update(_obj.attributes)
        return dict((k, v) for k, v in attrs.items() if v is not None)

    def get_cell_methods(cell_methods_obj):
        _cell_methods = []
        for cell_method in cell_methods_obj:
            names = ''.join(['{}: '.format(n) for n in cell_method.coord_names])
            intervals = ' '.join(['interval: {}'.format(interval)
                                  for interval in cell_method.intervals])
            comments = ' '.join(['comment: {}'.format(comment)
                                 for comment in cell_method.comments])
            extra = ' '.join([intervals, comments]).strip()
            if extra:
                extra += ' '
            _cell_methods.append(names + cell_method.method + extra)
        return ' '.join(_cell_methods)

    name = cube.var_name
    dims = [dim.var_name for dim in cube.dim_coords]
    if not dims:
        dims = ["dim{}".format(i) for i in range(cube.data.ndim)]
    coords = OrderedDict()

    for coord in cube.coords():
        coord_attrs = get_attr(coord)
        coord_dims = [dims[i] for i in cube.coord_dims(coord)]
        if coord_dims:
            coords[coord.var_name] = (coord_dims, coord.points, coord_attrs)
        else:
            coords[coord.var_name] = ((),
                                      np.asscalar(coord.points), coord_attrs)

    array_attrs = get_attr(cube)
    cell_methods = get_cell_methods(cube.cell_methods)
    if cell_methods:
        array_attrs['cell_methods'] = cell_methods
    dataarray = DataArray(cube.data, coords=coords, name=name,
                          attrs=array_attrs, dims=dims)
    return decode_cf(dataarray.to_dataset())[dataarray.name]
