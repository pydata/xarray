"""Functions for converting to and from xarray objects
"""
from __future__ import absolute_import, division, print_function

import numpy as np

from .coding.times import CFDatetimeCoder, CFTimedeltaCoder
from .conventions import decode_cf
from .core import duck_array_ops
from .core.dataarray import DataArray
from .core.dtypes import get_fill_value
from .core.pycompat import OrderedDict, range

cdms2_ignored_attrs = {'name', 'tileIndex'}
iris_forbidden_keys = {'standard_name', 'long_name', 'units', 'bounds', 'axis',
                       'calendar', 'leap_month', 'leap_year', 'month_lengths',
                       'coordinates', 'grid_mapping', 'climatology',
                       'cell_methods', 'formula_terms', 'compress',
                       'missing_value', 'add_offset', 'scale_factor',
                       'valid_max', 'valid_min', 'valid_range', '_FillValue'}
cell_methods_strings = {'point', 'sum', 'maximum', 'median', 'mid_range',
                        'minimum', 'mean', 'mode', 'standard_deviation',
                        'variance'}


def encode(var):
    return CFTimedeltaCoder().encode(CFDatetimeCoder().encode(var.variable))


def _filter_attrs(attrs, ignored_attrs):
    """ Return attrs that are not in ignored_attrs
    """
    return dict((k, v) for k, v in attrs.items() if k not in ignored_attrs)


def from_cdms2(variable):
    """Convert a cdms2 variable into an DataArray
    """
    values = np.asarray(variable)
    name = variable.id
    coords = [(v.id, np.asarray(v),
               _filter_attrs(v.attributes, cdms2_ignored_attrs))
              for v in variable.getAxisList()]
    attrs = _filter_attrs(variable.attributes, cdms2_ignored_attrs)
    dataarray = DataArray(values, coords=coords, name=name, attrs=attrs)
    return decode_cf(dataarray.to_dataset())[dataarray.name]


def to_cdms2(dataarray):
    """Convert a DataArray into a cdms2 variable
    """
    # we don't want cdms2 to be a hard dependency
    import cdms2

    def set_cdms2_attrs(var, attrs):
        for k, v in attrs.items():
            setattr(var, k, v)

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


def _pick_attrs(attrs, keys):
    """ Return attrs with keys in keys list
    """
    return dict((k, v) for k, v in attrs.items() if k in keys)


def _get_iris_args(attrs):
    """ Converts the xarray attrs into args that can be passed into Iris
    """
    # iris.unit is deprecated in Iris v1.9
    import cf_units
    args = {'attributes': _filter_attrs(attrs, iris_forbidden_keys)}
    args.update(_pick_attrs(attrs, ('standard_name', 'long_name',)))
    unit_args = _pick_attrs(attrs, ('calendar',))
    if 'units' in attrs:
        args['units'] = cf_units.Unit(attrs['units'], **unit_args)
    return args


# TODO: Add converting bounds from xarray to Iris and back
def to_iris(dataarray):
    """ Convert a DataArray into a Iris Cube
    """
    # Iris not a hard dependency
    import iris
    from iris.fileformats.netcdf import parse_cell_methods

    dim_coords = []
    aux_coords = []

    for coord_name in dataarray.coords:
        coord = encode(dataarray.coords[coord_name])
        coord_args = _get_iris_args(coord.attrs)
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

    args = _get_iris_args(dataarray.attrs)
    args['var_name'] = dataarray.name
    args['dim_coords_and_dims'] = dim_coords
    args['aux_coords_and_dims'] = aux_coords
    if 'cell_methods' in dataarray.attrs:
        args['cell_methods'] = \
            parse_cell_methods(dataarray.attrs['cell_methods'])

    masked_data = duck_array_ops.masked_invalid(dataarray.data)
    cube = iris.cube.Cube(masked_data, **args)

    return cube


def _iris_obj_to_attrs(obj):
    """ Return a dictionary of attrs when given a Iris object
    """
    attrs = {'standard_name': obj.standard_name,
             'long_name': obj.long_name}
    if obj.units.calendar:
        attrs['calendar'] = obj.units.calendar
    if obj.units.origin != '1':
        attrs['units'] = obj.units.origin
    attrs.update(obj.attributes)
    return dict((k, v) for k, v in attrs.items() if v is not None)


def _iris_cell_methods_to_str(cell_methods_obj):
    """ Converts a Iris cell methods into a string
    """
    cell_methods = []
    for cell_method in cell_methods_obj:
        names = ''.join(['{}: '.format(n) for n in cell_method.coord_names])
        intervals = ' '.join(['interval: {}'.format(interval)
                              for interval in cell_method.intervals])
        comments = ' '.join(['comment: {}'.format(comment)
                             for comment in cell_method.comments])
        extra = ' '.join([intervals, comments]).strip()
        if extra:
            extra = ' ({})'.format(extra)
        cell_methods.append(names + cell_method.method + extra)
    return ' '.join(cell_methods)


def from_iris(cube):
    """ Convert a Iris cube into an DataArray
    """
    import iris.exceptions
    from xarray.core.pycompat import dask_array_type

    name = cube.var_name
    dims = []
    for i in range(cube.ndim):
        try:
            dim_coord = cube.coord(dim_coords=True, dimensions=(i,))
            dims.append(dim_coord.var_name)
        except iris.exceptions.CoordinateNotFoundError:
            dims.append("dim_{}".format(i))

    coords = OrderedDict()

    for coord in cube.coords():
        coord_attrs = _iris_obj_to_attrs(coord)
        coord_dims = [dims[i] for i in cube.coord_dims(coord)]
        if not coord.var_name:
            raise ValueError("Coordinate '{}' has no "
                             "var_name attribute".format(coord.name()))
        if coord_dims:
            coords[coord.var_name] = (coord_dims, coord.points, coord_attrs)
        else:
            coords[coord.var_name] = ((),
                                      np.asscalar(coord.points), coord_attrs)

    array_attrs = _iris_obj_to_attrs(cube)
    cell_methods = _iris_cell_methods_to_str(cube.cell_methods)
    if cell_methods:
        array_attrs['cell_methods'] = cell_methods

    # Deal with iris 1.* and 2.*
    cube_data = cube.core_data() if hasattr(cube, 'core_data') else cube.data

    # Deal with dask and numpy masked arrays
    if isinstance(cube_data, dask_array_type):
        from dask.array import ma as dask_ma
        filled_data = dask_ma.filled(cube_data, get_fill_value(cube.dtype))
    elif isinstance(cube_data, np.ma.MaskedArray):
        filled_data = np.ma.filled(cube_data, get_fill_value(cube.dtype))
    else:
        filled_data = cube_data

    dataarray = DataArray(filled_data, coords=coords, name=name,
                          attrs=array_attrs, dims=dims)
    decoded_ds = decode_cf(dataarray._to_temp_dataset())
    return dataarray._from_temp_dataset(decoded_ds)
