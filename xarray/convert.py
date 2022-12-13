"""Functions for converting to and from xarray objects
"""
from collections import Counter

import numpy as np
import pandas as pd

from xarray.coding.times import CFDatetimeCoder, CFTimedeltaCoder
from xarray.conventions import decode_cf
from xarray.core import duck_array_ops
from xarray.core.dataarray import DataArray
from xarray.core.dtypes import get_fill_value
from xarray.core.pycompat import array_type

cdms2_ignored_attrs = {"name", "tileIndex"}
iris_forbidden_keys = {
    "standard_name",
    "long_name",
    "units",
    "bounds",
    "axis",
    "calendar",
    "leap_month",
    "leap_year",
    "month_lengths",
    "coordinates",
    "grid_mapping",
    "climatology",
    "cell_methods",
    "formula_terms",
    "compress",
    "missing_value",
    "add_offset",
    "scale_factor",
    "valid_max",
    "valid_min",
    "valid_range",
    "_FillValue",
}
cell_methods_strings = {
    "point",
    "sum",
    "maximum",
    "median",
    "mid_range",
    "minimum",
    "mean",
    "mode",
    "standard_deviation",
    "variance",
}


def encode(var):
    return CFTimedeltaCoder().encode(CFDatetimeCoder().encode(var.variable))


def _filter_attrs(attrs, ignored_attrs):
    """Return attrs that are not in ignored_attrs"""
    return {k: v for k, v in attrs.items() if k not in ignored_attrs}


def from_cdms2(variable):
    """Convert a cdms2 variable into an DataArray"""
    values = np.asarray(variable)
    name = variable.id
    dims = variable.getAxisIds()
    coords = {}
    for axis in variable.getAxisList():
        coords[axis.id] = DataArray(
            np.asarray(axis),
            dims=[axis.id],
            attrs=_filter_attrs(axis.attributes, cdms2_ignored_attrs),
        )
    grid = variable.getGrid()
    if grid is not None:
        ids = [a.id for a in grid.getAxisList()]
        for axis in grid.getLongitude(), grid.getLatitude():
            if axis.id not in variable.getAxisIds():
                coords[axis.id] = DataArray(
                    np.asarray(axis[:]),
                    dims=ids,
                    attrs=_filter_attrs(axis.attributes, cdms2_ignored_attrs),
                )
    attrs = _filter_attrs(variable.attributes, cdms2_ignored_attrs)
    dataarray = DataArray(values, dims=dims, coords=coords, name=name, attrs=attrs)
    return decode_cf(dataarray.to_dataset())[dataarray.name]


def to_cdms2(dataarray, copy=True):
    """Convert a DataArray into a cdms2 variable"""
    # we don't want cdms2 to be a hard dependency
    import cdms2

    def set_cdms2_attrs(var, attrs):
        for k, v in attrs.items():
            setattr(var, k, v)

    # 1D axes
    axes = []
    for dim in dataarray.dims:
        coord = encode(dataarray.coords[dim])
        axis = cdms2.createAxis(coord.values, id=dim)
        set_cdms2_attrs(axis, coord.attrs)
        axes.append(axis)

    # Data
    var = encode(dataarray)
    cdms2_var = cdms2.createVariable(
        var.values, axes=axes, id=dataarray.name, mask=pd.isnull(var.values), copy=copy
    )

    # Attributes
    set_cdms2_attrs(cdms2_var, var.attrs)

    # Curvilinear and unstructured grids
    if dataarray.name not in dataarray.coords:

        cdms2_axes = {}
        for coord_name in set(dataarray.coords.keys()) - set(dataarray.dims):

            coord_array = dataarray.coords[coord_name].to_cdms2()

            cdms2_axis_cls = (
                cdms2.coord.TransientAxis2D
                if coord_array.ndim
                else cdms2.auxcoord.TransientAuxAxis1D
            )
            cdms2_axis = cdms2_axis_cls(coord_array)
            if cdms2_axis.isLongitude():
                cdms2_axes["lon"] = cdms2_axis
            elif cdms2_axis.isLatitude():
                cdms2_axes["lat"] = cdms2_axis

        if "lon" in cdms2_axes and "lat" in cdms2_axes:
            if len(cdms2_axes["lon"].shape) == 2:
                cdms2_grid = cdms2.hgrid.TransientCurveGrid(
                    cdms2_axes["lat"], cdms2_axes["lon"]
                )
            else:
                cdms2_grid = cdms2.gengrid.AbstractGenericGrid(
                    cdms2_axes["lat"], cdms2_axes["lon"]
                )
            for axis in cdms2_grid.getAxisList():
                cdms2_var.setAxis(cdms2_var.getAxisIds().index(axis.id), axis)
            cdms2_var.setGrid(cdms2_grid)

    return cdms2_var


def _pick_attrs(attrs, keys):
    """Return attrs with keys in keys list"""
    return {k: v for k, v in attrs.items() if k in keys}


def _get_iris_args(attrs):
    """Converts the xarray attrs into args that can be passed into Iris"""
    # iris.unit is deprecated in Iris v1.9
    import cf_units

    args = {"attributes": _filter_attrs(attrs, iris_forbidden_keys)}
    args.update(_pick_attrs(attrs, ("standard_name", "long_name")))
    unit_args = _pick_attrs(attrs, ("calendar",))
    if "units" in attrs:
        args["units"] = cf_units.Unit(attrs["units"], **unit_args)
    return args


# TODO: Add converting bounds from xarray to Iris and back
def to_iris(dataarray):
    """Convert a DataArray into a Iris Cube"""
    # Iris not a hard dependency
    import iris
    from iris.fileformats.netcdf import parse_cell_methods

    dim_coords = []
    aux_coords = []

    for coord_name in dataarray.coords:
        coord = encode(dataarray.coords[coord_name])
        coord_args = _get_iris_args(coord.attrs)
        coord_args["var_name"] = coord_name
        axis = None
        if coord.dims:
            axis = dataarray.get_axis_num(coord.dims)
        if coord_name in dataarray.dims:
            try:
                iris_coord = iris.coords.DimCoord(coord.values, **coord_args)
                dim_coords.append((iris_coord, axis))
            except ValueError:
                iris_coord = iris.coords.AuxCoord(coord.values, **coord_args)
                aux_coords.append((iris_coord, axis))
        else:
            iris_coord = iris.coords.AuxCoord(coord.values, **coord_args)
            aux_coords.append((iris_coord, axis))

    args = _get_iris_args(dataarray.attrs)
    args["var_name"] = dataarray.name
    args["dim_coords_and_dims"] = dim_coords
    args["aux_coords_and_dims"] = aux_coords
    if "cell_methods" in dataarray.attrs:
        args["cell_methods"] = parse_cell_methods(dataarray.attrs["cell_methods"])

    masked_data = duck_array_ops.masked_invalid(dataarray.data)
    cube = iris.cube.Cube(masked_data, **args)

    return cube


def _iris_obj_to_attrs(obj):
    """Return a dictionary of attrs when given a Iris object"""
    attrs = {"standard_name": obj.standard_name, "long_name": obj.long_name}
    if obj.units.calendar:
        attrs["calendar"] = obj.units.calendar
    if obj.units.origin != "1" and not obj.units.is_unknown():
        attrs["units"] = obj.units.origin
    attrs.update(obj.attributes)
    return {k: v for k, v in attrs.items() if v is not None}


def _iris_cell_methods_to_str(cell_methods_obj):
    """Converts a Iris cell methods into a string"""
    cell_methods = []
    for cell_method in cell_methods_obj:
        names = "".join(f"{n}: " for n in cell_method.coord_names)
        intervals = " ".join(
            f"interval: {interval}" for interval in cell_method.intervals
        )
        comments = " ".join(f"comment: {comment}" for comment in cell_method.comments)
        extra = " ".join([intervals, comments]).strip()
        if extra:
            extra = f" ({extra})"
        cell_methods.append(names + cell_method.method + extra)
    return " ".join(cell_methods)


def _name(iris_obj, default="unknown"):
    """Mimics `iris_obj.name()` but with different name resolution order.

    Similar to iris_obj.name() method, but using iris_obj.var_name first to
    enable roundtripping.
    """
    return iris_obj.var_name or iris_obj.standard_name or iris_obj.long_name or default


def from_iris(cube):
    """Convert a Iris cube into an DataArray"""
    import iris.exceptions

    name = _name(cube)
    if name == "unknown":
        name = None
    dims = []
    for i in range(cube.ndim):
        try:
            dim_coord = cube.coord(dim_coords=True, dimensions=(i,))
            dims.append(_name(dim_coord))
        except iris.exceptions.CoordinateNotFoundError:
            dims.append(f"dim_{i}")

    if len(set(dims)) != len(dims):
        duplicates = [k for k, v in Counter(dims).items() if v > 1]
        raise ValueError(f"Duplicate coordinate name {duplicates}.")

    coords = {}

    for coord in cube.coords():
        coord_attrs = _iris_obj_to_attrs(coord)
        coord_dims = [dims[i] for i in cube.coord_dims(coord)]
        if coord_dims:
            coords[_name(coord)] = (coord_dims, coord.points, coord_attrs)
        else:
            coords[_name(coord)] = ((), coord.points.item(), coord_attrs)

    array_attrs = _iris_obj_to_attrs(cube)
    cell_methods = _iris_cell_methods_to_str(cube.cell_methods)
    if cell_methods:
        array_attrs["cell_methods"] = cell_methods

    # Deal with iris 1.* and 2.*
    cube_data = cube.core_data() if hasattr(cube, "core_data") else cube.data

    # Deal with dask and numpy masked arrays
    dask_array_type = array_type("dask")
    if isinstance(cube_data, dask_array_type):
        from dask.array import ma as dask_ma

        filled_data = dask_ma.filled(cube_data, get_fill_value(cube.dtype))
    elif isinstance(cube_data, np.ma.MaskedArray):
        filled_data = np.ma.filled(cube_data, get_fill_value(cube.dtype))
    else:
        filled_data = cube_data

    dataarray = DataArray(
        filled_data, coords=coords, name=name, attrs=array_attrs, dims=dims
    )
    decoded_ds = decode_cf(dataarray._to_temp_dataset())
    return dataarray._from_temp_dataset(decoded_ds)
