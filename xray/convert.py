"""Functions for converting to and from xray objects
"""
import numpy as np

from .core.dataarray import DataArray


ignored_attrs = set(['name', 'tileIndex'])


def _get_cdms2_attrs(var):
    return dict((k, v) for k, v in var.attributes.items()
                if k not in ignored_attrs)


def from_cdms2(variable):
    """Convert a cdms2 variable into an DataArray
    """
    values = np.asarray(variable)
    name = variable.id
    coords = [(v.id, np.asarray(v), _get_cdms2_attrs(v))
              for v in variable.getAxisList()]
    attrs = _get_cdms2_attrs(variable)
    return DataArray(values, coords=coords, name=name, attrs=attrs)


def _set_cdms2_attrs(var, attrs):
    for k, v in attrs.items():
        setattr(var, k, v)


def to_cdms2(dataarray):
    """Convert a DataArray into a cdms2 variable
    """
    # we don't want cdms2 to be a hard dependency
    import cdms2

    axes = []
    for dim in dataarray.dims:
        coord = dataarray.coords[dim]
        axis = cdms2.createAxis(coord.values, id=dim)
        _set_cdms2_attrs(axis, coord.attrs)
        axes.append(axis)

    var = cdms2.createVariable(dataarray.values, axes=axes, id=dataarray.name)
    _set_cdms2_attrs(var, dataarray.attrs)
    return var
