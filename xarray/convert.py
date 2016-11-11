"""Functions for converting to and from xarray objects
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from .core.dataarray import DataArray
from .conventions import (
    maybe_encode_timedelta, maybe_encode_datetime, decode_cf)

ignored_attrs = set(['name', 'tileIndex'])


def from_cdms2(variable):
    """Convert a cdms2 variable into an DataArray
    """
    def get_cdms2_attrs(var):
        return dict((k, v) for k, v in var.attributes.items()
                    if k not in ignored_attrs)

    values = np.asarray(variable)
    name = variable.id
    coords = [(v.id, np.asarray(v), get_cdms2_attrs(v))
              for v in variable.getAxisList()]
    attrs = get_cdms2_attrs(variable)
    dataarray = DataArray(values, coords=coords, name=name, attrs=attrs)
    return decode_cf(dataarray.to_dataset())[dataarray.name]


def to_cdms2(dataarray):
    """Convert a DataArray into a cdms2 variable
    """
    # we don't want cdms2 to be a hard dependency
    import cdms2

    def encode(var):
        return maybe_encode_timedelta(maybe_encode_datetime(var.variable))

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
