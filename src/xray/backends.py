"""Backend objects for saving and loading data

DataStores provide a uniform interface for saving and loading data in different
formats. They should not be used directly, but rather through Dataset objects.
"""
# TODO: implement backend logic directly in OrderedDict subclasses, to allow
# for directly manipulating Dataset.variables and the like?
import netCDF4 as nc4
import numpy as np
import pandas as pd

from scipy.io import netcdf
from collections import OrderedDict

import xarray
from conventions import (decode_cf_variable, encode_cf_variable,
                         is_valid_nc3_name, coerce_nc3_dtype)
from utils import FrozenOrderedDict, Frozen, datetimeindex2num


class AbstractDataStore(object):
    def set_dimensions(self, dimensions):
        for d, l in dimensions.iteritems():
            self.set_dimension(d, l)

    def set_attributes(self, attributes):
        for k, v in attributes.iteritems():
            self.set_attribute(k, v)

    def set_variables(self, variables):
        for vn, v in variables.iteritems():
            self.set_variable(vn, v)


class InMemoryDataStore(AbstractDataStore):
    """
    Stores dimensions, variables and attributes
    in ordered dictionaries, making this store
    fast compared to stores which store to disk.
    """
    def __init__(self):
        self.dimensions = OrderedDict()
        self.variables = OrderedDict()
        self.attributes = OrderedDict()

    def set_dimension(self, name, length):
        self.dimensions[name] = length

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def set_variable(self, name, variable):
        self.variables[name] = variable
        return self.variables[name]

    def del_attribute(self, key):
        del self.attributes[key]

    def sync(self):
        pass


class ScipyDataStore(AbstractDataStore):
    """
    Stores data using the scipy.io.netcdf package.
    This store has the advantage of being able to
    be initialized with a StringIO object, allow for
    serialization.
    """
    def __init__(self, filename_or_obj, mode='r', mmap=None, version=1,
                 mask_and_scale=True):
        self.ds = netcdf.netcdf_file(filename_or_obj, mode=mode, mmap=None,
                                     version=version)
        self.mask_and_scale = mask_and_scale

    @property
    def variables(self):
        return FrozenOrderedDict((k, decode_cf_variable(v.dimensions, v.data,
                                                        v._attributes))
                                 for k, v in self.ds.variables.iteritems())

    @property
    def attributes(self):
        return Frozen(self.ds._attributes)

    @property
    def dimensions(self):
        return Frozen(self.ds.dimensions)

    def set_dimension(self, name, length):
        if name in self.dimensions:
            raise ValueError('%s does not support modifying dimensions'
                             % type(self).__name__)
        self.ds.createDimension(name, length)

    def _validate_attr_key(self, key):
        if not is_valid_nc3_name(key):
            raise ValueError("Not a valid attribute name")

    def _cast_attr_value(self, value):
        if isinstance(value, basestring):
            value = unicode(value)
        else:
            value = coerce_nc3_dtype(np.atleast_1d(value))
            if value.ndim > 1:
                raise ValueError("netCDF attributes must be 1-dimensional")
        return value

    def set_attribute(self, key, value):
        self._validate_attr_key(key)
        setattr(self.ds, key, self._cast_attr_value(value))

    def set_variable(self, name, variable):
        variable = encode_cf_variable(variable)
        data = coerce_nc3_dtype(variable.data)
        self.ds.createVariable(name, data.dtype, variable.dimensions)
        scipy_var = self.ds.variables[name]
        scipy_var[:] = data[:]
        for k, v in variable.attributes.iteritems():
            self._validate_attr_key(k)
            setattr(scipy_var, k, self._cast_attr_value(v))

    def del_attribute(self, key):
        delattr(self.ds, key)

    def sync(self):
        self.ds.flush()


class NetCDF4DataStore(AbstractDataStore):
    def __init__(self, filename, mode='r', clobber=True, diskless=False,
                 persist=False, format='NETCDF4', mask_and_scale=True):
        self.ds = nc4.Dataset(filename, mode=mode, clobber=clobber,
                              diskless=diskless, persist=persist,
                              format=format)
        self.mask_and_scale = mask_and_scale

    @property
    def variables(self):
        def convert_variable(var):
            attr = OrderedDict((k, var.getncattr(k)) for k in var.ncattrs())
            var.set_auto_maskandscale(False)
            return decode_cf_variable(
                var.dimensions, var, attr, indexing_mode='orthogonal',
                mask_and_scale=self.mask_and_scale)
        return FrozenOrderedDict((k, convert_variable(v))
                                 for k, v in self.ds.variables.iteritems())

    @property
    def attributes(self):
        return FrozenOrderedDict((k, self.ds.getncattr(k))
                                 for k in self.ds.ncattrs())

    @property
    def dimensions(self):
        return FrozenOrderedDict((k, len(v)) for k, v in self.ds.dimensions.iteritems())

    def set_dimension(self, name, length):
        self.ds.createDimension(name, size=length)

    def set_attribute(self, key, value):
        self.ds.setncatts({key: value})

    def set_variable(self, name, variable):
        variable = encode_cf_variable(variable)
        # netCDF4 will automatically assign a fill value
        # depending on the datatype of the variable.  Here
        # we let the package handle the _FillValue attribute
        # instead of setting it ourselves.
        fill_value = variable.encoding.get('_FillValue')
        self.ds.createVariable(varname=name,
                               datatype=variable.dtype,
                               dimensions=variable.dimensions,
                               fill_value=fill_value)
        nc4_var = self.ds.variables[name]
        nc4_var.set_auto_maskandscale(False)
        nc4_var[:] = variable.data[:]
        nc4_var.setncatts(variable.attributes)

    def del_attribute(self, key):
        self.ds.delncattr(key)

    def sync(self):
        self.ds.sync()
