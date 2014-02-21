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
import conventions
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


def convert_to_cf_variable(array):
    data = array.data
    attributes = array.attributes.copy()
    if isinstance(array.data, pd.DatetimeIndex):
        (data, units, calendar) = datetimeindex2num(array.data)
        attributes['units'] = units
        attributes['calendar'] = calendar
    return xarray.XArray(array.dimensions, data, attributes)


def convert_scipy_variable(var):
    return xarray.XArray(var.dimensions, var.data, var._attributes)


class ScipyDataStore(AbstractDataStore):
    """
    Stores data using the scipy.io.netcdf package.
    This store has the advantage of being able to
    be initialized with a StringIO object, allow for
    serialization.
    """
    def __init__(self, fobj, *args, **kwdargs):
        self.ds = netcdf.netcdf_file(fobj, *args, **kwdargs)

    @property
    def variables(self):
        return FrozenOrderedDict((k, convert_scipy_variable(v))
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
        if not conventions.is_valid_name(key):
            raise ValueError("Not a valid attribute name")

    def _cast_attr_value(self, value):
        # Strings get special handling because netCDF treats them as
        # character arrays. Everything else gets coerced to a numpy
        # vector. netCDF treats scalars as 1-element vectors. Arrays of
        # non-numeric type are not allowed.
        if isinstance(value, basestring):
            # netcdf attributes should be unicode
            value = unicode(value)
        else:
            try:
                value = conventions.coerce_type(np.atleast_1d(np.asarray(value)))
            except:
                raise ValueError("Not a valid value for a netCDF attribute")
            if value.ndim > 1:
                raise ValueError("netCDF attributes must be vectors " +
                        "(1-dimensional)")
            value = conventions.coerce_type(value)
            if str(value.dtype) not in conventions.TYPEMAP:
                # A plain string attribute is okay, but an array of
                # string objects is not okay!
                raise ValueError("Can not convert to a valid netCDF type")
        return value

    def set_attribute(self, key, value):
        self._validate_attr_key(key)
        setattr(self.ds, key, self._cast_attr_value(value))

    def set_variable(self, name, variable):
        variable = convert_to_cf_variable(variable)
        data = variable.data
        dtype_convert = {'int64': 'int32', 'float64': 'float32'}
        if str(data.dtype) in dtype_convert:
            data = np.asarray(data, dtype=dtype_convert[str(data.dtype)])
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


def convert_nc4_variable(var):
    # we don't want to see scale_factor and add_offset in the attributes
    # since the netCDF4 package automatically scales the data on read.
    # If we kept scale_factor and add_offset around and did this:
    #
    # foo = ncdf4.Dataset('foo.nc')
    # ncdf4.dump(foo, 'bar.nc')
    # bar = ncdf4.Dataset('bar.nc')
    #
    # you would find that any packed variables in the original
    # netcdf file would now have been scaled twice!
    attr = OrderedDict((k, var.getncattr(k)) for k in var.ncattrs()
                       if k not in ['scale_factor', 'add_offset'])
    return xarray.XArray(var.dimensions, var, attr, indexing_mode='orthogonal')


class NetCDF4DataStore(AbstractDataStore):
    def __init__(self, filename, *args, **kwdargs):
        # TODO: set auto_maskandscale=True so we can handle the array
        # packing/unpacking ourselves (using NaN instead of masked arrays)
        self.ds = nc4.Dataset(filename, *args, **kwdargs)

    @property
    def variables(self):
        return FrozenOrderedDict((k, convert_nc4_variable(v))
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

    def _cast_data(self, data):
        if isinstance(data, pd.DatetimeIndex):
            data = datetimeindex2num(data)
        return data

    def set_variable(self, name, variable):
        variable = convert_to_cf_variable(variable)
        # netCDF4 will automatically assign a fill value
        # depending on the datatype of the variable.  Here
        # we let the package handle the _FillValue attribute
        # instead of setting it ourselves.
        fill_value = variable.attributes.pop('_FillValue', None)
        self.ds.createVariable(varname=name,
                               datatype=variable.dtype,
                               dimensions=variable.dimensions,
                               fill_value=fill_value)
        nc4_var = self.ds.variables[name]
        nc4_var[:] = variable.data[:]
        nc4_var.setncatts(variable.attributes)

    def del_attribute(self, key):
        self.ds.delncattr(key)

    def sync(self):
        self.ds.sync()
