#TODO: refactor this module so all the stores just expose dimension, variables
# and attributes with the OrderedDict API that handle all the storage logic

import netCDF4 as nc4
import numpy as np

from scipy.io import netcdf
from collections import OrderedDict

from utils import FrozenOrderedDict
import conventions
import utils
import variable


class AbstractDataStore(object):
    def unchecked_set_dimensions(self, dimensions):
        """Set the dimensions without checking validity"""
        for d, l in dimensions.iteritems():
            self.unchecked_set_dimension(d, l)

    def unchecked_set_attributes(self, attributes):
        """Set the attributes without checking validity"""
        for k, v in attributes.iteritems():
            self.unchecked_set_attribute(k, v)

    def unchecked_set_variables(self, variables):
        """Set the variables without checking validity"""
        for vn, v in variables.iteritems():
            self.unchecked_set_variable(vn, v)


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

    def unchecked_set_dimension(self, name, length):
        """Set a dimension length"""
        self.dimensions[name] = length

    def unchecked_set_attribute(self, key, value):
        """Set the attributes without checking validity"""
        self.attributes[key] = value

    def unchecked_set_variable(self, name, variable):
        """Set a variable without checks"""
        self.variables[name] = variable
        return self.variables[name]

    def sync(self):
        pass


class ScipyVariable(variable.Variable):
    def __init__(self, scipy_var):
        self._dimensions = scipy_var.dimensions
        self._data = scipy_var.data
        self._attributes = scipy_var._attributes


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
        return FrozenOrderedDict((k, ScipyVariable(v))
                                 for k, v in self.ds.variables.iteritems())

    @property
    def attributes(self):
        return self.ds._attributes

    @property
    def dimensions(self):
        return self.ds.dimensions

    def unchecked_set_dimension(self, name, length):
        """Set a dimension length"""
        if name in self.ds.dimensions:
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

    def unchecked_set_attribute(self, key, value):
        self._validate_attr_key(key)
        setattr(self.ds, key, self._cast_attr_value(value))

    def unchecked_set_variable(self, name, variable):
        """Add a variable without checks"""
        if name not in self.ds.variables:
            self.ds.createVariable(name, variable.dtype, variable.dimensions)
        scipy_var = self.ds.variables[name]
        scipy_var[:] = variable.data[:]
        for k, v in variable.attributes.iteritems():
            self._validate_attr_key(k)
            setattr(scipy_var, k, self._cast_attr_value(v))
        return ScipyVariable(scipy_var)

    def sync(self):
        self.ds.flush()


class NetCDF4Variable(variable.Variable):
    def __init__(self, nc4_variable):
        self._nc4_variable = nc4_variable
        self._dimensions = nc4_variable.dimensions
        self._data = nc4_variable
        self._attributes = None

    def _remap_indexer(self, key):
        # netCDF4-python already does orthogonal indexing, so just expand
        # the indexer
        return utils.expanded_indexer(key, self.ndim)

    @property
    def attributes(self):
        if self._attributes is None:
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
            packing_attributes = ['scale_factor', 'add_offset']
            keys = [k for k in self._nc4_variable.ncattrs()
                    if not k in packing_attributes]
            attr_dict = OrderedDict(
                (k, self._nc4_variable.getncattr(k)) for k in keys)
            self._attributes = attr_dict
        return self._attributes


class NetCDF4DataStore(AbstractDataStore):
    def __init__(self, filename, *args, **kwdargs):
        self.ds = nc4.Dataset(filename, *args, **kwdargs)

    @property
    def variables(self):
        return FrozenOrderedDict((k, NetCDF4Variable(v))
                                 for k, v in self.ds.variables.iteritems())

    @property
    def attributes(self):
        return FrozenOrderedDict((k, self.ds.getncattr(k))
                                 for k in self.ds.ncattrs())

    @property
    def dimensions(self):
        return FrozenOrderedDict((k, len(v)) for k, v in self.ds.dimensions.iteritems())

    def unchecked_set_dimension(self, name, length):
        """Set a dimension length"""
        self.ds.createDimension(name, size=length)

    def unchecked_set_attribute(self, key, value):
        self.ds.setncatts({key: value})

    def unchecked_set_variable(self, name, variable):
        """Set a variable without checks"""
        # netCDF4 will automatically assign a fill value
        # depending on the datatype of the variable.  Here
        # we let the package handle the _FillValue attribute
        # instead of setting it ourselves.
        fill_value = variable.attributes.pop('_FillValue', None)
        if name not in self.ds.variables:
            self.ds.createVariable(varname=name,
                                   datatype=variable.dtype,
                                   dimensions=variable.dimensions,
                                   fill_value=fill_value)
        nc4_var = self.ds.variables[name]
        nc4_var[:] = variable.data[:]
        nc4_var.setncatts(variable.attributes)
        return NetCDF4Variable(nc4_var)

    def sync(self):
        self.ds.sync()
