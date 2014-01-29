import iris
import netCDF4 as nc4

from scipy.io import netcdf
from collections import OrderedDict

import variable


class InMemoryDataStore(object):
    """
    Stores dimensions, variables and attributes
    in ordered dictionaries, making this store
    fast compared to stores which store to disk.
    """

    def __init__(self):
        self.unchecked_set_attributes(variable.AttributesDict())
        self.unchecked_set_dimensions(OrderedDict())
        self.unchecked_set_variables(OrderedDict())

    def unchecked_set_dimensions(self, dimensions):
        """Set the dimensions without checking validity"""
        self.dimensions = dimensions

    def unchecked_set_attributes(self, attributes):
        """Set the attributes without checking validity"""
        self.attributes = attributes

    def unchecked_set_variables(self, variables):
        """Set the variables without checking validity"""
        self.variables = variables

    def unchecked_create_dimension(self, name, length):
        """Set a dimension length"""
        self.dimensions[name] = length

    def unchecked_add_variable(self, name, variable):
        """Add a variable without checks"""
        self.variables[name] = variable
        return self.variables[name]

    def unchecked_create_variable(self, name, dims, data, attributes):
        """Creates a variable without checks"""
        v = variable.Variable(dims=dims, data=data,
                              attributes=attributes)
        self._unchecked_add_variable(name, v)
        return v

    def unchecked_create_coordinate(self, name, data, attributes):
        """Creates a coordinate (dim and var) without checks"""
        self._unchecked_create_dimension(name, data.size)
        return self._unchecked_create_variable(name, (name,), data, attributes)

    def sync(self):
        pass


class ScipyVariable(variable.Variable):
    def __init__(self, scipy_var):
        self._dimensions = scipy_var.dimensions
        self._data = scipy_var.data
        self._attributes = scipy_var._attributes


class ScipyDataStore(object):
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
        return OrderedDict((k, ScipyVariable(v))
                           for k, v in self.ds.variables.iteritems())

    @property
    def attributes(self):
        return self.ds._attributes

    @property
    def dimensions(self):
        return self.ds.dimensions

    def unchecked_set_dimensions(self, dimensions):
        """Set the dimensions without checking validity"""
        for d, l in dimensions.iteritems():
            self.unchecked_create_dimension(d, l)

    def unchecked_set_attributes(self, attributes):
        """Set the attributes without checking validity"""
        for k, v in attributes.iteritems():
            setattr(self.ds, k, v)

    def unchecked_set_variables(self, variables):
        """Set the variables without checking validity"""
        for vn, v in variables.iteritems():
            self.unchecked_add_variable(vn, v)

    def unchecked_create_dimension(self, name, length):
        """Set a dimension length"""
        self.ds.createDimension(name, length)

    def unchecked_add_variable(self, name, variable):
        """Add a variable without checks"""
        self.ds.createVariable(name, variable.dtype,
                               variable.dimensions)
        self.ds.variables[name][:] = variable.data[:]
        for k, v in variable.attributes.iteritems():
            setattr(self.ds.variables[name], k, v)

    def unchecked_create_coordinate(self, name, data, attributes):
        """Creates a coordinate (dim and var) without checks"""
        self.unchecked_create_dimension(name, data.size)
        return self.unchecked_create_variable(name, (name,), data, attributes)

    def sync(self):
        self.ds.flush()


class NetCDF4Variable(variable.Variable):

    def __init__(self, nc4_variable):
        self._nc4_variable = nc4_variable
        self._dimensions = nc4_variable.dimensions
        self._data = nc4_variable
        self._attributes = None

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
            attr_dict = variable.AttributesDict(
                (k, self._nc4_variable.getncattr(k)) for k in keys)
            self._attributes = attr_dict
        return self._attributes


class NetCDF4DataStore(object):

    def __init__(self, filename, *args, **kwdargs):
        self.ds = nc4.Dataset(filename, *args, **kwdargs)

    @property
    def variables(self):
        return OrderedDict((k, NetCDF4Variable(v))
                           for k, v in self.ds.variables.iteritems())

    @property
    def attributes(self):
        return variable.AttributesDict((k, self.ds.getncattr(k))
                                       for k in self.ds.ncattrs())

    @property
    def dimensions(self):
        return OrderedDict((k, len(v)) for k, v in self.ds.dimensions.iteritems())

    def unchecked_set_dimensions(self, dimensions):
        """Set the dimensions without checking validity"""
        for d, l in dimensions.iteritems():
            self.unchecked_create_dimension(d, l)

    def unchecked_set_attributes(self, attributes):
        """Set the attributes without checking validity"""
        self.ds.setncatts(attributes)

    def unchecked_set_variables(self, variables):
        """Set the variables without checking validity"""
        for vn, v in variables.iteritems():
            self.unchecked_add_variable(vn, v)

    def unchecked_create_dimension(self, name, length):
        """Set a dimension length"""
        self.ds.createDimension(name, size=length)

    def unchecked_add_variable(self, name, variable):
        """Add a variable without checks"""
        # netCDF4 will automatically assign a fill value
        # depending on the datatype of the variable.  Here
        # we let the package handle the _FillValue attribute
        # instead of setting it ourselves.
        fill_value = variable.attributes.pop('_FillValue', None)
        self.ds.createVariable(varname=name,
                               datatype=variable.dtype,
                               dimensions=variable.dimensions,
                               fill_value=fill_value)
        self.ds.variables[name][:] = variable.data[:]
        self.ds.variables[name].setncatts(variable.attributes)

    def unchecked_create_coordinate(self, name, data, attributes):
        """Creates a coordinate (dim and var) without checks"""
        self.unchecked_create_dimension(name, data.size)
        return self.unchecked_create_variable(name, (name,), data, attributes)

    def sync(self):
        self.ds.sync()
