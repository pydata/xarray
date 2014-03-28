"""Backend objects for saving and loading data

DataStores provide a uniform interface for saving and loading data in different
formats. They should not be used directly, but rather through Dataset objects.
"""
# TODO: implement backend logic directly in OrderedDict subclasses, to allow
# for directly manipulating Dataset.variables and the like?
import os
import numpy as np
import netCDF4 as nc4

from cStringIO import StringIO
from scipy.io import netcdf
from collections import OrderedDict

import xarray

from utils import FrozenOrderedDict, Frozen
from conventions import is_valid_nc3_name, coerce_nc3_dtype, encode_cf_variable


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

    def set_necessary_dimensions(self, variable):
        for d, l in zip(variable.dimensions, variable.shape):
            if d not in self.ds.dimensions:
                self.set_dimension(d, l)


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
    def __init__(self, filename_or_obj, mode='r', mmap=None, version=1):
        # if filename is a NetCDF3 bytestring we store it in a StringIO
        if (isinstance(filename_or_obj, basestring)
            and filename_or_obj.startswith('CDF')):
            # TODO: this check has the unfortunate side-effect that
            # paths to files cannot start with 'CDF'.
            filename_or_obj = StringIO(filename_or_obj)
        self.ds = netcdf.netcdf_file(filename_or_obj, mode=mode, mmap=mmap,
                                     version=version)

    @property
    def variables(self):
        return FrozenOrderedDict((k, xarray.XArray(v.dimensions, v.data,
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
        self.set_necessary_dimensions(variable)
        self.ds.createVariable(name, data.dtype, variable.dimensions)
        scipy_var = self.ds.variables[name]
        if data.ndim == 0:
            scipy_var.assignValue(data)
        else:
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
                 persist=False, format='NETCDF4'):
        self.ds = nc4.Dataset(filename, mode=mode, clobber=clobber,
                              diskless=diskless, persist=persist,
                              format=format)

    @property
    def variables(self):
        def convert_variable(var):
            var.set_auto_maskandscale(False)
            dimensions = var.dimensions
            data = var
            if var.ndim == 0:
                # work around for netCDF4-python's broken handling of 0-d
                # arrays (slicing them always returns a 1-dimensional array):
                # https://github.com/Unidata/netcdf4-python/pull/220
                data = np.asscalar(var[...])
            attributes = OrderedDict((k, var.getncattr(k))
                                     for k in var.ncattrs())
            # netCDF4 specific encoding; save _FillValue for later
            encoding = {}
            filters = var.filters()
            if filters is not None:
                encoding.update(filters)
            chunking = var.chunking()
            if chunking is not None:
                if chunking == 'contiguous':
                    encoding['contiguous'] = True
                    encoding['chunksizes'] = None
                else:
                    encoding['contiguous'] = False
                    encoding['chunksizes'] = tuple(chunking)
            encoding['endian'] = var.endian()
            encoding['least_significant_digit'] = \
                attributes.pop('least_significant_digit', None)
            return xarray.XArray(dimensions, data, attributes, encoding,
                                 indexing_mode='orthogonal')
        return FrozenOrderedDict((k, convert_variable(v))
                                 for k, v in self.ds.variables.iteritems())

    @property
    def attributes(self):
        return FrozenOrderedDict((k, self.ds.getncattr(k))
                                 for k in self.ds.ncattrs())

    @property
    def dimensions(self):
        return FrozenOrderedDict((k, len(v))
                                 for k, v in self.ds.dimensions.iteritems())

    def set_dimension(self, name, length):
        self.ds.createDimension(name, size=length)

    def set_attribute(self, key, value):
        self.ds.setncatts({key: value})

    def set_variable(self, name, variable):
        variable = encode_cf_variable(variable)
        self.set_necessary_dimensions(variable)
        # netCDF4 will automatically assign a fill value
        # depending on the datatype of the variable.  Here
        # we let the package handle the _FillValue attribute
        # instead of setting it ourselves.
        fill_value = variable.attributes.pop('_FillValue', None)
        encoding = variable.encoding
        self.ds.createVariable(
            varname=name,
            datatype=variable.dtype,
            dimensions=variable.dimensions,
            zlib=encoding.get('zlib', False),
            complevel=encoding.get('complevel', 4),
            shuffle=encoding.get('shuffle', True),
            fletcher32=encoding.get('fletcher32', False),
            contiguous=encoding.get('contiguous', False),
            chunksizes=encoding.get('chunksizes'),
            endian=encoding.get('endian', 'native'),
            least_significant_digit=encoding.get('least_significant_digit'),
            fill_value=fill_value)
        nc4_var = self.ds.variables[name]
        nc4_var.set_auto_maskandscale(False)
        if variable.data.ndim == 0:
            nc4_var[:] = variable.data
        else:
            nc4_var[:] = variable.data[:]
        nc4_var.setncatts(variable.attributes)

    def del_attribute(self, key):
        self.ds.delncattr(key)

    def sync(self):
        self.ds.sync()
