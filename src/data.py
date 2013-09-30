import os
import netCDF4 as nc4

from scipy.io import netcdf
from cStringIO import StringIO
from collections import OrderedDict

class Attributes(dict):
    pass

class Variable(object):
    """
    A netcdf-like variable consisting of dimensions, data and attributes
    which describe a single variable.  A single variable object is not
    fully described outside the context of its parent Dataset.
    """
    def __init__(self, dims, data, attributes):
        self.dimensions = dims
        self.data = data
        self.attributes = attributes

    def __getattribute__(self, key):
        """
        We want Variable to inherit some of the attributes of
        the underlaying data.
        """
        if key in ['dtype', 'shape', 'size']:
            return getattr(self.data, key)
        else:
            return object.__getattribute__(self, key)

class Dataset(object):
    """
    A netcdf-like data object consisting of dimensions, variables and
    attributes which together form a self describing data set.
    """

    def _load_scipy(self, scipy_nc, *args, **kwdargs):
        """
        Interprets a netcdf file-like object using scipy.io.netcdf.
        The contents of the netcdf object are loaded into memory.
        """
        try:
            nc = netcdf.netcdf_file(scipy_nc, mode='r', *args, **kwdargs)
        except:
            scipy_nc = StringIO(scipy_nc)
            scipy_nc.seek(0)
            nc = netcdf.netcdf_file(scipy_nc, mode='r', *args, **kwdargs)

        def from_scipy_variable(sci_var):
            return Variable(dims = sci_var.dimensions,
                            data = sci_var.data,
                            attributes = sci_var._attributes)

        object.__setattr__(self, 'attributes', Attributes())
        self.attributes.update(nc._attributes)

        object.__setattr__(self, 'dimensions', OrderedDict())
        dimensions = OrderedDict((k, len(d))
                                 for k, d in nc.dimensions.iteritems())
        self.dimensions.update(dimensions)

        object.__setattr__(self, 'variables', OrderedDict())
        OrderedDict = OrderedDict((vn, from_scipy_variable(v))
                                   for vn, v in nc.variables.iteritems())
        self.variables.update()

    def _load_netcdf4(self, netcdf_path, *args, **kwdargs):
        """
        Interprets the contents of netcdf_path using the netCDF4
        package.
        """
        nc = nc4.Dataset(netcdf_path, *args, **kwdargs)

        def from_netcdf4_variable(nc4_var):
            attributes = dict((k, nc4_var.getncattr(k)) for k in nc4_var.ncattrs())
            return Variable(dims = tuple(nc4_var.dimensions),
                            data = nc4_var[:],
                            attributes = attributes)

        object.__setattr__(self, 'attributes', Attributes())
        self.attributes.update(dict((k.encode(), nc.getncattr(k)) for k in nc.ncattrs()))

        object.__setattr__(self, 'dimensions', OrderedDict())
        dimensions = OrderedDict((k.encode(), len(d)) for k, d in nc.dimensions.iteritems())
        self.dimensions.update(dimensions)

        object.__setattr__(self, 'variables', OrderedDict())
        self.variables.update(dict((vn.encode(), from_netcdf4_variable(v))
                                   for vn, v in nc.variables.iteritems()))

    def __init__(self, nc, *args, **kwdargs):
        if isinstance(nc, basestring) and not nc.startswith('CDF'):
            """
            If the initialization nc is a string and it doesn't
            appear to be the contents of a netcdf file we load
            it using the netCDF4 package
            """
            self._load_netcdf4(nc, *args, **kwdargs)
        else:
            """
            If nc is a file-like object we read it using
            the scipy.io.netcdf package
            """
            self._load_scipy(nc)

    def __setattr__(self, attr, value):
        """"__setattr__ is overloaded to prevent operations that could
        cause loss of data consistency. If you really intend to update
        dir(self), use the self.__dict__.update method or the
        super(type(a), self).__setattr__ method to bypass."""
        raise AttributeError("__setattr__ is disabled")

    def dump(self, filepath, *args, **kwdargs):
        """
        Dump the contents to a location on disk using
        the netCDF4 package
        """
        nc = nc4.Dataset(filepath, mode='w', *args, **kwdargs)
        for d, l in self.dimensions.iteritems():
            nc.createDimension(d, size=l)
        for vn, v in self.variables.iteritems():
            nc.createVariable(vn, v.dtype, v.dimensions)
            nc.variables[vn][:] = v.data[:]
            for k, a in v.attributes.iteritems():
                try:
                    nc.variables[vn].setncattr(k, a)
                except:
                    import pdb; pdb.set_trace()

        nc.setncatts(self.attributes)
        return nc

    def dumps(self):
        """
        Serialize the contents to a string.  The serialization
        creates an in memory netcdf version 3 string using
        the scipy.io.netcdf package.
        """
        fobj = StringIO()
        nc = netcdf.netcdf_file(fobj, mode='w')
        for d, l in self.dimensions.iteritems():
            nc.createDimension(d, l)

        for vn, v in self.variables.iteritems():

            nc.createVariable(vn, v.dtype, v.dimensions)
            nc.variables[vn][:] = v.data[:]
            for k, a in v.attributes.iteritems():
                setattr(nc.variables[vn], k, a)
        for k, a in self.attributes.iteritems():
            setattr(nc, k, a)
        nc.flush()
        return fobj.getvalue()

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    test_dir = os.path.join(base_dir, '..', 'test', )
    write_test_path = os.path.join(test_dir, 'test_output.nc')
    ecmwf_netcdf = os.path.join(test_dir, 'ECMWF_ERA-40_subset.nc')

    import time
    st = time.time()
    nc = Dataset(ecmwf_netcdf)
    print "Seconds to read from filepath : ", time.time() - st

    st = time.time()
    nc.dump(write_test_path)
    print "Seconds to write : ", time.time() - st

    st = time.time()
    nc_string = nc.dumps()
    print "Seconds to serialize : ", time.time() - st

    st = time.time()
    nc = Dataset(nc_string)
    print "Seconds to deserialize : ", time.time() - st

    st = time.time()
    with open(ecmwf_netcdf, 'r') as f:
        nc = Dataset(f)
    print "Seconds to read from fobj : ", time.time() - st

