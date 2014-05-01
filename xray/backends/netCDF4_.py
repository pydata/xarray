from collections import OrderedDict
import warnings

import numpy as np

from common import AbstractWritableDataStore
import xray
from xray.conventions import encode_cf_variable
from xray.utils import FrozenOrderedDict, NDArrayMixin, as_array_or_item
from xray import indexing


class NetCDF4ArrayWrapper(NDArrayMixin):
    def __init__(self, array):
        self.array = array

    @property
    def dtype(self):
        dtype = self.array.dtype
        if dtype is str:
            # return object dtype because that's the only way in numpy to
            # represent variable length strings; it also prevents automatic
            # string concatenation via conventions.decode_cf_variable
            dtype = np.dtype('O')
        return dtype

    def __getitem__(self, key):
        if self.ndim == 0:
            # work around for netCDF4-python's broken handling of 0-d
            # arrays (slicing them always returns a 1-dimensional array):
            # https://github.com/Unidata/netcdf4-python/pull/220
            data = as_array_or_item(np.asscalar(self.array[key]))
        else:
            data = self.array[key]
        return data


class NetCDF4DataStore(AbstractWritableDataStore):
    """Store for reading and writing data via the Python-NetCDF4 library.

    This store supports NetCDF3, NetCDF4 and OpenDAP datasets.
    """
    def __init__(self, filename, mode='r', clobber=True, diskless=False,
                 persist=False, format='NETCDF4'):
        import netCDF4 as nc4
        if nc4.__version__ < (1, 0, 6):
            warnings.warn('python-netCDF4 %s detected; '
                          'the minimal recommended version is 1.0.6.'
                          % nc4.__version__, ImportWarning)

        self.ds = nc4.Dataset(filename, mode=mode, clobber=clobber,
                              diskless=diskless, persist=persist,
                              format=format)

    def open_store_variable(self, var):
        var.set_auto_maskandscale(False)
        dimensions = var.dimensions
        data = indexing.LazilyIndexedArray(NetCDF4ArrayWrapper(var))
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
        # TODO: figure out how to round-trip "endian-ness" without raising
        # warnings from netCDF4
        # encoding['endian'] = var.endian()
        encoding['least_significant_digit'] = \
            attributes.pop('least_significant_digit', None)
        return xray.Variable(dimensions, data, attributes, encoding)

    @property
    def attrs(self):
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
        fill_value = variable.attrs.pop('_FillValue', None)
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
        if variable.values.ndim == 0:
            nc4_var[:] = variable.values
        else:
            nc4_var[:] = variable.values[:]
        nc4_var.setncatts(variable.attrs)

    def del_attribute(self, key):
        self.ds.delncattr(key)

    def sync(self):
        self.ds.sync()

    def close(self):
        self.ds.close()

    def __exit__(self, type, value, tb):
        self.close()
