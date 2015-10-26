import warnings

import numpy as np

from .. import Variable
from ..conventions import pop_to, cf_encoder
from ..core import indexing
from ..core.utils import FrozenOrderedDict, NDArrayMixin, close_on_error
from ..core.pycompat import iteritems, basestring, OrderedDict

from .common import AbstractWritableDataStore
from .netcdf3 import (encode_nc3_attr_value, encode_nc3_variable,
                      maybe_convert_to_char_array)

# This lookup table maps from dtype.byteorder to a readable endian
# string used by netCDF4.
_endian_lookup = {'=': 'native',
                  '>': 'big',
                  '<': 'little',
                  '|': 'native'}


class NetCDF4ArrayWrapper(NDArrayMixin):
    def __init__(self, array):
        self.array = array

    @property
    def dtype(self):
        dtype = np.dtype(self.array.typecode())
        if dtype is str:
            # return object dtype because that's the only way in numpy to
            # represent variable length strings; it also prevents automatic
            # string concatenation via conventions.decode_cf_variable
            dtype = np.dtype('O')
        return dtype

    def __getitem__(self, key):
        print 'in pynio getitem', key
        print self.array.name
        
        if self.ndim == 0:
            # work around for netCDF4-python's broken handling of 0-d
            # arrays (slicing them always returns a 1-dimensional array):
            # https://github.com/Unidata/netcdf4-python/pull/220
            
            data = np.asscalar(self.array.get_value())
            #data = None
        else:
            data = self.array[key]
            print data
        return data


def _nc4_values_and_dtype(var):
    if var.dtype.kind == 'U':
        # this entire clause should not be necessary with netCDF4>=1.0.9
        if len(var) > 0:
            var = var.astype('O')
        dtype = str
    elif var.dtype.kind in ['i', 'u', 'f', 'S']:
        # use character arrays instead of unicode, because unicode suppot in
        # netCDF4 is still rather buggy
        data, dims = maybe_convert_to_char_array(var.data, var.dims)
        var = Variable(dims, data, var.attrs, var.encoding)
        dtype = var.dtype
    else:
        raise ValueError('cannot infer dtype for netCDF4 variable')
    return var, dtype


def _nc4_group(ds, group, mode):
    if group in set([None, '', '/']):
        # use the root group
        return ds
    else:
        # make sure it's a string
        if not isinstance(group, basestring):
            raise ValueError('group must be a string or None')
        # support path-like syntax
        path = group.strip('/').split('/')
        for key in path:
            try:
                ds = ds.groups[key]
            except KeyError as e:
                if mode != 'r':
                    ds = ds.createGroup(key)
                else:
                    # wrap error to provide slightly more helpful message
                    raise IOError('group not found: %s' % key, e)
        return ds


def _ensure_fill_value_valid(data, attributes):
    # work around for netCDF4/scipy issue where _FillValue has the wrong type:
    # https://github.com/Unidata/netcdf4-python/issues/271
    if data.dtype.kind == 'S' and '_FillValue' in attributes:
        attributes['_FillValue'] = np.string_(attributes['_FillValue'])


def _force_native_endianness(var):
    # possible values for byteorder are:
    #     =    native
    #     <    little-endian
    #     >    big-endian
    #     |    not applicable
    # Below we check if the data type is not native or NA
    if var.dtype.byteorder not in ['=', '|']:
        # if endianness is specified explicitly, convert to the native type
        data = var.data.astype(var.dtype.newbyteorder('='))
        var = Variable(var.dims, data, var.attrs, var.encoding)
        # if endian exists, remove it from the encoding.
        var.encoding.pop('endian', None)
    # check to see if encoding has a value for endian its 'native'
    if not var.encoding.get('endian', 'native') is 'native':
        raise NotImplementedError("Attempt to write non-native endian type, "
                                  "this is not supported by the netCDF4 python "
                                  "library.")
    return var


class NioDataStore(AbstractWritableDataStore):
    """Store for reading and writing data via the PyNIO library.

    This store supports NetCDF3, NetCDF4 and OpenDAP datasets.
    """
    def __init__(self, filename, mode='r', clobber=True, diskless=False,
                 persist=False, format='NETCDF4', group=None):
        import Nio
        ds = Nio.open_file(filename, mode=mode)

#, clobber=clobber,
#                         diskless=diskless, persist=persist,
#                         format=format)
        with close_on_error(ds):
            self.ds = _nc4_group(ds, group, mode)
        self.format = format
        self._filename = filename

    def store(self, variables, attributes):
        # All NetCDF files get CF encoded by default, without this attempting
        # to write times, for example, would fail.
        cf_variables, cf_attrs = cf_encoder(variables, attributes)
        AbstractWritableDataStore.store(self, cf_variables, cf_attrs)

    def open_store_variable(self, var):
        #var.set_auto_maskandscale(False)
        dimensions = var.dimensions
        data = indexing.LazilyIndexedArray(NetCDF4ArrayWrapper(var))
        attributes = OrderedDict((k, var.attributes[k])
                                 for k in var.attributes.keys())
        _ensure_fill_value_valid(data, attributes)
        # netCDF4 specific encoding; save _FillValue for later
        encoding = {}
        #filters = var.filters()
        filters = None
        if filters is not None:
            encoding.update(filters)
        if hasattr(var,'chunk_dimensions'):
            chunking = var.chunk_dimensions
        else:
            chunking = None
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
        #pop_to(attributes, encoding, 'least_significant_digit')
        # save source so __repr__ can detect if it's local or not
        encoding['source'] = self._filename
        return Variable(dimensions, data, attributes, encoding)

    def get_variables(self):
        return FrozenOrderedDict((k, self.open_store_variable(v))
                                 for k, v in iteritems(self.ds.variables))

    def get_attrs(self):
        print self.ds.attributes
        return FrozenOrderedDict((k, self.ds.attributes[k])
                                 for k in self.ds.attributes.keys())

    def get_dimensions(self):
        return FrozenOrderedDict((k, len(v))
                                 for k, v in iteritems(self.ds.dimensions))

    def set_dimension(self, name, length):
        self.ds.createDimension(name, size=length)

    def set_attribute(self, key, value):
        if self.format != 'NETCDF4':
            value = encode_nc3_attr_value(value)
        self.ds.setncattr(key, value)

    def prepare_variable(self, name, variable):
        attrs = variable.attrs.copy()

        variable = _force_native_endianness(variable)

        if self.format == 'NETCDF4':
            variable, datatype = _nc4_values_and_dtype(variable)
        else:
            variable = encode_nc3_variable(variable)
            datatype = variable.dtype

        self.set_necessary_dimensions(variable)

        fill_value = attrs.pop('_FillValue', None)
        if fill_value in ['', '\x00']:
            # these are equivalent to the default FillValue, but netCDF4
            # doesn't like setting fill_value to an empty string
            fill_value = None

        encoding = variable.encoding
        nc4_var = self.ds.createVariable(
            varname=name,
            datatype=datatype,
            dimensions=variable.dims,
            zlib=encoding.get('zlib', False),
            complevel=encoding.get('complevel', 4),
            shuffle=encoding.get('shuffle', True),
            fletcher32=encoding.get('fletcher32', False),
            contiguous=encoding.get('contiguous', False),
            chunksizes=encoding.get('chunksizes'),
            endian='native',
            least_significant_digit=encoding.get('least_significant_digit'),
            fill_value=fill_value)
        nc4_var.set_auto_maskandscale(False)

        for k, v in iteritems(attrs):
            # set attributes one-by-one since netCDF4<1.0.10 can't handle
            # OrderedDict as the input to setncatts
            nc4_var.setncattr(k, v)
        return nc4_var, variable.data

    def sync(self):
        self.ds.sync()

    def close(self):
        ds = self.ds
        # netCDF4 only allows closing the root group
        while ds.parent is not None:
            ds = ds.parent
        ds.close()
