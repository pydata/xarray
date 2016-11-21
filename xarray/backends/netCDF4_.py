from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import operator

import numpy as np

from .. import Variable
from ..conventions import pop_to, cf_encoder
from ..core import indexing
from ..core.utils import (FrozenOrderedDict, NDArrayMixin,
                          close_on_error, is_remote_uri)
from ..core.pycompat import iteritems, basestring, OrderedDict, PY3

from .common import WritableCFDataStore, robust_getitem, DataStorePickleMixin
from .netcdf3 import (encode_nc3_attr_value, encode_nc3_variable,
                      maybe_convert_to_char_array)

# This lookup table maps from dtype.byteorder to a readable endian
# string used by netCDF4.
_endian_lookup = {'=': 'native',
                  '>': 'big',
                  '<': 'little',
                  '|': 'native'}


class BaseNetCDF4Array(NDArrayMixin):
    def __init__(self, variable_name, datastore):
        self.datastore = datastore
        self.variable_name = variable_name

    @property
    def array(self):
        return self.datastore.ds.variables[self.variable_name]

    @property
    def dtype(self):
        dtype = self.array.dtype
        if dtype is str:
            # return object dtype because that's the only way in numpy to
            # represent variable length strings; it also prevents automatic
            # string concatenation via conventions.decode_cf_variable
            dtype = np.dtype('O')
        return dtype


class NetCDF4ArrayWrapper(BaseNetCDF4Array):
    def __getitem__(self, key):
        if self.datastore.is_remote:  # pragma: no cover
            getitem = functools.partial(robust_getitem, catch=RuntimeError)
        else:
            getitem = operator.getitem

        try:
            data = getitem(self.array, key)
        except IndexError:
            # Catch IndexError in netCDF4 and return a more informative error
            # message.  This is most often called when an unsorted indexer is
            # used before the data is loaded from disk.
            msg = ('The indexing operation you are attempting to perform is '
                   'not valid on netCDF4.Variable object. Try loading your '
                   'data into memory first by calling .load().')
            if not PY3:
                import traceback
                msg += '\n\nOriginal traceback:\n' + traceback.format_exc()
            raise IndexError(msg)

        if self.ndim == 0:
            # work around for netCDF4-python's broken handling of 0-d
            # arrays (slicing them always returns a 1-dimensional array):
            # https://github.com/Unidata/netcdf4-python/pull/220
            data = np.asscalar(data)
        return data


def _nc4_values_and_dtype(var):
    if var.dtype.kind == 'U':
        # this entire clause should not be necessary with netCDF4>=1.0.9
        if len(var) > 0:
            var = var.astype('O')
        dtype = str
    elif var.dtype.kind == 'S':
        # use character arrays instead of unicode, because unicode support in
        # netCDF4 is still rather buggy
        data, dims = maybe_convert_to_char_array(var.data, var.dims)
        var = Variable(dims, data, var.attrs, var.encoding)
        dtype = var.dtype
    elif var.dtype.kind in ['i', 'u', 'f', 'c']:
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


def _extract_nc4_encoding(variable, raise_on_invalid=False, lsd_okay=True,
                          backend='netCDF4'):
    encoding = variable.encoding.copy()

    safe_to_drop = set(['source', 'original_shape'])
    valid_encodings = set(['zlib', 'complevel', 'fletcher32', 'contiguous',
                           'chunksizes'])
    if lsd_okay:
        valid_encodings.add('least_significant_digit')

    if (encoding.get('chunksizes') is not None and
            (encoding.get('original_shape', variable.shape)
             != variable.shape) and
            not raise_on_invalid):
        del encoding['chunksizes']

    for k in safe_to_drop:
        if k in encoding:
            del encoding[k]

    if raise_on_invalid:
        invalid = [k for k in encoding if k not in valid_encodings]
        if invalid:
            raise ValueError('unexpected encoding parameters for %r backend: '
                             ' %r' % (backend, invalid))
    else:
        for k in list(encoding):
            if k not in valid_encodings:
                del encoding[k]

    return encoding


def _open_netcdf4_group(filename, mode, group=None, **kwargs):
    import netCDF4 as nc4

    ds = nc4.Dataset(filename, mode=mode, **kwargs)

    with close_on_error(ds):
        ds = _nc4_group(ds, group, mode)

    for var in ds.variables.values():
        # we handle masking and scaling ourselves
        var.set_auto_maskandscale(False)
    return ds


class NetCDF4DataStore(WritableCFDataStore, DataStorePickleMixin):
    """Store for reading and writing data via the Python-NetCDF4 library.

    This store supports NetCDF3, NetCDF4 and OpenDAP datasets.
    """
    def __init__(self, filename, mode='r', format='NETCDF4', group=None,
                 writer=None, clobber=True, diskless=False, persist=False):
        if format is None:
            format = 'NETCDF4'
        opener = functools.partial(_open_netcdf4_group, filename, mode=mode,
                                   group=group, clobber=clobber,
                                   diskless=diskless, persist=persist,
                                   format=format)
        self.ds = opener()
        self.format = format
        self.is_remote = is_remote_uri(filename)
        self._opener = opener
        self._filename = filename
        self._mode = 'a' if mode == 'w' else mode
        super(NetCDF4DataStore, self).__init__(writer)

    def open_store_variable(self, name, var):
        dimensions = var.dimensions
        data = indexing.LazilyIndexedArray(NetCDF4ArrayWrapper(name, self))
        attributes = OrderedDict((k, var.getncattr(k))
                                 for k in var.ncattrs())
        _ensure_fill_value_valid(data, attributes)
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
        pop_to(attributes, encoding, 'least_significant_digit')
        # save source so __repr__ can detect if it's local or not
        encoding['source'] = self._filename
        encoding['original_shape'] = var.shape
        return Variable(dimensions, data, attributes, encoding)

    def get_variables(self):
        return FrozenOrderedDict((k, self.open_store_variable(k, v))
                                 for k, v in iteritems(self.ds.variables))

    def get_attrs(self):
        return FrozenOrderedDict((k, self.ds.getncattr(k))
                                 for k in self.ds.ncattrs())

    def get_dimensions(self):
        return FrozenOrderedDict((k, len(v))
                                 for k, v in iteritems(self.ds.dimensions))

    def set_dimension(self, name, length):
        self.ds.createDimension(name, size=length)

    def set_attribute(self, key, value):
        if self.format != 'NETCDF4':
            value = encode_nc3_attr_value(value)
        self.ds.setncattr(key, value)

    def prepare_variable(self, name, variable, check_encoding=False):
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

        encoding = _extract_nc4_encoding(variable,
                                         raise_on_invalid=check_encoding)
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
        super(NetCDF4DataStore, self).sync()
        self.ds.sync()

    def close(self):
        ds = self.ds
        # netCDF4 only allows closing the root group
        while ds.parent is not None:
            ds = ds.parent
        ds.close()
