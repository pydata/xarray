from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import operator

import numpy as np

from .. import conventions
from .. import Variable
from ..conventions import pop_to
from ..core import indexing
from ..core.utils import (FrozenOrderedDict, close_on_error, is_remote_uri)
from ..core.pycompat import iteritems, basestring, OrderedDict, PY3, suppress

from .common import (WritableCFDataStore, robust_getitem, BackendArray,
                     DataStorePickleMixin, find_root)
from .netcdf3 import (encode_nc3_attr_value, encode_nc3_variable)

# This lookup table maps from dtype.byteorder to a readable endian
# string used by netCDF4.
_endian_lookup = {'=': 'native',
                  '>': 'big',
                  '<': 'little',
                  '|': 'native'}


class BaseNetCDF4Array(BackendArray):
    def __init__(self, variable_name, datastore):
        self.datastore = datastore
        self.variable_name = variable_name

        array = self.get_array()
        self.shape = array.shape

        dtype = array.dtype
        if dtype is str:
            # use object dtype because that's the only way in numpy to
            # represent variable length strings; it also prevents automatic
            # string concatenation via conventions.decode_cf_variable
            dtype = np.dtype('O')
        self.dtype = dtype

    def get_array(self):
        self.datastore.assert_open()
        return self.datastore.ds.variables[self.variable_name]


class NetCDF4ArrayWrapper(BaseNetCDF4Array):
    def __getitem__(self, key):
        key = indexing.unwrap_explicit_indexer(
            key, self, allow=(indexing.BasicIndexer, indexing.OuterIndexer))

        if self.datastore.is_remote:  # pragma: no cover
            getitem = functools.partial(robust_getitem, catch=RuntimeError)
        else:
            getitem = operator.getitem

        with self.datastore.ensure_open(autoclose=True):
            try:
                data = getitem(self.get_array(), key)
            except IndexError:
                # Catch IndexError in netCDF4 and return a more informative
                # error message.  This is most often called when an unsorted
                # indexer is used before the data is loaded from disk.
                msg = ('The indexing operation you are attempting to perform '
                       'is not valid on netCDF4.Variable object. Try loading '
                       'your data into memory first by calling .load().')
                if not PY3:
                    import traceback
                    msg += '\n\nOriginal traceback:\n' + traceback.format_exc()
                raise IndexError(msg)

        return data


def _nc4_values_and_dtype(var):
    if var.dtype.kind == 'U':
        dtype = str
    elif var.dtype.kind == 'S':
        # use character arrays instead of unicode, because unicode support in
        # netCDF4 is still rather buggy
        var = conventions.maybe_encode_as_char_array(var)
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
                                  "this is not supported by the netCDF4 "
                                  "python library.")
    return var


def _extract_nc4_variable_encoding(variable, raise_on_invalid=False,
                                   lsd_okay=True, backend='netCDF4',
                                   unlimited_dims=None):
    if unlimited_dims is None:
        unlimited_dims = ()

    encoding = variable.encoding.copy()

    safe_to_drop = set(['source', 'original_shape'])
    valid_encodings = set(['zlib', 'complevel', 'fletcher32', 'contiguous',
                           'chunksizes', 'shuffle'])
    if lsd_okay:
        valid_encodings.add('least_significant_digit')

    if not raise_on_invalid and encoding.get('chunksizes') is not None:
        # It's possible to get encoded chunksizes larger than a dimension size
        # if the original file had an unlimited dimension. This is problematic
        # if the new file no longer has an unlimited dimension.
        chunksizes = encoding['chunksizes']
        chunks_too_big = any(
            c > d and dim not in unlimited_dims
            for c, d, dim in zip(chunksizes, variable.shape, variable.dims))
        changed_shape = encoding.get('original_shape') != variable.shape
        if chunks_too_big or changed_shape:
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

    _disable_auto_decode_group(ds)

    return ds


def _disable_auto_decode_variable(var):
    """Disable automatic decoding on a netCDF4.Variable.

    We handle these types of decoding ourselves.
    """
    var.set_auto_maskandscale(False)

    # only added in netCDF4-python v1.2.8
    with suppress(AttributeError):
        var.set_auto_chartostring(False)


def _disable_auto_decode_group(ds):
    """Disable automatic decoding on all variables in a netCDF4.Group."""
    for var in ds.variables.values():
        _disable_auto_decode_variable(var)


class NetCDF4DataStore(WritableCFDataStore, DataStorePickleMixin):
    """Store for reading and writing data via the Python-NetCDF4 library.

    This store supports NetCDF3, NetCDF4 and OpenDAP datasets.
    """
    def __init__(self, netcdf4_dataset, mode='r', writer=None, opener=None,
                 autoclose=False):

        if autoclose and opener is None:
            raise ValueError('autoclose requires an opener')

        _disable_auto_decode_group(netcdf4_dataset)

        self.ds = netcdf4_dataset
        self._autoclose = autoclose
        self._isopen = True
        self.format = self.ds.data_model
        self._filename = self.ds.filepath()
        self.is_remote = is_remote_uri(self._filename)
        self._mode = mode = 'a' if mode == 'w' else mode
        if opener:
            self._opener = functools.partial(opener, mode=self._mode)
        else:
            self._opener = opener
        super(NetCDF4DataStore, self).__init__(writer)

    @classmethod
    def open(cls, filename, mode='r', format='NETCDF4', group=None,
             writer=None, clobber=True, diskless=False, persist=False,
             autoclose=False):
        if format is None:
            format = 'NETCDF4'
        opener = functools.partial(_open_netcdf4_group, filename, mode=mode,
                                   group=group, clobber=clobber,
                                   diskless=diskless, persist=persist,
                                   format=format)
        ds = opener()
        return cls(ds, mode=mode, writer=writer, opener=opener,
                   autoclose=autoclose)

    def open_store_variable(self, name, var):
        with self.ensure_open(autoclose=False):
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
        with self.ensure_open(autoclose=False):
            dsvars = FrozenOrderedDict((k, self.open_store_variable(k, v))
                                       for k, v in
                                       iteritems(self.ds.variables))
        return dsvars

    def get_attrs(self):
        with self.ensure_open(autoclose=True):
            attrs = FrozenOrderedDict((k, self.ds.getncattr(k))
                                      for k in self.ds.ncattrs())
        return attrs

    def get_dimensions(self):
        with self.ensure_open(autoclose=True):
            dims = FrozenOrderedDict((k, len(v))
                                     for k, v in iteritems(self.ds.dimensions))
        return dims

    def get_encoding(self):
        with self.ensure_open(autoclose=True):
            encoding = {}
            encoding['unlimited_dims'] = {
                k for k, v in self.ds.dimensions.items() if v.isunlimited()}
        return encoding

    def set_dimension(self, name, length, is_unlimited=False):
        with self.ensure_open(autoclose=False):
            dim_length = length if not is_unlimited else None
            self.ds.createDimension(name, size=dim_length)

    def set_attribute(self, key, value):
        with self.ensure_open(autoclose=False):
            if self.format != 'NETCDF4':
                value = encode_nc3_attr_value(value)
            self.ds.setncattr(key, value)

    def set_variables(self, *args, **kwargs):
        with self.ensure_open(autoclose=False):
            super(NetCDF4DataStore, self).set_variables(*args, **kwargs)

    def prepare_variable(self, name, variable, check_encoding=False,
                         unlimited_dims=None):
        variable = _force_native_endianness(variable)

        if self.format == 'NETCDF4':
            variable, datatype = _nc4_values_and_dtype(variable)
        else:
            variable = encode_nc3_variable(variable)
            datatype = variable.dtype

        self.set_necessary_dimensions(variable, unlimited_dims=unlimited_dims)

        attrs = variable.attrs.copy()

        fill_value = attrs.pop('_FillValue', None)

        if datatype is str and fill_value is not None:
            raise NotImplementedError(
                'netCDF4 does not yet support setting a fill value for '
                'variable-length strings '
                '(https://github.com/Unidata/netcdf4-python/issues/730). '
                "Either remove '_FillValue' from encoding on variable %r "
                "or set {'dtype': 'S1'} in encoding to use the fixed width "
                'NC_CHAR type.' % name)

        encoding = _extract_nc4_variable_encoding(
            variable, raise_on_invalid=check_encoding,
            unlimited_dims=unlimited_dims)
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
        _disable_auto_decode_variable(nc4_var)

        for k, v in iteritems(attrs):
            # set attributes one-by-one since netCDF4<1.0.10 can't handle
            # OrderedDict as the input to setncatts
            nc4_var.setncattr(k, v)

        return nc4_var, variable.data

    def sync(self):
        with self.ensure_open(autoclose=True):
            super(NetCDF4DataStore, self).sync()
            self.ds.sync()

    def close(self):
        if self._isopen:
            # netCDF4 only allows closing the root group
            ds = find_root(self.ds)
            if ds._isopen:
                ds.close()
            self._isopen = False
