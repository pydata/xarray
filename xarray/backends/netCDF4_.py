from __future__ import absolute_import, division, print_function

import functools
import operator
import warnings
from distutils.version import LooseVersion

import numpy as np

from .. import Variable, coding
from ..coding.variables import pop_to
from ..core import indexing
from ..core.pycompat import (
    PY3, OrderedDict, basestring, iteritems, suppress)
from ..core.utils import FrozenOrderedDict, close_on_error, is_remote_uri
from .common import (
    HDF5_LOCK, BackendArray, DataStorePickleMixin, WritableCFDataStore,
    find_root, robust_getitem)
from .netcdf3 import encode_nc3_attr_value, encode_nc3_variable

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

    def __setitem__(self, key, value):
        with self.datastore.ensure_open(autoclose=True):
            data = self.get_array()
            data[key] = value

    def get_array(self):
        self.datastore.assert_open()
        return self.datastore.ds.variables[self.variable_name]


class NetCDF4ArrayWrapper(BaseNetCDF4Array):
    def __getitem__(self, key):
        key, np_inds = indexing.decompose_indexer(
            key, self.shape, indexing.IndexingSupport.OUTER)
        if self.datastore.is_remote:  # pragma: no cover
            getitem = functools.partial(robust_getitem, catch=RuntimeError)
        else:
            getitem = operator.getitem

        with self.datastore.ensure_open(autoclose=True):
            try:
                array = getitem(self.get_array(), key.tuple)
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

        if len(np_inds.tuple) > 0:
            array = indexing.NumpyIndexingAdapter(array)[np_inds]

        return array


def _encode_nc4_variable(var):
    for coder in [coding.strings.EncodedStringCoder(allows_unicode=True),
                  coding.strings.CharacterArrayCoder()]:
        var = coder.encode(var)
    return var


def _check_encoding_dtype_is_vlen_string(dtype):
    if dtype is not str:
        raise AssertionError(  # pragma: no cover
            "unexpected dtype encoding %r. This shouldn't happen: please "
            "file a bug report at github.com/pydata/xarray" % dtype)


def _get_datatype(var, nc_format='NETCDF4', raise_on_invalid_encoding=False):
    if nc_format == 'NETCDF4':
        datatype = _nc4_dtype(var)
    else:
        if 'dtype' in var.encoding:
            encoded_dtype = var.encoding['dtype']
            _check_encoding_dtype_is_vlen_string(encoded_dtype)
            if raise_on_invalid_encoding:
                raise ValueError(
                    'encoding dtype=str for vlen strings is only supported '
                    'with format=\'NETCDF4\'.')
        datatype = var.dtype
    return datatype


def _nc4_dtype(var):
    if 'dtype' in var.encoding:
        dtype = var.encoding.pop('dtype')
        _check_encoding_dtype_is_vlen_string(dtype)
    elif coding.strings.is_unicode_dtype(var.dtype):
        dtype = str
    elif var.dtype.kind in ['i', 'u', 'f', 'c', 'S']:
        dtype = var.dtype
    else:
        raise ValueError('unsupported dtype for netCDF4 variable: {}'
                         .format(var.dtype))
    return dtype


def _netcdf4_create_group(dataset, name):
    return dataset.createGroup(name)


def _nc4_require_group(ds, group, mode, create_group=_netcdf4_create_group):
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
                    ds = create_group(ds, key)
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
                                   lsd_okay=True, h5py_okay=False,
                                   backend='netCDF4', unlimited_dims=None):
    if unlimited_dims is None:
        unlimited_dims = ()

    encoding = variable.encoding.copy()

    safe_to_drop = set(['source', 'original_shape'])
    valid_encodings = set(['zlib', 'complevel', 'fletcher32', 'contiguous',
                           'chunksizes', 'shuffle', '_FillValue', 'dtype'])
    if lsd_okay:
        valid_encodings.add('least_significant_digit')
    if h5py_okay:
        valid_encodings.add('compression')
        valid_encodings.add('compression_opts')

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
        ds = _nc4_require_group(ds, group, mode)

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


def _is_list_of_strings(value):
    if (np.asarray(value).dtype.kind in ['U', 'S'] and
            np.asarray(value).size > 1):
        return True
    else:
        return False


def _set_nc_attribute(obj, key, value):
    if _is_list_of_strings(value):
        # encode as NC_STRING if attr is list of strings
        try:
            obj.setncattr_string(key, value)
        except AttributeError:
            # Inform users with old netCDF that does not support
            # NC_STRING that we can't serialize lists of strings
            # as attrs
            msg = ('Attributes which are lists of strings are not '
                   'supported with this version of netCDF. Please '
                   'upgrade to netCDF4-python 1.2.4 or greater.')
            raise AttributeError(msg)
    else:
        obj.setncattr(key, value)


class NetCDF4DataStore(WritableCFDataStore, DataStorePickleMixin):
    """Store for reading and writing data via the Python-NetCDF4 library.

    This store supports NetCDF3, NetCDF4 and OpenDAP datasets.
    """

    def __init__(self, netcdf4_dataset, mode='r', writer=None, opener=None,
                 autoclose=False, lock=HDF5_LOCK):

        if autoclose and opener is None:
            raise ValueError('autoclose requires an opener')

        _disable_auto_decode_group(netcdf4_dataset)

        self._ds = netcdf4_dataset
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
        super(NetCDF4DataStore, self).__init__(writer, lock=lock)

    @classmethod
    def open(cls, filename, mode='r', format='NETCDF4', group=None,
             writer=None, clobber=True, diskless=False, persist=False,
             autoclose=False, lock=HDF5_LOCK):
        import netCDF4 as nc4
        if (len(filename) == 88 and
                LooseVersion(nc4.__version__) < "1.3.1"):
            warnings.warn(
                'A segmentation fault may occur when the '
                'file path has exactly 88 characters as it does '
                'in this case. The issue is known to occur with '
                'version 1.2.4 of netCDF4 and can be addressed by '
                'upgrading netCDF4 to at least version 1.3.1. '
                'More details can be found here: '
                'https://github.com/pydata/xarray/issues/1745')
        if format is None:
            format = 'NETCDF4'
        opener = functools.partial(_open_netcdf4_group, filename, mode=mode,
                                   group=group, clobber=clobber,
                                   diskless=diskless, persist=persist,
                                   format=format)
        ds = opener()
        return cls(ds, mode=mode, writer=writer, opener=opener,
                   autoclose=autoclose, lock=lock)

    def open_store_variable(self, name, var):
        with self.ensure_open(autoclose=False):
            dimensions = var.dimensions
            data = indexing.LazilyOuterIndexedArray(
                NetCDF4ArrayWrapper(name, self))
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
            encoding['dtype'] = var.dtype

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
            _set_nc_attribute(self.ds, key, value)

    def set_variables(self, *args, **kwargs):
        with self.ensure_open(autoclose=False):
            super(NetCDF4DataStore, self).set_variables(*args, **kwargs)

    def encode_variable(self, variable):
        variable = _force_native_endianness(variable)
        if self.format == 'NETCDF4':
            variable = _encode_nc4_variable(variable)
        else:
            variable = encode_nc3_variable(variable)
        return variable

    def prepare_variable(self, name, variable, check_encoding=False,
                         unlimited_dims=None):
        datatype = _get_datatype(variable, self.format,
                                 raise_on_invalid_encoding=check_encoding)
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
        if name in self.ds.variables:
            nc4_var = self.ds.variables[name]
        else:
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
                least_significant_digit=encoding.get(
                    'least_significant_digit'),
                fill_value=fill_value)
            _disable_auto_decode_variable(nc4_var)

        for k, v in iteritems(attrs):
            # set attributes one-by-one since netCDF4<1.0.10 can't handle
            # OrderedDict as the input to setncatts
            _set_nc_attribute(nc4_var, k, v)

        target = NetCDF4ArrayWrapper(name, self)

        return target, variable.data

    def sync(self, compute=True):
        with self.ensure_open(autoclose=True):
            super(NetCDF4DataStore, self).sync(compute=compute)
            self.ds.sync()

    def close(self):
        if self._isopen:
            # netCDF4 only allows closing the root group
            ds = find_root(self.ds)
            if ds._isopen:
                ds.close()
            self._isopen = False
