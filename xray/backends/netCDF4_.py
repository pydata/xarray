import warnings

import numpy as np

from .common import AbstractWritableDataStore
from .netcdf3 import encode_nc3_variable
import xray
from xray.conventions import encode_cf_variable
from xray.utils import FrozenOrderedDict, NDArrayMixin
from xray import indexing
from xray.pycompat import iteritems, basestring, bytes_type, OrderedDict


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
            data = np.asscalar(self.array[key])
        else:
            data = self.array[key]
        return data


def _version_check(actual, required):
    actual_tup = tuple(int(p) if p.isdigit() else p for p in actual.split('.'))
    try:
        return actual_tup >= required
    except TypeError:
        return True


def _nc4_values_and_dtype(variable):
    if variable.dtype.kind in ['i', 'u', 'f'] or variable.dtype == 'S1':
        values = variable.values
        dtype = variable.dtype
    elif (variable.dtype.kind == 'U' or
              (variable.dtype.kind == 'S' and variable.dtype.itemsize > 1)):
        # this entire clause should not be necessary with netCDF4>=1.0.9
        if len(variable) > 0:
            values = variable.values.astype('O')
        else:
            values = variable.values
        dtype = str
    else:
        raise ValueError('cannot infer dtype for netCDF4 variable')
    return values, dtype


def _nc4_group(ds, group):
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
                # wrap error to provide slightly more helpful message
                raise IOError('group not found: %s' % key, e)
        return ds


def _ensure_fill_value_valid(data, attributes):
    # work around for netCDF4/scipy issue where _FillValue has the wrong type:
    # https://github.com/Unidata/netcdf4-python/issues/271
    if data.dtype.kind == 'S' and '_FillValue' in attributes:
        attributes['_FillValue'] = np.string_(attributes['_FillValue'])


class NetCDF4DataStore(AbstractWritableDataStore):
    """Store for reading and writing data via the Python-NetCDF4 library.

    This store supports NetCDF3, NetCDF4 and OpenDAP datasets.
    """
    def __init__(self, filename, mode='r', clobber=True, diskless=False,
                 persist=False, format='NETCDF4', group=None):
        import netCDF4 as nc4
        if not _version_check(nc4.__version__, (1, 0, 6)):
            warnings.warn('python-netCDF4 %s detected; '
                          'the minimal recommended version is 1.0.6.'
                          % nc4.__version__, ImportWarning)

        ds = nc4.Dataset(filename, mode=mode, clobber=clobber,
                         diskless=diskless, persist=persist,
                         format=format)
        # support use of groups
        self.ds = _nc4_group(ds, group)
        self.format = format

    def open_store_variable(self, var):
        var.set_auto_maskandscale(False)
        dimensions = var.dimensions
        data = indexing.LazilyIndexedArray(NetCDF4ArrayWrapper(var))
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
                                 for k, v in iteritems(self.ds.dimensions))

    def set_dimension(self, name, length):
        self.ds.createDimension(name, size=length)

    def set_attribute(self, key, value):
        self.ds.setncattr(key, value)

    def set_variable(self, name, variable):
        variable = encode_cf_variable(variable)
        if self.format == 'NETCDF4':
            values, datatype = _nc4_values_and_dtype(variable)
        else:
            variable = encode_nc3_variable(variable)
            values = variable.values
            datatype = variable.dtype

        self.set_necessary_dimensions(variable)

        fill_value = variable.attrs.pop('_FillValue', None)
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
            endian=encoding.get('endian', 'native'),
            least_significant_digit=encoding.get('least_significant_digit'),
            fill_value=fill_value)
        nc4_var.set_auto_maskandscale(False)
        nc4_var[:] = values
        for k, v in iteritems(variable.attrs):
            # set attributes one-by-one since netCDF4<1.0.10 can't handle
            # OrderedDict as the input to setncatts
            nc4_var.setncattr(k, v)

    def del_attribute(self, key):
        self.ds.delncattr(key)

    def sync(self):
        self.ds.sync()

    def close(self):
        ds = self.ds
        # netCDF4 only allows closing the root group
        while ds.parent is not None:
            ds = ds.parent
        ds.close()
