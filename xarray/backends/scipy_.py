from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
from io import BytesIO

import numpy as np
import warnings

from .. import Variable
from ..core.pycompat import iteritems, basestring, OrderedDict
from ..core.utils import Frozen, FrozenOrderedDict
from ..core.indexing import NumpyIndexingAdapter

from .common import WritableCFDataStore, DataStorePickleMixin
from .netcdf3 import (is_valid_nc3_name, encode_nc3_attr_value,
                      encode_nc3_variable)


def _decode_string(s):
    if isinstance(s, bytes):
        return s.decode('utf-8', 'replace')
    return s


def _decode_attrs(d):
    # don't decode _FillValue from bytes -> unicode, because we want to ensure
    # that its type matches the data exactly
    return OrderedDict((k, v if k == '_FillValue' else _decode_string(v))
                       for (k, v) in iteritems(d))


class ScipyArrayWrapper(NumpyIndexingAdapter):
    def __init__(self, variable_name, datastore):
        self.datastore = datastore
        self.variable_name = variable_name

    @property
    def array(self):
        return self.datastore.ds.variables[self.variable_name].data

    @property
    def dtype(self):
        # always use native endianness
        return np.dtype(self.array.dtype.kind + str(self.array.dtype.itemsize))

    def __getitem__(self, key):
        data = super(ScipyArrayWrapper, self).__getitem__(key)
        # Copy data if the source file is mmapped. This makes things consistent
        # with the netCDF4 library by ensuring we can safely read arrays even
        # after closing associated files.
        copy = self.datastore.ds.use_mmap
        data = np.array(data, dtype=self.dtype, copy=copy)
        return data


def _open_scipy_netcdf(filename, mode, mmap, version):
    import scipy.io

    if isinstance(filename, bytes) and filename.startswith(b'CDF'):
        # it's a NetCDF3 bytestring
        filename = BytesIO(filename)

    return scipy.io.netcdf_file(filename, mode=mode, mmap=mmap,
                                version=version)


class ScipyDataStore(WritableCFDataStore, DataStorePickleMixin):
    """Store for reading and writing data via scipy.io.netcdf.

    This store has the advantage of being able to be initialized with a
    StringIO object, allow for serialization without writing to disk.

    It only supports the NetCDF3 file-format.
    """
    def __init__(self, filename_or_obj, mode='r', format=None, group=None,
                 writer=None, mmap=None):
        import scipy
        import scipy.io
        if mode != 'r' and scipy.__version__ < '0.13':  # pragma: no cover
            warnings.warn('scipy %s detected; '
                          'the minimal recommended version is 0.13. '
                          'Older version of this library do not reliably '
                          'read and write files.'
                          % scipy.__version__, ImportWarning)

        if group is not None:
            raise ValueError('cannot save to a group with the '
                             'scipy.io.netcdf backend')

        if format is None or format == 'NETCDF3_64BIT':
            version = 2
        elif format == 'NETCDF3_CLASSIC':
            version = 1
        else:
            raise ValueError('invalid format for scipy.io.netcdf backend: %r'
                             % format)

        opener = functools.partial(_open_scipy_netcdf,
                                   filename=filename_or_obj,
                                   mode=mode, mmap=mmap, version=version)
        self.ds = opener()
        self._opener = opener
        self._mode = mode

        super(ScipyDataStore, self).__init__(writer)

    def open_store_variable(self, name, var):
        return Variable(var.dimensions, ScipyArrayWrapper(name, self),
                        _decode_attrs(var._attributes))

    def get_variables(self):
        return FrozenOrderedDict((k, self.open_store_variable(k, v))
                                 for k, v in iteritems(self.ds.variables))

    def get_attrs(self):
        return Frozen(_decode_attrs(self.ds._attributes))

    def get_dimensions(self):
        return Frozen(self.ds.dimensions)

    def set_dimension(self, name, length):
        if name in self.dimensions:
            raise ValueError('%s does not support modifying dimensions'
                             % type(self).__name__)
        self.ds.createDimension(name, length)

    def _validate_attr_key(self, key):
        if not is_valid_nc3_name(key):
            raise ValueError("Not a valid attribute name")

    def set_attribute(self, key, value):
        self._validate_attr_key(key)
        value = encode_nc3_attr_value(value)
        setattr(self.ds, key, value)

    def prepare_variable(self, name, variable, check_encoding=False):
        variable = encode_nc3_variable(variable)
        if check_encoding and variable.encoding:
            raise ValueError('unexpected encoding for scipy backend: %r'
                             % list(variable.encoding))

        self.set_necessary_dimensions(variable)
        data = variable.data
        # nb. this still creates a numpy array in all memory, even though we
        # don't write the data yet; scipy.io.netcdf does not not support
        # incremental writes.
        self.ds.createVariable(name, data.dtype, variable.dims)
        scipy_var = self.ds.variables[name]
        for k, v in iteritems(variable.attrs):
            self._validate_attr_key(k)
            setattr(scipy_var, k, v)
        return scipy_var, data

    def sync(self):
        super(ScipyDataStore, self).sync()
        self.ds.flush()

    def close(self):
        self.ds.close()

    def __exit__(self, type, value, tb):
        self.close()

    def __setstate__(self, state):
        filename = state['_opener'].keywords['filename']
        if hasattr(filename, 'seek'):
            # it's a file-like object
            # seek to the start of the file so scipy can read it
            filename.seek(0)
        super(ScipyDataStore, self).__setstate__(state)
