from __future__ import absolute_import, division, print_function

import functools
import warnings
from distutils.version import LooseVersion
from io import BytesIO

import numpy as np

from .. import Variable
from ..core.indexing import NumpyIndexingAdapter
from ..core.pycompat import OrderedDict, basestring, iteritems
from ..core.utils import Frozen, FrozenOrderedDict
from .common import BackendArray, DataStorePickleMixin, WritableCFDataStore
from .netcdf3 import (
    encode_nc3_attr_value, encode_nc3_variable, is_valid_nc3_name)


def _decode_string(s):
    if isinstance(s, bytes):
        return s.decode('utf-8', 'replace')
    return s


def _decode_attrs(d):
    # don't decode _FillValue from bytes -> unicode, because we want to ensure
    # that its type matches the data exactly
    return OrderedDict((k, v if k == '_FillValue' else _decode_string(v))
                       for (k, v) in iteritems(d))


class ScipyArrayWrapper(BackendArray):

    def __init__(self, variable_name, datastore):
        self.datastore = datastore
        self.variable_name = variable_name
        array = self.get_array()
        self.shape = array.shape
        self.dtype = np.dtype(array.dtype.kind +
                              str(array.dtype.itemsize))

    def get_array(self):
        self.datastore.assert_open()
        return self.datastore.ds.variables[self.variable_name].data

    def __getitem__(self, key):
        with self.datastore.ensure_open(autoclose=True):
            data = NumpyIndexingAdapter(self.get_array())[key]
            # Copy data if the source file is mmapped.
            # This makes things consistent
            # with the netCDF4 library by ensuring
            # we can safely read arrays even
            # after closing associated files.
            copy = self.datastore.ds.use_mmap
            return np.array(data, dtype=self.dtype, copy=copy)

    def __setitem__(self, key, value):
        with self.datastore.ensure_open(autoclose=True):
            data = self.datastore.ds.variables[self.variable_name]
            try:
                data[key] = value
            except TypeError:
                if key is Ellipsis:
                    # workaround for GH: scipy/scipy#6880
                    data[:] = value
                else:
                    raise


def _open_scipy_netcdf(filename, mode, mmap, version):
    import scipy.io
    import gzip

    # if the string ends with .gz, then gunzip and open as netcdf file
    if isinstance(filename, basestring) and filename.endswith('.gz'):
        try:
            return scipy.io.netcdf_file(gzip.open(filename), mode=mode,
                                        mmap=mmap, version=version)
        except TypeError as e:
            # TODO: gzipped loading only works with NetCDF3 files.
            if 'is not a valid NetCDF 3 file' in e.message:
                raise ValueError('gzipped file loading only supports '
                                 'NetCDF 3 files.')
            else:
                raise

    if isinstance(filename, bytes) and filename.startswith(b'CDF'):
        # it's a NetCDF3 bytestring
        filename = BytesIO(filename)

    try:
        return scipy.io.netcdf_file(filename, mode=mode, mmap=mmap,
                                    version=version)
    except TypeError as e:  # netcdf3 message is obscure in this case
        errmsg = e.args[0]
        if 'is not a valid NetCDF 3 file' in errmsg:
            msg = """
            If this is a NetCDF4 file, you may need to install the
            netcdf4 library, e.g.,

            $ pip install netcdf4
            """
            errmsg += msg
            raise TypeError(errmsg)
        else:
            raise


class ScipyDataStore(WritableCFDataStore, DataStorePickleMixin):
    """Store for reading and writing data via scipy.io.netcdf.

    This store has the advantage of being able to be initialized with a
    StringIO object, allow for serialization without writing to disk.

    It only supports the NetCDF3 file-format.
    """

    def __init__(self, filename_or_obj, mode='r', format=None, group=None,
                 writer=None, mmap=None, autoclose=False, lock=None):
        import scipy
        import scipy.io

        if (mode != 'r' and
                scipy.__version__ < LooseVersion('0.13')):  # pragma: no cover
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
        self._ds = opener()
        self._autoclose = autoclose
        self._isopen = True
        self._opener = opener
        self._mode = mode

        super(ScipyDataStore, self).__init__(writer, lock=lock)

    def open_store_variable(self, name, var):
        with self.ensure_open(autoclose=False):
            return Variable(var.dimensions, ScipyArrayWrapper(name, self),
                            _decode_attrs(var._attributes))

    def get_variables(self):
        with self.ensure_open(autoclose=False):
            return FrozenOrderedDict((k, self.open_store_variable(k, v))
                                     for k, v in iteritems(self.ds.variables))

    def get_attrs(self):
        with self.ensure_open(autoclose=True):
            return Frozen(_decode_attrs(self.ds._attributes))

    def get_dimensions(self):
        with self.ensure_open(autoclose=True):
            return Frozen(self.ds.dimensions)

    def get_encoding(self):
        encoding = {}
        encoding['unlimited_dims'] = {
            k for k, v in self.ds.dimensions.items() if v is None}
        return encoding

    def set_dimension(self, name, length, is_unlimited=False):
        with self.ensure_open(autoclose=False):
            if name in self.ds.dimensions:
                raise ValueError('%s does not support modifying dimensions'
                                 % type(self).__name__)
            dim_length = length if not is_unlimited else None
            self.ds.createDimension(name, dim_length)

    def _validate_attr_key(self, key):
        if not is_valid_nc3_name(key):
            raise ValueError("Not a valid attribute name")

    def set_attribute(self, key, value):
        with self.ensure_open(autoclose=False):
            self._validate_attr_key(key)
            value = encode_nc3_attr_value(value)
            setattr(self.ds, key, value)

    def encode_variable(self, variable):
        variable = encode_nc3_variable(variable)
        return variable

    def prepare_variable(self, name, variable, check_encoding=False,
                         unlimited_dims=None):
        if check_encoding and variable.encoding:
            if variable.encoding != {'_FillValue': None}:
                raise ValueError('unexpected encoding for scipy backend: %r'
                                 % list(variable.encoding))

        data = variable.data
        # nb. this still creates a numpy array in all memory, even though we
        # don't write the data yet; scipy.io.netcdf does not not support
        # incremental writes.
        if name not in self.ds.variables:
            self.ds.createVariable(name, data.dtype, variable.dims)
        scipy_var = self.ds.variables[name]
        for k, v in iteritems(variable.attrs):
            self._validate_attr_key(k)
            setattr(scipy_var, k, v)

        target = ScipyArrayWrapper(name, self)

        return target, data

    def sync(self):
        with self.ensure_open(autoclose=True):
            super(ScipyDataStore, self).sync()
            self.ds.flush()

    def close(self):
        self.ds.close()
        self._isopen = False

    def __exit__(self, type, value, tb):
        self.close()

    def __setstate__(self, state):
        filename = state['_opener'].keywords['filename']
        if hasattr(filename, 'seek'):
            # it's a file-like object
            # seek to the start of the file so scipy can read it
            filename.seek(0)
        super(ScipyDataStore, self).__setstate__(state)
        self._ds = None
        self._isopen = False
