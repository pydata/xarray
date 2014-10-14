from io import BytesIO

import numpy as np
import warnings

from .. import Variable
from ..conventions import cf_encoder
from ..core.pycompat import iteritems, basestring, unicode_type, OrderedDict
from ..core.utils import Frozen, FrozenOrderedDict

from .common import AbstractWritableDataStore
from .netcdf3 import is_valid_nc3_name, coerce_nc3_dtype, encode_nc3_variable
from xray.conventions import cf_decoder


def _decode_string(s):
    if isinstance(s, bytes):
        return s.decode('utf-8', 'replace')
    return s


def _decode_attrs(d):
    # don't decode _FillValue from bytes -> unicode, because we want to ensure
    # that its type matches the data exactly
    return OrderedDict((k, v if k == '_FillValue' else _decode_string(v))
                       for (k, v) in iteritems(d))


class ScipyDataStore(AbstractWritableDataStore):
    """Store for reading and writing data via scipy.io.netcdf.

    This store has the advantage of being able to be initialized with a
    StringIO object, allow for serialization without writing to disk.

    It only supports the NetCDF3 file-format.
    """
    def __init__(self, filename_or_obj, mode='r', mmap=None, version=1):
        import scipy
        if mode != 'r' and scipy.__version__ < '0.13':
            warnings.warn('scipy %s detected; '
                          'the minimal recommended version is 0.13. '
                          'Older version of this library do not reliably '
                          'read and write files.'
                          % scipy.__version__, ImportWarning)

        import scipy.io
        # if filename is a NetCDF3 bytestring we store it in a StringIO
        if (isinstance(filename_or_obj, basestring)
                and filename_or_obj.startswith('CDF')):
            # TODO: this check has the unfortunate side-effect that
            # paths to files cannot start with 'CDF'.
            filename_or_obj = BytesIO(filename_or_obj)
        self.ds = scipy.io.netcdf.netcdf_file(
            filename_or_obj, mode=mode, mmap=mmap, version=version)

    def store(self, variables, attributes):
        # All Scipy objects get CF encoded by default, without this attempting
        # to write times, for example, would fail.
        cf_variables, cf_attrs = cf_encoder(variables, attributes)
        AbstractWritableDataStore.store(self, cf_variables, cf_attrs)

    def open_store_variable(self, var):
        return Variable(var.dimensions, var.data,
                        _decode_attrs(var._attributes))

    def get_variables(self):
        return FrozenOrderedDict((k, self.open_store_variable(v))
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

    def _cast_attr_value(self, value):
        if isinstance(value, basestring):
            if not isinstance(value, unicode_type):
                value = value.decode('utf-8')
        else:
            value = coerce_nc3_dtype(np.atleast_1d(value))
            if value.ndim > 1:
                raise ValueError("netCDF attributes must be 1-dimensional")
        return value

    def set_attribute(self, key, value):
        self._validate_attr_key(key)
        setattr(self.ds, key, self._cast_attr_value(value))

    def set_variable(self, name, variable):
        # TODO, create a netCDF3 encoder
        variable = encode_nc3_variable(variable)
        self.set_necessary_dimensions(variable)
        data = variable.values
        self.ds.createVariable(name, data.dtype, variable.dims)
        scipy_var = self.ds.variables[name]
        if data.ndim == 0:
            scipy_var.assignValue(data)
        else:
            scipy_var[:] = data[:]
        for k, v in iteritems(variable.attrs):
            self._validate_attr_key(k)
            setattr(scipy_var, k, self._cast_attr_value(v))

    def del_attribute(self, key):
        delattr(self.ds, key)

    def sync(self):
        self.ds.flush()

    def close(self):
        self.ds.close()

    def __exit__(self, type, value, tb):
        self.close()
