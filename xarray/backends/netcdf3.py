from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import unicodedata

import numpy as np

from .. import conventions, Variable
from ..core import duck_array_ops
from ..core.pycompat import basestring, unicode_type, OrderedDict


# Special characters that are permitted in netCDF names except in the
# 0th position of the string
_specialchars = '_.@+- !"#$%&\()*,:;<=>?[]^`{|}~'

# The following are reserved names in CDL and may not be used as names of
# variables, dimension, attributes
_reserved_names = set(['byte', 'char', 'short', 'ushort', 'int', 'uint',
                       'int64', 'uint64', 'float' 'real', 'double', 'bool',
                       'string'])

# These data-types aren't supported by netCDF3, so they are automatically
# coerced instead as indicated by the "coerce_nc3_dtype" function
_nc3_dtype_coercions = {'int64': 'int32', 'bool': 'int8'}


def coerce_nc3_dtype(arr):
    """Coerce an array to a data type that can be stored in a netCDF-3 file

    This function performs the following dtype conversions:
        int64 -> int32
        float64 -> float32
        bool -> int8
        unicode -> string

    Data is checked for equality, or equivalence (non-NaN values) with
    `np.allclose` with the default keyword arguments.
    """
    dtype = str(arr.dtype)
    if dtype in _nc3_dtype_coercions:
        new_dtype = _nc3_dtype_coercions[dtype]
        # TODO: raise a warning whenever casting the data-type instead?
        cast_arr = arr.astype(new_dtype)
        if ((('int' in dtype or 'U' in dtype) and
                not (cast_arr == arr).all()) or
                ('float' in dtype and
                    not duck_array_ops.allclose_or_equiv(cast_arr, arr))):
            raise ValueError('could not safely cast array from dtype %s to %s'
                             % (dtype, new_dtype))
        arr = cast_arr
    elif arr.dtype.kind == 'U':
        arr = np.core.defchararray.encode(arr, 'utf-8')
    return arr


def maybe_convert_to_char_array(data, dims):
    if data.dtype.kind == 'S' and data.dtype.itemsize > 1:
        data = conventions.string_to_char(data)
        dims = dims + ('string%s' % data.shape[-1],)
    return data, dims


def encode_nc3_attr_value(value):
    if isinstance(value, basestring):
        if not isinstance(value, unicode_type):
            value = value.decode('utf-8')
    else:
        value = coerce_nc3_dtype(np.atleast_1d(value))
        if value.ndim > 1:
            raise ValueError("netCDF attributes must be 1-dimensional")
    return value


def encode_nc3_attrs(attrs):
    return OrderedDict([(k, encode_nc3_attr_value(v))
                        for k, v in attrs.items()])


def encode_nc3_variable(var):
    data = coerce_nc3_dtype(var.data)
    data, dims = maybe_convert_to_char_array(data, var.dims)
    attrs = encode_nc3_attrs(var.attrs)
    return Variable(dims, data, attrs, var.encoding)


def _isalnumMUTF8(c):
    """Return True if the given UTF-8 encoded character is alphanumeric
    or multibyte.

    Input is not checked!
    """
    return c.isalnum() or (len(c.encode('utf-8')) > 1)


def is_valid_nc3_name(s):
    """Test whether an object can be validly converted to a netCDF-3
    dimension, variable or attribute name

    Earlier versions of the netCDF C-library reference implementation
    enforced a more restricted set of characters in creating new names,
    but permitted reading names containing arbitrary bytes. This
    specification extends the permitted characters in names to include
    multi-byte UTF-8 encoded Unicode and additional printing characters
    from the US-ASCII alphabet. The first character of a name must be
    alphanumeric, a multi-byte UTF-8 character, or '_' (reserved for
    special names with meaning to implementations, such as the
    "_FillValue" attribute). Subsequent characters may also include
    printing special characters, except for '/' which is not allowed in
    names. Names that have trailing space characters are also not
    permitted.
    """
    if not isinstance(s, basestring):
        return False
    if not isinstance(s, unicode_type):
        s = s.decode('utf-8')
    num_bytes = len(s.encode('utf-8'))
    return ((unicodedata.normalize('NFC', s) == s) and
            (s not in _reserved_names) and
            (num_bytes >= 0) and
            ('/' not in s) and
            (s[-1] != ' ') and
            (_isalnumMUTF8(s[0]) or (s[0] == '_')) and
            all((_isalnumMUTF8(c) or c in _specialchars for c in s)))
