import numpy as np
import unicodedata

NULL          = '\x00'
NC_BYTE       = '\x00\x00\x00\x01'
NC_CHAR       = '\x00\x00\x00\x02'
NC_SHORT      = '\x00\x00\x00\x03'
# netCDF-3 only supports 32-bit integers
NC_INT        = '\x00\x00\x00\x04'
NC_FLOAT      = '\x00\x00\x00\x05'
NC_DOUBLE     = '\x00\x00\x00\x06'

# Map between netCDF type and numpy dtype and vice versa. Due to a bug
# in the __hash__() method of numpy dtype objects (fixed in development
# release of numpy), we need to explicitly match byteorder for dict
# lookups to succeed. Here we normalize to native byte order.
#
# NC_CHAR is a special case because netCDF represents strings as
# character arrays. When NC_CHAR is encountered as the type of an
# attribute value, this TYPEMAP is not consulted and the data is read
# as a string. However, when NC_CHAR is encountered as the type of a
# variable, then the data is read is a numpy array of 1-char elements
# (equivalently, length-1 raw "strings"). There is no support for numpy
# arrays of multi-character strings.
TYPEMAP = {
        # we could use np.dtype's as key/values except __hash__ comparison of
        # numpy.dtype is broken in older versions of numpy.  If you must compare
        # and cannot upgrade, use __eq__.This bug is
        # known to be fixed in numpy version 1.3
        NC_BYTE: 'int8',
        NC_CHAR: '|S1',
        NC_SHORT: 'int16',
        NC_INT: 'int32',
        NC_FLOAT: 'float32',
        NC_DOUBLE: 'float64',
        }
for k in TYPEMAP.keys():
    TYPEMAP[TYPEMAP[k]] = k

# Special characters that are permitted in netCDF names except in the
# 0th position of the string
_specialchars = '_.@+- !"#$%&\()*,:;<=>?[]^`{|}~'

# The following are reserved names in CDL and may not be used as names of
# variables, dimension, attributes
_reserved_names = set([
        'byte',
        'char',
        'short',
        'ushort',
        'int',
        'uint',
        'int64',
        'uint64',
        'float'
        'real',
        'double',
        'bool',
        'string',
        ])

def pretty_print(x, numchars):
    """Given an object x, call x.__str__() and format the returned
    string so that it is numchars long, padding with trailing spaces or
    truncating with ellipses as necessary"""
    s = str(x).rstrip(NULL)
    if len(s) > numchars:
        return s[:(numchars - 3)] + '...'
    else:
        return s

def coerce_type(arr):
    """Coerce a numeric data type to a type that is compatible with
    netCDF-3

    netCDF-3 can not handle 64-bit integers, but on most platforms
    Python integers are int64. To work around this discrepancy, this
    helper function coerces int64 arrays to int32. An exception is
    raised if this coercion is not safe.

    netCDF-3 can not handle booleans, but booleans can be trivially
    (albeit wastefully) represented as bytes. To work around this
    discrepancy, this helper function coerces bool arrays to int8.
    """
    # Comparing the char attributes of numpy dtypes is inelegant, but this is
    # the fastest test of equivalence that is invariant to endianness
    if arr.dtype.char == 'l': # np.dtype('int64')
        cast_arr = arr.astype(
                np.dtype('int32').newbyteorder(arr.dtype.byteorder))
        if not (cast_arr == arr).all():
            raise ValueError("array contains integer values that " +
                    "are not representable as 32-bit signed integers")
        return cast_arr
    elif arr.dtype.char == '?': # np.dtype('bool')
        # bool
        cast_arr = arr.astype(
                np.dtype('int8').newbyteorder(arr.dtype.byteorder))
        return cast_arr
    else:
        return arr

def _isalnumMUTF8(c):
    """Return True if the given UTF-8 encoded character is alphanumeric
    or multibyte.

    Input is not checked!
    """
    return (c.isalnum() or (len(c.encode('utf-8')) > 1))

def is_valid_name(s):
    """Test whether an object can be validly converted to a netCDF
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
    if not isinstance(s, unicode):
        s = unicode(s, 'utf-8')
    num_bytes = len(s.encode('utf-8'))
    return ((unicodedata.normalize('NFC', s) == s) and
            (s not in _reserved_names) and
            (num_bytes >= 0) and
            ('/' not in s) and
            (s[-1] != ' ') and
            (_isalnumMUTF8(s[0]) or (s[0] == '_')) and
            all((_isalnumMUTF8(c) or c in _specialchars for c in s))
            )