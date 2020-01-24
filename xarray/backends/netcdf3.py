import unicodedata

import numpy as np

from .. import coding
from ..core.variable import Variable

# Special characters that are permitted in netCDF names except in the
# 0th position of the string
_specialchars = '_.@+- !"#$%&\\()*,:;<=>?[]^`{|}~'

# The following are reserved names in CDL and may not be used as names of
# variables, dimension, attributes
_reserved_names = {
    "byte",
    "char",
    "short",
    "ushort",
    "int",
    "uint",
    "int64",
    "uint64",
    "float" "real",
    "double",
    "bool",
    "string",
}

# These data-types aren't supported by netCDF3, so they are automatically
# coerced instead as indicated by the "coerce_nc3_dtype" function
_nc3_dtype_coercions = {"int64": "int32", "bool": "int8"}

# encode all strings as UTF-8
STRING_ENCODING = "utf-8"


def coerce_nc3_dtype(arr):
    """Coerce an array to a data type that can be stored in a netCDF-3 file

    This function performs the following dtype conversions:
        int64 -> int32
        bool -> int8

    Data is checked for equality, or equivalence (non-NaN values) with
    `np.allclose` with the default keyword arguments.
    """
    dtype = str(arr.dtype)
    if dtype in _nc3_dtype_coercions:
        new_dtype = _nc3_dtype_coercions[dtype]
        # TODO: raise a warning whenever casting the data-type instead?
        cast_arr = arr.astype(new_dtype)
        if not (cast_arr == arr).all():
            raise ValueError(
                f"could not safely cast array from dtype {dtype} to {new_dtype}"
            )
        arr = cast_arr
    return arr


def encode_nc3_attr_value(value):
    if isinstance(value, bytes):
        pass
    elif isinstance(value, str):
        value = value.encode(STRING_ENCODING)
    else:
        value = coerce_nc3_dtype(np.atleast_1d(value))
        if value.ndim > 1:
            raise ValueError("netCDF attributes must be 1-dimensional")
    return value


def encode_nc3_attrs(attrs):
    return {k: encode_nc3_attr_value(v) for k, v in attrs.items()}


def encode_nc3_variable(var):
    for coder in [
        coding.strings.EncodedStringCoder(allows_unicode=False),
        coding.strings.CharacterArrayCoder(),
    ]:
        var = coder.encode(var)
    data = coerce_nc3_dtype(var.data)
    attrs = encode_nc3_attrs(var.attrs)
    return Variable(var.dims, data, attrs, var.encoding)


def _isalnumMUTF8(c):
    """Return True if the given UTF-8 encoded character is alphanumeric
    or multibyte.

    Input is not checked!
    """
    return c.isalnum() or (len(c.encode("utf-8")) > 1)


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
    if not isinstance(s, str):
        return False
    if not isinstance(s, str):
        s = s.decode("utf-8")
    num_bytes = len(s.encode("utf-8"))
    return (
        (unicodedata.normalize("NFC", s) == s)
        and (s not in _reserved_names)
        and (num_bytes >= 0)
        and ("/" not in s)
        and (s[-1] != " ")
        and (_isalnumMUTF8(s[0]) or (s[0] == "_"))
        and all(_isalnumMUTF8(c) or c in _specialchars for c in s)
    )
