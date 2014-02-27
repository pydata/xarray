import unicodedata

import numpy as np
import pandas as pd

import xarray
import utils


# Special characters that are permitted in netCDF names except in the
# 0th position of the string
_specialchars = '_.@+- !"#$%&\()*,:;<=>?[]^`{|}~'

# The following are reserved names in CDL and may not be used as names of
# variables, dimension, attributes
_reserved_names = set(['byte', 'char', 'short', 'ushort', 'int', 'uint',
                       'int64', 'uint64', 'float' 'real', 'double', 'bool',
                       'string'])


def pretty_print(x, numchars):
    """Given an object x, call x.__str__() and format the returned
    string so that it is numchars long, padding with trailing spaces or
    truncating with ellipses as necessary"""
    s = str(x)
    if len(s) > numchars:
        return s[:(numchars - 3)] + '...'
    else:
        return s


def coerce_nc3_dtype(arr):
    """Coerce an array to a data type that can be stored in a netCDF-3 file

    This function performs the following dtype conversions:
        int64 -> int32
        float64 -> float32
        bool -> int8

    Data is checked for equality, or equivalence with the default values of
    `np.allclose`.
    """
    dtype = str(arr.dtype)
    dtype_map = {'int64': 'int32', 'float64': 'float32', 'bool': 'int8'}
    if dtype in dtype_map:
        new_dtype = dtype_map[dtype]
        cast_arr = arr.astype(new_dtype)
        if (('int' in dtype and not (cast_arr == arr).all())
                or ('float' in dtype and not np.allclose(cast_arr, arr))):
            raise ValueError('could not safely cast array from dtype %s to %s'
                             % (dtype, new_dtype))
        arr = cast_arr
    return arr


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
    if not isinstance(s, unicode):
        s = unicode(s, 'utf-8')
    num_bytes = len(s.encode('utf-8'))
    return ((unicodedata.normalize('NFC', s) == s) and
            (s not in _reserved_names) and
            (num_bytes >= 0) and
            ('/' not in s) and
            (s[-1] != ' ') and
            (_isalnumMUTF8(s[0]) or (s[0] == '_')) and
            all((_isalnumMUTF8(c) or c in _specialchars for c in s)))


class MaskedAndScaledArray(object):
    """Wrapper around array-like objects to create a new indexable object where
    values, when accessesed, are automatically scaled and masked according to
    CF conventions for packed and missing data values

    New values are given by the formula:
        original_values * scale_factor + add_offset

    Values can only be accessed via `__getitem__`:

    >>> x = _MaskedAndScaledArray(np.array([-99, -1, 0, 1, 2]), -99, 0.01, 1)
    >>> x
    _MaskedAndScaledArray(array([-99, -1,  0,  1,  2]), fill_value=-99,
    scale_factor=0.01, add_offset=1)
    >>> x[:]
    array([  nan,  0.99,  1.  ,  1.01,  1.02]

    References
    ----------
    http://www.unidata.ucar.edu/software/netcdf/docs/BestPractices.html
    """
    def __init__(self, array, fill_value=None, scale_factor=None,
                 add_offset=None):
        """
        Parameters
        ----------
        array : array-like
            Original array of values to wrap
        fill_value : number, optional
            All values equal to fill_value in the original array are replaced
            by NaN.
        scale_factor : number, optional
            Multiply entries in the original array by this number.
        add_offset : number, optional
            After applying scale_factor, add this number to entries in the
            original array.
        """
        self.array = array
        self.scale_factor = scale_factor
        self.add_offset = add_offset
        self.fill_value = fill_value

    @property
    def dtype(self):
        return np.dtype('float')

    @property
    def shape(self):
        return self.array.shape

    @property
    def size(self):
        return self.array.size

    @property
    def ndim(self):
        return self.array.ndim

    def __len__(self):
        return len(self.array)

    def __array__(self):
        return self[...]

    def __getitem__(self, key):
        # cast to float to insure NaN is meaningful
        values = np.array(self.array[key], dtype=float, copy=True)
        if self.fill_value is not None:
            values[values == self.fill_value] = np.nan
        if self.scale_factor is not None:
            values *= self.scale_factor
        if self.add_offset is not None:
            values += self.add_offset
        return values

    def __repr__(self):
        return ("%s(%r, fill_value=%r, scale_factor=%r, add_offset=%r)" %
                (type(self).__name__, self.array, self.fill_value,
                 self.scale_factor, self.add_offset))


def encode_cf_variable(array):
    """Converts an XArray into an XArray suitable for saving as a netCDF
    variable
    """
    data = array.data
    attributes = array.attributes.copy()
    if isinstance(data, pd.DatetimeIndex):
        # DatetimeIndex objects need to be encoded into numeric arrays
        (data, units, calendar) = utils.datetimeindex2num(data)
        attributes['units'] = units
        attributes['calendar'] = calendar
    elif data.dtype == np.dtype('O'):
        # Unfortunately, pandas.Index arrays often have dtype=object even if
        # they were created from an array with a sensible datatype (e.g.,
        # pandas.Float64Index always has dtype=object for some reason). Because
        # we allow for doing math with coordinates, these object arrays can
        # propagate onward to other variables, which is why we don't only apply
        # this check to XArrays with data that is a pandas.Index.
        dtype = np.array(data.reshape(-1)[0]).dtype
        # N.B. the "astype" call will fail if data cannot be cast to the type
        # of its first element (which is probably the only sensible thing to
        # do).
        data = np.asarray(data).astype(dtype)

    # unscale/mask
    if any(k in attributes for k in ['add_offset', 'scale_factor']):
        data = np.array(data, dtype=float, copy=True)
        if 'add_offset' in attributes:
            data -= attributes['add_offset']
        if 'scale_factor' in attributes:
            data /= attributes['scale_factor']

    # restore original dtype
    if 'encoded_dtype' in attributes:
        data = data.astype(attributes.pop('encoded_dtype'))

    return xarray.XArray(array.dimensions, data, attributes)


def decode_cf_variable(dimensions, data, attributes, indexing_mode='numpy'):
    attributes = attributes.copy()
    attributes['encoded_dtype'] = data.dtype

    mask_and_scale_attrs = ['_FillValue', 'scale_factor', 'add_offset']
    if any(k in attributes for k in mask_and_scale_attrs):
        data = MaskedAndScaledArray(data, attributes.get('_FillValue'),
                                    attributes.get('scale_factor'),
                                    attributes.get('add_offset'))

    return xarray.XArray(dimensions, data, attributes, indexing_mode)
