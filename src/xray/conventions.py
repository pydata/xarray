import unicodedata

import netCDF4 as nc4
import numpy as np
from datetime import datetime

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

# These data-types aren't supported by netCDF3, so they are automatically
# coerced instead as indicated by the "coerce_nc3_dtype" function
_nc3_dtype_coercions = {'int64': 'int32', 'float64': 'float32', 'bool': 'int8'}


def coerce_nc3_dtype(arr):
    """Coerce an array to a data type that can be stored in a netCDF-3 file

    This function performs the following dtype conversions:
        int64 -> int32
        float64 -> float32
        bool -> int8

    Data is checked for equality, or equivalence (non-NaN values) with
    `np.allclose` with the default keyword arguments.
    """
    dtype = str(arr.dtype)
    if dtype in _nc3_dtype_coercions:
        new_dtype = _nc3_dtype_coercions[dtype]
        # TODO: raise a warning whenever casting the data-type instead?
        cast_arr = arr.astype(new_dtype)
        if (('int' in dtype and not (cast_arr == arr).all())
                or ('float' in dtype and
                    not utils.allclose_or_equiv(cast_arr, arr))):
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
        if self.fill_value is not None and not np.isnan(self.fill_value):
            if self.ndim > 0:
                values[values == self.fill_value] = np.nan
            elif values == self.fill_value:
                values = np.array(np.nan)
        if self.scale_factor is not None:
            values *= self.scale_factor
        if self.add_offset is not None:
            values += self.add_offset
        return values

    def __repr__(self):
        return ("%s(%r, fill_value=%r, scale_factor=%r, add_offset=%r)" %
                (type(self).__name__, self.array, self.fill_value,
                 self.scale_factor, self.add_offset))


class CharToStringArray(object):
    """Wrapper around array-like objects to create a new indexable object where
    values, when accessesed, are automatically concatenated along the last
    dimension

    >>> CharToStringArray(np.array(['a', 'b', 'c']))[:]
    array('abc',
          dtype='|S3')
    """
    def __init__(self, array):
        """
        Parameters
        ----------
        array : array-like
            Original array of values to wrap.
        """
        self.array = array

    @property
    def dtype(self):
        return np.dtype(str(self.array.dtype)[:2] + str(self.array.shape[-1]))

    @property
    def shape(self):
        return self.array.shape[:-1]

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def ndim(self):
        return self.array.ndim - 1

    def __len__(self):
        if self.ndim > 0:
            return len(self.array)
        else:
            raise TypeError('len() of unsized object')

    def __str__(self):
        if self.ndim == 0:
            return str(self[...])
        else:
            return repr(self)

    def __repr__(self):
        return '%s(%r)' % (type(self).__name__, self.array)

    def __array__(self):
        return self[...]

    def __getitem__(self, key):
        # require slicing the last dimension completely
        key = utils.expanded_indexer(key, self.array.ndim)
        if key[-1] != slice(None):
            raise IndexError('too many indices')
        return nc4.chartostring(self.array[key])


def encode_cf_variable(array):
    """Converts an XArray into an XArray suitable for saving as a netCDF
    variable
    """
    dimensions = array.dimensions
    data = array.data
    attributes = array.attributes.copy()
    encoding = array.encoding.copy()

    if (np.issubdtype(data.dtype, np.datetime64)
            or (data.dtype.kind == 'O'
                and isinstance(data.reshape(-1)[0], datetime))):
        # encode datetime arrays into numeric arrays
        (data, units, calendar) = utils.encode_cf_datetime(
            data, encoding.pop('units', None), encoding.pop('calendar', None))
        attributes['units'] = units
        attributes['calendar'] = calendar
    elif data.dtype == np.dtype('O'):
        # Occasionally, one will end up with variables with dtype=object
        # (likely because they were created from pandas objects which don't
        # maintain dtype careful). Thie code makes a best effort attempt to
        # encode them into a dtype that NETCDF can handle by inspecting the
        # dtype of the first element.
        dtype = np.array(data.reshape(-1)[0]).dtype
        # N.B. the "astype" call below will fail if data cannot be cast to the
        # type of its first element (which is probably the only sensible thing
        # to do).
        data = np.asarray(data).astype(dtype)

    def get_to(source, dest, k):
        v = source.get(k)
        dest[k] = v
        return v

    # encode strings as character arrays
    if np.issubdtype(data.dtype, (str, unicode)):
        data = nc4.stringtochar(data)
        dimensions = dimensions + ('string%s' % data.shape[-1],)

    # unscale/mask
    if any(k in encoding for k in ['add_offset', 'scale_factor']):
        data = np.array(data, dtype=float, copy=True)
        if 'add_offset' in encoding:
            data -= get_to(encoding, attributes, 'add_offset')
        if 'scale_factor' in encoding:
            data /= get_to(encoding, attributes, 'scale_factor')

    # replace NaN with the fill value
    if '_FillValue' in encoding:
        if encoding['_FillValue'] is np.nan:
            attributes['_FillValue'] = np.nan
        else:
            nans = np.isnan(data)
            if nans.any():
                data[nans] = get_to(encoding, attributes, '_FillValue')

    # restore original dtype
    if 'dtype' in encoding:
        if np.issubdtype(encoding['dtype'], int):
            data = data.round()
        data = data.astype(encoding['dtype'])

    return xarray.XArray(dimensions, data, attributes, encoding=encoding)


def decode_cf_variable(var, mask_and_scale=True):
    # use _data instead of data so as not to trigger loading data
    data = var._data
    dimensions = var.dimensions
    attributes = var.attributes.copy()
    encoding = var.encoding.copy()
    indexing_mode = var._indexing_mode

    def pop_to(source, dest, k):
        """
        A convenience function which pops a key k from source to dest.
        None values are not passed on.  If k already exists in dest an
        error is raised.
        """
        v = source.pop(k, None)
        if v is not None:
            if k in dest:
                raise ValueError("Failed hard to prevent overwriting key %s" % k)
            dest[k] = v
        return v

    if 'dtype' in encoding:
        if var.data.dtype != encoding['dtype']:
            raise ValueError("Refused to overwrite dtype")
    encoding['dtype'] = data.dtype

    if np.issubdtype(data.dtype, (str, unicode)):
        # TODO: add some sort of check instead of just assuming that the last
        # dimension on a character array is always the string dimension
        dimensions = dimensions[:-1]
        data = CharToStringArray(data)
    elif mask_and_scale:
        fill_value = pop_to(attributes, encoding, '_FillValue')
        scale_factor = pop_to(attributes, encoding, 'scale_factor')
        add_offset = pop_to(attributes, encoding, 'add_offset')
        if ((fill_value is not None and not np.isnan(fill_value))
                or scale_factor is not None or add_offset is not None):
            data = MaskedAndScaledArray(data, fill_value, scale_factor,
                                        add_offset)

    if 'units' in attributes and 'since' in attributes['units']:
        # convert CF times to datetimes
        # TODO: make this lazy
        data = var.data
        units = pop_to(attributes, encoding, 'units')
        calendar = pop_to(attributes, encoding, 'calendar')
        data = utils.decode_cf_datetime(data, units=units, calendar=calendar)
        indexing_mode = 'numpy'

    return xarray.XArray(dimensions, data, attributes, encoding=encoding,
                         indexing_mode=indexing_mode)
