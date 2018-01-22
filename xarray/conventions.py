from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
from collections import defaultdict

import numpy as np

import pandas as pd

from .coding import times
from .coding import variables
from .coding.variables import SerializationWarning
from .core import duck_array_ops, indexing
from .core.pycompat import OrderedDict, basestring, iteritems
from .core.variable import IndexVariable, Variable, as_variable


class StackedBytesArray(indexing.ExplicitlyIndexedNDArrayMixin):
    """Wrapper around array-like objects to create a new indexable object where
    values, when accessed, are automatically stacked along the last dimension.

    >>> StackedBytesArray(np.array(['a', 'b', 'c']))[:]
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
        if array.dtype != 'S1':
            raise ValueError(
                "can only use StackedBytesArray if argument has dtype='S1'")
        self.array = indexing.as_indexable(array)

    @property
    def dtype(self):
        return np.dtype('S' + str(self.array.shape[-1]))

    @property
    def shape(self):
        return self.array.shape[:-1]

    def __str__(self):
        # TODO(shoyer): figure out why we need this special case?
        if self.ndim == 0:
            return str(np.array(self).item())
        else:
            return repr(self)

    def __repr__(self):
        return ('%s(%r)' % (type(self).__name__, self.array))

    def __getitem__(self, key):
        # require slicing the last dimension completely
        key = type(key)(indexing.expanded_indexer(key.tuple, self.array.ndim))
        if key.tuple[-1] != slice(None):
            raise IndexError('too many indices')
        return char_to_bytes(self.array[key])


class BytesToStringArray(indexing.ExplicitlyIndexedNDArrayMixin):
    """Wrapper that decodes bytes to unicode when values are read.

    >>> BytesToStringArray(np.array([b'abc']))[:]
    array(['abc'],
          dtype=object)
    """

    def __init__(self, array, encoding='utf-8'):
        """
        Parameters
        ----------
        array : array-like
            Original array of values to wrap.
        encoding : str
            String encoding to use.
        """
        self.array = indexing.as_indexable(array)
        self.encoding = encoding

    @property
    def dtype(self):
        # variable length string
        return np.dtype(object)

    def __str__(self):
        # TODO(shoyer): figure out why we need this special case?
        if self.ndim == 0:
            return str(np.array(self).item())
        else:
            return repr(self)

    def __repr__(self):
        return ('%s(%r, encoding=%r)'
                % (type(self).__name__, self.array, self.encoding))

    def __getitem__(self, key):
        return decode_bytes_array(self.array[key], self.encoding)


class NativeEndiannessArray(indexing.ExplicitlyIndexedNDArrayMixin):
    """Decode arrays on the fly from non-native to native endianness

    This is useful for decoding arrays from netCDF3 files (which are all
    big endian) into native endianness, so they can be used with Cython
    functions, such as those found in bottleneck and pandas.

    >>> x = np.arange(5, dtype='>i2')

    >>> x.dtype
    dtype('>i2')

    >>> NativeEndianArray(x).dtype
    dtype('int16')

    >>> NativeEndianArray(x)[:].dtype
    dtype('int16')
    """

    def __init__(self, array):
        self.array = indexing.as_indexable(array)

    @property
    def dtype(self):
        return np.dtype(self.array.dtype.kind + str(self.array.dtype.itemsize))

    def __getitem__(self, key):
        return np.asarray(self.array[key], dtype=self.dtype)


class BoolTypeArray(indexing.ExplicitlyIndexedNDArrayMixin):
    """Decode arrays on the fly from integer to boolean datatype

    This is useful for decoding boolean arrays from integer typed netCDF
    variables.

    >>> x = np.array([1, 0, 1, 1, 0], dtype='i1')

    >>> x.dtype
    dtype('>i2')

    >>> BoolTypeArray(x).dtype
    dtype('bool')

    >>> BoolTypeArray(x)[:].dtype
    dtype('bool')
    """

    def __init__(self, array):
        self.array = indexing.as_indexable(array)

    @property
    def dtype(self):
        return np.dtype('bool')

    def __getitem__(self, key):
        return np.asarray(self.array[key], dtype=self.dtype)


def bytes_to_char(arr):
    """Like netCDF4.stringtochar, but faster and more flexible.
    """
    # ensure the array is contiguous
    arr = np.array(arr, copy=False, order='C')
    kind = arr.dtype.kind
    if kind not in ['U', 'S']:
        raise ValueError('argument must be a string array')
    return arr.reshape(arr.shape + (1,)).view(kind + '1')


def char_to_bytes(arr):
    """Like netCDF4.chartostring, but faster and more flexible.
    """
    # based on: http://stackoverflow.com/a/10984878/809705
    arr = np.array(arr, copy=False, order='C')

    kind = arr.dtype.kind
    if kind not in ['U', 'S']:
        raise ValueError('argument must be a string array')

    if not arr.ndim:
        # no dimension to concatenate along
        return arr

    size = arr.shape[-1]
    if not size:
        # can't make an S0 dtype
        return np.zeros(arr.shape[:-1], dtype=kind)

    dtype = kind + str(size)
    return arr.view(dtype).reshape(arr.shape[:-1])


def decode_bytes_array(bytes_array, encoding='utf-8'):
    # This is faster than using np.char.decode() or np.vectorize()
    bytes_array = np.asarray(bytes_array)
    decoded = [x.decode(encoding) for x in bytes_array.ravel()]
    return np.array(decoded, dtype=object).reshape(bytes_array.shape)


def encode_string_array(string_array, encoding='utf-8'):
    string_array = np.asarray(string_array)
    encoded = [x.encode(encoding) for x in string_array.ravel()]
    return np.array(encoded, dtype=bytes).reshape(string_array.shape)


def safe_setitem(dest, key, value, name=None):
    if key in dest:
        var_str = ' on variable {!r}'.format(name) if name else ''
        raise ValueError(
            'failed to prevent overwriting existing key {} in attrs{}. '
            'This is probably an encoding field used by xarray to describe '
            'how a variable is serialized. To proceed, remove this key from '
            "the variable's attributes manually.".format(key, var_str))
    dest[key] = value


def pop_to(source, dest, key, name=None):
    """
    A convenience function which pops a key k from source to dest.
    None values are not passed on.  If k already exists in dest an
    error is raised.
    """
    value = source.pop(key, None)
    if value is not None:
        safe_setitem(dest, key, value, name=name)
    return value


def _var_as_tuple(var):
    return var.dims, var.data, var.attrs.copy(), var.encoding.copy()


def maybe_encode_as_char_array(var, name=None):
    if var.dtype.kind in {'S', 'U'}:
        dims, data, attrs, encoding = _var_as_tuple(var)
        if data.dtype.kind == 'U':
            if '_FillValue' in attrs:
                raise NotImplementedError(
                    'variable {!r} has a _FillValue specified, but '
                    '_FillValue is yet supported on unicode strings: '
                    'https://github.com/pydata/xarray/issues/1647'
                    .format(name))

            string_encoding = encoding.pop('_Encoding', 'utf-8')
            safe_setitem(attrs, '_Encoding', string_encoding, name=name)
            data = encode_string_array(data, string_encoding)

        if data.dtype.itemsize > 1:
            data = bytes_to_char(data)
            dims = dims + ('string%s' % data.shape[-1],)

        var = Variable(dims, data, attrs, encoding)
    return var


def maybe_encode_string_dtype(var, name=None):
    # need to apply after ensure_dtype_not_object()
    if 'dtype' in var.encoding and var.encoding['dtype'] == 'S1':
        assert var.dtype.kind in {'S', 'U'}
        var = maybe_encode_as_char_array(var, name=name)
        del var.encoding['dtype']
    return var


def maybe_encode_nonstring_dtype(var, name=None):
    if 'dtype' in var.encoding and var.encoding['dtype'] != 'S1':
        dims, data, attrs, encoding = _var_as_tuple(var)
        dtype = np.dtype(encoding.pop('dtype'))
        if dtype != var.dtype:
            if np.issubdtype(dtype, np.integer):
                if (np.issubdtype(var.dtype, np.floating) and
                        '_FillValue' not in var.attrs):
                    warnings.warn('saving variable %s with floating '
                                  'point data as an integer dtype without '
                                  'any _FillValue to use for NaNs' % name,
                                  SerializationWarning, stacklevel=3)
                data = duck_array_ops.around(data)[...]
            data = data.astype(dtype=dtype)
        var = Variable(dims, data, attrs, encoding)
    return var


def maybe_default_fill_value(var):
    # make NaN the fill value for float types:
    if ('_FillValue' not in var.attrs and
            '_FillValue' not in var.encoding and
            np.issubdtype(var.dtype, np.floating)):
        var.attrs['_FillValue'] = var.dtype.type(np.nan)
    return var


def maybe_encode_bools(var):
    if ((var.dtype == np.bool) and
            ('dtype' not in var.encoding) and ('dtype' not in var.attrs)):
        dims, data, attrs, encoding = _var_as_tuple(var)
        attrs['dtype'] = 'bool'
        data = data.astype(dtype='i1', copy=True)
        var = Variable(dims, data, attrs, encoding)
    return var


def _infer_dtype(array, name=None):
    """Given an object array with no missing values, infer its dtype from its
    first element
    """
    if array.size == 0:
        dtype = np.dtype(float)
    else:
        dtype = np.array(array[(0,) * array.ndim]).dtype
        if dtype.kind in ['S', 'U']:
            # don't just use inferred dtype to avoid truncating arrays to
            # the length of their first element
            dtype = np.dtype(dtype.kind)
        elif dtype.kind == 'O':
            raise ValueError('unable to infer dtype on variable {!r}; xarray '
                             'cannot serialize arbitrary Python objects'
                             .format(name))
    return dtype


def ensure_not_multiindex(var, name=None):
    if (isinstance(var, IndexVariable) and
            isinstance(var.to_index(), pd.MultiIndex)):
        raise NotImplementedError(
            'variable {!r} is a MultiIndex, which cannot yet be '
            'serialized to netCDF files '
            '(https://github.com/pydata/xarray/issues/1077). Use '
            'reset_index() to convert MultiIndex levels into coordinate '
            'variables instead.'.format(name))


def ensure_dtype_not_object(var, name=None):
    # TODO: move this from conventions to backends? (it's not CF related)
    if var.dtype.kind == 'O':
        dims, data, attrs, encoding = _var_as_tuple(var)
        missing = pd.isnull(data)
        if missing.any():
            # nb. this will fail for dask.array data
            non_missing_values = data[~missing]
            inferred_dtype = _infer_dtype(non_missing_values, name)

            # There is no safe bit-pattern for NA in typical binary string
            # formats, we so can't set a fill_value. Unfortunately, this means
            # we can't distinguish between missing values and empty strings.
            if inferred_dtype.kind == 'S':
                fill_value = b''
            elif inferred_dtype.kind == 'U':
                fill_value = u''
            else:
                # insist on using float for numeric values
                if not np.issubdtype(inferred_dtype, np.floating):
                    inferred_dtype = np.dtype(float)
                fill_value = inferred_dtype.type(np.nan)

            data = np.array(data, dtype=inferred_dtype, copy=True)
            data[missing] = fill_value
        else:
            data = data.astype(dtype=_infer_dtype(data, name))
        var = Variable(dims, data, attrs, encoding)
    return var


def encode_cf_variable(var, needs_copy=True, name=None):
    """
    Converts an Variable into an Variable which follows some
    of the CF conventions:

        - Nans are masked using _FillValue (or the deprecated missing_value)
        - Rescaling via: scale_factor and add_offset
        - datetimes are converted to the CF 'units since time' format
        - dtype encodings are enforced.

    Parameters
    ----------
    var : xarray.Variable
        A variable holding un-encoded data.

    Returns
    -------
    out : xarray.Variable
        A variable which has been encoded as described above.
    """
    ensure_not_multiindex(var, name=name)

    for coder in [times.CFDatetimeCoder(),
                  times.CFTimedeltaCoder(),
                  variables.CFScaleOffsetCoder(),
                  variables.CFMaskCoder(),
                  variables.UnsignedIntegerCoder()]:
        var = coder.encode(var, name=name)

    # TODO(shoyer): convert all of these to use coders, too:
    var = maybe_encode_nonstring_dtype(var, name=name)
    var = maybe_default_fill_value(var)
    var = maybe_encode_bools(var)
    var = ensure_dtype_not_object(var, name=name)
    var = maybe_encode_string_dtype(var, name=name)
    return var


def decode_cf_variable(name, var, concat_characters=True, mask_and_scale=True,
                       decode_times=True, decode_endianness=True,
                       stack_char_dim=True):
    """
    Decodes a variable which may hold CF encoded information.

    This includes variables that have been masked and scaled, which
    hold CF style time variables (this is almost always the case if
    the dataset has been serialized) and which have strings encoded
    as character arrays.

    Parameters
    ----------
    name: str
        Name of the variable. Used for better error messages.
    var : Variable
        A variable holding potentially CF encoded information.
    concat_characters : bool
        Should character arrays be concatenated to strings, for
        example: ['h', 'e', 'l', 'l', 'o'] -> 'hello'
    mask_and_scale: bool
        Lazily scale (using scale_factor and add_offset) and mask
        (using _FillValue). If the _Unsigned attribute is present
        treat integer arrays as unsigned.
    decode_times : bool
        Decode cf times ('hours since 2000-01-01') to np.datetime64.
    decode_endianness : bool
        Decode arrays from non-native to native endianness.
    stack_char_dim : bool
        Whether to stack characters into bytes along the last dimension of this
        array. Passed as an argument because we need to look at the full
        dataset to figure out if this is appropriate.

    Returns
    -------
    out : Variable
        A variable holding the decoded equivalent of var.
    """
    # use _data instead of data so as not to trigger loading data
    var = as_variable(var)
    data = var._data
    dimensions = var.dims
    attributes = var.attrs.copy()
    encoding = var.encoding.copy()

    original_dtype = data.dtype

    if concat_characters and data.dtype.kind == 'S':
        if stack_char_dim:
            dimensions = dimensions[:-1]
            data = StackedBytesArray(data)

        string_encoding = pop_to(attributes, encoding, '_Encoding')
        if string_encoding is not None:
            data = BytesToStringArray(data, string_encoding)

    # TODO(shoyer): convert everything above to use coders
    var = Variable(dimensions, data, attributes, encoding)

    if mask_and_scale:
        for coder in [variables.UnsignedIntegerCoder(),
                      variables.CFMaskCoder(),
                      variables.CFScaleOffsetCoder()]:
            var = coder.decode(var, name=name)
    if decode_times:
        for coder in [times.CFTimedeltaCoder(),
                      times.CFDatetimeCoder()]:
            var = coder.decode(var, name=name)

    dimensions, data, attributes, encoding = (
        variables.unpack_for_decoding(var))
    # TODO(shoyer): convert everything below to use coders

    if decode_endianness and not data.dtype.isnative:
        # do this last, so it's only done if we didn't already unmask/scale
        data = NativeEndiannessArray(data)
        original_dtype = data.dtype

    if 'dtype' in encoding:
        if original_dtype != encoding['dtype']:
            warnings.warn("CF decoding is overwriting dtype on variable {!r}"
                          .format(name))
    else:
        encoding['dtype'] = original_dtype

    if 'dtype' in attributes and attributes['dtype'] == 'bool':
        del attributes['dtype']
        data = BoolTypeArray(data)

    return Variable(dimensions, indexing.LazilyIndexedArray(data),
                    attributes, encoding=encoding)


def decode_cf_variables(variables, attributes, concat_characters=True,
                        mask_and_scale=True, decode_times=True,
                        decode_coords=True, drop_variables=None):
    """
    Decode a several CF encoded variables.

    See: decode_cf_variable
    """
    dimensions_used_by = defaultdict(list)
    for v in variables.values():
        for d in v.dims:
            dimensions_used_by[d].append(v)

    def stackable(dim):
        # figure out if a dimension can be concatenated over
        if dim in variables:
            return False
        for v in dimensions_used_by[dim]:
            if v.dtype.kind != 'S' or dim != v.dims[-1]:
                return False
        return True

    coord_names = set()

    if isinstance(drop_variables, basestring):
        drop_variables = [drop_variables]
    elif drop_variables is None:
        drop_variables = []
    drop_variables = set(drop_variables)

    new_vars = OrderedDict()
    for k, v in iteritems(variables):
        if k in drop_variables:
            continue
        stack_char_dim = (concat_characters and v.dtype == 'S1' and
                          v.ndim > 0 and stackable(v.dims[-1]))
        new_vars[k] = decode_cf_variable(
            k, v, concat_characters=concat_characters,
            mask_and_scale=mask_and_scale, decode_times=decode_times,
            stack_char_dim=stack_char_dim)
        if decode_coords:
            var_attrs = new_vars[k].attrs
            if 'coordinates' in var_attrs:
                coord_str = var_attrs['coordinates']
                var_coord_names = coord_str.split()
                if all(k in variables for k in var_coord_names):
                    new_vars[k].encoding['coordinates'] = coord_str
                    del var_attrs['coordinates']
                    coord_names.update(var_coord_names)

    if decode_coords and 'coordinates' in attributes:
        attributes = OrderedDict(attributes)
        coord_names.update(attributes.pop('coordinates').split())

    return new_vars, attributes, coord_names


def decode_cf(obj, concat_characters=True, mask_and_scale=True,
              decode_times=True, decode_coords=True, drop_variables=None):
    """Decode the given Dataset or Datastore according to CF conventions into
    a new Dataset.

    Parameters
    ----------
    obj : Dataset or DataStore
        Object to decode.
    concat_characters : bool, optional
        Should character arrays be concatenated to strings, for
        example: ['h', 'e', 'l', 'l', 'o'] -> 'hello'
    mask_and_scale: bool, optional
        Lazily scale (using scale_factor and add_offset) and mask
        (using _FillValue).
    decode_times : bool, optional
        Decode cf times (e.g., integers since 'hours since 2000-01-01') to
        np.datetime64.
    decode_coords : bool, optional
        Use the 'coordinates' attribute on variable (or the dataset itself) to
        identify coordinates.
    drop_variables: string or iterable, optional
        A variable or list of variables to exclude from being parsed from the
        dataset. This may be useful to drop variables with problems or
        inconsistent values.

    Returns
    -------
    decoded : Dataset
    """
    from .core.dataset import Dataset
    from .backends.common import AbstractDataStore

    if isinstance(obj, Dataset):
        vars = obj._variables
        attrs = obj.attrs
        extra_coords = set(obj.coords)
        file_obj = obj._file_obj
        encoding = obj.encoding
    elif isinstance(obj, AbstractDataStore):
        vars, attrs = obj.load()
        extra_coords = set()
        file_obj = obj
        encoding = obj.get_encoding()
    else:
        raise TypeError('can only decode Dataset or DataStore objects')

    vars, attrs, coord_names = decode_cf_variables(
        vars, attrs, concat_characters, mask_and_scale, decode_times,
        decode_coords, drop_variables=drop_variables)
    ds = Dataset(vars, attrs=attrs)
    ds = ds.set_coords(coord_names.union(extra_coords).intersection(vars))
    ds._file_obj = file_obj
    ds.encoding = encoding

    return ds


def cf_decoder(variables, attributes,
               concat_characters=True, mask_and_scale=True,
               decode_times=True):
    """
    Decode a set of CF encoded variables and attributes.

    See Also, decode_cf_variable

    Parameters
    ----------
    variables : dict
        A dictionary mapping from variable name to xarray.Variable
    attributes : dict
        A dictionary mapping from attribute name to value
    concat_characters : bool
        Should character arrays be concatenated to strings, for
        example: ['h', 'e', 'l', 'l', 'o'] -> 'hello'
    mask_and_scale: bool
        Lazily scale (using scale_factor and add_offset) and mask
        (using _FillValue).
    decode_times : bool
        Decode cf times ('hours since 2000-01-01') to np.datetime64.

    Returns
    -------
    decoded_variables : dict
        A dictionary mapping from variable name to xarray.Variable objects.
    decoded_attributes : dict
        A dictionary mapping from attribute name to values.
    """
    variables, attributes, _ = decode_cf_variables(
        variables, attributes, concat_characters, mask_and_scale, decode_times)
    return variables, attributes


def _encode_coordinates(variables, attributes, non_dim_coord_names):
    # calculate global and variable specific coordinates
    non_dim_coord_names = set(non_dim_coord_names)

    for name in list(non_dim_coord_names):
        if isinstance(name, basestring) and ' ' in name:
            warnings.warn(
                'coordinate {!r} has a space in its name, which means it '
                'cannot be marked as a coordinate on disk and will be '
                'saved as a data variable instead'.format(name),
                SerializationWarning, stacklevel=6)
            non_dim_coord_names.discard(name)

    global_coordinates = non_dim_coord_names.copy()
    variable_coordinates = defaultdict(set)
    for coord_name in non_dim_coord_names:
        target_dims = variables[coord_name].dims
        for k, v in variables.items():
            if (k not in non_dim_coord_names and k not in v.dims and
                    set(target_dims) <= set(v.dims)):
                variable_coordinates[k].add(coord_name)
                global_coordinates.discard(coord_name)

    variables = OrderedDict((k, v.copy(deep=False))
                            for k, v in variables.items())

    # These coordinates are saved according to CF conventions
    for var_name, coord_names in variable_coordinates.items():
        attrs = variables[var_name].attrs
        if 'coordinates' in attrs:
            raise ValueError('cannot serialize coordinates because variable '
                             "%s already has an attribute 'coordinates'"
                             % var_name)
        attrs['coordinates'] = ' '.join(map(str, coord_names))

    # These coordinates are not associated with any particular variables, so we
    # save them under a global 'coordinates' attribute so xarray can roundtrip
    # the dataset faithfully. Because this serialization goes beyond CF
    # conventions, only do it if necessary.
    # Reference discussion:
    # http://mailman.cgd.ucar.edu/pipermail/cf-metadata/2014/057771.html
    if global_coordinates:
        attributes = OrderedDict(attributes)
        if 'coordinates' in attributes:
            raise ValueError('cannot serialize coordinates because the global '
                             "attribute 'coordinates' already exists")
        attributes['coordinates'] = ' '.join(map(str, global_coordinates))

    return variables, attributes


def encode_dataset_coordinates(dataset):
    """Encode coordinates on the given dataset object into variable specific
    and global attributes.

    When possible, this is done according to CF conventions.

    Parameters
    ----------
    dataset : Dataset
        Object to encode.

    Returns
    -------
    variables : dict
    attrs : dict
    """
    non_dim_coord_names = set(dataset.coords) - set(dataset.dims)
    return _encode_coordinates(dataset._variables, dataset.attrs,
                               non_dim_coord_names=non_dim_coord_names)


def cf_encoder(variables, attributes):
    """
    A function which takes a dicts of variables and attributes
    and encodes them to conform to CF conventions as much
    as possible.  This includes masking, scaling, character
    array handling, and CF-time encoding.

    Decode a set of CF encoded variables and attributes.

    See Also, decode_cf_variable

    Parameters
    ----------
    variables : dict
        A dictionary mapping from variable name to xarray.Variable
    attributes : dict
        A dictionary mapping from attribute name to value

    Returns
    -------
    encoded_variables : dict
        A dictionary mapping from variable name to xarray.Variable,
    encoded_attributes : dict
        A dictionary mapping from attribute name to value

    See also: encode_cf_variable
    """
    new_vars = OrderedDict((k, encode_cf_variable(v, name=k))
                           for k, v in iteritems(variables))
    return new_vars, attributes
