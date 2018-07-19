"""Coders for individual Variable objects."""
from __future__ import absolute_import, division, print_function

import warnings
from functools import partial

import numpy as np
import pandas as pd

from ..core import dtypes, duck_array_ops, indexing
from ..core.pycompat import dask_array_type
from ..core.variable import Variable


class SerializationWarning(RuntimeWarning):
    """Warnings about encoding/decoding issues in serialization."""


class VariableCoder(object):
    """Base class for encoding and decoding transformations on variables.

    We use coders for transforming variables between xarray's data model and
    a format suitable for serialization. For example, coders apply CF
    conventions for how data should be represented in netCDF files.

    Subclasses should implement encode() and decode(), which should satisfy
    the identity ``coder.decode(coder.encode(variable)) == variable``. If any
    options are necessary, they should be implemented as arguments to the
    __init__ method.

    The optional name argument to encode() and decode() exists solely for the
    sake of better error messages, and should correspond to the name of
    variables in the underlying store.
    """

    def encode(self, variable, name=None):
        # type: (Variable, Any) -> Variable
        """Convert an encoded variable to a decoded variable."""
        raise NotImplementedError

    def decode(self, variable, name=None):
        # type: (Variable, Any) -> Variable
        """Convert an decoded variable to a encoded variable."""
        raise NotImplementedError


class _ElementwiseFunctionArray(indexing.ExplicitlyIndexedNDArrayMixin):
    """Lazily computed array holding values of elemwise-function.

    Do not construct this object directly: call lazy_elemwise_func instead.

    Values are computed upon indexing or coercion to a NumPy array.
    """

    def __init__(self, array, func, dtype):
        assert not isinstance(array, dask_array_type)
        self.array = indexing.as_indexable(array)
        self.func = func
        self._dtype = dtype

    @property
    def dtype(self):
        return np.dtype(self._dtype)

    def __getitem__(self, key):
        return type(self)(self.array[key], self.func, self.dtype)

    def __array__(self, dtype=None):
        return self.func(self.array)

    def __repr__(self):
        return ("%s(%r, func=%r, dtype=%r)" %
                (type(self).__name__, self.array, self.func, self.dtype))


def lazy_elemwise_func(array, func, dtype):
    """Lazily apply an element-wise function to an array.

    Parameters
    ----------
    array : any valid value of Variable._data
    func : callable
        Function to apply to indexed slices of an array. For use with dask,
        this should be a pickle-able object.
    dtype : coercible to np.dtype
        Dtype for the result of this function.

    Returns
    -------
    Either a dask.array.Array or _ElementwiseFunctionArray.
    """
    if isinstance(array, dask_array_type):
        return array.map_blocks(func, dtype=dtype)
    else:
        return _ElementwiseFunctionArray(array, func, dtype)


def unpack_for_encoding(var):
    return var.dims, var.data, var.attrs.copy(), var.encoding.copy()


def unpack_for_decoding(var):
    return var.dims, var._data, var.attrs.copy(), var.encoding.copy()


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


def _apply_mask(data,  # type: np.ndarray
                encoded_fill_values,  # type: list
                decoded_fill_value,  # type: Any
                dtype,  # type: Any
                ):  # type: np.ndarray
    """Mask all matching values in a NumPy arrays."""
    data = np.asarray(data, dtype=dtype)
    condition = False
    for fv in encoded_fill_values:
        condition |= data == fv
    return np.where(condition, decoded_fill_value, data)


class CFMaskCoder(VariableCoder):
    """Mask or unmask fill values according to CF conventions."""

    def encode(self, variable, name=None):
        dims, data, attrs, encoding = unpack_for_encoding(variable)

        if encoding.get('_FillValue') is not None:
            fill_value = pop_to(encoding, attrs, '_FillValue', name=name)
            if not pd.isnull(fill_value):
                data = duck_array_ops.fillna(data, fill_value)

        return Variable(dims, data, attrs, encoding)

    def decode(self, variable, name=None):
        dims, data, attrs, encoding = unpack_for_decoding(variable)

        raw_fill_values = [pop_to(attrs, encoding, attr, name=name)
                           for attr in ('missing_value', '_FillValue')]
        if raw_fill_values:
            encoded_fill_values = {fv for option in raw_fill_values
                                   for fv in np.ravel(option)
                                   if not pd.isnull(fv)}

            if len(encoded_fill_values) > 1:
                warnings.warn("variable {!r} has multiple fill values {}, "
                              "decoding all values to NaN."
                              .format(name, encoded_fill_values),
                              SerializationWarning, stacklevel=3)

            dtype, decoded_fill_value = dtypes.maybe_promote(data.dtype)

            if encoded_fill_values:
                transform = partial(_apply_mask,
                                    encoded_fill_values=encoded_fill_values,
                                    decoded_fill_value=decoded_fill_value,
                                    dtype=dtype)
                data = lazy_elemwise_func(data, transform, dtype)

        return Variable(dims, data, attrs, encoding)


def _scale_offset_decoding(data, scale_factor, add_offset, dtype):
    data = np.array(data, dtype=dtype, copy=True)
    if scale_factor is not None:
        data *= scale_factor
    if add_offset is not None:
        data += add_offset
    return data


def _choose_float_dtype(dtype, has_offset):
    """Return a float dtype that can losslessly represent `dtype` values."""
    # Keep float32 as-is.  Upcast half-precision to single-precision,
    # because float16 is "intended for storage but not computation"
    if dtype.itemsize <= 4 and np.issubdtype(dtype, np.floating):
        return np.float32
    # float32 can exactly represent all integers up to 24 bits
    if dtype.itemsize <= 2 and np.issubdtype(dtype, np.integer):
        # A scale factor is entirely safe (vanishing into the mantissa),
        # but a large integer offset could lead to loss of precision.
        # Sensitivity analysis can be tricky, so we just use a float64
        # if there's any offset at all - better unoptimised than wrong!
        if not has_offset:
            return np.float32
    # For all other types and circumstances, we just use float64.
    # (safe because eg. complex numbers are not supported in NetCDF)
    return np.float64


class CFScaleOffsetCoder(VariableCoder):
    """Scale and offset variables according to CF conventions.

    Follows the formula:
        decode_values = encoded_values * scale_factor + add_offset
    """

    def encode(self, variable, name=None):
        dims, data, attrs, encoding = unpack_for_encoding(variable)

        if 'scale_factor' in encoding or 'add_offset' in encoding:
            dtype = _choose_float_dtype(data.dtype, 'add_offset' in encoding)
            data = data.astype(dtype=dtype, copy=True)
            if 'add_offset' in encoding:
                data -= pop_to(encoding, attrs, 'add_offset', name=name)
            if 'scale_factor' in encoding:
                data /= pop_to(encoding, attrs, 'scale_factor', name=name)

        return Variable(dims, data, attrs, encoding)

    def decode(self, variable, name=None):
        dims, data, attrs, encoding = unpack_for_decoding(variable)

        if 'scale_factor' in attrs or 'add_offset' in attrs:
            scale_factor = pop_to(attrs, encoding, 'scale_factor', name=name)
            add_offset = pop_to(attrs, encoding, 'add_offset', name=name)
            dtype = _choose_float_dtype(data.dtype, 'add_offset' in attrs)
            transform = partial(_scale_offset_decoding,
                                scale_factor=scale_factor,
                                add_offset=add_offset,
                                dtype=dtype)
            data = lazy_elemwise_func(data, transform, dtype)

        return Variable(dims, data, attrs, encoding)


class UnsignedIntegerCoder(VariableCoder):

    def encode(self, variable, name=None):
        dims, data, attrs, encoding = unpack_for_encoding(variable)

        if encoding.get('_Unsigned', False):
            pop_to(encoding, attrs, '_Unsigned')
            signed_dtype = np.dtype('i%s' % data.dtype.itemsize)
            if '_FillValue' in attrs:
                new_fill = signed_dtype.type(attrs['_FillValue'])
                attrs['_FillValue'] = new_fill
            data = duck_array_ops.around(data).astype(signed_dtype)

        return Variable(dims, data, attrs, encoding)

    def decode(self, variable, name=None):
        dims, data, attrs, encoding = unpack_for_decoding(variable)

        if '_Unsigned' in attrs:
            unsigned = pop_to(attrs, encoding, '_Unsigned')

            if data.dtype.kind == 'i':
                if unsigned:
                    unsigned_dtype = np.dtype('u%s' % data.dtype.itemsize)
                    transform = partial(np.asarray, dtype=unsigned_dtype)
                    data = lazy_elemwise_func(data, transform, unsigned_dtype)
                    if '_FillValue' in attrs:
                        new_fill = unsigned_dtype.type(attrs['_FillValue'])
                        attrs['_FillValue'] = new_fill
            else:
                warnings.warn("variable %r has _Unsigned attribute but is not "
                              "of integer type. Ignoring attribute." % name,
                              SerializationWarning, stacklevel=3)

        return Variable(dims, data, attrs, encoding)
