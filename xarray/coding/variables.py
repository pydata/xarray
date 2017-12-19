"""Coders for individual Variable objects."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import warnings

import numpy as np
import pandas as pd

from ..core import dtypes
from ..core import duck_array_ops
from ..core import indexing
from ..core import utils
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
        return self.func(self.array[key])

    def __repr__(self):
        return ("%s(%r, func=%r, dtype=%r)" %
                (type(self).__name__, self.array, self._func, self._dtype))


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
                decoded_fill_value  # type: Any
                ):  # type: npndarray
    """Mask all matching values in a NumPy arrays."""
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
            variable = Variable(dims, data, attrs, encoding)

        if ('_FillValue' not in attrs and '_FillValue' not in encoding and
                np.issubdtype(data.dtype, np.floating)):
            attrs['_FillValue'] = data.dtype.type(np.nan)

        return Variable(dims, data, attrs, encoding)

    def decode(self, variable, name=None):
        dims, data, attrs, encoding = unpack_for_decoding(variable)

        if 'missing_value' in attrs:
            # missing_value is deprecated, but we still want to support it as
            # an alias for _FillValue.
            if ('_FillValue' in attrs and
                not utils.equivalent(attrs['_FillValue'],
                                     attrs['missing_value'])):
                raise ValueError("Conflicting _FillValue and missing_value "
                                 "attrs on a variable {!r}: {} vs. {}\n\n"
                                 "Consider opening the offending dataset "
                                 "using decode_cf=False, correcting the "
                                 "attrs and decoding explicitly using "
                                 "xarray.decode_cf()."
                                 .format(name, attrs['_FillValue'],
                                         attrs['missing_value']))
            attrs['_FillValue'] = attrs.pop('missing_value')

        if '_FillValue' in attrs:
            raw_fill_value = pop_to(attrs, encoding, '_FillValue', name=name)
            encoded_fill_values = [
                fv for fv in np.ravel(raw_fill_value) if not pd.isnull(fv)]

            if len(encoded_fill_values) > 1:
                warnings.warn("variable {!r} has multiple fill values {}, "
                              "decoding all values to NaN."
                              .format(name, encoded_fill_values),
                              SerializationWarning, stacklevel=3)

            dtype, decoded_fill_value = dtypes.maybe_promote(data.dtype)

            if encoded_fill_values:
                transform = partial(_apply_mask,
                                    encoded_fill_values=encoded_fill_values,
                                    decoded_fill_value=decoded_fill_value)
                data = lazy_elemwise_func(data, transform, dtype)

        return Variable(dims, data, attrs, encoding)
