"""Coders for individual Variable objects."""
from __future__ import annotations

import warnings
from functools import partial
from typing import Any, Hashable

import numpy as np
import pandas as pd

from ..core import dtypes, duck_array_ops, indexing
from ..core.pycompat import is_duck_dask_array
from ..core.variable import Variable


class SerializationWarning(RuntimeWarning):
    """Warnings about encoding/decoding issues in serialization."""


class VariableCoder:
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

    def encode(
        self, variable: Variable, name: Hashable = None
    ) -> Variable:  # pragma: no cover
        """Convert an encoded variable to a decoded variable"""
        raise NotImplementedError()

    def decode(
        self, variable: Variable, name: Hashable = None
    ) -> Variable:  # pragma: no cover
        """Convert an decoded variable to a encoded variable"""
        raise NotImplementedError()


class _ElementwiseFunctionArray(indexing.ExplicitlyIndexedNDArrayMixin):
    """Lazily computed array holding values of elemwise-function.

    Do not construct this object directly: call lazy_elemwise_func instead.

    Values are computed upon indexing or coercion to a NumPy array.
    """

    def __init__(self, array, func, dtype):
        assert not is_duck_dask_array(array)
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
        return "{}({!r}, func={!r}, dtype={!r})".format(
            type(self).__name__, self.array, self.func, self.dtype
        )


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
    if is_duck_dask_array(array):
        import dask.array as da

        return da.map_blocks(func, array, dtype=dtype)
    else:
        return _ElementwiseFunctionArray(array, func, dtype)


def unpack_for_encoding(var):
    return var.dims, var.data, var.attrs.copy(), var.encoding.copy()


def unpack_for_decoding(var):
    return var.dims, var._data, var.attrs.copy(), var.encoding.copy()


def safe_setitem(dest, key, value, name=None):
    if key in dest:
        var_str = f" on variable {name!r}" if name else ""
        raise ValueError(
            "failed to prevent overwriting existing key {} in attrs{}. "
            "This is probably an encoding field used by xarray to describe "
            "how a variable is serialized. To proceed, remove this key from "
            "the variable's attributes manually.".format(key, var_str)
        )
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


def _apply_mask(
    data: np.ndarray, encoded_fill_values: list, decoded_fill_value: Any, dtype: Any
) -> np.ndarray:
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

        dtype = np.dtype(encoding.get("dtype", data.dtype))
        fv = encoding.get("_FillValue")
        mv = encoding.get("missing_value")

        if (
            fv is not None
            and mv is not None
            and not duck_array_ops.allclose_or_equiv(fv, mv)
        ):
            raise ValueError(
                f"Variable {name!r} has conflicting _FillValue ({fv}) and missing_value ({mv}). Cannot encode data."
            )

        if fv is not None:
            # Ensure _FillValue is cast to same dtype as data's
            encoding["_FillValue"] = dtype.type(fv)
            fill_value = pop_to(encoding, attrs, "_FillValue", name=name)
            if not pd.isnull(fill_value):
                data = duck_array_ops.fillna(data, fill_value)

        if mv is not None:
            # Ensure missing_value is cast to same dtype as data's
            encoding["missing_value"] = dtype.type(mv)
            fill_value = pop_to(encoding, attrs, "missing_value", name=name)
            if not pd.isnull(fill_value) and fv is None:
                data = duck_array_ops.fillna(data, fill_value)

        return Variable(dims, data, attrs, encoding)

    def decode(self, variable, name=None):
        dims, data, attrs, encoding = unpack_for_decoding(variable)

        raw_fill_values = [
            pop_to(attrs, encoding, attr, name=name)
            for attr in ("missing_value", "_FillValue")
        ]
        if raw_fill_values:
            encoded_fill_values = {
                fv
                for option in raw_fill_values
                for fv in np.ravel(option)
                if not pd.isnull(fv)
            }

            if len(encoded_fill_values) > 1:
                warnings.warn(
                    "variable {!r} has multiple fill values {}, "
                    "decoding all values to NaN.".format(name, encoded_fill_values),
                    SerializationWarning,
                    stacklevel=3,
                )

            dtype, decoded_fill_value = dtypes.maybe_promote(data.dtype)

            if encoded_fill_values:
                transform = partial(
                    _apply_mask,
                    encoded_fill_values=encoded_fill_values,
                    decoded_fill_value=decoded_fill_value,
                    dtype=dtype,
                )
                data = lazy_elemwise_func(data, transform, dtype)

        return Variable(dims, data, attrs, encoding)


def _scale_offset_decoding(data, scale_factor, add_offset, dtype):
    data = np.array(data, dtype=dtype, copy=True)
    if scale_factor is not None:
        data *= scale_factor
    if add_offset is not None:
        data += add_offset
    return data


def _choose_float_dtype_encoding(dtype, has_offset):
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
        if has_offset:
            return np.float64
        else:
            return np.float32
    # For all other types and circumstances, we just use float64.
    # (safe because eg. complex numbers are not supported in NetCDF)
    return np.float64


def _choose_float_dtype_decoding(dtype, scale_factor, add_offset):
    """Return a dtype per CF convention standards."""

    # From CF standard:
    # If the scale_factor and add_offset attributes are of the same data type as the
    # associated variable, the unpacked data is assumed to be of the same data type as
    # the packed data.
    if (dtype == type(scale_factor) if scale_factor is not None else dtype) and (
        dtype == (type(add_offset) if add_offset is not None else dtype)
    ):
        ret_dtype = dtype

    # If here, one or both of scale_factor and add_offset exist, and do not match dtype

    # From CF standard:
    # However, if the scale_factor and add_offset attributes are of a different data
    # type from the variable (containing the packed data) then the unpacked data should
    # match the type of these attributes, which must both be of type float or both be of
    # type double.

    # Check that scale_factor and add_offset are float or double and the same.
    if (scale_factor is not None) and (add_offset is not None):
        if not np.issubdtype(type(scale_factor), np.double):
            print("Warning - non-compliant CF file: scale_factor not float or double")
        if not np.issubdtype(type(add_offset), np.double):
            print("Warning - non-compliant CF file: add_offset not float or double")
        if type(scale_factor) != type(add_offset):
            print(
                "Warning - non-compliant CF file: scale_factor and add_offset are different type"
            )

    # From CF standard:
    # An additional restriction in this case is that the variable containing the packed
    # data must be of type byte, short or int. It is not advised to unpack an int into a
    # float as there is a potential precision loss.

    # Check that dtype is byte, short, or int.
    if not np.issubdtype(dtype, np.integer):
        print("Warning - non-compliant CF file: dtype is not byte, short, or int")

    if scale_factor is None:
        ret_dtype = np.find_common_type([dtype, type(add_offset)], [])
    elif add_offset is None:
        ret_dtype = np.find_common_type([dtype, type(scale_factor)], [])
    else:
        ret_dtype = np.find_common_type(
            [
                dtype,
                type(scale_factor) if scale_factor is not None else None,
                type(add_offset) if add_offset is not None else None,
            ],
            [],
        )

    # Upcast half-precision to single-precision, because float16 is "intended for
    # storage but not computation"
    if ret_dtype.itemsize <= 4 and np.issubdtype(ret_dtype, np.floating):
        ret_dtype = np.float32
    return ret_dtype


class CFScaleOffsetCoder(VariableCoder):
    """Scale and offset variables according to CF conventions.

    Follows the formula:
        decode_values = encoded_values * scale_factor + add_offset
    """

    def encode(self, variable, name=None):
        dims, data, attrs, encoding = unpack_for_encoding(variable)

        scale_factor = pop_to(encoding, attrs, "scale_factor", name=name)
        add_offset = pop_to(encoding, attrs, "add_offset", name=name)
        dtype = _choose_float_dtype_encoding(data.dtype, add_offset in attrs)
        data = data.astype(dtype=dtype, copy=True)
        if add_offset is not None:
            data = data - add_offset
        if scale_factor is not None:
            data = data / scale_factor

        return Variable(dims, data, attrs, encoding)

    def decode(self, variable, name=None):
        dims, data, attrs, encoding = unpack_for_decoding(variable)

        if "scale_factor" in attrs or "add_offset" in attrs:
            scale_factor = pop_to(attrs, encoding, "scale_factor", name=name)
            add_offset = pop_to(attrs, encoding, "add_offset", name=name)
            dtype = _choose_float_dtype_decoding(data.dtype, scale_factor, add_offset)
            if np.ndim(scale_factor) > 0:
                scale_factor = np.asarray(scale_factor).item()
            if np.ndim(add_offset) > 0:
                add_offset = np.asarray(add_offset).item()
            transform = partial(
                _scale_offset_decoding,
                scale_factor=scale_factor,
                add_offset=add_offset,
                dtype=dtype,
            )
            data = lazy_elemwise_func(data, transform, dtype)

        return Variable(dims, data, attrs, encoding)


class UnsignedIntegerCoder(VariableCoder):
    def encode(self, variable, name=None):
        dims, data, attrs, encoding = unpack_for_encoding(variable)

        # from netCDF best practices
        # https://www.unidata.ucar.edu/software/netcdf/docs/BestPractices.html
        #     "_Unsigned = "true" to indicate that
        #      integer data should be treated as unsigned"
        if encoding.get("_Unsigned", "false") == "true":
            pop_to(encoding, attrs, "_Unsigned")
            signed_dtype = np.dtype(f"i{data.dtype.itemsize}")
            if "_FillValue" in attrs:
                new_fill = signed_dtype.type(attrs["_FillValue"])
                attrs["_FillValue"] = new_fill
            data = duck_array_ops.around(data).astype(signed_dtype)

        return Variable(dims, data, attrs, encoding)

    def decode(self, variable, name=None):
        dims, data, attrs, encoding = unpack_for_decoding(variable)

        if "_Unsigned" in attrs:
            unsigned = pop_to(attrs, encoding, "_Unsigned")

            if data.dtype.kind == "i":
                if unsigned == "true":
                    unsigned_dtype = np.dtype(f"u{data.dtype.itemsize}")
                    transform = partial(np.asarray, dtype=unsigned_dtype)
                    data = lazy_elemwise_func(data, transform, unsigned_dtype)
                    if "_FillValue" in attrs:
                        new_fill = unsigned_dtype.type(attrs["_FillValue"])
                        attrs["_FillValue"] = new_fill
            elif data.dtype.kind == "u":
                if unsigned == "false":
                    signed_dtype = np.dtype(f"i{data.dtype.itemsize}")
                    transform = partial(np.asarray, dtype=signed_dtype)
                    data = lazy_elemwise_func(data, transform, signed_dtype)
                    if "_FillValue" in attrs:
                        new_fill = signed_dtype.type(attrs["_FillValue"])
                        attrs["_FillValue"] = new_fill
            else:
                warnings.warn(
                    f"variable {name!r} has _Unsigned attribute but is not "
                    "of integer type. Ignoring attribute.",
                    SerializationWarning,
                    stacklevel=3,
                )

        return Variable(dims, data, attrs, encoding)
