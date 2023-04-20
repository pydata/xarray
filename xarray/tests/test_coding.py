from __future__ import annotations

from contextlib import suppress

import numpy as np
import pandas as pd
import pytest

from xarray import (
    DataArray,
    SerializationWarning,
    Variable,
)
from xarray.coding import variables
from xarray.conventions import (
    cf_encoder,
    decode_cf,
    decode_cf_variable,
    encode_cf_variable,
)
from xarray.tests import assert_allclose, assert_equal, assert_identical, requires_dask

with suppress(ImportError):
    import dask.array as da


def test_CFMaskCoder_decode() -> None:
    original = Variable(("x",), [0, -1, 1], {"_FillValue": -1})
    expected = Variable(("x",), [0, np.nan, 1])
    coder = variables.CFMaskCoder()
    encoded = coder.decode(original)
    assert_identical(expected, encoded)


encoding_with_dtype = {
    "dtype": np.dtype("float64"),
    "_FillValue": np.float32(1e20),
    "missing_value": np.float64(1e20),
}
encoding_without_dtype = {
    "_FillValue": np.float32(1e20),
    "missing_value": np.float64(1e20),
}
CFMASKCODER_ENCODE_DTYPE_CONFLICT_TESTS = {
    "numeric-with-dtype": ([0.0, -1.0, 1.0], encoding_with_dtype),
    "numeric-without-dtype": ([0.0, -1.0, 1.0], encoding_without_dtype),
    "times-with-dtype": (pd.date_range("2000", periods=3), encoding_with_dtype),
}


@pytest.mark.parametrize(
    ("data", "encoding"),
    CFMASKCODER_ENCODE_DTYPE_CONFLICT_TESTS.values(),
    ids=list(CFMASKCODER_ENCODE_DTYPE_CONFLICT_TESTS.keys()),
)
def test_CFMaskCoder_encode_missing_fill_values_conflict(data, encoding) -> None:
    original = Variable(("x",), data, encoding=encoding)
    encoded = encode_cf_variable(original)

    assert encoded.dtype == encoded.attrs["missing_value"].dtype
    assert encoded.dtype == encoded.attrs["_FillValue"].dtype

    with pytest.warns(variables.SerializationWarning):
        roundtripped = decode_cf_variable("foo", encoded)
        assert_identical(roundtripped, original)


def test_CFMaskCoder_missing_value() -> None:
    expected = DataArray(
        np.array([[26915, 27755, -9999, 27705], [25595, -9999, 28315, -9999]]),
        dims=["npts", "ntimes"],
        name="tmpk",
    )
    expected.attrs["missing_value"] = -9999

    decoded = decode_cf(expected.to_dataset())
    encoded, _ = cf_encoder(decoded.variables, decoded.attrs)

    assert_equal(encoded["tmpk"], expected.variable)

    decoded.tmpk.encoding["_FillValue"] = -9940
    with pytest.raises(ValueError):
        encoded, _ = cf_encoder(decoded.variables, decoded.attrs)


@requires_dask
def test_CFMaskCoder_decode_dask() -> None:
    original = Variable(("x",), [0, -1, 1], {"_FillValue": -1}).chunk()
    expected = Variable(("x",), [0, np.nan, 1])
    coder = variables.CFMaskCoder()
    encoded = coder.decode(original)
    assert isinstance(encoded.data, da.Array)
    assert_identical(expected, encoded)


# TODO(shoyer): port other fill-value tests


# TODO(shoyer): parameterize when we have more coders
def test_coder_roundtrip() -> None:
    original = Variable(("x",), [0.0, np.nan, 1.0])
    coder = variables.CFMaskCoder()
    roundtripped = coder.decode(coder.encode(original))
    assert_identical(original, roundtripped)


@pytest.mark.parametrize("dtype", "i1 i2 f2 f4".split())
def test_scaling_converts_to_float32(dtype) -> None:
    original = Variable(
        ("x",),
        np.arange(10, dtype=dtype),
        encoding=dict(scale_factor=np.float64(10), dtype=dtype),
    )
    coder = variables.CFScaleOffsetCoder()
    encoded = coder.encode(original)
    assert encoded.dtype == np.float32
    roundtripped = coder.decode(encoded)
    assert_identical(original, roundtripped)
    assert roundtripped.dtype == np.float32


@pytest.mark.parametrize("scale_factor", (10.0, [10.0]))
@pytest.mark.parametrize("add_offset", (0.1, [0.1]))
def test_scaling_offset_as_list(scale_factor, add_offset) -> None:
    # test for #4631
    encoding = dict(scale_factor=scale_factor, add_offset=add_offset, dtype="i2")
    original = Variable(("x",), np.arange(10.0), encoding=encoding)
    coder = variables.CFScaleOffsetCoder()
    encoded = coder.encode(original)
    roundtripped = coder.decode(encoded)
    assert_allclose(original, roundtripped)


@pytest.mark.parametrize("bits", [1, 2, 4, 8])
def test_decode_unsigned_from_signed(bits) -> None:
    unsigned_dtype = np.dtype(f"u{bits}")
    signed_dtype = np.dtype(f"i{bits}")
    original_values = np.array([np.iinfo(unsigned_dtype).max], dtype=unsigned_dtype)
    encoded = Variable(
        ("x",), original_values.astype(signed_dtype), attrs={"_Unsigned": "true"}
    )
    coder = variables.UnsignedIntegerCoder()
    decoded = coder.decode(encoded)
    assert decoded.dtype == unsigned_dtype
    assert decoded.values == original_values


@pytest.mark.parametrize("bits", [1, 2, 4, 8])
def test_decode_signed_from_unsigned(bits) -> None:
    unsigned_dtype = np.dtype(f"u{bits}")
    signed_dtype = np.dtype(f"i{bits}")
    original_values = np.array([-1], dtype=signed_dtype)
    encoded = Variable(
        ("x",), original_values.astype(unsigned_dtype), attrs={"_Unsigned": "false"}
    )
    coder = variables.UnsignedIntegerCoder()
    decoded = coder.decode(encoded)
    assert decoded.dtype == signed_dtype
    assert decoded.values == original_values


def test_ensure_scale_offset_conformance():
    # scale offset dtype mismatch
    mapping = dict(scale_factor=np.float16(10), add_offset=np.float64(-10), dtype="i2")
    with pytest.raises(ValueError, match="float16 is not allowed"):
        variables._ensure_scale_offset_conformance(mapping)

    # do nothing
    mapping = dict()
    assert variables._ensure_scale_offset_conformance(mapping) is None

    # do nothing
    mapping = dict(dtype="i2")
    assert variables._ensure_scale_offset_conformance(mapping) is None

    # mandatory packing information missing
    mapping = dict(scale_factor=np.float32(10))
    with pytest.raises(ValueError, match="Packed dtype information is missing!"):
        variables._ensure_scale_offset_conformance(mapping)

    # scale offset dtype mismatch 1
    mapping = dict(scale_factor=np.int32(10), add_offset=np.int32(10), dtype="i2")
    with pytest.warns(
        SerializationWarning, match="Must be either float32 or float64 dtype."
    ):
        variables._ensure_scale_offset_conformance(mapping, strict=False)
    with pytest.raises(ValueError, match="Must be either float32 or float64 dtype."):
        variables._ensure_scale_offset_conformance(mapping, strict=True)

    # packed dtype mismatch
    mapping = dict(scale_factor=np.float32(10), add_offset=np.float32(10), dtype="u2")
    with pytest.warns(
        SerializationWarning, match="Must be of type byte, short or int."
    ):
        variables._ensure_scale_offset_conformance(mapping, strict=False)
    with pytest.raises(ValueError, match="Must be of type byte, short or int."):
        variables._ensure_scale_offset_conformance(mapping, strict=True)

    # pack float32 into int32
    mapping = dict(scale_factor=np.float32(10), add_offset=np.float32(10), dtype="i4")
    with pytest.warns(SerializationWarning, match="Trying to pack float32 into int32."):
        variables._ensure_scale_offset_conformance(mapping)

    # scale offset dtype mismatch
    mapping = dict(scale_factor=np.float32(10), add_offset=np.float64(-10), dtype="i2")
    with pytest.warns(
        SerializationWarning,
        match="scale_factor dtype float32 and add_offset dtype float64 mismatch!",
    ):
        variables._ensure_scale_offset_conformance(mapping, strict=False)
    with pytest.raises(
        ValueError,
        match="scale_factor dtype float32 and add_offset dtype float64 mismatch!",
    ):
        variables._ensure_scale_offset_conformance(mapping, strict=True)
