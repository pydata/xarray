from __future__ import annotations

from contextlib import suppress

import numpy as np
import pandas as pd
import pytest

import xarray as xr
from xarray.coding import variables
from xarray.conventions import decode_cf_variable, encode_cf_variable
from xarray.tests import assert_allclose, assert_equal, assert_identical, requires_dask

with suppress(ImportError):
    import dask.array as da


def test_CFMaskCoder_decode() -> None:
    original = xr.Variable(("x",), [0, -1, 1], {"_FillValue": -1})
    expected = xr.Variable(("x",), [0, np.nan, 1])
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
    original = xr.Variable(("x",), data, encoding=encoding)
    encoded = encode_cf_variable(original)

    assert encoded.dtype == encoded.attrs["missing_value"].dtype
    assert encoded.dtype == encoded.attrs["_FillValue"].dtype


def test_CFMaskCoder_multiple_missing_values_conflict():
    data = np.array([0.0, -1.0, 1.0])
    attrs = dict(_FillValue=np.float64(1e20), missing_value=np.float64(1e21))
    original = xr.Variable(("x",), data, attrs=attrs)
    with pytest.warns(variables.SerializationWarning):
        decoded = decode_cf_variable("foo", original)
    with pytest.raises(ValueError):
        encode_cf_variable(decoded)


def test_CFMaskCoder_missing_value() -> None:
    expected = xr.DataArray(
        np.array([[26915, 27755, -9999, 27705], [25595, -9999, 28315, -9999]]),
        dims=["npts", "ntimes"],
        name="tmpk",
    )
    expected.attrs["missing_value"] = -9999

    decoded = xr.decode_cf(expected.to_dataset())
    encoded, _ = xr.conventions.cf_encoder(decoded.variables, decoded.attrs)

    assert_equal(encoded["tmpk"], expected.variable)

    decoded.tmpk.encoding["_FillValue"] = -9940
    with pytest.raises(ValueError):
        encoded, _ = xr.conventions.cf_encoder(decoded.variables, decoded.attrs)


@requires_dask
def test_CFMaskCoder_decode_dask() -> None:
    original = xr.Variable(("x",), [0, -1, 1], {"_FillValue": -1}).chunk()
    expected = xr.Variable(("x",), [0, np.nan, 1])
    coder = variables.CFMaskCoder()
    encoded = coder.decode(original)
    assert isinstance(encoded.data, da.Array)
    assert_identical(expected, encoded)


# TODO(shoyer): port other fill-value tests


# TODO(shoyer): parameterize when we have more coders
def test_coder_roundtrip() -> None:
    original = xr.Variable(("x",), [0.0, np.nan, 1.0])
    coder = variables.CFMaskCoder()
    roundtripped = coder.decode(coder.encode(original))
    assert_identical(original, roundtripped)


@pytest.mark.parametrize("ptype", "u1 u2 u4 i1 i2 i4".split())
@pytest.mark.parametrize("utype", "f4 f8".split())
def test_mask_scale_roundtrip(utype: str, ptype: str) -> None:
    # this tests cf conforming packing/unpacking via
    # encode_cf_variable/decode_cf_variable
    # f4->i4 packing is skipped as non-conforming
    if utype[1] == "4" and ptype[1] == "4":
        pytest.skip("Can't pack float32 into int32/uint32")
    # fillvalues according to netCDF4
    filldict = {
        "i1": -127,
        "u1": 255,
        "i2": -32767,
        "u2": 65535,
        "i4": -2147483647,
        "u4": 4294967295,
    }
    fillvalue = filldict[ptype]
    unpacked_dtype = np.dtype(utype).type
    packed_dtype = np.dtype(ptype).type
    info = np.iinfo(packed_dtype)

    # create original "encoded" Variable
    packed_data = np.array(
        [info.min, fillvalue, info.max - 1, info.max], dtype=packed_dtype
    )
    attrs = dict(
        scale_factor=unpacked_dtype(1),
        add_offset=unpacked_dtype(0),
        _FillValue=packed_dtype(fillvalue),
    )
    original = xr.Variable(("x",), packed_data, attrs=attrs)

    # create wanted "decoded" Variable
    unpacked_data = np.array(
        [info.min, fillvalue, info.max - 1, info.max], dtype=unpacked_dtype
    )
    encoding = dict(
        scale_factor=unpacked_dtype(1),
        add_offset=unpacked_dtype(0),
        _FillValue=packed_dtype(fillvalue),
    )
    wanted = xr.Variable(("x"), unpacked_data, encoding=encoding)
    wanted = wanted.where(wanted != fillvalue)

    # decode original and compare with wanted
    decoded = decode_cf_variable("x", original)
    assert wanted.dtype == decoded.dtype
    xr.testing.assert_identical(wanted, decoded)

    # encode again and compare with original
    encoded = encode_cf_variable(decoded)
    assert original.dtype == encoded.dtype
    xr.testing.assert_identical(original, encoded)


@pytest.mark.parametrize("unpacked_dtype", "f4 f8 i4".split())
@pytest.mark.parametrize("packed_dtype", "u1 u2 i1 i2 f2 f4".split())
def test_scaling_converts_to_float32(packed_dtype: str, unpacked_dtype: str) -> None:
    # if scale_factor but no add_offset is given transform to float32 in any case
    # this minimizes memory usage, see #1840, #1842
    original = xr.Variable(
        ("x",),
        np.arange(10, dtype=packed_dtype),
        encoding=dict(scale_factor=np.dtype(unpacked_dtype).type(10)),
    )
    coder = variables.CFScaleOffsetCoder()
    encoded = coder.encode(original)
    assert encoded.dtype == np.float32
    roundtripped = coder.decode(encoded)
    assert_identical(original, roundtripped)
    assert roundtripped.dtype == np.float32


@pytest.mark.parametrize("unpacked_dtype", "f4 f8 i4".split())
@pytest.mark.parametrize("packed_dtype", "u1 u2 i1 i2 f2 f4".split())
def test_scaling_converts_to_float64(
    packed_dtype: str, unpacked_dtype: type[np.number]
) -> None:
    # if add_offset is given, but no scale_factor transform to float64 in any case
    # to prevent precision issues
    original = xr.Variable(
        ("x",),
        np.arange(10, dtype=packed_dtype),
        encoding=dict(add_offset=np.dtype(unpacked_dtype).type(10)),
    )
    coder = variables.CFScaleOffsetCoder()
    encoded = coder.encode(original)
    assert encoded.dtype == np.float64
    roundtripped = coder.decode(encoded)
    assert_identical(original, roundtripped)
    assert roundtripped.dtype == np.float64


@pytest.mark.parametrize("scale_factor", (10, [10]))
@pytest.mark.parametrize("add_offset", (0.1, [0.1]))
def test_scaling_offset_as_list(scale_factor, add_offset) -> None:
    # test for #4631
    # attention: scale_factor and add_offset are not conforming to cf specs here
    encoding = dict(scale_factor=scale_factor, add_offset=add_offset)
    original = xr.Variable(("x",), np.arange(10.0), encoding=encoding)
    coder = variables.CFScaleOffsetCoder()
    encoded = coder.encode(original)
    roundtripped = coder.decode(encoded)
    assert_allclose(original, roundtripped)


@pytest.mark.parametrize("bits", [1, 2, 4, 8])
def test_decode_unsigned_from_signed(bits) -> None:
    unsigned_dtype = np.dtype(f"u{bits}")
    signed_dtype = np.dtype(f"i{bits}")
    original_values = np.array([np.iinfo(unsigned_dtype).max], dtype=unsigned_dtype)
    encoded = xr.Variable(
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
    encoded = xr.Variable(
        ("x",), original_values.astype(unsigned_dtype), attrs={"_Unsigned": "false"}
    )
    coder = variables.UnsignedIntegerCoder()
    decoded = coder.decode(encoded)
    assert decoded.dtype == signed_dtype
    assert decoded.values == original_values
