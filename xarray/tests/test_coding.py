from contextlib import suppress

import numpy as np
import pytest

import xarray as xr
from xarray.coding import variables

from . import assert_identical, requires_dask

with suppress(ImportError):
    import dask.array as da


def reverse_list_of_tuple(plist):
    return [tuple(reversed(t)) for t in plist]


def test_CFMaskCoder_decode():
    original = xr.Variable(('x',), [0, -1, 1], {'_FillValue': -1})
    expected = xr.Variable(('x',), [0, np.nan, 1])
    coder = variables.CFMaskCoder()
    encoded = coder.decode(original)
    assert_identical(expected, encoded)


@requires_dask
def test_CFMaskCoder_decode_dask():
    original = xr.Variable(('x',), [0, -1, 1], {'_FillValue': -1}).chunk()
    expected = xr.Variable(('x',), [0, np.nan, 1])
    coder = variables.CFMaskCoder()
    encoded = coder.decode(original)
    assert isinstance(encoded.data, da.Array)
    assert_identical(expected, encoded)

# TODO(shoyer): port other fill-value tests


# TODO(shoyer): parameterize when we have more coders
def test_coder_roundtrip():
    original = xr.Variable(('x',), [0.0, np.nan, 1.0])
    coder = variables.CFMaskCoder()
    roundtripped = coder.decode(coder.encode(original))
    assert_identical(original, roundtripped)


@pytest.mark.parametrize('dtype', 'u1 u2 i1 i2 f2 f4'.split())
def test_scaling_converts_to_float32(dtype):
    original = xr.Variable(('x',), np.arange(10, dtype=dtype),
                           encoding=dict(scale_factor=np.float32(10)))
    coder = variables.CFScaleOffsetCoder()
    encoded = coder.encode(original)
    assert encoded.dtype == np.float32
    roundtripped = coder.decode(encoded)
    assert roundtripped.dtype == np.float32
    assert_identical(original, roundtripped)


all_possible_types = [np.uint8(8),
                      np.uint16(16),
                      np.uint32(32),
                      np.uint64(64),
                      np.int8(80),
                      np.int16(160),
                      np.int32(320),
                      np.int64(640),
                      1,
                      0.01,
                      np.float16(1600),
                      np.float32(3200),
                      np.float64(6400)]

# In all cases encoding returns either np.float32 or np.float64
# Encoding only cares about existence of add_offset when the
# variable is an integer
# If the variable is a float then the encoded dtype is np.float32
# If the variable is an integer with add_offset
# then the encoded dtype is np.float64
# If the variable is an integer with no add_affset
# then the encoded dtype is np.float32
# In all other cases the encoded dtype is np.float64
# decoding is the equivalent of unpacking mentioned in the cf-convention
# in all cases decoding takes the encoded dtype which
# is either np.float32 or np.float64
# then decoded type is the largest between scale_factor, add_offset
# and encoded type and not original

#############################
# Case 1: variable has offset
#############################

# Case 1.1: variable is float
# encoded should be np.float32
# if (scale_factor, add_offset) is in the following list
# decoded should be np.float32
# if not decoded should be np.float64
combinations_for_float32 = [
    (np.uint8(8), np.uint16(16)),
    (np.uint8(8), np.int8(80)),
    (np.uint8(8), np.int16(160)),
    (np.uint8(8), np.float16(1600)),
    (np.uint8(8), np.float32(3200)),
    (np.uint16(16), np.float16(1600)),
    (np.uint16(16), np.float32(3200)),
    (np.int8(80), np.int16(160)),
    (np.int8(80), np.float16(1600)),
    (np.int8(80), np.float32(3200)),
    (np.int16(160), np.float16(1600)),
    (np.int16(160), np.float32(3200)),
    (np.float16(1600), np.float32(3200)),
    (np.float32(3200), np.float32(3200)),
    (np.float16(1600), np.float16(1600)),
    (np.int16(160), np.int16(160)),
    (np.int8(80), np.int8(80)),
    (np.uint16(16), np.uint16(16)),
    (np.uint8(8), np.uint8(8))
]
(combinations_for_float32.extend(
    reverse_list_of_tuple(combinations_for_float32)))


@pytest.mark.parametrize('dtype', 'f2 f4'.split())
@pytest.mark.parametrize('scale_factor', all_possible_types)
@pytest.mark.parametrize('add_offset', all_possible_types)
def test_scaling_case_1_float_var(dtype, scale_factor, add_offset):
    original = xr.Variable(('x',), np.arange(10, dtype=dtype),
                           encoding=dict(scale_factor=scale_factor,
                           add_offset=add_offset))

    coder = variables.CFScaleOffsetCoder()
    encoded = coder.encode(original)
    assert encoded.dtype == np.float32
    roundtripped = coder.decode(encoded)

    if (scale_factor, add_offset) in combinations_for_float32:
        assert roundtripped.dtype == np.float32
    else:
        assert roundtripped.dtype == np.float64


# Case 1.2: variable is integer
# encoded should be np.float64 as we have offset
# decoded should always be np.float64
@pytest.mark.parametrize('dtype', 'u1 u2 i1 i2'.split())
@pytest.mark.parametrize('scale_factor', all_possible_types)
@pytest.mark.parametrize('add_offset', all_possible_types)
def test_scaling_cf_convention_case_1_int_var(dtype, scale_factor, add_offset):
    original = xr.Variable(('x',), np.arange(10, dtype=dtype),
                           encoding=dict(scale_factor=scale_factor,
                           add_offset=add_offset))

    coder = variables.CFScaleOffsetCoder()
    encoded = coder.encode(original)
    assert encoded.dtype == np.float64
    roundtripped = coder.decode(encoded)
    assert roundtripped.dtype == np.float64


####################################
# Case 2: variable has no add_offset
####################################

# Case 2.1:
# for any variable dtype
# encoded should be np.float32
# if scale_factor in the following list
# decoded should be np.float32
# if not decoded should be np.float64
types_for_float32 = [np.uint8(8),
                     np.uint16(16),
                     np.int8(80),
                     np.int16(160),
                     np.float16(1600),
                     np.float32(3200)]


@pytest.mark.parametrize('dtype', 'u1 u2 i1 i2 f2 f4'.split())
@pytest.mark.parametrize('scale_factor', all_possible_types)
def test_scaling_case_2(dtype, scale_factor):

    original = xr.Variable(('x',), np.arange(10, dtype=dtype),
                           encoding=dict(scale_factor=scale_factor))

    coder = variables.CFScaleOffsetCoder()
    encoded = coder.encode(original)
    assert encoded.dtype == np.float32
    roundtripped = coder.decode(encoded)
    if scale_factor in types_for_float32:
        assert roundtripped.dtype == np.float32
    else:
        assert roundtripped.dtype == np.float64
