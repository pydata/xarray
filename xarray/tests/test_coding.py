from contextlib import suppress

import numpy as np
import pytest

import xarray as xr
from xarray.coding import variables

from . import assert_identical, requires_dask

with suppress(ImportError):
    import dask.array as da


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
                           encoding=dict(scale_factor=10))
    coder = variables.CFScaleOffsetCoder()
    encoded = coder.encode(original)
    assert encoded.dtype == np.float32
    roundtripped = coder.decode(encoded)
    assert_identical(original, roundtripped)
    assert roundtripped.dtype == np.float32


@pytest.mark.parametrize('dtype', 'u1 u2 i1 i2 f2 f4'.split())
@pytest.mark.parametrize('scale_factor', [10, 0.01,
                                          np.float16(10),
                                          np.float32(10),
                                          np.float64(10),
                                          np.int8(10),
                                          np.int16(10), np.int32(10),
                                          np.int64(10), np.uint8(10),
                                          np.uint16(10), np.uint32(10),
                                          np.uint64(10), np.uint64(10)])
@pytest.mark.parametrize('add_offset', [10, 0.01,
                                        np.float16(10),
                                        np.float32(10),
                                        np.float64(10),
                                        np.int8(10),
                                        np.int16(10), np.int32(10),
                                        np.int64(10), np.uint8(10),
                                        np.uint16(10), np.uint32(10),
                                        np.uint64(10), np.uint64(10)])
def test_scaling_according_to_cf_convention(dtype, scale_factor, add_offset):
    original = xr.Variable(('x',), np.arange(10, dtype=dtype),
                           encoding=dict(scale_factor=scale_factor,
                           add_offset=add_offset))
    coder = variables.CFScaleOffsetCoder()
    encoded = coder.encode(original)
    assert encoded.dtype.itemsize >= np.dtype(dtype).itemsize
    assert encoded.dtype.itemsize >= 4 and np.issubdtype(encoded, np.floating)

    roundtripped = coder.decode(encoded)

    # We make sure that roundtripped is larger than
    # the original
    assert roundtripped.dtype.itemsize >= original.dtype.itemsize
    assert (roundtripped.dtype is np.dtype(np.float64)
            or roundtripped.dtype is np.dtype(np.float32))

    np.testing.assert_array_almost_equal(roundtripped, original)
