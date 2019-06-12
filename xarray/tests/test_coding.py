from contextlib import suppress

import numpy as np
import pytest

import xarray as xr
from xarray.coding import variables

from . import assert_equal, assert_identical, requires_dask

with suppress(ImportError):
    import dask.array as da


def test_CFMaskCoder_decode():
    original = xr.Variable(('x',), [0, -1, 1], {'_FillValue': -1})
    expected = xr.Variable(('x',), [0, np.nan, 1])
    coder = variables.CFMaskCoder()
    encoded = coder.decode(original)
    assert_identical(expected, encoded)


def test_CFMaskCoder_missing_value():
    expected = xr.DataArray(np.array([[26915, 27755, -9999, 27705],
                                      [25595, -9999, 28315, -9999]]),
                            dims=['npts', 'ntimes'],
                            name='tmpk')
    expected.attrs['missing_value'] = -9999

    decoded = xr.decode_cf(expected.to_dataset())
    encoded, _ = xr.conventions.cf_encoder(decoded, decoded.attrs)

    assert_equal(encoded['tmpk'], expected.variable)

    decoded.tmpk.encoding['_FillValue'] = -9940
    with pytest.raises(ValueError):
        encoded, _ = xr.conventions.cf_encoder(decoded, decoded.attrs)


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
