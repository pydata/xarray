import numpy as np
import pytest

import xarray as xr
from xarray.core.pycompat import suppress
from xarray.coding import variables

from . import requires_dask, raises_regex, assert_identical

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
