"""
Property-based tests for encoding/decoding methods.

These ones pass, just as you'd hope!

"""

import pytest

pytest.importorskip("hypothesis")
# isort: split

import hypothesis.extra.numpy as npst
import numpy as np
from hypothesis import given

import xarray as xr
from xarray.testing.strategies import cftime_arrays, variables
from xarray.tests import requires_cftime


@pytest.mark.slow
@given(original=variables())
def test_CFMask_coder_roundtrip(original) -> None:
    coder = xr.coding.variables.CFMaskCoder()
    roundtripped = coder.decode(coder.encode(original))
    xr.testing.assert_identical(original, roundtripped)


@pytest.mark.xfail
@pytest.mark.slow
@given(var=variables(dtype=npst.floating_dtypes()))
def test_CFMask_coder_decode(var) -> None:
    var[0] = -99
    var.attrs["_FillValue"] = -99
    coder = xr.coding.variables.CFMaskCoder()
    decoded = coder.decode(var)
    assert np.isnan(decoded[0])


@pytest.mark.slow
@given(original=variables())
def test_CFScaleOffset_coder_roundtrip(original) -> None:
    coder = xr.coding.variables.CFScaleOffsetCoder()
    roundtripped = coder.decode(coder.encode(original))
    xr.testing.assert_identical(original, roundtripped)


@requires_cftime
@given(original_array=cftime_arrays(shapes=npst.array_shapes(max_dims=1)))
def test_CFDatetime_coder_roundtrip_cftime(original_array) -> None:
    original = xr.Variable("time", original_array)
    coder = xr.coding.times.CFDatetimeCoder(use_cftime=True)
    roundtripped = coder.decode(coder.encode(original))
    xr.testing.assert_identical(original, roundtripped)


@given(
    original_array=npst.arrays(
        dtype=npst.datetime64_dtypes(endianness="=", max_period="ns"),
        shape=npst.array_shapes(max_dims=1),
    )
)
def test_CFDatetime_coder_roundtrip_numpy(original_array) -> None:
    original = xr.Variable("time", original_array)
    coder = xr.coding.times.CFDatetimeCoder(use_cftime=False)
    roundtripped = coder.decode(coder.encode(original))
    xr.testing.assert_identical(original, roundtripped)


# datetime_arrays =, shape=npst.array_shapes(min_dims=1, max_dims=1)) | cftime_arrays()
