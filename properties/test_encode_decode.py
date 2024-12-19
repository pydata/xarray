"""
Property-based tests for encoding/decoding methods.

These ones pass, just as you'd hope!

"""

import pytest

pytest.importorskip("hypothesis")
# isort: split

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
from hypothesis import given

import xarray as xr
from xarray.coding.times import _parse_iso8601_without_reso
from xarray.testing.strategies import variables


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


# TODO: add cftime.datetime
@given(dt=st.datetimes())
def test_iso8601_decode(dt):
    iso = dt.isoformat()
    assert dt == _parse_iso8601_without_reso(type(dt), iso)
