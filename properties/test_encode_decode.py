"""
Property-based tests for encoding/decoding methods.

These ones pass, just as you'd hope!

"""

import warnings

import pytest

pytest.importorskip("hypothesis")
# isort: split

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
from hypothesis import given

import xarray as xr
from xarray.coding.times import _parse_iso8601
from xarray.testing.strategies import CFTimeStrategyISO8601, variables
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
@given(dt=st.datetimes() | CFTimeStrategyISO8601())
def test_iso8601_decode(dt):
    iso = dt.isoformat()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*date/calendar/year zero.*")
        parsed, _ = _parse_iso8601(type(dt), iso)
        assert dt == parsed
