"""
Property-based tests for encoding/decoding methods.

These ones pass, just as you'd hope!

"""
import pytest

pytest.importorskip("hypothesis")
# isort: split

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
from hypothesis import given

import xarray as xr

an_array = npst.arrays(
    dtype=st.one_of(
        npst.unsigned_integer_dtypes(), npst.integer_dtypes(), npst.floating_dtypes()
    ),
    shape=npst.array_shapes(max_side=3),  # max_side specified for performance
)


@pytest.mark.slow
@given(st.data(), an_array)
def test_CFMask_coder_roundtrip(data, arr) -> None:
    names = data.draw(
        st.lists(st.text(), min_size=arr.ndim, max_size=arr.ndim, unique=True).map(
            tuple
        )
    )
    original = xr.Variable(names, arr)
    coder = xr.coding.variables.CFMaskCoder()
    roundtripped = coder.decode(coder.encode(original))
    xr.testing.assert_identical(original, roundtripped)


@pytest.mark.slow
@given(st.data(), an_array)
def test_CFScaleOffset_coder_roundtrip(data, arr) -> None:
    names = data.draw(
        st.lists(st.text(), min_size=arr.ndim, max_size=arr.ndim, unique=True).map(
            tuple
        )
    )
    original = xr.Variable(names, arr)
    coder = xr.coding.variables.CFScaleOffsetCoder()
    roundtripped = coder.decode(coder.encode(original))
    xr.testing.assert_identical(original, roundtripped)
