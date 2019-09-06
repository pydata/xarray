"""
Property-based tests for roundtripping between xarray and pandas objects.
"""
import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
from hypothesis import given

import numpy as np
import xarray as xr

an_array = npst.arrays(
    dtype=st.one_of(
        npst.unsigned_integer_dtypes(), npst.integer_dtypes(), npst.floating_dtypes()
    ),
    shape=npst.array_shapes(max_dims=2),  # can only convert 1D/2D to pandas
)


@given(st.data(), an_array)
def test_roundtrip_dataarray(data, arr):
    names = data.draw(
        st.lists(st.text(), min_size=arr.ndim, max_size=arr.ndim, unique=True).map(
            tuple
        )
    )
    coords = {name: np.arange(n) for (name, n) in zip(names, arr.shape)}
    original = xr.DataArray(arr, dims=names, coords=coords)
    roundtripped = xr.DataArray(original.to_pandas())
    xr.testing.assert_identical(original, roundtripped)
