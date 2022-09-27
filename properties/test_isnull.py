import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
import pandas as pd
from hypothesis import given, settings

import xarray as xr

# Run for a while - arrays are a bigger search space than usual
settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


an_array = npst.arrays(
    dtype=npst.nested_dtypes(),
    shape=npst.array_shapes(max_side=3),  # max_side specified for performance
)


@given(st.data(), an_array)
def test_isnull_consistency(data, array):
    actual = xr.core.duck_array_ops.isnull(array)
    if array.dtype.kind == "V":
        # not supported by pandas
        expected = np.zeros_like(array, dtype=bool)
    else:
        expected = pd.isnull(array)
    np.testing.assert_equal(actual, expected)
