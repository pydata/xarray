"""
Property-based tests for roundtripping between xarray and pandas objects.
"""
import hypothesis.extra.numpy as npst
import hypothesis.extra.pandas as pdst
import hypothesis.strategies as st
from hypothesis import given

import numpy as np
import pandas as pd
import xarray as xr

numeric_dtypes = st.one_of(
    npst.unsigned_integer_dtypes(), npst.integer_dtypes(), npst.floating_dtypes()
)

numeric_series = numeric_dtypes.flatmap(lambda dt: pdst.series(dtype=dt))

an_array = npst.arrays(
    dtype=numeric_dtypes,
    shape=npst.array_shapes(max_dims=2),  # can only convert 1D/2D to pandas
)


@st.composite
def datasets_1d_vars(draw):
    """Generate datasets with only 1D variables

    Suitable for converting to pandas dataframes.
    """
    n_vars = draw(st.integers(min_value=1, max_value=3))
    n_entries = draw(st.integers(min_value=0, max_value=100))
    dims = ("rows",)
    vars = {}
    for _ in range(n_vars):
        name = draw(st.text(min_size=0))
        dt = draw(numeric_dtypes)
        arr = draw(npst.arrays(dtype=dt, shape=(n_entries,)))
        vars[name] = xr.Variable(dims, arr)

    coords = {
        dims[0]: draw(pdst.indexes(dtype="u8", min_size=n_entries, max_size=n_entries))
    }

    return xr.Dataset(vars, coords=coords)


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


@given(datasets_1d_vars())
def test_roundtrip_dataset(dataset):
    df = dataset.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    roundtripped = xr.Dataset(df)
    xr.testing.assert_identical(dataset, roundtripped)


@given(numeric_series, st.text())
def test_roundtrip_pandas_series(ser, ix_name):
    # Need to name the index, otherwise Xarray calls it 'dim_0'.
    ser.index.name = ix_name
    arr = xr.DataArray(ser)
    roundtripped = arr.to_pandas()
    pd.testing.assert_series_equal(ser, roundtripped)


# Dataframes with columns of all the same dtype - for roundtrip to DataArray
numeric_homogeneous_dataframe = numeric_dtypes.flatmap(
    lambda dt: pdst.data_frames(columns=pdst.columns(["a", "b", "c"], dtype=dt))
)


@given(numeric_homogeneous_dataframe)
def test_roundtrip_pandas_dataframe(df):
    # Need to name the indexes, otherwise Xarray names them 'dim_0', 'dim_1'.
    df.index.name = "rows"
    df.columns.name = "cols"
    arr = xr.DataArray(df)
    roundtripped = arr.to_pandas()
    pd.testing.assert_frame_equal(df, roundtripped)
