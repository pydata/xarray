"""
Property-based tests for roundtripping between xarray and pandas objects.
"""

from functools import partial

import numpy as np
import pandas as pd
import pytest

import xarray as xr

pytest.importorskip("hypothesis")
import hypothesis.extra.numpy as npst  # isort:skip
import hypothesis.extra.pandas as pdst  # isort:skip
import hypothesis.strategies as st  # isort:skip
from hypothesis import given  # isort:skip

numeric_dtypes = st.one_of(
    npst.unsigned_integer_dtypes(endianness="="),
    npst.integer_dtypes(endianness="="),
    npst.floating_dtypes(endianness="="),
)

numeric_series = numeric_dtypes.flatmap(lambda dt: pdst.series(dtype=dt))


@st.composite
def dataframe_strategy(draw):
    tz = draw(st.timezones())
    dtype = pd.DatetimeTZDtype(unit="ns", tz=tz)

    datetimes = st.datetimes(
        min_value=pd.Timestamp("1677-09-21T00:12:43.145224193"),
        max_value=pd.Timestamp("2262-04-11T23:47:16.854775807"),
        timezones=st.just(tz),
    )

    df = pdst.data_frames(
        [
            pdst.column("datetime_col", elements=datetimes),
            pdst.column("other_col", elements=st.integers()),
        ],
        index=pdst.range_indexes(min_size=1, max_size=10),
    )
    return draw(df).astype({"datetime_col": dtype})


an_array = npst.arrays(
    dtype=numeric_dtypes,
    shape=npst.array_shapes(max_dims=2),  # can only convert 1D/2D to pandas
)


@st.composite
def datasets_1d_vars(draw) -> xr.Dataset:
    """Generate datasets with only 1D variables

    Suitable for converting to pandas dataframes.
    """
    # Generate an index for the dataset
    idx = draw(pdst.indexes(dtype="u8", min_size=0, max_size=100))

    # Generate 1-3 variables, 1D with the same length as the index
    vars_strategy = st.dictionaries(
        keys=st.text(),
        values=npst.arrays(dtype=numeric_dtypes, shape=len(idx)).map(
            partial(xr.Variable, ("rows",))
        ),
        min_size=1,
        max_size=3,
    )
    return xr.Dataset(draw(vars_strategy), coords={"rows": idx})


@given(st.data(), an_array)
def test_roundtrip_dataarray(data, arr) -> None:
    names = data.draw(
        st.lists(st.text(), min_size=arr.ndim, max_size=arr.ndim, unique=True).map(
            tuple
        )
    )
    coords = {name: np.arange(n) for (name, n) in zip(names, arr.shape, strict=True)}
    original = xr.DataArray(arr, dims=names, coords=coords)
    roundtripped = xr.DataArray(original.to_pandas())
    xr.testing.assert_identical(original, roundtripped)


@given(datasets_1d_vars())
def test_roundtrip_dataset(dataset) -> None:
    df = dataset.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    roundtripped = xr.Dataset(df)
    xr.testing.assert_identical(dataset, roundtripped)


@given(numeric_series, st.text())
def test_roundtrip_pandas_series(ser, ix_name) -> None:
    # Need to name the index, otherwise Xarray calls it 'dim_0'.
    ser.index.name = ix_name
    arr = xr.DataArray(ser)
    roundtripped = arr.to_pandas()
    pd.testing.assert_series_equal(ser, roundtripped)
    xr.testing.assert_identical(arr, roundtripped.to_xarray())


# Dataframes with columns of all the same dtype - for roundtrip to DataArray
numeric_homogeneous_dataframe = numeric_dtypes.flatmap(
    lambda dt: pdst.data_frames(columns=pdst.columns(["a", "b", "c"], dtype=dt))
)


@pytest.mark.xfail
@given(numeric_homogeneous_dataframe)
def test_roundtrip_pandas_dataframe(df) -> None:
    # Need to name the indexes, otherwise Xarray names them 'dim_0', 'dim_1'.
    df.index.name = "rows"
    df.columns.name = "cols"
    arr = xr.DataArray(df)
    roundtripped = arr.to_pandas()
    pd.testing.assert_frame_equal(df, roundtripped)
    xr.testing.assert_identical(arr, roundtripped.to_xarray())


@given(df=dataframe_strategy())
def test_roundtrip_pandas_dataframe_datetime(df) -> None:
    # Need to name the indexes, otherwise Xarray names them 'dim_0', 'dim_1'.
    df.index.name = "rows"
    df.columns.name = "cols"
    dataset = xr.Dataset.from_dataframe(df)
    roundtripped = dataset.to_dataframe()
    roundtripped.columns.name = "cols"  # why?
    pd.testing.assert_frame_equal(df, roundtripped)
    xr.testing.assert_identical(dataset, roundtripped.to_xarray())


def test_roundtrip_1d_pandas_extension_array() -> None:
    df = pd.DataFrame({"cat": pd.Categorical(["a", "b", "c"])})
    arr = xr.Dataset.from_dataframe(df)["cat"]
    roundtripped = arr.to_pandas()
    assert (df["cat"] == roundtripped).all()
    assert df["cat"].dtype == roundtripped.dtype
    xr.testing.assert_identical(arr, roundtripped.to_xarray())
