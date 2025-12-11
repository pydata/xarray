"""Property tests comparing CoordinateTransformIndex to PandasIndex."""

from collections.abc import Hashable
from typing import Any

import numpy as np
import pytest

pytest.importorskip("hypothesis")

import hypothesis.strategies as st
from hypothesis import given

import xarray as xr
import xarray.testing.strategies as xrst
from xarray.core.coordinate_transform import CoordinateTransform
from xarray.core.indexes import CoordinateTransformIndex
from xarray.testing import assert_identical

DATA_VAR_NAME = "_test_data_"


class IdentityTransform(CoordinateTransform):
    """Identity transform that returns dimension positions as coordinate labels."""

    def forward(self, dim_positions: dict[str, Any]) -> dict[Hashable, Any]:
        return dim_positions

    def reverse(self, coord_labels: dict[Hashable, Any]) -> dict[str, Any]:
        return coord_labels

    def equals(
        self, other: CoordinateTransform, exclude: frozenset[Hashable] | None = None
    ) -> bool:
        if not isinstance(other, IdentityTransform):
            return False
        return self.dim_size == other.dim_size


def create_transform_da(sizes: dict[str, int]) -> xr.DataArray:
    """Create a DataArray with IdentityTransform CoordinateTransformIndex."""
    dims = list(sizes.keys())
    shape = tuple(sizes.values())
    data = np.arange(np.prod(shape)).reshape(shape)

    # Create dataset with transform index for each dimension
    ds = xr.Dataset({DATA_VAR_NAME: (dims, data)})
    for dim, size in sizes.items():
        transform = IdentityTransform((dim,), {dim: size}, dtype=np.dtype(np.int64))
        index = CoordinateTransformIndex(transform)
        ds = ds.assign_coords(xr.Coordinates.from_xindex(index))

    return ds[DATA_VAR_NAME]


def create_pandas_da(sizes: dict[str, int]) -> xr.DataArray:
    """Create a DataArray with standard PandasIndex (range index)."""
    shape = tuple(sizes.values())
    data = np.arange(np.prod(shape)).reshape(shape)
    coords = {dim: np.arange(size) for dim, size in sizes.items()}
    return xr.DataArray(
        data, dims=list(sizes.keys()), coords=coords, name=DATA_VAR_NAME
    )


@given(
    st.data(),
    xrst.dimension_sizes(min_dims=1, max_dims=3, min_side=1, max_side=5),
)
def test_basic_indexing(data, sizes):
    """Test basic indexing produces identical results for transform and pandas index."""
    pandas_da = create_pandas_da(sizes)
    transform_da = create_transform_da(sizes)
    idxr = data.draw(xrst.basic_indexers(sizes=sizes))
    pandas_result = pandas_da.isel(idxr)
    transform_result = transform_da.isel(idxr)
    assert_identical(pandas_result, transform_result)


@given(
    st.data(),
    xrst.dimension_sizes(min_dims=1, max_dims=3, min_side=1, max_side=5),
)
def test_outer_indexing(data, sizes):
    """Test outer indexing produces identical results for transform and pandas index."""
    pandas_da = create_pandas_da(sizes)
    transform_da = create_transform_da(sizes)
    idxr = data.draw(xrst.outer_array_indexers(sizes=sizes, min_dims=1))
    pandas_result = pandas_da.isel(idxr)
    transform_result = transform_da.isel(idxr)
    assert_identical(pandas_result, transform_result)


@given(
    st.data(),
    xrst.dimension_sizes(min_dims=2, max_dims=3, min_side=1, max_side=5),
)
def test_vectorized_indexing(data, sizes):
    """Test vectorized indexing produces identical results for transform and pandas index."""
    pandas_da = create_pandas_da(sizes)
    transform_da = create_transform_da(sizes)
    idxr = data.draw(xrst.vectorized_indexers(sizes=sizes))
    pandas_result = pandas_da.isel(idxr)
    transform_result = transform_da.isel(idxr)
    assert_identical(pandas_result, transform_result)
