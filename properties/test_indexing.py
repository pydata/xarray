import pytest

pytest.importorskip("hypothesis")

import hypothesis.strategies as st
from hypothesis import given

import xarray as xr
import xarray.testing.strategies as xrst


def _slice_size(s: slice, dim_size: int) -> int:
    """Compute the size of a slice applied to a dimension."""
    return len(range(*s.indices(dim_size)))


@given(
    st.data(),
    xrst.variables(dims=xrst.dimension_sizes(min_dims=1, max_dims=4, min_side=1)),
)
def test_basic_indexing(data, var):
    """Test that basic indexers produce expected output shape."""
    idxr = data.draw(xrst.basic_indexers(sizes=var.sizes))
    result = var.isel(idxr)
    expected_shape = tuple(
        _slice_size(idxr[d], var.sizes[d]) if d in idxr else var.sizes[d]
        for d in result.dims
    )
    assert result.shape == expected_shape


@given(
    st.data(),
    xrst.variables(dims=xrst.dimension_sizes(min_dims=1, max_dims=4, min_side=1)),
)
def test_outer_indexing(data, var):
    """Test that outer array indexers produce expected output shape."""
    idxr = data.draw(xrst.outer_array_indexers(sizes=var.sizes, min_dims=1))
    result = var.isel(idxr)
    expected_shape = tuple(
        len(idxr[d]) if d in idxr else var.sizes[d] for d in result.dims
    )
    assert result.shape == expected_shape


@given(
    st.data(),
    xrst.variables(dims=xrst.dimension_sizes(min_dims=2, max_dims=4, min_side=1)),
)
def test_vectorized_indexing(data, var):
    """Test that vectorized indexers produce expected output shape."""
    da = xr.DataArray(var)
    idxr = data.draw(xrst.vectorized_indexers(sizes=var.sizes))
    result = da.isel(idxr)

    # TODO: this logic works because the dims in idxr don't overlap with da.dims
    # Compute expected shape from result dims
    # Non-indexed dims keep their original size, indexed dims get broadcast size
    broadcast_result = xr.broadcast(*idxr.values())
    broadcast_sizes = dict(
        zip(broadcast_result[0].dims, broadcast_result[0].shape, strict=True)
    )
    expected_shape = tuple(
        var.sizes[d] if d in var.sizes else broadcast_sizes[d] for d in result.dims
    )
    assert result.shape == expected_shape
