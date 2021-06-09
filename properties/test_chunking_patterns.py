"""
Property-based tests that dask-powered algorithms will succeed regardless of the
pattern of chunks on the input arrays.
"""

import uuid

import pytest

pytest.importorskip("hypothesis")

import hypothesis.strategies as st
import numpy as np
from hypothesis import given

import xarray as xr
from xarray.tests import requires_dask


def example_dataarray(shape=(5, 20)):
    data = np.random.randn(*shape)
    dims = [f"dim_{i}" for i in range(len(shape))]
    da = xr.DataArray(data, dims=dims, name="T")
    return da


def example_dataset(n_dim=2, n_vars=2):
    """Random dataset with every variable having the same shape"""

    shape = tuple(range(8, 8 + n_dim))
    dims = [f"dim_{i}" for i in range(len(shape))]
    var_names = [uuid.uuid4().hex for _ in range(n_vars)]
    ds = xr.Dataset()
    for i in range(n_vars):
        name = var_names[i]
        data = np.random.randn(*shape)
        da = xr.DataArray(data, dims=dims, name=name)
        ds[name] = da
    return ds


@st.composite
def chunk_shapes(draw, n_dim=3, max_arr_len=10):
    """Generate different chunking patterns for an N-D array of data."""
    chunks = []
    for n in range(n_dim):
        shape = draw(st.integers(min_value=1, max_value=max_arr_len))
        chunks.append(shape)
    return tuple(chunks)


@requires_dask
class TestChunkingHypotheses:
    @given(chunk_shapes(n_dim=1, max_arr_len=20))
    def test_all_chunking_patterns_1d(self, chunks):

        data = example_dataarray(shape=(20,)).chunk(chunks)

        nbins_a = 8
        bins = np.linspace(-4, 4, nbins_a + 1)

        h = xr.hist(data, bins=[bins])

        assert h.shape == (nbins_a,)

        hist, _ = np.histogram(
            data.values,
            bins=bins,
        )

        np.testing.assert_allclose(hist, h)

    @pytest.mark.slow
    @given(chunk_shapes(n_dim=2, max_arr_len=8))
    def test_all_chunking_patterns_2d(self, chunks):

        data_a = example_dataarray(shape=(5, 20)).chunk(chunks)
        data_b = example_dataarray(shape=(5, 20)).chunk(chunks)

        nbins_a = 8
        nbins_b = 9
        bins_a = np.linspace(-4, 4, nbins_a + 1)
        bins_b = np.linspace(-4, 4, nbins_b + 1)

        h = xr.hist(data_a, data_b, bins=[bins_a, bins_b])

        assert h.shape == (nbins_a, nbins_b)

        hist, _, _ = np.histogram2d(
            data_a.values.ravel(),
            data_b.values.ravel(),
            bins=[bins_a, bins_b],
        )

        np.testing.assert_allclose(hist, h.values)

    @pytest.mark.slow
    @pytest.mark.parametrize("n_vars", [1, 2, 3, 4])
    @given(chunk_shapes(n_dim=2, max_arr_len=7))
    def test_all_chunking_patterns_dd_hist(self, n_vars, chunk_shapes):
        ds = example_dataset(n_dim=2, n_vars=n_vars)
        ds = ds.chunk({d: c for d, c in zip(ds.dims.keys(), chunk_shapes)})

        n_bins = (7, 8, 9, 10)[:n_vars]
        bins = [np.linspace(-4, 4, n + 1) for n in n_bins]

        h = xr.hist(*[da for name, da in ds.data_vars.items()], bins=bins)

        assert h.shape == n_bins

        input_data = np.stack(
            [da.values.ravel() for name, da in ds.data_vars.items()], axis=-1
        )
        hist, _ = np.histogramdd(input_data, bins=bins)

        np.testing.assert_allclose(hist, h.values)
