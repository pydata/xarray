from distutils.version import LooseVersion

import numpy as np
import pandas as pd
import pytest

import xarray as xr
from xarray.core import formatting_html as fh


@pytest.fixture
def dataarray():
    return xr.DataArray(np.random.RandomState(0).randn(4, 6))


@pytest.fixture
def dask_dataarray(dataarray):
    pytest.importorskip("dask")
    return dataarray.chunk()


@pytest.fixture
def multiindex():
    mindex = pd.MultiIndex.from_product(
        [["a", "b"], [1, 2]], names=("level_1", "level_2")
    )
    return xr.Dataset({}, {"x": mindex})


@pytest.fixture
def dataset():
    times = pd.date_range("2000-01-01", "2001-12-31", name="time")
    annual_cycle = np.sin(2 * np.pi * (times.dayofyear.values / 365.25 - 0.28))

    base = 10 + 15 * annual_cycle.reshape(-1, 1)
    tmin_values = base + 3 * np.random.randn(annual_cycle.size, 3)
    tmax_values = base + 10 + 3 * np.random.randn(annual_cycle.size, 3)

    return xr.Dataset(
        {
            "tmin": (("time", "location"), tmin_values),
            "tmax": (("time", "location"), tmax_values),
        },
        {"time": times, "location": ["<IA>", "IN", "IL"]},
        attrs={"description": "Test data."},
    )


def test_short_data_repr_html(dataarray):
    data_repr = fh.short_data_repr_html(dataarray)
    assert data_repr.startswith("array")


def test_short_data_repr_html_non_str_keys(dataset):
    ds = dataset.assign({2: lambda x: x["tmin"]})
    fh.dataset_repr(ds)


def test_short_data_repr_html_dask(dask_dataarray):
    import dask

    if LooseVersion(dask.__version__) < "2.0.0":
        assert not hasattr(dask_dataarray.data, "_repr_html_")
        data_repr = fh.short_data_repr_html(dask_dataarray)
        assert (
            data_repr
            == "dask.array&lt;xarray-&lt;this-array&gt;, shape=(4, 6), dtype=float64, chunksize=(4, 6)&gt;"
        )
    else:
        assert hasattr(dask_dataarray.data, "_repr_html_")
        data_repr = fh.short_data_repr_html(dask_dataarray)
        assert data_repr == dask_dataarray.data._repr_html_()


def test_format_dims_no_dims():
    dims, coord_names = {}, []
    formatted = fh.format_dims(dims, coord_names)
    assert formatted == ""


def test_format_dims_unsafe_dim_name():
    dims, coord_names = {"<x>": 3, "y": 2}, []
    formatted = fh.format_dims(dims, coord_names)
    assert "&lt;x&gt;" in formatted


def test_format_dims_non_index():
    dims, coord_names = {"x": 3, "y": 2}, ["time"]
    formatted = fh.format_dims(dims, coord_names)
    assert "class='xr-has-index'" not in formatted


def test_format_dims_index():
    dims, coord_names = {"x": 3, "y": 2}, ["x"]
    formatted = fh.format_dims(dims, coord_names)
    assert "class='xr-has-index'" in formatted


def test_summarize_attrs_with_unsafe_attr_name_and_value():
    attrs = {"<x>": 3, "y": "<pd.DataFrame>"}
    formatted = fh.summarize_attrs(attrs)
    assert "<dt><span>&lt;x&gt; :</span></dt>" in formatted
    assert "<dt><span>y :</span></dt>" in formatted
    assert "<dd>3</dd>" in formatted
    assert "<dd>&lt;pd.DataFrame&gt;</dd>" in formatted


def test_repr_of_dataarray(dataarray):
    formatted = fh.array_repr(dataarray)
    assert "dim_0" in formatted
    # has an expandable data section
    assert formatted.count("class='xr-array-in' type='checkbox' >") == 1
    # coords and attrs don't have an items so they'll be be disabled and collapsed
    assert (
        formatted.count("class='xr-section-summary-in' type='checkbox' disabled >") == 2
    )


def test_summary_of_multiindex_coord(multiindex):
    idx = multiindex.x.variable.to_index_variable()
    formatted = fh._summarize_coord_multiindex("foo", idx)
    assert "(level_1, level_2)" in formatted
    assert "MultiIndex" in formatted
    assert "<span class='xr-has-index'>foo</span>" in formatted


def test_repr_of_multiindex(multiindex):
    formatted = fh.dataset_repr(multiindex)
    assert "(x)" in formatted


def test_repr_of_dataset(dataset):
    formatted = fh.dataset_repr(dataset)
    # coords, attrs, and data_vars are expanded
    assert (
        formatted.count("class='xr-section-summary-in' type='checkbox'  checked>") == 3
    )
    assert "&lt;U4" in formatted or "&gt;U4" in formatted
    assert "&lt;IA&gt;" in formatted
