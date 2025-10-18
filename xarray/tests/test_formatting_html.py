from __future__ import annotations

import re
from functools import partial

import numpy as np
import pandas as pd
import pytest

import xarray as xr
from xarray.core import formatting_html as fh
from xarray.core.coordinates import Coordinates


def drop_fallback_text_repr(html: str) -> str:
    pattern = (
        re.escape("<pre class='xr-text-repr-fallback'>") + "[^<]*" + re.escape("</pre>")
    )
    return re.sub(pattern, "", html)


XarrayTypes = xr.DataTree | xr.Dataset | xr.DataArray | xr.Variable


def xarray_html_only_repr(obj: XarrayTypes) -> str:
    return drop_fallback_text_repr(obj._repr_html_())


def assert_consistent_text_and_html(
    obj: XarrayTypes, section_headers: list[str]
) -> None:
    actual_html = xarray_html_only_repr(obj)
    actual_text = repr(obj)
    for section_header in section_headers:
        assert actual_html.count(section_header) == actual_text.count(section_header), (
            section_header
        )


assert_consistent_text_and_html_dataarray = partial(
    assert_consistent_text_and_html,
    section_headers=[
        "Coordinates",
        "Indexes",
        "Attributes",
    ],
)


assert_consistent_text_and_html_dataset = partial(
    assert_consistent_text_and_html,
    section_headers=[
        "Dimensions",
        "Coordinates",
        "Data variables",
        "Indexes",
        "Attributes",
    ],
)


assert_consistent_text_and_html_datatree = partial(
    assert_consistent_text_and_html,
    section_headers=[
        "Dimensions",
        "Coordinates",
        "Inherited coordinates",
        "Data variables",
        "Indexes",
        "Attributes",
    ],
)


@pytest.fixture
def dataarray() -> xr.DataArray:
    return xr.DataArray(np.random.default_rng(0).random((4, 6)))


@pytest.fixture
def dask_dataarray(dataarray: xr.DataArray) -> xr.DataArray:
    pytest.importorskip("dask")
    return dataarray.chunk()


@pytest.fixture
def multiindex() -> xr.Dataset:
    midx = pd.MultiIndex.from_product(
        [["a", "b"], [1, 2]], names=("level_1", "level_2")
    )
    midx_coords = Coordinates.from_pandas_multiindex(midx, "x")
    return xr.Dataset({}, midx_coords)


@pytest.fixture
def dataset() -> xr.Dataset:
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
        {"location": ["<IA>", "IN", "IL"], "time": times},
        attrs={"description": "Test data."},
    )


def test_short_data_repr_html(dataarray: xr.DataArray) -> None:
    data_repr = fh.short_data_repr_html(dataarray)
    assert data_repr.startswith("<pre>array")


def test_short_data_repr_html_non_str_keys(dataset: xr.Dataset) -> None:
    ds = dataset.assign({2: lambda x: x["tmin"]})
    fh.dataset_repr(ds)


def test_short_data_repr_html_dask(dask_dataarray: xr.DataArray) -> None:
    assert hasattr(dask_dataarray.data, "_repr_html_")
    data_repr = fh.short_data_repr_html(dask_dataarray)
    assert data_repr == dask_dataarray.data._repr_html_()


def test_format_dims_no_dims() -> None:
    dims: dict = {}
    dims_with_index: list = []
    formatted = fh.format_dims(dims, dims_with_index)
    assert formatted == ""


def test_format_dims_unsafe_dim_name() -> None:
    dims = {"<x>": 3, "y": 2}
    dims_with_index: list = []
    formatted = fh.format_dims(dims, dims_with_index)
    assert "&lt;x&gt;" in formatted


def test_format_dims_non_index() -> None:
    dims, dims_with_index = {"x": 3, "y": 2}, ["time"]
    formatted = fh.format_dims(dims, dims_with_index)
    assert "class='xr-has-index'" not in formatted


def test_format_dims_index() -> None:
    dims, dims_with_index = {"x": 3, "y": 2}, ["x"]
    formatted = fh.format_dims(dims, dims_with_index)
    assert "class='xr-has-index'" in formatted


def test_summarize_attrs_with_unsafe_attr_name_and_value() -> None:
    attrs = {"<x>": 3, "y": "<pd.DataFrame>"}
    formatted = fh.summarize_attrs(attrs)
    assert "<dt><span>&lt;x&gt; :</span></dt>" in formatted
    assert "<dt><span>y :</span></dt>" in formatted
    assert "<dd>3</dd>" in formatted
    assert "<dd>&lt;pd.DataFrame&gt;</dd>" in formatted


def test_repr_of_dataarray() -> None:
    dataarray = xr.DataArray(np.random.default_rng(0).random((4, 6)))
    formatted = xarray_html_only_repr(dataarray)
    assert "dim_0" in formatted
    # has an expanded data section
    assert formatted.count("class='xr-array-in' type='checkbox' checked>") == 1
    # coords, indexes and attrs don't have an items so they'll be omitted
    assert "Coordinates" not in formatted
    assert "Indexes" not in formatted
    assert "Attributes" not in formatted

    assert_consistent_text_and_html_dataarray(dataarray)

    with xr.set_options(display_expand_data=False):
        formatted = xarray_html_only_repr(dataarray)
        assert "dim_0" in formatted
        # has a collapsed data section
        assert formatted.count("class='xr-array-in' type='checkbox' checked>") == 0
        # coords, indexes and attrs don't have an items so they'll be omitted
        assert "Coordinates" not in formatted
        assert "Indexes" not in formatted
        assert "Attributes" not in formatted


def test_repr_coords_order_of_datarray() -> None:
    da1 = xr.DataArray(
        np.empty((2, 2)),
        coords={"foo": [0, 1], "bar": [0, 1]},
        dims=["foo", "bar"],
    )
    da2 = xr.DataArray(
        np.empty((2, 2)),
        coords={"bar": [0, 1], "foo": [0, 1]},
        dims=["bar", "foo"],
    )
    ds = xr.Dataset({"da1": da1, "da2": da2})

    bar_line = (
        "<span class='xr-has-index'>bar</span></div><div class='xr-var-dims'>(bar)"
    )
    foo_line = (
        "<span class='xr-has-index'>foo</span></div><div class='xr-var-dims'>(foo)"
    )

    formatted_da1 = fh.array_repr(ds.da1)
    assert formatted_da1.index(foo_line) < formatted_da1.index(bar_line)

    formatted_da2 = fh.array_repr(ds.da2)
    assert formatted_da2.index(bar_line) < formatted_da2.index(foo_line)


def test_repr_of_multiindex(multiindex: xr.Dataset) -> None:
    formatted = fh.dataset_repr(multiindex)
    assert "(x)" in formatted

    assert_consistent_text_and_html_dataset(multiindex)


def test_repr_of_dataset(dataset: xr.Dataset) -> None:
    formatted = xarray_html_only_repr(dataset)
    # coords, attrs, and data_vars are expanded
    assert (
        formatted.count("class='xr-section-summary-in' type='checkbox'  checked>") == 3
    )
    # indexes is omitted
    assert "Indexes" not in formatted
    assert "&lt;U4" in formatted or "&gt;U4" in formatted
    assert "&lt;IA&gt;" in formatted

    assert_consistent_text_and_html_dataset(dataset)

    with xr.set_options(
        display_expand_coords=False,
        display_expand_data_vars=False,
        display_expand_attrs=False,
        display_expand_indexes=True,
        display_default_indexes=True,
    ):
        formatted = xarray_html_only_repr(dataset)
        # coords, attrs, and data_vars are collapsed, indexes is shown & expanded
        assert (
            formatted.count("class='xr-section-summary-in' type='checkbox'  checked>")
            == 1
        )
        assert "Indexes" in formatted
        assert "&lt;U4" in formatted or "&gt;U4" in formatted
        assert "&lt;IA&gt;" in formatted


def test_repr_text_fallback(dataset: xr.Dataset) -> None:
    formatted = fh.dataset_repr(dataset)

    # Just test that the "pre" block used for fallback to plain text is present.
    assert "<pre class='xr-text-repr-fallback'>" in formatted


def test_repr_coords_order_of_dataset() -> None:
    ds = xr.Dataset()
    ds.coords["as"] = 10
    ds["var"] = xr.DataArray(np.ones((10,)), dims="x", coords={"x": np.arange(10)})
    formatted = fh.dataset_repr(ds)

    x_line = "<span class='xr-has-index'>x</span></div><div class='xr-var-dims'>(x)"
    as_line = "<span>as</span></div><div class='xr-var-dims'>()"
    assert formatted.index(x_line) < formatted.index(as_line)


def test_variable_repr_html() -> None:
    v = xr.Variable(["time", "x"], [[1, 2, 3], [4, 5, 6]], {"foo": "bar"})
    assert hasattr(v, "_repr_html_")
    with xr.set_options(display_style="html"):
        html = v._repr_html_().strip()
    # We don't do a complete string identity since
    # html output is probably subject to change, is long and... reasons.
    # Just test that something reasonable was produced.
    assert html.startswith("<div") and html.endswith("</div>")
    assert "xarray.Variable" in html


def test_repr_of_nonstr_dataset(dataset: xr.Dataset) -> None:
    ds = dataset.copy()
    ds.attrs[1] = "Test value"
    ds[2] = ds["tmin"]
    formatted = fh.dataset_repr(ds)
    assert "<dt><span>1 :</span></dt><dd>Test value</dd>" in formatted
    assert "<div class='xr-var-name'><span>2</span>" in formatted


def test_repr_of_nonstr_dataarray(dataarray: xr.DataArray) -> None:
    da = dataarray.rename(dim_0=15)
    da.attrs[1] = "value"
    formatted = fh.array_repr(da)
    assert "<dt><span>1 :</span></dt><dd>value</dd>" in formatted
    assert "<li><span>15</span>: 4</li>" in formatted


def test_nonstr_variable_repr_html() -> None:
    v = xr.Variable(["time", 10], [[1, 2, 3], [4, 5, 6]], {22: "bar"})
    assert hasattr(v, "_repr_html_")
    with xr.set_options(display_style="html"):
        html = v._repr_html_().strip()
    assert "<dt><span>22 :</span></dt><dd>bar</dd>" in html
    assert "<li><span>10</span>: 3</li></ul>" in html


class TestDataTreeTruncatesNodes:
    def test_many_nodes(self) -> None:
        # construct a datatree with 500 nodes
        number_of_files = 20
        number_of_groups = 25
        tree_dict = {}
        for f in range(number_of_files):
            for g in range(number_of_groups):
                tree_dict[f"file_{f}/group_{g}"] = xr.Dataset({"g": f * g})

        tree = xr.DataTree.from_dict(tree_dict)
        with xr.set_options(display_style="html"):
            result = tree._repr_html_()

        assert "6/20" in result
        for i in range(number_of_files):
            if i < 3 or i >= (number_of_files - 3):
                assert f"file_{i}</div>" in result
            else:
                assert f"file_{i}</div>" not in result

        assert "6/25" in result
        for i in range(number_of_groups):
            if i < 3 or i >= (number_of_groups - 3):
                assert f"group_{i}</div>" in result
            else:
                assert f"group_{i}</div>" not in result

        with xr.set_options(display_style="html", display_max_children=3):
            result = tree._repr_html_()

        assert "3/20" in result
        for i in range(number_of_files):
            if i < 2 or i >= (number_of_files - 1):
                assert f"file_{i}</div>" in result
            else:
                assert f"file_{i}</div>" not in result

        assert "3/25" in result
        for i in range(number_of_groups):
            if i < 2 or i >= (number_of_groups - 1):
                assert f"group_{i}</div>" in result
            else:
                assert f"group_{i}</div>" not in result


class TestDataTreeInheritance:
    def test_inherited_section_present(self) -> None:
        dt = xr.DataTree.from_dict(data={"a/b/c": None}, coords={"x": [1]})

        root_html = dt._repr_html_()
        assert "Inherited coordinates" not in root_html

        child_html = xarray_html_only_repr(dt["a"])
        assert child_html.count("Inherited coordinates") == 1

    def test_repr_consistency(self) -> None:
        dt = xr.DataTree.from_dict({"/a/b/c": None})
        assert_consistent_text_and_html_datatree(dt)
        assert_consistent_text_and_html_datatree(dt["a"])
        assert_consistent_text_and_html_datatree(dt["a/b"])
        assert_consistent_text_and_html_datatree(dt["a/b/c"])

    def test_no_repeated_style_or_fallback_text(self) -> None:
        dt = xr.DataTree.from_dict({"/a/b/c": None})
        html = dt._repr_html_()
        assert html.count("<style>") == 1
        assert html.count("<pre class='xr-text-repr-fallback'>") == 1
