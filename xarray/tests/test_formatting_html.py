from __future__ import annotations

import re
from functools import partial

import numpy as np
import pandas as pd
import pytest

import xarray as xr
from xarray.core import formatting_html as fh
from xarray.core.coordinates import Coordinates
from xarray.tests import requires_dask


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
    annual_cycle = np.sin(2 * np.pi * (times.day_of_year.values / 365.25 - 0.28))

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


@requires_dask
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
    assert "<dt><span>&lt;x&gt;</span></dt>" in formatted
    assert "<dt><span>y</span></dt>" in formatted
    assert "<dd>3</dd>" in formatted
    assert "<dd>&lt;pd.DataFrame&gt;</dd>" in formatted


def test_repr_toggle_all() -> None:
    ds = xr.Dataset(
        {"a": ("x", [1, 2]), "b": ("x", [3, 4])},
        coords={"x": ("x", [0, 1], {"units": "m"})},
    )
    with xr.set_options(display_default_indexes=True):
        formatted = xarray_html_only_repr(ds)
    by_name = {}
    for section in formatted.split("<li class='xr-section-item'>")[1:]:
        match = re.search(r"class='xr-section-summary'[^>]*>([^:<]*)", section)
        assert match is not None
        by_name[match.group(1)] = section

    # coordinates have attributes, so both toggles are enabled
    coords = by_name["Coordinates"]
    assert "class='xr-all-attrs-in' type='checkbox'>" in coords
    assert "class='xr-all-data-in' type='checkbox'>" in coords
    # no data variable has attributes
    assert (
        "class='xr-all-attrs-in' type='checkbox' disabled>" in by_name["Data variables"]
    )
    # indexes have no attributes toggle
    assert "xr-all-attrs-in" not in by_name["Indexes"]
    assert "class='xr-all-data-in' type='checkbox'>" in by_name["Indexes"]
    for name in ("Dimensions", "Attributes"):
        assert "xr-all-" not in by_name.get(name, "")


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
        formatted.count("class='xr-section-summary-in' type='checkbox' checked />") == 3
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
            formatted.count("class='xr-section-summary-in' type='checkbox' checked />")
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


@pytest.mark.parametrize("kind", ["Variable", "DataArray", "Dataset", "DataTree"])
def test_repr_of_nonstr_names(kind: str) -> None:
    # names of dimensions, coordinates, indexes, variables and attributes
    # can be any hashable, e.g. int or tuple (GH10613)
    ds = xr.Dataset(
        {1: ((("a", "b"), 10), np.zeros((2, 3)), {2: "variable attr"})},
        coords={10: [5, 6, 7], 7: ((("a", "b"),), ["x", "y"])},
        attrs={3: "int key", ("a", "b"): "tuple key"},
    )
    objs: dict[str, XarrayTypes] = {
        "Variable": ds[1].variable,
        "DataArray": ds[1],
        "Dataset": ds,
        "DataTree": xr.DataTree(ds),
    }
    obj = objs[kind]
    with xr.set_options(display_style="html", display_default_indexes=True):
        html = obj._repr_html_()

    tup = "(&#x27;a&#x27;, &#x27;b&#x27;)"
    expected = [f"<li><span>{tup}</span>: 2</li>"]
    if kind == "Variable":
        expected += [
            "<li><span>10</span>: 3</li>",
            "<dt><span>2</span></dt><dd>variable attr</dd>",
        ]
    else:
        expected += [
            "<li><span class='xr-has-index'>10</span>: 3</li>",
            "<div class='xr-var-name'><span class='xr-has-index'>10</span>",
            "<div class='xr-var-name'><span>7</span>",
            f"<div class='xr-var-dims'>({tup})</div>",
            "<div class='xr-index-name'><div>10</div></div>",
        ]
    if kind == "DataArray":
        expected.append("<dt><span>2</span></dt><dd>variable attr</dd>")
    if kind in ("Dataset", "DataTree"):
        expected += [
            "<div class='xr-var-name'><span>1</span>",
            f"<div class='xr-var-dims'>({tup}, 10)</div>",
            "<dt><span>3</span></dt><dd>int key</dd>",
            f"<dt><span>{tup}</span></dt><dd>tuple key</dd>",
        ]
    for fragment in expected:
        assert fragment in html


class TestDataTreeTruncatesNodes:
    def test_many_nodes(self) -> None:
        number_of_files = 10
        number_of_groups = 10
        tree_dict = {}
        for f in range(number_of_files):
            for g in range(number_of_groups):
                tree_dict[f"file_{f}/group_{g}"] = xr.Dataset({"g": f * g})

        tree = xr.DataTree.from_dict(tree_dict)

        with xr.set_options(display_max_html_elements=25):
            result = xarray_html_only_repr(tree)
        assert result.count("file_0/group_9") == 1
        assert result.count("file_1/group_0") == 0  # disabled
        assert result.count("Too many items to display") == 9 + 10

        with xr.set_options(display_max_html_elements=1000):
            result = xarray_html_only_repr(tree)
        assert result.count("Too many items to display") == 0

    def test_many_children_truncated(self) -> None:
        # Create tree with 20 children at root level
        tree_dict = {f"child_{i:02d}": xr.Dataset({"x": i}) for i in range(20)}
        tree = xr.DataTree.from_dict(tree_dict)

        # With max_children=5: show first 3, ellipsis, last 2
        with xr.set_options(display_max_children=5, display_max_html_elements=1000):
            result = xarray_html_only_repr(tree)

        # First 3 children should appear
        assert "/child_00" in result
        assert "/child_01" in result
        assert "/child_02" in result

        # Middle children should NOT appear
        assert "/child_03" not in result
        assert "/child_10" not in result
        assert "/child_17" not in result

        # Last 2 children should appear
        assert "/child_18" in result
        assert "/child_19" in result

        # Vertical ellipsis should appear
        assert "⋮" in result

    def test_few_children_not_truncated(self) -> None:
        # Create tree with 5 children (at the limit)
        tree_dict = {f"child_{i}": xr.Dataset({"x": i}) for i in range(5)}
        tree = xr.DataTree.from_dict(tree_dict)

        with xr.set_options(display_max_children=5, display_max_html_elements=1000):
            result = xarray_html_only_repr(tree)

        # All children should appear
        for i in range(5):
            assert f"/child_{i}" in result

        # No ellipsis
        assert "⋮" not in result

    def test_nested_children_truncated(self) -> None:
        # Create tree with nested children: root → 10 children → each with 2 grandchildren
        tree_dict = {}
        for i in range(10):
            for j in range(2):
                tree_dict[f"child_{i:02d}/grandchild_{j}"] = xr.Dataset({"x": i * j})
        tree = xr.DataTree.from_dict(tree_dict)

        with xr.set_options(display_max_children=5, display_max_html_elements=1000):
            result = xarray_html_only_repr(tree)

        # Root level: first 3 and last 2 of 10 children should appear
        assert "/child_00" in result
        assert "/child_01" in result
        assert "/child_02" in result
        assert "/child_05" not in result  # truncated
        assert "/child_08" in result
        assert "/child_09" in result

        # Ellipsis should appear for truncated children
        assert "⋮" in result

    def test_node_item_count_displayed(self) -> None:
        # Create tree with known item counts
        tree = xr.DataTree.from_dict(
            {
                "node_a": xr.Dataset({"var1": 1, "var2": 2}),  # 2 vars
                "node_b": xr.Dataset(
                    {"var1": 1}, attrs={"attr1": "x", "attr2": "y"}
                ),  # 1 var + 2 attrs
            }
        )

        with xr.set_options(display_max_html_elements=1000):
            result = xarray_html_only_repr(tree)

        # Item counts should appear in parentheses
        assert "(2)" in result  # node_a: 2 variables
        assert "(3)" in result  # node_b: 1 variable + 2 attrs

    def test_collapsible_group_checkbox(self) -> None:
        # Create simple tree with children
        tree = xr.DataTree.from_dict(
            {
                "child_a": xr.Dataset({"x": 1}),
                "child_b": xr.Dataset({"y": 2}),
            }
        )

        with xr.set_options(display_max_html_elements=1000):
            result = xarray_html_only_repr(tree)

        # Group nodes should have checkbox inputs for collapsing
        assert "<input" in result
        assert "type='checkbox'" in result


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
