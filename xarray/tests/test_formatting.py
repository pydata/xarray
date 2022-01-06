import sys
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest
from numpy.core import defchararray

import xarray as xr
from xarray.core import formatting

from . import requires_netCDF4


class TestFormatting:
    def test_get_indexer_at_least_n_items(self) -> None:
        cases = [
            ((20,), (slice(10),), (slice(-10, None),)),
            ((3, 20), (0, slice(10)), (-1, slice(-10, None))),
            ((2, 10), (0, slice(10)), (-1, slice(-10, None))),
            ((2, 5), (slice(2), slice(None)), (slice(-2, None), slice(None))),
            ((1, 2, 5), (0, slice(2), slice(None)), (-1, slice(-2, None), slice(None))),
            ((2, 3, 5), (0, slice(2), slice(None)), (-1, slice(-2, None), slice(None))),
            (
                (1, 10, 1),
                (0, slice(10), slice(None)),
                (-1, slice(-10, None), slice(None)),
            ),
            (
                (2, 5, 1),
                (slice(2), slice(None), slice(None)),
                (slice(-2, None), slice(None), slice(None)),
            ),
            ((2, 5, 3), (0, slice(4), slice(None)), (-1, slice(-4, None), slice(None))),
            (
                (2, 3, 3),
                (slice(2), slice(None), slice(None)),
                (slice(-2, None), slice(None), slice(None)),
            ),
        ]
        for shape, start_expected, end_expected in cases:
            actual = formatting._get_indexer_at_least_n_items(shape, 10, from_end=False)
            assert start_expected == actual
            actual = formatting._get_indexer_at_least_n_items(shape, 10, from_end=True)
            assert end_expected == actual

    def test_first_n_items(self) -> None:
        array = np.arange(100).reshape(10, 5, 2)
        for n in [3, 10, 13, 100, 200]:
            actual = formatting.first_n_items(array, n)
            expected = array.flat[:n]
            assert (expected == actual).all()

        with pytest.raises(ValueError, match=r"at least one item"):
            formatting.first_n_items(array, 0)

    def test_last_n_items(self) -> None:
        array = np.arange(100).reshape(10, 5, 2)
        for n in [3, 10, 13, 100, 200]:
            actual = formatting.last_n_items(array, n)
            expected = array.flat[-n:]
            assert (expected == actual).all()

        with pytest.raises(ValueError, match=r"at least one item"):
            formatting.first_n_items(array, 0)

    def test_last_item(self) -> None:
        array = np.arange(100)

        reshape = ((10, 10), (1, 100), (2, 2, 5, 5))
        expected = np.array([99])

        for r in reshape:
            result = formatting.last_item(array.reshape(r))
            assert result == expected

    def test_format_item(self) -> None:
        cases = [
            (pd.Timestamp("2000-01-01T12"), "2000-01-01T12:00:00"),
            (pd.Timestamp("2000-01-01"), "2000-01-01"),
            (pd.Timestamp("NaT"), "NaT"),
            (pd.Timedelta("10 days 1 hour"), "10 days 01:00:00"),
            (pd.Timedelta("-3 days"), "-3 days +00:00:00"),
            (pd.Timedelta("3 hours"), "0 days 03:00:00"),
            (pd.Timedelta("NaT"), "NaT"),
            ("foo", "'foo'"),
            (b"foo", "b'foo'"),
            (1, "1"),
            (1.0, "1.0"),
            (np.float16(1.1234), "1.123"),
            (np.float32(1.0111111), "1.011"),
            (np.float64(22.222222), "22.22"),
        ]
        for item, expected in cases:
            actual = formatting.format_item(item)
            assert expected == actual

    def test_format_items(self) -> None:
        cases = [
            (np.arange(4) * np.timedelta64(1, "D"), "0 days 1 days 2 days 3 days"),
            (
                np.arange(4) * np.timedelta64(3, "h"),
                "00:00:00 03:00:00 06:00:00 09:00:00",
            ),
            (
                np.arange(4) * np.timedelta64(500, "ms"),
                "00:00:00 00:00:00.500000 00:00:01 00:00:01.500000",
            ),
            (pd.to_timedelta(["NaT", "0s", "1s", "NaT"]), "NaT 00:00:00 00:00:01 NaT"),
            (
                pd.to_timedelta(["1 day 1 hour", "1 day", "0 hours"]),
                "1 days 01:00:00 1 days 00:00:00 0 days 00:00:00",
            ),
            ([1, 2, 3], "1 2 3"),
        ]
        for item, expected in cases:
            actual = " ".join(formatting.format_items(item))
            assert expected == actual

    def test_format_array_flat(self) -> None:
        actual = formatting.format_array_flat(np.arange(100), 2)
        expected = "..."
        assert expected == actual

        actual = formatting.format_array_flat(np.arange(100), 9)
        expected = "0 ... 99"
        assert expected == actual

        actual = formatting.format_array_flat(np.arange(100), 10)
        expected = "0 1 ... 99"
        assert expected == actual

        actual = formatting.format_array_flat(np.arange(100), 13)
        expected = "0 1 ... 98 99"
        assert expected == actual

        actual = formatting.format_array_flat(np.arange(100), 15)
        expected = "0 1 2 ... 98 99"
        assert expected == actual

        # NB: Probably not ideal; an alternative would be cutting after the
        # first ellipsis
        actual = formatting.format_array_flat(np.arange(100.0), 11)
        expected = "0.0 ... ..."
        assert expected == actual

        actual = formatting.format_array_flat(np.arange(100.0), 12)
        expected = "0.0 ... 99.0"
        assert expected == actual

        actual = formatting.format_array_flat(np.arange(3), 5)
        expected = "0 1 2"
        assert expected == actual

        actual = formatting.format_array_flat(np.arange(4.0), 11)
        expected = "0.0 ... 3.0"
        assert expected == actual

        actual = formatting.format_array_flat(np.arange(0), 0)
        expected = ""
        assert expected == actual

        actual = formatting.format_array_flat(np.arange(1), 1)
        expected = "0"
        assert expected == actual

        actual = formatting.format_array_flat(np.arange(2), 3)
        expected = "0 1"
        assert expected == actual

        actual = formatting.format_array_flat(np.arange(4), 7)
        expected = "0 1 2 3"
        assert expected == actual

        actual = formatting.format_array_flat(np.arange(5), 7)
        expected = "0 ... 4"
        assert expected == actual

        long_str = [" ".join(["hello world" for _ in range(100)])]
        actual = formatting.format_array_flat(np.asarray([long_str]), 21)
        expected = "'hello world hello..."
        assert expected == actual

    def test_pretty_print(self) -> None:
        assert formatting.pretty_print("abcdefghij", 8) == "abcde..."
        assert formatting.pretty_print("ß", 1) == "ß"

    def test_maybe_truncate(self) -> None:
        assert formatting.maybe_truncate("ß", 10) == "ß"

    def test_format_timestamp_invalid_pandas_format(self) -> None:
        expected = "2021-12-06 17:00:00 00"
        with pytest.raises(ValueError):
            formatting.format_timestamp(expected)

    def test_format_timestamp_out_of_bounds(self) -> None:
        from datetime import datetime

        date = datetime(1300, 12, 1)
        expected = "1300-12-01"
        result = formatting.format_timestamp(date)
        assert result == expected

        date = datetime(2300, 12, 1)
        expected = "2300-12-01"
        result = formatting.format_timestamp(date)
        assert result == expected

    def test_attribute_repr(self) -> None:
        short = formatting.summarize_attr("key", "Short string")
        long = formatting.summarize_attr("key", 100 * "Very long string ")
        newlines = formatting.summarize_attr("key", "\n\n\n")
        tabs = formatting.summarize_attr("key", "\t\t\t")
        assert short == "    key: Short string"
        assert len(long) <= 80
        assert long.endswith("...")
        assert "\n" not in newlines
        assert "\t" not in tabs

    def test_diff_array_repr(self) -> None:
        da_a = xr.DataArray(
            np.array([[1, 2, 3], [4, 5, 6]], dtype="int64"),
            dims=("x", "y"),
            coords={
                "x": np.array(["a", "b"], dtype="U1"),
                "y": np.array([1, 2, 3], dtype="int64"),
            },
            attrs={"units": "m", "description": "desc"},
        )

        da_b = xr.DataArray(
            np.array([1, 2], dtype="int64"),
            dims="x",
            coords={
                "x": np.array(["a", "c"], dtype="U1"),
                "label": ("x", np.array([1, 2], dtype="int64")),
            },
            attrs={"units": "kg"},
        )

        byteorder = "<" if sys.byteorder == "little" else ">"
        expected = dedent(
            """\
        Left and right DataArray objects are not identical
        Differing dimensions:
            (x: 2, y: 3) != (x: 2)
        Differing values:
        L
            array([[1, 2, 3],
                   [4, 5, 6]], dtype=int64)
        R
            array([1, 2], dtype=int64)
        Differing coordinates:
        L * x        (x) %cU1 'a' 'b'
        R * x        (x) %cU1 'a' 'c'
        Coordinates only on the left object:
          * y        (y) int64 1 2 3
        Coordinates only on the right object:
            label    (x) int64 1 2
        Differing attributes:
        L   units: m
        R   units: kg
        Attributes only on the left object:
            description: desc"""
            % (byteorder, byteorder)
        )

        actual = formatting.diff_array_repr(da_a, da_b, "identical")
        try:
            assert actual == expected
        except AssertionError:
            # depending on platform, dtype may not be shown in numpy array repr
            assert actual == expected.replace(", dtype=int64", "")

        va = xr.Variable(
            "x", np.array([1, 2, 3], dtype="int64"), {"title": "test Variable"}
        )
        vb = xr.Variable(("x", "y"), np.array([[1, 2, 3], [4, 5, 6]], dtype="int64"))

        expected = dedent(
            """\
        Left and right Variable objects are not equal
        Differing dimensions:
            (x: 3) != (x: 2, y: 3)
        Differing values:
        L
            array([1, 2, 3], dtype=int64)
        R
            array([[1, 2, 3],
                   [4, 5, 6]], dtype=int64)"""
        )

        actual = formatting.diff_array_repr(va, vb, "equals")
        try:
            assert actual == expected
        except AssertionError:
            assert actual == expected.replace(", dtype=int64", "")

    @pytest.mark.filterwarnings("error")
    def test_diff_attrs_repr_with_array(self) -> None:
        attrs_a = {"attr": np.array([0, 1])}

        attrs_b = {"attr": 1}
        expected = dedent(
            """\
            Differing attributes:
            L   attr: [0 1]
            R   attr: 1
            """
        ).strip()
        actual = formatting.diff_attrs_repr(attrs_a, attrs_b, "equals")
        assert expected == actual

        attrs_c = {"attr": np.array([-3, 5])}
        expected = dedent(
            """\
            Differing attributes:
            L   attr: [0 1]
            R   attr: [-3  5]
            """
        ).strip()
        actual = formatting.diff_attrs_repr(attrs_a, attrs_c, "equals")
        assert expected == actual

        # should not raise a warning
        attrs_c = {"attr": np.array([0, 1, 2])}
        expected = dedent(
            """\
            Differing attributes:
            L   attr: [0 1]
            R   attr: [0 1 2]
            """
        ).strip()
        actual = formatting.diff_attrs_repr(attrs_a, attrs_c, "equals")
        assert expected == actual

    def test_diff_dataset_repr(self) -> None:
        ds_a = xr.Dataset(
            data_vars={
                "var1": (("x", "y"), np.array([[1, 2, 3], [4, 5, 6]], dtype="int64")),
                "var2": ("x", np.array([3, 4], dtype="int64")),
            },
            coords={
                "x": np.array(["a", "b"], dtype="U1"),
                "y": np.array([1, 2, 3], dtype="int64"),
            },
            attrs={"units": "m", "description": "desc"},
        )

        ds_b = xr.Dataset(
            data_vars={"var1": ("x", np.array([1, 2], dtype="int64"))},
            coords={
                "x": ("x", np.array(["a", "c"], dtype="U1"), {"source": 0}),
                "label": ("x", np.array([1, 2], dtype="int64")),
            },
            attrs={"units": "kg"},
        )

        byteorder = "<" if sys.byteorder == "little" else ">"
        expected = dedent(
            """\
        Left and right Dataset objects are not identical
        Differing dimensions:
            (x: 2, y: 3) != (x: 2)
        Differing coordinates:
        L * x        (x) %cU1 'a' 'b'
        R * x        (x) %cU1 'a' 'c'
            source: 0
        Coordinates only on the left object:
          * y        (y) int64 1 2 3
        Coordinates only on the right object:
            label    (x) int64 1 2
        Differing data variables:
        L   var1     (x, y) int64 1 2 3 4 5 6
        R   var1     (x) int64 1 2
        Data variables only on the left object:
            var2     (x) int64 3 4
        Differing attributes:
        L   units: m
        R   units: kg
        Attributes only on the left object:
            description: desc"""
            % (byteorder, byteorder)
        )

        actual = formatting.diff_dataset_repr(ds_a, ds_b, "identical")
        assert actual == expected

    def test_array_repr(self) -> None:
        ds = xr.Dataset(coords={"foo": [1, 2, 3], "bar": [1, 2, 3]})
        ds[(1, 2)] = xr.DataArray([0], dims="test")
        actual = formatting.array_repr(ds[(1, 2)])
        expected = dedent(
            """\
        <xarray.DataArray (1, 2) (test: 1)>
        array([0])
        Dimensions without coordinates: test"""
        )

        assert actual == expected

        with xr.set_options(display_expand_data=False):
            actual = formatting.array_repr(ds[(1, 2)])
            expected = dedent(
                """\
            <xarray.DataArray (1, 2) (test: 1)>
            0
            Dimensions without coordinates: test"""
            )

            assert actual == expected

    def test_array_repr_variable(self) -> None:
        var = xr.Variable("x", [0, 1])

        formatting.array_repr(var)

        with xr.set_options(display_expand_data=False):
            formatting.array_repr(var)


def test_inline_variable_array_repr_custom_repr() -> None:
    class CustomArray:
        def __init__(self, value, attr):
            self.value = value
            self.attr = attr

        def _repr_inline_(self, width):
            formatted = f"({self.attr}) {self.value}"
            if len(formatted) > width:
                formatted = f"({self.attr}) ..."

            return formatted

        def __array_function__(self, *args, **kwargs):
            return NotImplemented

        @property
        def shape(self):
            return self.value.shape

        @property
        def dtype(self):
            return self.value.dtype

        @property
        def ndim(self):
            return self.value.ndim

    value = CustomArray(np.array([20, 40]), "m")
    variable = xr.Variable("x", value)

    max_width = 10
    actual = formatting.inline_variable_array_repr(variable, max_width=10)

    assert actual == value._repr_inline_(max_width)


def test_set_numpy_options() -> None:
    original_options = np.get_printoptions()
    with formatting.set_numpy_options(threshold=10):
        assert len(repr(np.arange(500))) < 200
    # original options are restored
    assert np.get_printoptions() == original_options


def test_short_numpy_repr() -> None:
    cases = [
        np.random.randn(500),
        np.random.randn(20, 20),
        np.random.randn(5, 10, 15),
        np.random.randn(5, 10, 15, 3),
        np.random.randn(100, 5, 1),
    ]
    # number of lines:
    # for default numpy repr: 167, 140, 254, 248, 599
    # for short_numpy_repr: 1, 7, 24, 19, 25
    for array in cases:
        num_lines = formatting.short_numpy_repr(array).count("\n") + 1
        assert num_lines < 30


def test_large_array_repr_length() -> None:

    da = xr.DataArray(np.random.randn(100, 5, 1))

    result = repr(da).splitlines()
    assert len(result) < 50


@requires_netCDF4
def test_repr_file_collapsed(tmp_path) -> None:
    arr = xr.DataArray(np.arange(300), dims="test")
    arr.to_netcdf(tmp_path / "test.nc", engine="netcdf4")

    with xr.open_dataarray(tmp_path / "test.nc") as arr, xr.set_options(
        display_expand_data=False
    ):
        actual = formatting.array_repr(arr)
        expected = dedent(
            """\
        <xarray.DataArray (test: 300)>
        array([  0,   1,   2, ..., 297, 298, 299])
        Dimensions without coordinates: test"""
        )

        assert actual == expected


@pytest.mark.parametrize(
    "display_max_rows, n_vars, n_attr",
    [(50, 40, 30), (35, 40, 30), (11, 40, 30), (1, 40, 30)],
)
def test__mapping_repr(display_max_rows, n_vars, n_attr) -> None:
    long_name = "long_name"
    a = defchararray.add(long_name, np.arange(0, n_vars).astype(str))
    b = defchararray.add("attr_", np.arange(0, n_attr).astype(str))
    c = defchararray.add("coord", np.arange(0, n_vars).astype(str))
    attrs = {k: 2 for k in b}
    coords = {_c: np.array([0, 1]) for _c in c}
    data_vars = dict()
    for (v, _c) in zip(a, coords.items()):
        data_vars[v] = xr.DataArray(
            name=v,
            data=np.array([3, 4]),
            dims=[_c[0]],
            coords=dict([_c]),
        )
    ds = xr.Dataset(data_vars)
    ds.attrs = attrs

    with xr.set_options(display_max_rows=display_max_rows):

        # Parse the data_vars print and show only data_vars rows:
        summary = formatting.dataset_repr(ds).split("\n")
        summary = [v for v in summary if long_name in v]
        # The length should be less than or equal to display_max_rows:
        len_summary = len(summary)
        data_vars_print_size = min(display_max_rows, len_summary)
        assert len_summary == data_vars_print_size

        summary = formatting.data_vars_repr(ds.data_vars).split("\n")
        summary = [v for v in summary if long_name in v]
        # The length should be equal to the number of data variables
        len_summary = len(summary)
        assert len_summary == n_vars

        summary = formatting.coords_repr(ds.coords).split("\n")
        summary = [v for v in summary if "coord" in v]
        # The length should be equal to the number of data variables
        len_summary = len(summary)
        assert len_summary == n_vars

    with xr.set_options(
        display_max_rows=display_max_rows,
        display_expand_coords=False,
        display_expand_data_vars=False,
        display_expand_attrs=False,
    ):
        actual = formatting.dataset_repr(ds)
        col_width = formatting._calculate_col_width(
            formatting._get_col_items(ds.variables)
        )
        dims_start = formatting.pretty_print("Dimensions:", col_width)
        dims_values = formatting.dim_summary_limited(
            ds, col_width=col_width + 1, max_rows=display_max_rows
        )
        expected = f"""\
<xarray.Dataset>
{dims_start}({dims_values})
Coordinates: ({n_vars})
Data variables: ({n_vars})
Attributes: ({n_attr})"""
        expected = dedent(expected)
        assert actual == expected


def test__element_formatter(n_elements: int = 100) -> None:
    expected = """\
    Dimensions without coordinates: dim_0: 3, dim_1: 3, dim_2: 3, dim_3: 3,
                                    dim_4: 3, dim_5: 3, dim_6: 3, dim_7: 3,
                                    dim_8: 3, dim_9: 3, dim_10: 3, dim_11: 3,
                                    dim_12: 3, dim_13: 3, dim_14: 3, dim_15: 3,
                                    dim_16: 3, dim_17: 3, dim_18: 3, dim_19: 3,
                                    dim_20: 3, dim_21: 3, dim_22: 3, dim_23: 3,
                                    ...
                                    dim_76: 3, dim_77: 3, dim_78: 3, dim_79: 3,
                                    dim_80: 3, dim_81: 3, dim_82: 3, dim_83: 3,
                                    dim_84: 3, dim_85: 3, dim_86: 3, dim_87: 3,
                                    dim_88: 3, dim_89: 3, dim_90: 3, dim_91: 3,
                                    dim_92: 3, dim_93: 3, dim_94: 3, dim_95: 3,
                                    dim_96: 3, dim_97: 3, dim_98: 3, dim_99: 3"""
    expected = dedent(expected)

    intro = "Dimensions without coordinates: "
    elements = [
        f"{k}: {v}" for k, v in {f"dim_{k}": 3 for k in np.arange(n_elements)}.items()
    ]
    values = xr.core.formatting._element_formatter(
        elements, col_width=len(intro), max_rows=12
    )
    actual = intro + values
    assert expected == actual
