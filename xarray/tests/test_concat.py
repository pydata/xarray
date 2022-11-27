from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from xarray import DataArray, Dataset, Variable, concat
from xarray.core import dtypes, merge
from xarray.core.indexes import PandasIndex

from . import (
    InaccessibleArray,
    assert_array_equal,
    assert_equal,
    assert_identical,
    requires_dask,
)
from .test_dataset import create_test_data

if TYPE_CHECKING:
    from xarray.core.types import CombineAttrsOptions, JoinOptions


def test_concat_compat() -> None:
    ds1 = Dataset(
        {
            "has_x_y": (("y", "x"), [[1, 2]]),
            "has_x": ("x", [1, 2]),
            "no_x_y": ("z", [1, 2]),
        },
        coords={"x": [0, 1], "y": [0], "z": [-1, -2]},
    )
    ds2 = Dataset(
        {
            "has_x_y": (("y", "x"), [[3, 4]]),
            "has_x": ("x", [1, 2]),
            "no_x_y": (("q", "z"), [[1, 2]]),
        },
        coords={"x": [0, 1], "y": [1], "z": [-1, -2], "q": [0]},
    )

    result = concat([ds1, ds2], dim="y", data_vars="minimal", compat="broadcast_equals")
    assert_equal(ds2.no_x_y, result.no_x_y.transpose())

    for var in ["has_x", "no_x_y"]:
        assert "y" not in result[var].dims and "y" not in result[var].coords
    with pytest.raises(
        ValueError, match=r"coordinates in some datasets but not others"
    ):
        concat([ds1, ds2], dim="q")
    with pytest.raises(ValueError, match=r"'q' is not present in all datasets"):
        concat([ds2, ds1], dim="q")


class TestConcatDataset:
    @pytest.fixture
    def data(self) -> Dataset:
        return create_test_data().drop_dims("dim3")

    def rectify_dim_order(self, data, dataset) -> Dataset:
        # return a new dataset with all variable dimensions transposed into
        # the order in which they are found in `data`
        return Dataset(
            {k: v.transpose(*data[k].dims) for k, v in dataset.data_vars.items()},
            dataset.coords,
            attrs=dataset.attrs,
        )

    @pytest.mark.parametrize("coords", ["different", "minimal"])
    @pytest.mark.parametrize("dim", ["dim1", "dim2"])
    def test_concat_simple(self, data, dim, coords) -> None:
        datasets = [g for _, g in data.groupby(dim, squeeze=False)]
        assert_identical(data, concat(datasets, dim, coords=coords))

    def test_concat_merge_variables_present_in_some_datasets(self, data) -> None:
        # coordinates present in some datasets but not others
        ds1 = Dataset(data_vars={"a": ("y", [0.1])}, coords={"x": 0.1})
        ds2 = Dataset(data_vars={"a": ("y", [0.2])}, coords={"z": 0.2})
        actual = concat([ds1, ds2], dim="y", coords="minimal")
        expected = Dataset({"a": ("y", [0.1, 0.2])}, coords={"x": 0.1, "z": 0.2})
        assert_identical(expected, actual)

        # data variables present in some datasets but not others
        split_data = [data.isel(dim1=slice(3)), data.isel(dim1=slice(3, None))]
        data0, data1 = deepcopy(split_data)
        data1["foo"] = ("bar", np.random.randn(10))
        actual = concat([data0, data1], "dim1")
        expected = data.copy().assign(foo=data1.foo)
        assert_identical(expected, actual)

    def test_concat_2(self, data) -> None:
        dim = "dim2"
        datasets = [g for _, g in data.groupby(dim, squeeze=True)]
        concat_over = [k for k, v in data.coords.items() if dim in v.dims and k != dim]
        actual = concat(datasets, data[dim], coords=concat_over)
        assert_identical(data, self.rectify_dim_order(data, actual))

    @pytest.mark.parametrize("coords", ["different", "minimal", "all"])
    @pytest.mark.parametrize("dim", ["dim1", "dim2"])
    def test_concat_coords_kwarg(self, data, dim, coords) -> None:
        data = data.copy(deep=True)
        # make sure the coords argument behaves as expected
        data.coords["extra"] = ("dim4", np.arange(3))
        datasets = [g for _, g in data.groupby(dim, squeeze=True)]

        actual = concat(datasets, data[dim], coords=coords)
        if coords == "all":
            expected = np.array([data["extra"].values for _ in range(data.dims[dim])])
            assert_array_equal(actual["extra"].values, expected)

        else:
            assert_equal(data["extra"], actual["extra"])

    def test_concat(self, data) -> None:
        split_data = [
            data.isel(dim1=slice(3)),
            data.isel(dim1=3),
            data.isel(dim1=slice(4, None)),
        ]
        assert_identical(data, concat(split_data, "dim1"))

    def test_concat_dim_precedence(self, data) -> None:
        # verify that the dim argument takes precedence over
        # concatenating dataset variables of the same name
        dim = (2 * data["dim1"]).rename("dim1")
        datasets = [g for _, g in data.groupby("dim1", squeeze=False)]
        expected = data.copy()
        expected["dim1"] = dim
        assert_identical(expected, concat(datasets, dim))

    def test_concat_data_vars_typing(self) -> None:
        # Testing typing, can be removed if the next function works with annotations.
        data = Dataset({"foo": ("x", np.random.randn(10))})
        objs: list[Dataset] = [data.isel(x=slice(5)), data.isel(x=slice(5, None))]
        actual = concat(objs, dim="x", data_vars="minimal")
        assert_identical(data, actual)

    def test_concat_data_vars(self):
        # TODO: annotating this func fails
        data = Dataset({"foo": ("x", np.random.randn(10))})
        objs: list[Dataset] = [data.isel(x=slice(5)), data.isel(x=slice(5, None))]
        for data_vars in ["minimal", "different", "all", [], ["foo"]]:
            actual = concat(objs, dim="x", data_vars=data_vars)
            assert_identical(data, actual)

    def test_concat_coords(self):
        # TODO: annotating this func fails
        data = Dataset({"foo": ("x", np.random.randn(10))})
        expected = data.assign_coords(c=("x", [0] * 5 + [1] * 5))
        objs = [
            data.isel(x=slice(5)).assign_coords(c=0),
            data.isel(x=slice(5, None)).assign_coords(c=1),
        ]
        for coords in ["different", "all", ["c"]]:
            actual = concat(objs, dim="x", coords=coords)
            assert_identical(expected, actual)
        for coords in ["minimal", []]:
            with pytest.raises(merge.MergeError, match="conflicting values"):
                concat(objs, dim="x", coords=coords)

    def test_concat_constant_index(self):
        # TODO: annotating this func fails
        # GH425
        ds1 = Dataset({"foo": 1.5}, {"y": 1})
        ds2 = Dataset({"foo": 2.5}, {"y": 1})
        expected = Dataset({"foo": ("y", [1.5, 2.5]), "y": [1, 1]})
        for mode in ["different", "all", ["foo"]]:
            actual = concat([ds1, ds2], "y", data_vars=mode)
            assert_identical(expected, actual)
        with pytest.raises(merge.MergeError, match="conflicting values"):
            # previously dim="y", and raised error which makes no sense.
            # "foo" has dimension "y" so minimal should concatenate it?
            concat([ds1, ds2], "new_dim", data_vars="minimal")

    def test_concat_size0(self) -> None:
        data = create_test_data()
        split_data = [data.isel(dim1=slice(0, 0)), data]
        actual = concat(split_data, "dim1")
        assert_identical(data, actual)

        actual = concat(split_data[::-1], "dim1")
        assert_identical(data, actual)

    def test_concat_autoalign(self) -> None:
        ds1 = Dataset({"foo": DataArray([1, 2], coords=[("x", [1, 2])])})
        ds2 = Dataset({"foo": DataArray([1, 2], coords=[("x", [1, 3])])})
        actual = concat([ds1, ds2], "y")
        expected = Dataset(
            {
                "foo": DataArray(
                    [[1, 2, np.nan], [1, np.nan, 2]],
                    dims=["y", "x"],
                    coords={"x": [1, 2, 3]},
                )
            }
        )
        assert_identical(expected, actual)

    def test_concat_errors(self):
        # TODO: annotating this func fails
        data = create_test_data()
        split_data = [data.isel(dim1=slice(3)), data.isel(dim1=slice(3, None))]

        with pytest.raises(ValueError, match=r"must supply at least one"):
            concat([], "dim1")

        with pytest.raises(ValueError, match=r"Cannot specify both .*='different'"):
            concat(
                [data, data], dim="concat_dim", data_vars="different", compat="override"
            )

        with pytest.raises(ValueError, match=r"must supply at least one"):
            concat([], "dim1")

        with pytest.raises(ValueError, match=r"are not coordinates"):
            concat([data, data], "new_dim", coords=["not_found"])

        with pytest.raises(ValueError, match=r"global attributes not"):
            # call deepcopy seperately to get unique attrs
            data0 = deepcopy(split_data[0])
            data1 = deepcopy(split_data[1])
            data1.attrs["foo"] = "bar"
            concat([data0, data1], "dim1", compat="identical")
        assert_identical(data, concat([data0, data1], "dim1", compat="equals"))

        with pytest.raises(ValueError, match=r"compat.* invalid"):
            concat(split_data, "dim1", compat="foobar")

        with pytest.raises(ValueError, match=r"unexpected value for"):
            concat([data, data], "new_dim", coords="foobar")

        with pytest.raises(
            ValueError, match=r"coordinate in some datasets but not others"
        ):
            concat([Dataset({"x": 0}), Dataset({"x": [1]})], dim="z")

        with pytest.raises(
            ValueError, match=r"coordinate in some datasets but not others"
        ):
            concat([Dataset({"x": 0}), Dataset({}, {"x": 1})], dim="z")

    def test_concat_join_kwarg(self) -> None:
        ds1 = Dataset({"a": (("x", "y"), [[0]])}, coords={"x": [0], "y": [0]})
        ds2 = Dataset({"a": (("x", "y"), [[0]])}, coords={"x": [1], "y": [0.0001]})

        expected: dict[JoinOptions, Any] = {}
        expected["outer"] = Dataset(
            {"a": (("x", "y"), [[0, np.nan], [np.nan, 0]])},
            {"x": [0, 1], "y": [0, 0.0001]},
        )
        expected["inner"] = Dataset(
            {"a": (("x", "y"), [[], []])}, {"x": [0, 1], "y": []}
        )
        expected["left"] = Dataset(
            {"a": (("x", "y"), np.array([0, np.nan], ndmin=2).T)},
            coords={"x": [0, 1], "y": [0]},
        )
        expected["right"] = Dataset(
            {"a": (("x", "y"), np.array([np.nan, 0], ndmin=2).T)},
            coords={"x": [0, 1], "y": [0.0001]},
        )
        expected["override"] = Dataset(
            {"a": (("x", "y"), np.array([0, 0], ndmin=2).T)},
            coords={"x": [0, 1], "y": [0]},
        )

        with pytest.raises(ValueError, match=r"cannot align.*exact.*dimensions.*'y'"):
            actual = concat([ds1, ds2], join="exact", dim="x")

        for join in expected:
            actual = concat([ds1, ds2], join=join, dim="x")
            assert_equal(actual, expected[join])

        # regression test for #3681
        actual = concat(
            [ds1.drop_vars("x"), ds2.drop_vars("x")], join="override", dim="y"
        )
        expected2 = Dataset(
            {"a": (("x", "y"), np.array([0, 0], ndmin=2))}, coords={"y": [0, 0.0001]}
        )
        assert_identical(actual, expected2)

    @pytest.mark.parametrize(
        "combine_attrs, var1_attrs, var2_attrs, expected_attrs, expect_exception",
        [
            (
                "no_conflicts",
                {"a": 1, "b": 2},
                {"a": 1, "c": 3},
                {"a": 1, "b": 2, "c": 3},
                False,
            ),
            ("no_conflicts", {"a": 1, "b": 2}, {}, {"a": 1, "b": 2}, False),
            ("no_conflicts", {}, {"a": 1, "c": 3}, {"a": 1, "c": 3}, False),
            (
                "no_conflicts",
                {"a": 1, "b": 2},
                {"a": 4, "c": 3},
                {"a": 1, "b": 2, "c": 3},
                True,
            ),
            ("drop", {"a": 1, "b": 2}, {"a": 1, "c": 3}, {}, False),
            ("identical", {"a": 1, "b": 2}, {"a": 1, "b": 2}, {"a": 1, "b": 2}, False),
            ("identical", {"a": 1, "b": 2}, {"a": 1, "c": 3}, {"a": 1, "b": 2}, True),
            (
                "override",
                {"a": 1, "b": 2},
                {"a": 4, "b": 5, "c": 3},
                {"a": 1, "b": 2},
                False,
            ),
            (
                "drop_conflicts",
                {"a": 41, "b": 42, "c": 43},
                {"b": 2, "c": 43, "d": 44},
                {"a": 41, "c": 43, "d": 44},
                False,
            ),
            (
                lambda attrs, context: {"a": -1, "b": 0, "c": 1} if any(attrs) else {},
                {"a": 41, "b": 42, "c": 43},
                {"b": 2, "c": 43, "d": 44},
                {"a": -1, "b": 0, "c": 1},
                False,
            ),
        ],
    )
    def test_concat_combine_attrs_kwarg(
        self, combine_attrs, var1_attrs, var2_attrs, expected_attrs, expect_exception
    ):
        ds1 = Dataset({"a": ("x", [0])}, coords={"x": [0]}, attrs=var1_attrs)
        ds2 = Dataset({"a": ("x", [0])}, coords={"x": [1]}, attrs=var2_attrs)

        if expect_exception:
            with pytest.raises(ValueError, match=f"combine_attrs='{combine_attrs}'"):
                concat([ds1, ds2], dim="x", combine_attrs=combine_attrs)
        else:
            actual = concat([ds1, ds2], dim="x", combine_attrs=combine_attrs)
            expected = Dataset(
                {"a": ("x", [0, 0])}, {"x": [0, 1]}, attrs=expected_attrs
            )

            assert_identical(actual, expected)

    @pytest.mark.parametrize(
        "combine_attrs, attrs1, attrs2, expected_attrs, expect_exception",
        [
            (
                "no_conflicts",
                {"a": 1, "b": 2},
                {"a": 1, "c": 3},
                {"a": 1, "b": 2, "c": 3},
                False,
            ),
            ("no_conflicts", {"a": 1, "b": 2}, {}, {"a": 1, "b": 2}, False),
            ("no_conflicts", {}, {"a": 1, "c": 3}, {"a": 1, "c": 3}, False),
            (
                "no_conflicts",
                {"a": 1, "b": 2},
                {"a": 4, "c": 3},
                {"a": 1, "b": 2, "c": 3},
                True,
            ),
            ("drop", {"a": 1, "b": 2}, {"a": 1, "c": 3}, {}, False),
            ("identical", {"a": 1, "b": 2}, {"a": 1, "b": 2}, {"a": 1, "b": 2}, False),
            ("identical", {"a": 1, "b": 2}, {"a": 1, "c": 3}, {"a": 1, "b": 2}, True),
            (
                "override",
                {"a": 1, "b": 2},
                {"a": 4, "b": 5, "c": 3},
                {"a": 1, "b": 2},
                False,
            ),
            (
                "drop_conflicts",
                {"a": 41, "b": 42, "c": 43},
                {"b": 2, "c": 43, "d": 44},
                {"a": 41, "c": 43, "d": 44},
                False,
            ),
            (
                lambda attrs, context: {"a": -1, "b": 0, "c": 1} if any(attrs) else {},
                {"a": 41, "b": 42, "c": 43},
                {"b": 2, "c": 43, "d": 44},
                {"a": -1, "b": 0, "c": 1},
                False,
            ),
        ],
    )
    def test_concat_combine_attrs_kwarg_variables(
        self, combine_attrs, attrs1, attrs2, expected_attrs, expect_exception
    ):
        """check that combine_attrs is used on data variables and coords"""
        ds1 = Dataset({"a": ("x", [0], attrs1)}, coords={"x": ("x", [0], attrs1)})
        ds2 = Dataset({"a": ("x", [0], attrs2)}, coords={"x": ("x", [1], attrs2)})

        if expect_exception:
            with pytest.raises(ValueError, match=f"combine_attrs='{combine_attrs}'"):
                concat([ds1, ds2], dim="x", combine_attrs=combine_attrs)
        else:
            actual = concat([ds1, ds2], dim="x", combine_attrs=combine_attrs)
            expected = Dataset(
                {"a": ("x", [0, 0], expected_attrs)},
                {"x": ("x", [0, 1], expected_attrs)},
            )

            assert_identical(actual, expected)

    def test_concat_promote_shape(self) -> None:
        # mixed dims within variables
        objs = [Dataset({}, {"x": 0}), Dataset({"x": [1]})]
        actual = concat(objs, "x")
        expected = Dataset({"x": [0, 1]})
        assert_identical(actual, expected)

        objs = [Dataset({"x": [0]}), Dataset({}, {"x": 1})]
        actual = concat(objs, "x")
        assert_identical(actual, expected)

        # mixed dims between variables
        objs = [Dataset({"x": [2], "y": 3}), Dataset({"x": [4], "y": 5})]
        actual = concat(objs, "x")
        expected = Dataset({"x": [2, 4], "y": ("x", [3, 5])})
        assert_identical(actual, expected)

        # mixed dims in coord variable
        objs = [Dataset({"x": [0]}, {"y": -1}), Dataset({"x": [1]}, {"y": ("x", [-2])})]
        actual = concat(objs, "x")
        expected = Dataset({"x": [0, 1]}, {"y": ("x", [-1, -2])})
        assert_identical(actual, expected)

        # scalars with mixed lengths along concat dim -- values should repeat
        objs = [Dataset({"x": [0]}, {"y": -1}), Dataset({"x": [1, 2]}, {"y": -2})]
        actual = concat(objs, "x")
        expected = Dataset({"x": [0, 1, 2]}, {"y": ("x", [-1, -2, -2])})
        assert_identical(actual, expected)

        # broadcast 1d x 1d -> 2d
        objs = [
            Dataset({"z": ("x", [-1])}, {"x": [0], "y": [0]}),
            Dataset({"z": ("y", [1])}, {"x": [1], "y": [0]}),
        ]
        actual = concat(objs, "x")
        expected = Dataset({"z": (("x", "y"), [[-1], [1]])}, {"x": [0, 1], "y": [0]})
        assert_identical(actual, expected)

        # regression GH6384
        objs = [
            Dataset({}, {"x": pd.Interval(-1, 0, closed="right")}),
            Dataset({"x": [pd.Interval(0, 1, closed="right")]}),
        ]
        actual = concat(objs, "x")
        expected = Dataset(
            {
                "x": [
                    pd.Interval(-1, 0, closed="right"),
                    pd.Interval(0, 1, closed="right"),
                ]
            }
        )
        assert_identical(actual, expected)

        # regression GH6416 (coord dtype) and GH6434
        time_data1 = np.array(["2022-01-01", "2022-02-01"], dtype="datetime64[ns]")
        time_data2 = np.array("2022-03-01", dtype="datetime64[ns]")
        time_expected = np.array(
            ["2022-01-01", "2022-02-01", "2022-03-01"], dtype="datetime64[ns]"
        )
        objs = [Dataset({}, {"time": time_data1}), Dataset({}, {"time": time_data2})]
        actual = concat(objs, "time")
        expected = Dataset({}, {"time": time_expected})
        assert_identical(actual, expected)
        assert isinstance(actual.indexes["time"], pd.DatetimeIndex)

    def test_concat_do_not_promote(self) -> None:
        # GH438
        objs = [
            Dataset({"y": ("t", [1])}, {"x": 1, "t": [0]}),
            Dataset({"y": ("t", [2])}, {"x": 1, "t": [0]}),
        ]
        expected = Dataset({"y": ("t", [1, 2])}, {"x": 1, "t": [0, 0]})
        actual = concat(objs, "t")
        assert_identical(expected, actual)

        objs = [
            Dataset({"y": ("t", [1])}, {"x": 1, "t": [0]}),
            Dataset({"y": ("t", [2])}, {"x": 2, "t": [0]}),
        ]
        with pytest.raises(ValueError):
            concat(objs, "t", coords="minimal")

    def test_concat_dim_is_variable(self) -> None:
        objs = [Dataset({"x": 0}), Dataset({"x": 1})]
        coord = Variable("y", [3, 4], attrs={"foo": "bar"})
        expected = Dataset({"x": ("y", [0, 1]), "y": coord})
        actual = concat(objs, coord)
        assert_identical(actual, expected)

    def test_concat_dim_is_dataarray(self) -> None:
        objs = [Dataset({"x": 0}), Dataset({"x": 1})]
        coord = DataArray([3, 4], dims="y", attrs={"foo": "bar"})
        expected = Dataset({"x": ("y", [0, 1]), "y": coord})
        actual = concat(objs, coord)
        assert_identical(actual, expected)

    def test_concat_multiindex(self) -> None:
        x = pd.MultiIndex.from_product([[1, 2, 3], ["a", "b"]])
        expected = Dataset(coords={"x": x})
        actual = concat(
            [expected.isel(x=slice(2)), expected.isel(x=slice(2, None))], "x"
        )
        assert expected.equals(actual)
        assert isinstance(actual.x.to_index(), pd.MultiIndex)

    def test_concat_along_new_dim_multiindex(self) -> None:
        # see https://github.com/pydata/xarray/issues/6881
        level_names = ["x_level_0", "x_level_1"]
        x = pd.MultiIndex.from_product([[1, 2, 3], ["a", "b"]], names=level_names)
        ds = Dataset(coords={"x": x})
        concatenated = concat([ds], "new")
        actual = list(concatenated.xindexes.get_all_coords("x"))
        expected = ["x"] + level_names
        assert actual == expected

    @pytest.mark.parametrize("fill_value", [dtypes.NA, 2, 2.0, {"a": 2, "b": 1}])
    def test_concat_fill_value(self, fill_value) -> None:
        datasets = [
            Dataset({"a": ("x", [2, 3]), "b": ("x", [-2, 1]), "x": [1, 2]}),
            Dataset({"a": ("x", [1, 2]), "b": ("x", [3, -1]), "x": [0, 1]}),
        ]
        if fill_value == dtypes.NA:
            # if we supply the default, we expect the missing value for a
            # float array
            fill_value_a = fill_value_b = np.nan
        elif isinstance(fill_value, dict):
            fill_value_a = fill_value["a"]
            fill_value_b = fill_value["b"]
        else:
            fill_value_a = fill_value_b = fill_value
        expected = Dataset(
            {
                "a": (("t", "x"), [[fill_value_a, 2, 3], [1, 2, fill_value_a]]),
                "b": (("t", "x"), [[fill_value_b, -2, 1], [3, -1, fill_value_b]]),
            },
            {"x": [0, 1, 2]},
        )
        actual = concat(datasets, dim="t", fill_value=fill_value)
        assert_identical(actual, expected)

    @pytest.mark.parametrize("dtype", [str, bytes])
    @pytest.mark.parametrize("dim", ["x1", "x2"])
    def test_concat_str_dtype(self, dtype, dim) -> None:

        data = np.arange(4).reshape([2, 2])

        da1 = Dataset(
            {
                "data": (["x1", "x2"], data),
                "x1": [0, 1],
                "x2": np.array(["a", "b"], dtype=dtype),
            }
        )
        da2 = Dataset(
            {
                "data": (["x1", "x2"], data),
                "x1": np.array([1, 2]),
                "x2": np.array(["c", "d"], dtype=dtype),
            }
        )
        actual = concat([da1, da2], dim=dim)

        assert np.issubdtype(actual.x2.dtype, dtype)


class TestConcatDataArray:
    def test_concat(self) -> None:
        ds = Dataset(
            {
                "foo": (["x", "y"], np.random.random((2, 3))),
                "bar": (["x", "y"], np.random.random((2, 3))),
            },
            {"x": [0, 1]},
        )
        foo = ds["foo"]
        bar = ds["bar"]

        # from dataset array:
        expected = DataArray(
            np.array([foo.values, bar.values]),
            dims=["w", "x", "y"],
            coords={"x": [0, 1]},
        )
        actual = concat([foo, bar], "w")
        assert_equal(expected, actual)
        # from iteration:
        grouped = [g for _, g in foo.groupby("x")]
        stacked = concat(grouped, ds["x"])
        assert_identical(foo, stacked)
        # with an index as the 'dim' argument
        stacked = concat(grouped, pd.Index(ds["x"], name="x"))
        assert_identical(foo, stacked)

        actual2 = concat([foo[0], foo[1]], pd.Index([0, 1])).reset_coords(drop=True)
        expected = foo[:2].rename({"x": "concat_dim"})
        assert_identical(expected, actual2)

        actual3 = concat([foo[0], foo[1]], [0, 1]).reset_coords(drop=True)
        expected = foo[:2].rename({"x": "concat_dim"})
        assert_identical(expected, actual3)

        with pytest.raises(ValueError, match=r"not identical"):
            concat([foo, bar], dim="w", compat="identical")

        with pytest.raises(ValueError, match=r"not a valid argument"):
            concat([foo, bar], dim="w", data_vars="minimal")

    def test_concat_encoding(self) -> None:
        # Regression test for GH1297
        ds = Dataset(
            {
                "foo": (["x", "y"], np.random.random((2, 3))),
                "bar": (["x", "y"], np.random.random((2, 3))),
            },
            {"x": [0, 1]},
        )
        foo = ds["foo"]
        foo.encoding = {"complevel": 5}
        ds.encoding = {"unlimited_dims": "x"}
        assert concat([foo, foo], dim="x").encoding == foo.encoding
        assert concat([ds, ds], dim="x").encoding == ds.encoding

    @requires_dask
    def test_concat_lazy(self) -> None:
        import dask.array as da

        arrays = [
            DataArray(
                da.from_array(InaccessibleArray(np.zeros((3, 3))), 3), dims=["x", "y"]
            )
            for _ in range(2)
        ]
        # should not raise
        combined = concat(arrays, dim="z")
        assert combined.shape == (2, 3, 3)
        assert combined.dims == ("z", "x", "y")

    @pytest.mark.parametrize("fill_value", [dtypes.NA, 2, 2.0])
    def test_concat_fill_value(self, fill_value) -> None:
        foo = DataArray([1, 2], coords=[("x", [1, 2])])
        bar = DataArray([1, 2], coords=[("x", [1, 3])])
        if fill_value == dtypes.NA:
            # if we supply the default, we expect the missing value for a
            # float array
            fill_value = np.nan
        expected = DataArray(
            [[1, 2, fill_value], [1, fill_value, 2]],
            dims=["y", "x"],
            coords={"x": [1, 2, 3]},
        )
        actual = concat((foo, bar), dim="y", fill_value=fill_value)
        assert_identical(actual, expected)

    def test_concat_join_kwarg(self) -> None:
        ds1 = Dataset(
            {"a": (("x", "y"), [[0]])}, coords={"x": [0], "y": [0]}
        ).to_array()
        ds2 = Dataset(
            {"a": (("x", "y"), [[0]])}, coords={"x": [1], "y": [0.0001]}
        ).to_array()

        expected: dict[JoinOptions, Any] = {}
        expected["outer"] = Dataset(
            {"a": (("x", "y"), [[0, np.nan], [np.nan, 0]])},
            {"x": [0, 1], "y": [0, 0.0001]},
        )
        expected["inner"] = Dataset(
            {"a": (("x", "y"), [[], []])}, {"x": [0, 1], "y": []}
        )
        expected["left"] = Dataset(
            {"a": (("x", "y"), np.array([0, np.nan], ndmin=2).T)},
            coords={"x": [0, 1], "y": [0]},
        )
        expected["right"] = Dataset(
            {"a": (("x", "y"), np.array([np.nan, 0], ndmin=2).T)},
            coords={"x": [0, 1], "y": [0.0001]},
        )
        expected["override"] = Dataset(
            {"a": (("x", "y"), np.array([0, 0], ndmin=2).T)},
            coords={"x": [0, 1], "y": [0]},
        )

        with pytest.raises(ValueError, match=r"cannot align.*exact.*dimensions.*'y'"):
            actual = concat([ds1, ds2], join="exact", dim="x")

        for join in expected:
            actual = concat([ds1, ds2], join=join, dim="x")
            assert_equal(actual, expected[join].to_array())

    def test_concat_combine_attrs_kwarg(self) -> None:
        da1 = DataArray([0], coords=[("x", [0])], attrs={"b": 42})
        da2 = DataArray([0], coords=[("x", [1])], attrs={"b": 42, "c": 43})

        expected: dict[CombineAttrsOptions, Any] = {}
        expected["drop"] = DataArray([0, 0], coords=[("x", [0, 1])])
        expected["no_conflicts"] = DataArray(
            [0, 0], coords=[("x", [0, 1])], attrs={"b": 42, "c": 43}
        )
        expected["override"] = DataArray(
            [0, 0], coords=[("x", [0, 1])], attrs={"b": 42}
        )

        with pytest.raises(ValueError, match=r"combine_attrs='identical'"):
            actual = concat([da1, da2], dim="x", combine_attrs="identical")
        with pytest.raises(ValueError, match=r"combine_attrs='no_conflicts'"):
            da3 = da2.copy(deep=True)
            da3.attrs["b"] = 44
            actual = concat([da1, da3], dim="x", combine_attrs="no_conflicts")

        for combine_attrs in expected:
            actual = concat([da1, da2], dim="x", combine_attrs=combine_attrs)
            assert_identical(actual, expected[combine_attrs])

    @pytest.mark.parametrize("dtype", [str, bytes])
    @pytest.mark.parametrize("dim", ["x1", "x2"])
    def test_concat_str_dtype(self, dtype, dim) -> None:

        data = np.arange(4).reshape([2, 2])

        da1 = DataArray(
            data=data,
            dims=["x1", "x2"],
            coords={"x1": [0, 1], "x2": np.array(["a", "b"], dtype=dtype)},
        )
        da2 = DataArray(
            data=data,
            dims=["x1", "x2"],
            coords={"x1": np.array([1, 2]), "x2": np.array(["c", "d"], dtype=dtype)},
        )
        actual = concat([da1, da2], dim=dim)

        assert np.issubdtype(actual.x2.dtype, dtype)

    def test_concat_coord_name(self) -> None:

        da = DataArray([0], dims="a")
        da_concat = concat([da, da], dim=DataArray([0, 1], dims="b"))
        assert list(da_concat.coords) == ["b"]

        da_concat_std = concat([da, da], dim=DataArray([0, 1]))
        assert list(da_concat_std.coords) == ["dim_0"]


@pytest.mark.parametrize("attr1", ({"a": {"meta": [10, 20, 30]}}, {"a": [1, 2, 3]}, {}))
@pytest.mark.parametrize("attr2", ({"a": [1, 2, 3]}, {}))
def test_concat_attrs_first_variable(attr1, attr2) -> None:

    arrs = [
        DataArray([[1], [2]], dims=["x", "y"], attrs=attr1),
        DataArray([[3], [4]], dims=["x", "y"], attrs=attr2),
    ]

    concat_attrs = concat(arrs, "y").attrs
    assert concat_attrs == attr1


def test_concat_merge_single_non_dim_coord():
    # TODO: annotating this func fails
    da1 = DataArray([1, 2, 3], dims="x", coords={"x": [1, 2, 3], "y": 1})
    da2 = DataArray([4, 5, 6], dims="x", coords={"x": [4, 5, 6]})

    expected = DataArray(range(1, 7), dims="x", coords={"x": range(1, 7), "y": 1})

    for coords in ["different", "minimal"]:
        actual = concat([da1, da2], "x", coords=coords)
        assert_identical(actual, expected)

    with pytest.raises(ValueError, match=r"'y' is not present in all datasets."):
        concat([da1, da2], dim="x", coords="all")

    da1 = DataArray([1, 2, 3], dims="x", coords={"x": [1, 2, 3], "y": 1})
    da2 = DataArray([4, 5, 6], dims="x", coords={"x": [4, 5, 6]})
    da3 = DataArray([7, 8, 9], dims="x", coords={"x": [7, 8, 9], "y": 1})
    for coords in ["different", "all"]:
        with pytest.raises(ValueError, match=r"'y' not present in all datasets"):
            concat([da1, da2, da3], dim="x")


def test_concat_preserve_coordinate_order() -> None:
    x = np.arange(0, 5)
    y = np.arange(0, 10)
    time = np.arange(0, 4)
    data = np.zeros((4, 10, 5), dtype=bool)

    ds1 = Dataset(
        {"data": (["time", "y", "x"], data[0:2])},
        coords={"time": time[0:2], "y": y, "x": x},
    )
    ds2 = Dataset(
        {"data": (["time", "y", "x"], data[2:4])},
        coords={"time": time[2:4], "y": y, "x": x},
    )

    expected = Dataset(
        {"data": (["time", "y", "x"], data)},
        coords={"time": time, "y": y, "x": x},
    )

    actual = concat([ds1, ds2], dim="time")

    # check dimension order
    for act, exp in zip(actual.dims, expected.dims):
        assert act == exp
        assert actual.dims[act] == expected.dims[exp]

    # check coordinate order
    for act, exp in zip(actual.coords, expected.coords):
        assert act == exp
        assert_identical(actual.coords[act], expected.coords[exp])


def test_concat_typing_check() -> None:
    ds = Dataset({"foo": 1}, {"bar": 2})
    da = Dataset({"foo": 3}, {"bar": 4}).to_array(dim="foo")

    # concatenate a list of non-homogeneous types must raise TypeError
    with pytest.raises(
        TypeError,
        match="The elements in the input list need to be either all 'Dataset's or all 'DataArray's",
    ):
        concat([ds, da], dim="foo")  # type: ignore
    with pytest.raises(
        TypeError,
        match="The elements in the input list need to be either all 'Dataset's or all 'DataArray's",
    ):
        concat([da, ds], dim="foo")  # type: ignore


def test_concat_not_all_indexes() -> None:
    ds1 = Dataset(coords={"x": ("x", [1, 2])})
    # ds2.x has no default index
    ds2 = Dataset(coords={"x": ("y", [3, 4])})

    with pytest.raises(
        ValueError, match=r"'x' must have either an index or no index in all datasets.*"
    ):
        concat([ds1, ds2], dim="x")


def test_concat_index_not_same_dim() -> None:
    ds1 = Dataset(coords={"x": ("x", [1, 2])})
    ds2 = Dataset(coords={"x": ("y", [3, 4])})
    # TODO: use public API for setting a non-default index, when available
    ds2._indexes["x"] = PandasIndex([3, 4], "y")

    with pytest.raises(
        ValueError,
        match=r"Cannot concatenate along dimension 'x' indexes with dimensions.*",
    ):
        concat([ds1, ds2], dim="x")
