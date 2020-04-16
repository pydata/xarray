import numpy as np
import pandas as pd
import pytest

import xarray as xr
from xarray.core.groupby import _consolidate_slices

from . import assert_allclose, assert_equal, assert_identical, raises_regex


@pytest.fixture
def dataset():
    ds = xr.Dataset(
        {"foo": (("x", "y", "z"), np.random.randn(3, 4, 2))},
        {"x": ["a", "b", "c"], "y": [1, 2, 3, 4], "z": [1, 2]},
    )
    ds["boo"] = (("z", "y"), [["f", "g", "h", "j"]] * 2)

    return ds


@pytest.fixture
def array(dataset):
    return dataset["foo"]


def test_consolidate_slices():

    assert _consolidate_slices([slice(3), slice(3, 5)]) == [slice(5)]
    assert _consolidate_slices([slice(2, 3), slice(3, 6)]) == [slice(2, 6)]
    assert _consolidate_slices([slice(2, 3, 1), slice(3, 6, 1)]) == [slice(2, 6, 1)]

    slices = [slice(2, 3), slice(5, 6)]
    assert _consolidate_slices(slices) == slices

    with pytest.raises(ValueError):
        _consolidate_slices([slice(3), 4])


def test_groupby_dims_property(dataset):
    assert dataset.groupby("x").dims == dataset.isel(x=1).dims
    assert dataset.groupby("y").dims == dataset.isel(y=1).dims

    stacked = dataset.stack({"xy": ("x", "y")})
    assert stacked.groupby("xy").dims == stacked.isel(xy=0).dims


def test_multi_index_groupby_map(dataset):
    # regression test for GH873
    ds = dataset.isel(z=1, drop=True)[["foo"]]
    expected = 2 * ds
    actual = (
        ds.stack(space=["x", "y"])
        .groupby("space")
        .map(lambda x: 2 * x)
        .unstack("space")
    )
    assert_equal(expected, actual)


def test_multi_index_groupby_sum():
    # regression test for GH873
    ds = xr.Dataset(
        {"foo": (("x", "y", "z"), np.ones((3, 4, 2)))},
        {"x": ["a", "b", "c"], "y": [1, 2, 3, 4]},
    )
    expected = ds.sum("z")
    actual = ds.stack(space=["x", "y"]).groupby("space").sum("z").unstack("space")
    assert_equal(expected, actual)


def test_groupby_da_datetime():
    # test groupby with a DataArray of dtype datetime for GH1132
    # create test data
    times = pd.date_range("2000-01-01", periods=4)
    foo = xr.DataArray([1, 2, 3, 4], coords=dict(time=times), dims="time")
    # create test index
    dd = times.to_pydatetime()
    reference_dates = [dd[0], dd[2]]
    labels = reference_dates[0:1] * 2 + reference_dates[1:2] * 2
    ind = xr.DataArray(
        labels, coords=dict(time=times), dims="time", name="reference_date"
    )
    g = foo.groupby(ind)
    actual = g.sum(dim="time")
    expected = xr.DataArray(
        [3, 7], coords=dict(reference_date=reference_dates), dims="reference_date"
    )
    assert_equal(expected, actual)


def test_groupby_duplicate_coordinate_labels():
    # fix for http://stackoverflow.com/questions/38065129
    array = xr.DataArray([1, 2, 3], [("x", [1, 1, 2])])
    expected = xr.DataArray([3, 3], [("x", [1, 2])])
    actual = array.groupby("x").sum()
    assert_equal(expected, actual)


def test_groupby_input_mutation():
    # regression test for GH2153
    array = xr.DataArray([1, 2, 3], [("x", [2, 2, 1])])
    array_copy = array.copy()
    expected = xr.DataArray([3, 3], [("x", [1, 2])])
    actual = array.groupby("x").sum()
    assert_identical(expected, actual)
    assert_identical(array, array_copy)  # should not modify inputs


@pytest.mark.parametrize(
    "obj",
    [
        xr.DataArray([1, 2, 3, 4, 5, 6], [("x", [1, 1, 1, 2, 2, 2])]),
        xr.Dataset({"foo": ("x", [1, 2, 3, 4, 5, 6])}, {"x": [1, 1, 1, 2, 2, 2]}),
    ],
)
def test_groupby_map_shrink_groups(obj):
    expected = obj.isel(x=[0, 1, 3, 4])
    actual = obj.groupby("x").map(lambda f: f.isel(x=[0, 1]))
    assert_identical(expected, actual)


@pytest.mark.parametrize(
    "obj",
    [
        xr.DataArray([1, 2, 3], [("x", [1, 2, 2])]),
        xr.Dataset({"foo": ("x", [1, 2, 3])}, {"x": [1, 2, 2]}),
    ],
)
def test_groupby_map_change_group_size(obj):
    def func(group):
        if group.sizes["x"] == 1:
            result = group.isel(x=[0, 0])
        else:
            result = group.isel(x=[0])
        return result

    expected = obj.isel(x=[0, 0, 1])
    actual = obj.groupby("x").map(func)
    assert_identical(expected, actual)


def test_da_groupby_map_func_args():
    def func(arg1, arg2, arg3=0):
        return arg1 + arg2 + arg3

    array = xr.DataArray([1, 1, 1], [("x", [1, 2, 3])])
    expected = xr.DataArray([3, 3, 3], [("x", [1, 2, 3])])
    actual = array.groupby("x").map(func, args=(1,), arg3=1)
    assert_identical(expected, actual)


def test_ds_groupby_map_func_args():
    def func(arg1, arg2, arg3=0):
        return arg1 + arg2 + arg3

    dataset = xr.Dataset({"foo": ("x", [1, 1, 1])}, {"x": [1, 2, 3]})
    expected = xr.Dataset({"foo": ("x", [3, 3, 3])}, {"x": [1, 2, 3]})
    actual = dataset.groupby("x").map(func, args=(1,), arg3=1)
    assert_identical(expected, actual)


def test_da_groupby_empty():

    empty_array = xr.DataArray([], dims="dim")

    with pytest.raises(ValueError):
        empty_array.groupby("dim")


def test_da_groupby_quantile():

    array = xr.DataArray(
        data=[1, 2, 3, 4, 5, 6], coords={"x": [1, 1, 1, 2, 2, 2]}, dims="x"
    )

    # Scalar quantile
    expected = xr.DataArray(
        data=[2, 5], coords={"x": [1, 2], "quantile": 0.5}, dims="x"
    )
    actual = array.groupby("x").quantile(0.5)
    assert_identical(expected, actual)

    # Vector quantile
    expected = xr.DataArray(
        data=[[1, 3], [4, 6]],
        coords={"x": [1, 2], "quantile": [0, 1]},
        dims=("x", "quantile"),
    )
    actual = array.groupby("x").quantile([0, 1])
    assert_identical(expected, actual)

    # Multiple dimensions
    array = xr.DataArray(
        data=[[1, 11, 26], [2, 12, 22], [3, 13, 23], [4, 16, 24], [5, 15, 25]],
        coords={"x": [1, 1, 1, 2, 2], "y": [0, 0, 1]},
        dims=("x", "y"),
    )

    actual_x = array.groupby("x").quantile(0, dim=...)
    expected_x = xr.DataArray(
        data=[1, 4], coords={"x": [1, 2], "quantile": 0}, dims="x"
    )
    assert_identical(expected_x, actual_x)

    actual_y = array.groupby("y").quantile(0, dim=...)
    expected_y = xr.DataArray(
        data=[1, 22], coords={"y": [0, 1], "quantile": 0}, dims="y"
    )
    assert_identical(expected_y, actual_y)

    actual_xx = array.groupby("x").quantile(0)
    expected_xx = xr.DataArray(
        data=[[1, 11, 22], [4, 15, 24]],
        coords={"x": [1, 2], "y": [0, 0, 1], "quantile": 0},
        dims=("x", "y"),
    )
    assert_identical(expected_xx, actual_xx)

    actual_yy = array.groupby("y").quantile(0)
    expected_yy = xr.DataArray(
        data=[[1, 26], [2, 22], [3, 23], [4, 24], [5, 25]],
        coords={"x": [1, 1, 1, 2, 2], "y": [0, 1], "quantile": 0},
        dims=("x", "y"),
    )
    assert_identical(expected_yy, actual_yy)

    times = pd.date_range("2000-01-01", periods=365)
    x = [0, 1]
    foo = xr.DataArray(
        np.reshape(np.arange(365 * 2), (365, 2)),
        coords={"time": times, "x": x},
        dims=("time", "x"),
    )
    g = foo.groupby(foo.time.dt.month)

    actual = g.quantile(0, dim=...)
    expected = xr.DataArray(
        data=[
            0.0,
            62.0,
            120.0,
            182.0,
            242.0,
            304.0,
            364.0,
            426.0,
            488.0,
            548.0,
            610.0,
            670.0,
        ],
        coords={"month": np.arange(1, 13), "quantile": 0},
        dims="month",
    )
    assert_identical(expected, actual)

    actual = g.quantile(0, dim="time")[:2]
    expected = xr.DataArray(
        data=[[0.0, 1], [62.0, 63]],
        coords={"month": [1, 2], "x": [0, 1], "quantile": 0},
        dims=("month", "x"),
    )
    assert_identical(expected, actual)


def test_ds_groupby_quantile():
    ds = xr.Dataset(
        data_vars={"a": ("x", [1, 2, 3, 4, 5, 6])}, coords={"x": [1, 1, 1, 2, 2, 2]}
    )

    # Scalar quantile
    expected = xr.Dataset(
        data_vars={"a": ("x", [2, 5])}, coords={"quantile": 0.5, "x": [1, 2]}
    )
    actual = ds.groupby("x").quantile(0.5)
    assert_identical(expected, actual)

    # Vector quantile
    expected = xr.Dataset(
        data_vars={"a": (("x", "quantile"), [[1, 3], [4, 6]])},
        coords={"x": [1, 2], "quantile": [0, 1]},
    )
    actual = ds.groupby("x").quantile([0, 1])
    assert_identical(expected, actual)

    # Multiple dimensions
    ds = xr.Dataset(
        data_vars={
            "a": (
                ("x", "y"),
                [[1, 11, 26], [2, 12, 22], [3, 13, 23], [4, 16, 24], [5, 15, 25]],
            )
        },
        coords={"x": [1, 1, 1, 2, 2], "y": [0, 0, 1]},
    )

    actual_x = ds.groupby("x").quantile(0, dim=...)
    expected_x = xr.Dataset({"a": ("x", [1, 4])}, coords={"x": [1, 2], "quantile": 0})
    assert_identical(expected_x, actual_x)

    actual_y = ds.groupby("y").quantile(0, dim=...)
    expected_y = xr.Dataset({"a": ("y", [1, 22])}, coords={"y": [0, 1], "quantile": 0})
    assert_identical(expected_y, actual_y)

    actual_xx = ds.groupby("x").quantile(0)
    expected_xx = xr.Dataset(
        {"a": (("x", "y"), [[1, 11, 22], [4, 15, 24]])},
        coords={"x": [1, 2], "y": [0, 0, 1], "quantile": 0},
    )
    assert_identical(expected_xx, actual_xx)

    actual_yy = ds.groupby("y").quantile(0)
    expected_yy = xr.Dataset(
        {"a": (("x", "y"), [[1, 26], [2, 22], [3, 23], [4, 24], [5, 25]])},
        coords={"x": [1, 1, 1, 2, 2], "y": [0, 1], "quantile": 0},
    ).transpose()
    assert_identical(expected_yy, actual_yy)

    times = pd.date_range("2000-01-01", periods=365)
    x = [0, 1]
    foo = xr.Dataset(
        {"a": (("time", "x"), np.reshape(np.arange(365 * 2), (365, 2)))},
        coords=dict(time=times, x=x),
    )
    g = foo.groupby(foo.time.dt.month)

    actual = g.quantile(0, dim=...)
    expected = xr.Dataset(
        {
            "a": (
                "month",
                [
                    0.0,
                    62.0,
                    120.0,
                    182.0,
                    242.0,
                    304.0,
                    364.0,
                    426.0,
                    488.0,
                    548.0,
                    610.0,
                    670.0,
                ],
            )
        },
        coords={"month": np.arange(1, 13), "quantile": 0},
    )
    assert_identical(expected, actual)

    actual = g.quantile(0, dim="time").isel(month=slice(None, 2))
    expected = xr.Dataset(
        data_vars={"a": (("month", "x"), [[0.0, 1], [62.0, 63]])},
        coords={"month": [1, 2], "x": [0, 1], "quantile": 0},
    )
    assert_identical(expected, actual)


def test_da_groupby_assign_coords():
    actual = xr.DataArray(
        [[3, 4, 5], [6, 7, 8]], dims=["y", "x"], coords={"y": range(2), "x": range(3)}
    )
    actual1 = actual.groupby("x").assign_coords({"y": [-1, -2]})
    actual2 = actual.groupby("x").assign_coords(y=[-1, -2])
    expected = xr.DataArray(
        [[3, 4, 5], [6, 7, 8]], dims=["y", "x"], coords={"y": [-1, -2], "x": range(3)}
    )
    assert_identical(expected, actual1)
    assert_identical(expected, actual2)


repr_da = xr.DataArray(
    np.random.randn(10, 20, 6, 24),
    dims=["x", "y", "z", "t"],
    coords={
        "z": ["a", "b", "c", "a", "b", "c"],
        "x": [1, 1, 1, 2, 2, 3, 4, 5, 3, 4],
        "t": pd.date_range("2001-01-01", freq="M", periods=24),
        "month": ("t", list(range(1, 13)) * 2),
    },
)


@pytest.mark.parametrize("dim", ["x", "y", "z", "month"])
@pytest.mark.parametrize("obj", [repr_da, repr_da.to_dataset(name="a")])
def test_groupby_repr(obj, dim):
    actual = repr(obj.groupby(dim))
    expected = "%sGroupBy" % obj.__class__.__name__
    expected += ", grouped over %r " % dim
    expected += "\n%r groups with labels " % (len(np.unique(obj[dim])))
    if dim == "x":
        expected += "1, 2, 3, 4, 5."
    elif dim == "y":
        expected += "0, 1, 2, 3, 4, 5, ..., 15, 16, 17, 18, 19."
    elif dim == "z":
        expected += "'a', 'b', 'c'."
    elif dim == "month":
        expected += "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12."
    assert actual == expected


@pytest.mark.parametrize("obj", [repr_da, repr_da.to_dataset(name="a")])
def test_groupby_repr_datetime(obj):
    actual = repr(obj.groupby("t.month"))
    expected = "%sGroupBy" % obj.__class__.__name__
    expected += ", grouped over 'month' "
    expected += "\n%r groups with labels " % (len(np.unique(obj.t.dt.month)))
    expected += "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12."
    assert actual == expected


def test_groupby_drops_nans():
    # GH2383
    # nan in 2D data variable (requires stacking)
    ds = xr.Dataset(
        {
            "variable": (("lat", "lon", "time"), np.arange(60.0).reshape((4, 3, 5))),
            "id": (("lat", "lon"), np.arange(12.0).reshape((4, 3))),
        },
        coords={"lat": np.arange(4), "lon": np.arange(3), "time": np.arange(5)},
    )

    ds["id"].values[0, 0] = np.nan
    ds["id"].values[3, 0] = np.nan
    ds["id"].values[-1, -1] = np.nan

    grouped = ds.groupby(ds.id)

    # non reduction operation
    expected = ds.copy()
    expected.variable.values[0, 0, :] = np.nan
    expected.variable.values[-1, -1, :] = np.nan
    expected.variable.values[3, 0, :] = np.nan
    actual = grouped.map(lambda x: x).transpose(*ds.variable.dims)
    assert_identical(actual, expected)

    # reduction along grouped dimension
    actual = grouped.mean()
    stacked = ds.stack({"xy": ["lat", "lon"]})
    expected = (
        stacked.variable.where(stacked.id.notnull()).rename({"xy": "id"}).to_dataset()
    )
    expected["id"] = stacked.id.values
    assert_identical(actual, expected.dropna("id").transpose(*actual.dims))

    # reduction operation along a different dimension
    actual = grouped.mean("time")
    with pytest.warns(RuntimeWarning):  # mean of empty slice
        expected = ds.mean("time").where(ds.id.notnull())
    assert_identical(actual, expected)

    # NaN in non-dimensional coordinate
    array = xr.DataArray([1, 2, 3], [("x", [1, 2, 3])])
    array["x1"] = ("x", [1, 1, np.nan])
    expected = xr.DataArray(3, [("x1", [1])])
    actual = array.groupby("x1").sum()
    assert_equal(expected, actual)

    # NaT in non-dimensional coordinate
    array["t"] = (
        "x",
        [
            np.datetime64("2001-01-01"),
            np.datetime64("2001-01-01"),
            np.datetime64("NaT"),
        ],
    )
    expected = xr.DataArray(3, [("t", [np.datetime64("2001-01-01")])])
    actual = array.groupby("t").sum()
    assert_equal(expected, actual)

    # test for repeated coordinate labels
    array = xr.DataArray([0, 1, 2, 4, 3, 4], [("x", [np.nan, 1, 1, np.nan, 2, np.nan])])
    expected = xr.DataArray([3, 3], [("x", [1, 2])])
    actual = array.groupby("x").sum()
    assert_equal(expected, actual)


def test_groupby_grouping_errors():
    dataset = xr.Dataset({"foo": ("x", [1, 1, 1])}, {"x": [1, 2, 3]})
    with raises_regex(ValueError, "None of the data falls within bins with edges"):
        dataset.groupby_bins("x", bins=[0.1, 0.2, 0.3])

    with raises_regex(ValueError, "None of the data falls within bins with edges"):
        dataset.to_array().groupby_bins("x", bins=[0.1, 0.2, 0.3])

    with raises_regex(ValueError, "All bin edges are NaN."):
        dataset.groupby_bins("x", bins=[np.nan, np.nan, np.nan])

    with raises_regex(ValueError, "All bin edges are NaN."):
        dataset.to_array().groupby_bins("x", bins=[np.nan, np.nan, np.nan])

    with raises_regex(ValueError, "Failed to group data."):
        dataset.groupby(dataset.foo * np.nan)

    with raises_regex(ValueError, "Failed to group data."):
        dataset.to_array().groupby(dataset.foo * np.nan)


def test_groupby_reduce_dimension_error(array):
    grouped = array.groupby("y")
    with raises_regex(ValueError, "cannot reduce over dimensions"):
        grouped.mean()

    with raises_regex(ValueError, "cannot reduce over dimensions"):
        grouped.mean("huh")

    with raises_regex(ValueError, "cannot reduce over dimensions"):
        grouped.mean(("x", "y", "asd"))

    grouped = array.groupby("y", squeeze=False)
    assert_identical(array, grouped.mean())

    assert_identical(array.mean("x"), grouped.reduce(np.mean, "x"))
    assert_allclose(array.mean(["x", "z"]), grouped.reduce(np.mean, ["x", "z"]))


def test_groupby_multiple_string_args(array):
    with pytest.raises(TypeError):
        array.groupby("x", "y")


def test_groupby_bins_timeseries():
    ds = xr.Dataset()
    ds["time"] = xr.DataArray(
        pd.date_range("2010-08-01", "2010-08-15", freq="15min"), dims="time"
    )
    ds["val"] = xr.DataArray(np.ones(*ds["time"].shape), dims="time")
    time_bins = pd.date_range(start="2010-08-01", end="2010-08-15", freq="24H")
    actual = ds.groupby_bins("time", time_bins).sum()
    expected = xr.DataArray(
        96 * np.ones((14,)),
        dims=["time_bins"],
        coords={"time_bins": pd.cut(time_bins, time_bins).categories},
    ).to_dataset(name="val")
    assert_identical(actual, expected)


# TODO: move other groupby tests from test_dataset and test_dataarray over here
