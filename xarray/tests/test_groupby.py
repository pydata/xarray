import numpy as np
import pandas as pd
import pytest
from pandas.tseries.frequencies import to_offset

import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core.groupby import _consolidate_slices

from . import (
    assert_allclose,
    assert_array_equal,
    assert_equal,
    assert_identical,
    create_test_data,
    requires_dask,
    requires_scipy,
)


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


def test_consolidate_slices() -> None:

    assert _consolidate_slices([slice(3), slice(3, 5)]) == [slice(5)]
    assert _consolidate_slices([slice(2, 3), slice(3, 6)]) == [slice(2, 6)]
    assert _consolidate_slices([slice(2, 3, 1), slice(3, 6, 1)]) == [slice(2, 6, 1)]

    slices = [slice(2, 3), slice(5, 6)]
    assert _consolidate_slices(slices) == slices

    with pytest.raises(ValueError):
        _consolidate_slices([slice(3), 4])


def test_groupby_dims_property(dataset) -> None:
    assert dataset.groupby("x").dims == dataset.isel(x=1).dims
    assert dataset.groupby("y").dims == dataset.isel(y=1).dims

    stacked = dataset.stack({"xy": ("x", "y")})
    assert stacked.groupby("xy").dims == stacked.isel(xy=0).dims


def test_multi_index_groupby_map(dataset) -> None:
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


def test_multi_index_groupby_sum() -> None:
    # regression test for GH873
    ds = xr.Dataset(
        {"foo": (("x", "y", "z"), np.ones((3, 4, 2)))},
        {"x": ["a", "b", "c"], "y": [1, 2, 3, 4]},
    )
    expected = ds.sum("z")
    actual = ds.stack(space=["x", "y"]).groupby("space").sum("z").unstack("space")
    assert_equal(expected, actual)


def test_groupby_da_datetime() -> None:
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


def test_groupby_duplicate_coordinate_labels() -> None:
    # fix for http://stackoverflow.com/questions/38065129
    array = xr.DataArray([1, 2, 3], [("x", [1, 1, 2])])
    expected = xr.DataArray([3, 3], [("x", [1, 2])])
    actual = array.groupby("x").sum()
    assert_equal(expected, actual)


def test_groupby_input_mutation() -> None:
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
def test_groupby_map_shrink_groups(obj) -> None:
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
def test_groupby_map_change_group_size(obj) -> None:
    def func(group):
        if group.sizes["x"] == 1:
            result = group.isel(x=[0, 0])
        else:
            result = group.isel(x=[0])
        return result

    expected = obj.isel(x=[0, 0, 1])
    actual = obj.groupby("x").map(func)
    assert_identical(expected, actual)


def test_da_groupby_map_func_args() -> None:
    def func(arg1, arg2, arg3=0):
        return arg1 + arg2 + arg3

    array = xr.DataArray([1, 1, 1], [("x", [1, 2, 3])])
    expected = xr.DataArray([3, 3, 3], [("x", [1, 2, 3])])
    actual = array.groupby("x").map(func, args=(1,), arg3=1)
    assert_identical(expected, actual)


def test_ds_groupby_map_func_args() -> None:
    def func(arg1, arg2, arg3=0):
        return arg1 + arg2 + arg3

    dataset = xr.Dataset({"foo": ("x", [1, 1, 1])}, {"x": [1, 2, 3]})
    expected = xr.Dataset({"foo": ("x", [3, 3, 3])}, {"x": [1, 2, 3]})
    actual = dataset.groupby("x").map(func, args=(1,), arg3=1)
    assert_identical(expected, actual)


def test_da_groupby_empty() -> None:

    empty_array = xr.DataArray([], dims="dim")

    with pytest.raises(ValueError):
        empty_array.groupby("dim")


def test_da_groupby_quantile() -> None:

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


def test_ds_groupby_quantile() -> None:
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


def test_da_groupby_assign_coords() -> None:
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
def test_groupby_repr(obj, dim) -> None:
    actual = repr(obj.groupby(dim))
    expected = f"{obj.__class__.__name__}GroupBy"
    expected += ", grouped over %r" % dim
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
def test_groupby_repr_datetime(obj) -> None:
    actual = repr(obj.groupby("t.month"))
    expected = f"{obj.__class__.__name__}GroupBy"
    expected += ", grouped over 'month'"
    expected += "\n%r groups with labels " % (len(np.unique(obj.t.dt.month)))
    expected += "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12."
    assert actual == expected


def test_groupby_drops_nans() -> None:
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
    expected = ds.mean("time").where(ds.id.notnull())
    assert_identical(actual, expected)

    # NaN in non-dimensional coordinate
    array = xr.DataArray([1, 2, 3], [("x", [1, 2, 3])])
    array["x1"] = ("x", [1, 1, np.nan])
    expected_da = xr.DataArray(3, [("x1", [1])])
    actual = array.groupby("x1").sum()
    assert_equal(expected_da, actual)

    # NaT in non-dimensional coordinate
    array["t"] = (
        "x",
        [
            np.datetime64("2001-01-01"),
            np.datetime64("2001-01-01"),
            np.datetime64("NaT"),
        ],
    )
    expected_da = xr.DataArray(3, [("t", [np.datetime64("2001-01-01")])])
    actual = array.groupby("t").sum()
    assert_equal(expected_da, actual)

    # test for repeated coordinate labels
    array = xr.DataArray([0, 1, 2, 4, 3, 4], [("x", [np.nan, 1, 1, np.nan, 2, np.nan])])
    expected_da = xr.DataArray([3, 3], [("x", [1, 2])])
    actual = array.groupby("x").sum()
    assert_equal(expected_da, actual)


def test_groupby_grouping_errors() -> None:
    dataset = xr.Dataset({"foo": ("x", [1, 1, 1])}, {"x": [1, 2, 3]})
    with pytest.raises(
        ValueError, match=r"None of the data falls within bins with edges"
    ):
        dataset.groupby_bins("x", bins=[0.1, 0.2, 0.3])

    with pytest.raises(
        ValueError, match=r"None of the data falls within bins with edges"
    ):
        dataset.to_array().groupby_bins("x", bins=[0.1, 0.2, 0.3])

    with pytest.raises(ValueError, match=r"All bin edges are NaN."):
        dataset.groupby_bins("x", bins=[np.nan, np.nan, np.nan])

    with pytest.raises(ValueError, match=r"All bin edges are NaN."):
        dataset.to_array().groupby_bins("x", bins=[np.nan, np.nan, np.nan])

    with pytest.raises(ValueError, match=r"Failed to group data."):
        dataset.groupby(dataset.foo * np.nan)

    with pytest.raises(ValueError, match=r"Failed to group data."):
        dataset.to_array().groupby(dataset.foo * np.nan)


def test_groupby_reduce_dimension_error(array) -> None:
    grouped = array.groupby("y")
    with pytest.raises(ValueError, match=r"cannot reduce over dimensions"):
        grouped.mean()

    with pytest.raises(ValueError, match=r"cannot reduce over dimensions"):
        grouped.mean("huh")

    with pytest.raises(ValueError, match=r"cannot reduce over dimensions"):
        grouped.mean(("x", "y", "asd"))

    grouped = array.groupby("y", squeeze=False)
    assert_identical(array, grouped.mean())

    assert_identical(array.mean("x"), grouped.reduce(np.mean, "x"))
    assert_allclose(array.mean(["x", "z"]), grouped.reduce(np.mean, ["x", "z"]))


def test_groupby_multiple_string_args(array) -> None:
    with pytest.raises(TypeError):
        array.groupby("x", "y")


def test_groupby_bins_timeseries() -> None:
    ds = xr.Dataset()
    ds["time"] = xr.DataArray(
        pd.date_range("2010-08-01", "2010-08-15", freq="15min"), dims="time"
    )
    ds["val"] = xr.DataArray(np.ones(ds["time"].shape), dims="time")
    time_bins = pd.date_range(start="2010-08-01", end="2010-08-15", freq="24H")
    actual = ds.groupby_bins("time", time_bins).sum()
    expected = xr.DataArray(
        96 * np.ones((14,)),
        dims=["time_bins"],
        coords={"time_bins": pd.cut(time_bins, time_bins).categories},
    ).to_dataset(name="val")
    assert_identical(actual, expected)


def test_groupby_none_group_name() -> None:
    # GH158
    # xarray should not fail if a DataArray's name attribute is None

    data = np.arange(10) + 10
    da = xr.DataArray(data)  # da.name = None
    key = xr.DataArray(np.floor_divide(data, 2))

    mean = da.groupby(key).mean()
    assert "group" in mean.dims


def test_groupby_getitem(dataset) -> None:

    assert_identical(dataset.sel(x="a"), dataset.groupby("x")["a"])
    assert_identical(dataset.sel(z=1), dataset.groupby("z")[1])

    assert_identical(dataset.foo.sel(x="a"), dataset.foo.groupby("x")["a"])
    assert_identical(dataset.foo.sel(z=1), dataset.foo.groupby("z")[1])

    actual = dataset.groupby("boo")["f"].unstack().transpose("x", "y", "z")
    expected = dataset.sel(y=[1], z=[1, 2]).transpose("x", "y", "z")
    assert_identical(expected, actual)


def test_groupby_dataset() -> None:
    data = Dataset(
        {"z": (["x", "y"], np.random.randn(3, 5))},
        {"x": ("x", list("abc")), "c": ("x", [0, 1, 0]), "y": range(5)},
    )
    groupby = data.groupby("x")
    assert len(groupby) == 3
    expected_groups = {"a": 0, "b": 1, "c": 2}
    assert groupby.groups == expected_groups
    expected_items = [
        ("a", data.isel(x=0)),
        ("b", data.isel(x=1)),
        ("c", data.isel(x=2)),
    ]
    for actual, expected in zip(groupby, expected_items):
        assert actual[0] == expected[0]
        assert_equal(actual[1], expected[1])

    def identity(x):
        return x

    for k in ["x", "c", "y"]:
        actual = data.groupby(k, squeeze=False).map(identity)
        assert_equal(data, actual)


def test_groupby_dataset_returns_new_type() -> None:
    data = Dataset({"z": (["x", "y"], np.random.randn(3, 5))})

    actual = data.groupby("x").map(lambda ds: ds["z"])
    expected = data["z"]
    assert_identical(expected, actual)

    actual = data["z"].groupby("x").map(lambda x: x.to_dataset())
    expected_ds = data
    assert_identical(expected_ds, actual)


def test_groupby_dataset_iter() -> None:
    data = create_test_data()
    for n, (t, sub) in enumerate(list(data.groupby("dim1"))[:3]):
        assert data["dim1"][n] == t
        assert_equal(data["var1"][n], sub["var1"])
        assert_equal(data["var2"][n], sub["var2"])
        assert_equal(data["var3"][:, n], sub["var3"])


def test_groupby_dataset_errors() -> None:
    data = create_test_data()
    with pytest.raises(TypeError, match=r"`group` must be"):
        data.groupby(np.arange(10))
    with pytest.raises(ValueError, match=r"length does not match"):
        data.groupby(data["dim1"][:3])
    with pytest.raises(TypeError, match=r"`group` must be"):
        data.groupby(data.coords["dim1"].to_index())


def test_groupby_dataset_reduce() -> None:
    data = Dataset(
        {
            "xy": (["x", "y"], np.random.randn(3, 4)),
            "xonly": ("x", np.random.randn(3)),
            "yonly": ("y", np.random.randn(4)),
            "letters": ("y", ["a", "a", "b", "b"]),
        }
    )

    expected = data.mean("y")
    expected["yonly"] = expected["yonly"].variable.set_dims({"x": 3})
    actual = data.groupby("x").mean(...)
    assert_allclose(expected, actual)

    actual = data.groupby("x").mean("y")
    assert_allclose(expected, actual)

    letters = data["letters"]
    expected = Dataset(
        {
            "xy": data["xy"].groupby(letters).mean(...),
            "xonly": (data["xonly"].mean().variable.set_dims({"letters": 2})),
            "yonly": data["yonly"].groupby(letters).mean(),
        }
    )
    actual = data.groupby("letters").mean(...)
    assert_allclose(expected, actual)


@pytest.mark.parametrize("squeeze", [True, False])
def test_groupby_dataset_math(squeeze) -> None:
    def reorder_dims(x):
        return x.transpose("dim1", "dim2", "dim3", "time")

    ds = create_test_data()
    ds["dim1"] = ds["dim1"]
    grouped = ds.groupby("dim1", squeeze=squeeze)

    expected = reorder_dims(ds + ds.coords["dim1"])
    actual = grouped + ds.coords["dim1"]
    assert_identical(expected, reorder_dims(actual))

    actual = ds.coords["dim1"] + grouped
    assert_identical(expected, reorder_dims(actual))

    ds2 = 2 * ds
    expected = reorder_dims(ds + ds2)
    actual = grouped + ds2
    assert_identical(expected, reorder_dims(actual))

    actual = ds2 + grouped
    assert_identical(expected, reorder_dims(actual))


def test_groupby_math_more() -> None:
    ds = create_test_data()
    grouped = ds.groupby("numbers")
    zeros = DataArray([0, 0, 0, 0], [("numbers", range(4))])
    expected = (ds + Variable("dim3", np.zeros(10))).transpose(
        "dim3", "dim1", "dim2", "time"
    )
    actual = grouped + zeros
    assert_equal(expected, actual)

    actual = zeros + grouped
    assert_equal(expected, actual)

    with pytest.raises(ValueError, match=r"incompat.* grouped binary"):
        grouped + ds
    with pytest.raises(ValueError, match=r"incompat.* grouped binary"):
        ds + grouped
    with pytest.raises(TypeError, match=r"only support binary ops"):
        grouped + 1
    with pytest.raises(TypeError, match=r"only support binary ops"):
        grouped + grouped
    with pytest.raises(TypeError, match=r"in-place operations"):
        ds += grouped

    ds = Dataset(
        {
            "x": ("time", np.arange(100)),
            "time": pd.date_range("2000-01-01", periods=100),
        }
    )
    with pytest.raises(ValueError, match=r"incompat.* grouped binary"):
        ds + ds.groupby("time.month")


@pytest.mark.parametrize("indexed_coord", [True, False])
def test_groupby_bins_math(indexed_coord) -> None:
    N = 7
    da = DataArray(np.random.random((N, N)), dims=("x", "y"))
    if indexed_coord:
        da["x"] = np.arange(N)
        da["y"] = np.arange(N)
    g = da.groupby_bins("x", np.arange(0, N + 1, 3))
    mean = g.mean()
    expected = da.isel(x=slice(1, None)) - mean.isel(x_bins=("x", [0, 0, 0, 1, 1, 1]))
    actual = g - mean
    assert_identical(expected, actual)


def test_groupby_math_nD_group() -> None:
    N = 40
    da = DataArray(
        np.random.random((N, N)),
        dims=("x", "y"),
        coords={
            "labels": (
                "x",
                np.repeat(["a", "b", "c", "d", "e", "f", "g", "h"], repeats=N // 8),
            ),
        },
    )
    da["labels2d"] = xr.broadcast(da.labels, da)[0]

    g = da.groupby("labels2d")
    mean = g.mean()
    expected = da - mean.sel(labels2d=da.labels2d)
    expected["labels"] = expected.labels.broadcast_like(expected.labels2d)
    actual = g - mean
    assert_identical(expected, actual)

    da["num"] = (
        "x",
        np.repeat([1, 2, 3, 4, 5, 6, 7, 8], repeats=N // 8),
    )
    da["num2d"] = xr.broadcast(da.num, da)[0]
    g = da.groupby_bins("num2d", bins=[0, 4, 6])
    mean = g.mean()
    idxr = np.digitize(da.num2d, bins=(0, 4, 6), right=True)[:30, :] - 1
    expanded_mean = mean.drop("num2d_bins").isel(num2d_bins=(("x", "y"), idxr))
    expected = da.isel(x=slice(30)) - expanded_mean
    expected["labels"] = expected.labels.broadcast_like(expected.labels2d)
    expected["num"] = expected.num.broadcast_like(expected.num2d)
    expected["num2d_bins"] = (("x", "y"), mean.num2d_bins.data[idxr])
    actual = g - mean
    assert_identical(expected, actual)


def test_groupby_dataset_math_virtual() -> None:
    ds = Dataset({"x": ("t", [1, 2, 3])}, {"t": pd.date_range("20100101", periods=3)})
    grouped = ds.groupby("t.day")
    actual = grouped - grouped.mean(...)
    expected = Dataset({"x": ("t", [0, 0, 0])}, ds[["t", "t.day"]])
    assert_identical(actual, expected)


def test_groupby_dataset_nan() -> None:
    # nan should be excluded from groupby
    ds = Dataset({"foo": ("x", [1, 2, 3, 4])}, {"bar": ("x", [1, 1, 2, np.nan])})
    actual = ds.groupby("bar").mean(...)
    expected = Dataset({"foo": ("bar", [1.5, 3]), "bar": [1, 2]})
    assert_identical(actual, expected)


def test_groupby_dataset_order() -> None:
    # groupby should preserve variables order
    ds = Dataset()
    for vn in ["a", "b", "c"]:
        ds[vn] = DataArray(np.arange(10), dims=["t"])
    data_vars_ref = list(ds.data_vars.keys())
    ds = ds.groupby("t").mean(...)
    data_vars = list(ds.data_vars.keys())
    assert data_vars == data_vars_ref
    # coords are now at the end of the list, so the test below fails
    # all_vars = list(ds.variables.keys())
    # all_vars_ref = list(ds.variables.keys())
    # .assertEqual(all_vars, all_vars_ref)


def test_groupby_dataset_fillna():

    ds = Dataset({"a": ("x", [np.nan, 1, np.nan, 3])}, {"x": [0, 1, 2, 3]})
    expected = Dataset({"a": ("x", range(4))}, {"x": [0, 1, 2, 3]})
    for target in [ds, expected]:
        target.coords["b"] = ("x", [0, 0, 1, 1])
    actual = ds.groupby("b").fillna(DataArray([0, 2], dims="b"))
    assert_identical(expected, actual)

    actual = ds.groupby("b").fillna(Dataset({"a": ("b", [0, 2])}))
    assert_identical(expected, actual)

    # attrs with groupby
    ds.attrs["attr"] = "ds"
    ds.a.attrs["attr"] = "da"
    actual = ds.groupby("b").fillna(Dataset({"a": ("b", [0, 2])}))
    assert actual.attrs == ds.attrs
    assert actual.a.name == "a"
    assert actual.a.attrs == ds.a.attrs


def test_groupby_dataset_where():
    # groupby
    ds = Dataset({"a": ("x", range(5))}, {"c": ("x", [0, 0, 1, 1, 1])})
    cond = Dataset({"a": ("c", [True, False])})
    expected = ds.copy(deep=True)
    expected["a"].values = [0, 1] + [np.nan] * 3
    actual = ds.groupby("c").where(cond)
    assert_identical(expected, actual)

    # attrs with groupby
    ds.attrs["attr"] = "ds"
    ds.a.attrs["attr"] = "da"
    actual = ds.groupby("c").where(cond)
    assert actual.attrs == ds.attrs
    assert actual.a.name == "a"
    assert actual.a.attrs == ds.a.attrs


def test_groupby_dataset_assign():
    ds = Dataset({"a": ("x", range(3))}, {"b": ("x", ["A"] * 2 + ["B"])})
    actual = ds.groupby("b").assign(c=lambda ds: 2 * ds.a)
    expected = ds.merge({"c": ("x", [0, 2, 4])})
    assert_identical(actual, expected)

    actual = ds.groupby("b").assign(c=lambda ds: ds.a.sum())
    expected = ds.merge({"c": ("x", [1, 1, 2])})
    assert_identical(actual, expected)

    actual = ds.groupby("b").assign_coords(c=lambda ds: ds.a.sum())
    expected = expected.set_coords("c")
    assert_identical(actual, expected)


class TestDataArrayGroupBy:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.attrs = {"attr1": "value1", "attr2": 2929}
        self.x = np.random.random((10, 20))
        self.v = Variable(["x", "y"], self.x)
        self.va = Variable(["x", "y"], self.x, self.attrs)
        self.ds = Dataset({"foo": self.v})
        self.dv = self.ds["foo"]

        self.mindex = pd.MultiIndex.from_product(
            [["a", "b"], [1, 2]], names=("level_1", "level_2")
        )
        self.mda = DataArray([0, 1, 2, 3], coords={"x": self.mindex}, dims="x")

        self.da = self.dv.copy()
        self.da.coords["abc"] = ("y", np.array(["a"] * 9 + ["c"] + ["b"] * 10))
        self.da.coords["y"] = 20 + 100 * self.da["y"]

    def test_stack_groupby_unsorted_coord(self):
        data = [[0, 1], [2, 3]]
        data_flat = [0, 1, 2, 3]
        dims = ["x", "y"]
        y_vals = [2, 3]

        arr = xr.DataArray(data, dims=dims, coords={"y": y_vals})
        actual1 = arr.stack(z=dims).groupby("z").first()
        midx1 = pd.MultiIndex.from_product([[0, 1], [2, 3]], names=dims)
        expected1 = xr.DataArray(data_flat, dims=["z"], coords={"z": midx1})
        assert_equal(actual1, expected1)

        # GH: 3287.  Note that y coord values are not in sorted order.
        arr = xr.DataArray(data, dims=dims, coords={"y": y_vals[::-1]})
        actual2 = arr.stack(z=dims).groupby("z").first()
        midx2 = pd.MultiIndex.from_product([[0, 1], [3, 2]], names=dims)
        expected2 = xr.DataArray(data_flat, dims=["z"], coords={"z": midx2})
        assert_equal(actual2, expected2)

    def test_groupby_iter(self):
        for ((act_x, act_dv), (exp_x, exp_ds)) in zip(
            self.dv.groupby("y"), self.ds.groupby("y")
        ):
            assert exp_x == act_x
            assert_identical(exp_ds["foo"], act_dv)
        for ((_, exp_dv), act_dv) in zip(self.dv.groupby("x"), self.dv):
            assert_identical(exp_dv, act_dv)

    def test_groupby_properties(self):
        grouped = self.da.groupby("abc")
        expected_groups = {"a": range(0, 9), "c": [9], "b": range(10, 20)}
        assert expected_groups.keys() == grouped.groups.keys()
        for key in expected_groups:
            assert_array_equal(expected_groups[key], grouped.groups[key])
        assert 3 == len(grouped)

    def test_groupby_map_identity(self):
        expected = self.da
        idx = expected.coords["y"]

        def identity(x):
            return x

        for g in ["x", "y", "abc", idx]:
            for shortcut in [False, True]:
                for squeeze in [False, True]:
                    grouped = expected.groupby(g, squeeze=squeeze)
                    actual = grouped.map(identity, shortcut=shortcut)
                    assert_identical(expected, actual)

    def test_groupby_sum(self):
        array = self.da
        grouped = array.groupby("abc")

        expected_sum_all = Dataset(
            {
                "foo": Variable(
                    ["abc"],
                    np.array(
                        [
                            self.x[:, :9].sum(),
                            self.x[:, 10:].sum(),
                            self.x[:, 9:10].sum(),
                        ]
                    ).T,
                ),
                "abc": Variable(["abc"], np.array(["a", "b", "c"])),
            }
        )["foo"]
        assert_allclose(expected_sum_all, grouped.reduce(np.sum, dim=...))
        assert_allclose(expected_sum_all, grouped.sum(...))

        expected = DataArray(
            [
                array["y"].values[idx].sum()
                for idx in [slice(9), slice(10, None), slice(9, 10)]
            ],
            [["a", "b", "c"]],
            ["abc"],
        )
        actual = array["y"].groupby("abc").map(np.sum)
        assert_allclose(expected, actual)
        actual = array["y"].groupby("abc").sum(...)
        assert_allclose(expected, actual)

        expected_sum_axis1 = Dataset(
            {
                "foo": (
                    ["x", "abc"],
                    np.array(
                        [
                            self.x[:, :9].sum(1),
                            self.x[:, 10:].sum(1),
                            self.x[:, 9:10].sum(1),
                        ]
                    ).T,
                ),
                "abc": Variable(["abc"], np.array(["a", "b", "c"])),
            }
        )["foo"]
        assert_allclose(expected_sum_axis1, grouped.reduce(np.sum, "y"))
        assert_allclose(expected_sum_axis1, grouped.sum("y"))

    def test_groupby_sum_default(self):
        array = self.da
        grouped = array.groupby("abc")

        expected_sum_all = Dataset(
            {
                "foo": Variable(
                    ["x", "abc"],
                    np.array(
                        [
                            self.x[:, :9].sum(axis=-1),
                            self.x[:, 10:].sum(axis=-1),
                            self.x[:, 9:10].sum(axis=-1),
                        ]
                    ).T,
                ),
                "abc": Variable(["abc"], np.array(["a", "b", "c"])),
            }
        )["foo"]

        assert_allclose(expected_sum_all, grouped.sum(dim="y"))

    def test_groupby_count(self):
        array = DataArray(
            [0, 0, np.nan, np.nan, 0, 0],
            coords={"cat": ("x", ["a", "b", "b", "c", "c", "c"])},
            dims="x",
        )
        actual = array.groupby("cat").count()
        expected = DataArray([1, 1, 2], coords=[("cat", ["a", "b", "c"])])
        assert_identical(actual, expected)

    @pytest.mark.skip("needs to be fixed for shortcut=False, keep_attrs=False")
    def test_groupby_reduce_attrs(self):
        array = self.da
        array.attrs["foo"] = "bar"

        for shortcut in [True, False]:
            for keep_attrs in [True, False]:
                print(f"shortcut={shortcut}, keep_attrs={keep_attrs}")
                actual = array.groupby("abc").reduce(
                    np.mean, keep_attrs=keep_attrs, shortcut=shortcut
                )
                expected = array.groupby("abc").mean()
                if keep_attrs:
                    expected.attrs["foo"] = "bar"
                assert_identical(expected, actual)

    def test_groupby_map_center(self):
        def center(x):
            return x - np.mean(x)

        array = self.da
        grouped = array.groupby("abc")

        expected_ds = array.to_dataset()
        exp_data = np.hstack(
            [center(self.x[:, :9]), center(self.x[:, 9:10]), center(self.x[:, 10:])]
        )
        expected_ds["foo"] = (["x", "y"], exp_data)
        expected_centered = expected_ds["foo"]
        assert_allclose(expected_centered, grouped.map(center))

    def test_groupby_map_ndarray(self):
        # regression test for #326
        array = self.da
        grouped = array.groupby("abc")
        actual = grouped.map(np.asarray)
        assert_equal(array, actual)

    def test_groupby_map_changes_metadata(self):
        def change_metadata(x):
            x.coords["x"] = x.coords["x"] * 2
            x.attrs["fruit"] = "lemon"
            return x

        array = self.da
        grouped = array.groupby("abc")
        actual = grouped.map(change_metadata)
        expected = array.copy()
        expected = change_metadata(expected)
        assert_equal(expected, actual)

    def test_groupby_math(self):
        array = self.da
        for squeeze in [True, False]:
            grouped = array.groupby("x", squeeze=squeeze)

            expected = array + array.coords["x"]
            actual = grouped + array.coords["x"]
            assert_identical(expected, actual)

            actual = array.coords["x"] + grouped
            assert_identical(expected, actual)

            ds = array.coords["x"].to_dataset(name="X")
            expected = array + ds
            actual = grouped + ds
            assert_identical(expected, actual)

            actual = ds + grouped
            assert_identical(expected, actual)

        grouped = array.groupby("abc")
        expected_agg = (grouped.mean(...) - np.arange(3)).rename(None)
        actual = grouped - DataArray(range(3), [("abc", ["a", "b", "c"])])
        actual_agg = actual.groupby("abc").mean(...)
        assert_allclose(expected_agg, actual_agg)

        with pytest.raises(TypeError, match=r"only support binary ops"):
            grouped + 1
        with pytest.raises(TypeError, match=r"only support binary ops"):
            grouped + grouped
        with pytest.raises(TypeError, match=r"in-place operations"):
            array += grouped

    def test_groupby_math_not_aligned(self):
        array = DataArray(
            range(4), {"b": ("x", [0, 0, 1, 1]), "x": [0, 1, 2, 3]}, dims="x"
        )
        other = DataArray([10], coords={"b": [0]}, dims="b")
        actual = array.groupby("b") + other
        expected = DataArray([10, 11, np.nan, np.nan], array.coords)
        assert_identical(expected, actual)

        other = DataArray([10], coords={"c": 123, "b": [0]}, dims="b")
        actual = array.groupby("b") + other
        expected.coords["c"] = (["x"], [123] * 2 + [np.nan] * 2)
        assert_identical(expected, actual)

        other = Dataset({"a": ("b", [10])}, {"b": [0]})
        actual = array.groupby("b") + other
        expected = Dataset({"a": ("x", [10, 11, np.nan, np.nan])}, array.coords)
        assert_identical(expected, actual)

    def test_groupby_restore_dim_order(self):
        array = DataArray(
            np.random.randn(5, 3),
            coords={"a": ("x", range(5)), "b": ("y", range(3))},
            dims=["x", "y"],
        )
        for by, expected_dims in [
            ("x", ("x", "y")),
            ("y", ("x", "y")),
            ("a", ("a", "y")),
            ("b", ("x", "b")),
        ]:
            result = array.groupby(by).map(lambda x: x.squeeze())
            assert result.dims == expected_dims

    def test_groupby_restore_coord_dims(self):
        array = DataArray(
            np.random.randn(5, 3),
            coords={
                "a": ("x", range(5)),
                "b": ("y", range(3)),
                "c": (("x", "y"), np.random.randn(5, 3)),
            },
            dims=["x", "y"],
        )

        for by, expected_dims in [
            ("x", ("x", "y")),
            ("y", ("x", "y")),
            ("a", ("a", "y")),
            ("b", ("x", "b")),
        ]:
            result = array.groupby(by, restore_coord_dims=True).map(
                lambda x: x.squeeze()
            )["c"]
            assert result.dims == expected_dims

    def test_groupby_first_and_last(self):
        array = DataArray([1, 2, 3, 4, 5], dims="x")
        by = DataArray(["a"] * 2 + ["b"] * 3, dims="x", name="ab")

        expected = DataArray([1, 3], [("ab", ["a", "b"])])
        actual = array.groupby(by).first()
        assert_identical(expected, actual)

        expected = DataArray([2, 5], [("ab", ["a", "b"])])
        actual = array.groupby(by).last()
        assert_identical(expected, actual)

        array = DataArray(np.random.randn(5, 3), dims=["x", "y"])
        expected = DataArray(array[[0, 2]], {"ab": ["a", "b"]}, ["ab", "y"])
        actual = array.groupby(by).first()
        assert_identical(expected, actual)

        actual = array.groupby("x").first()
        expected = array  # should be a no-op
        assert_identical(expected, actual)

    def make_groupby_multidim_example_array(self):
        return DataArray(
            [[[0, 1], [2, 3]], [[5, 10], [15, 20]]],
            coords={
                "lon": (["ny", "nx"], [[30, 40], [40, 50]]),
                "lat": (["ny", "nx"], [[10, 10], [20, 20]]),
            },
            dims=["time", "ny", "nx"],
        )

    def test_groupby_multidim(self):
        array = self.make_groupby_multidim_example_array()
        for dim, expected_sum in [
            ("lon", DataArray([5, 28, 23], coords=[("lon", [30.0, 40.0, 50.0])])),
            ("lat", DataArray([16, 40], coords=[("lat", [10.0, 20.0])])),
        ]:
            actual_sum = array.groupby(dim).sum(...)
            assert_identical(expected_sum, actual_sum)

    def test_groupby_multidim_map(self):
        array = self.make_groupby_multidim_example_array()
        actual = array.groupby("lon").map(lambda x: x - x.mean())
        expected = DataArray(
            [[[-2.5, -6.0], [-5.0, -8.5]], [[2.5, 3.0], [8.0, 8.5]]],
            coords=array.coords,
            dims=array.dims,
        )
        assert_identical(expected, actual)

    def test_groupby_bins(self):
        array = DataArray(np.arange(4), dims="dim_0")
        # the first value should not be part of any group ("right" binning)
        array[0] = 99
        # bins follow conventions for pandas.cut
        # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
        bins = [0, 1.5, 5]
        bin_coords = pd.cut(array["dim_0"], bins).categories
        expected = DataArray(
            [1, 5], dims="dim_0_bins", coords={"dim_0_bins": bin_coords}
        )
        # the problem with this is that it overwrites the dimensions of array!
        # actual = array.groupby('dim_0', bins=bins).sum()
        actual = array.groupby_bins("dim_0", bins).map(lambda x: x.sum())
        assert_identical(expected, actual)
        # make sure original array dims are unchanged
        assert len(array.dim_0) == 4

    def test_groupby_bins_empty(self):
        array = DataArray(np.arange(4), [("x", range(4))])
        # one of these bins will be empty
        bins = [0, 4, 5]
        bin_coords = pd.cut(array["x"], bins).categories
        actual = array.groupby_bins("x", bins).sum()
        expected = DataArray([6, np.nan], dims="x_bins", coords={"x_bins": bin_coords})
        assert_identical(expected, actual)
        # make sure original array is unchanged
        # (was a problem in earlier versions)
        assert len(array.x) == 4

    def test_groupby_bins_multidim(self):
        array = self.make_groupby_multidim_example_array()
        bins = [0, 15, 20]
        bin_coords = pd.cut(array["lat"].values.flat, bins).categories
        expected = DataArray([16, 40], dims="lat_bins", coords={"lat_bins": bin_coords})
        actual = array.groupby_bins("lat", bins).map(lambda x: x.sum())
        assert_identical(expected, actual)
        # modify the array coordinates to be non-monotonic after unstacking
        array["lat"].data = np.array([[10.0, 20.0], [20.0, 10.0]])
        expected = DataArray([28, 28], dims="lat_bins", coords={"lat_bins": bin_coords})
        actual = array.groupby_bins("lat", bins).map(lambda x: x.sum())
        assert_identical(expected, actual)

    def test_groupby_bins_sort(self):
        data = xr.DataArray(
            np.arange(100), dims="x", coords={"x": np.linspace(-100, 100, num=100)}
        )
        binned_mean = data.groupby_bins("x", bins=11).mean()
        assert binned_mean.to_index().is_monotonic

    def test_groupby_assign_coords(self):

        array = DataArray([1, 2, 3, 4], {"c": ("x", [0, 0, 1, 1])}, dims="x")
        actual = array.groupby("c").assign_coords(d=lambda a: a.mean())
        expected = array.copy()
        expected.coords["d"] = ("x", [1.5, 1.5, 3.5, 3.5])
        assert_identical(actual, expected)

    def test_groupby_fillna(self):
        a = DataArray([np.nan, 1, np.nan, 3], coords={"x": range(4)}, dims="x")
        fill_value = DataArray([0, 1], dims="y")
        actual = a.fillna(fill_value)
        expected = DataArray(
            [[0, 1], [1, 1], [0, 1], [3, 3]], coords={"x": range(4)}, dims=("x", "y")
        )
        assert_identical(expected, actual)

        b = DataArray(range(4), coords={"x": range(4)}, dims="x")
        expected = b.copy()
        for target in [a, expected]:
            target.coords["b"] = ("x", [0, 0, 1, 1])
        actual = a.groupby("b").fillna(DataArray([0, 2], dims="b"))
        assert_identical(expected, actual)


class TestDataArrayResample:
    def test_resample(self):
        times = pd.date_range("2000-01-01", freq="6H", periods=10)
        array = DataArray(np.arange(10), [("time", times)])

        actual = array.resample(time="24H").mean()
        expected = DataArray(array.to_series().resample("24H").mean())
        assert_identical(expected, actual)

        actual = array.resample(time="24H").reduce(np.mean)
        assert_identical(expected, actual)

        # Our use of `loffset` may change if we align our API with pandas' changes.
        # ref https://github.com/pydata/xarray/pull/4537
        actual = array.resample(time="24H", loffset="-12H").mean()
        expected_ = array.to_series().resample("24H").mean()
        expected_.index += to_offset("-12H")
        expected = DataArray.from_series(expected_)
        assert_identical(actual, expected)

        with pytest.raises(ValueError, match=r"index must be monotonic"):
            array[[2, 0, 1]].resample(time="1D")

    def test_da_resample_func_args(self):
        def func(arg1, arg2, arg3=0.0):
            return arg1.mean("time") + arg2 + arg3

        times = pd.date_range("2000", periods=3, freq="D")
        da = xr.DataArray([1.0, 1.0, 1.0], coords=[times], dims=["time"])
        expected = xr.DataArray([3.0, 3.0, 3.0], coords=[times], dims=["time"])
        actual = da.resample(time="D").map(func, args=(1.0,), arg3=1.0)
        assert_identical(actual, expected)

    def test_resample_first(self):
        times = pd.date_range("2000-01-01", freq="6H", periods=10)
        array = DataArray(np.arange(10), [("time", times)])

        actual = array.resample(time="1D").first()
        expected = DataArray([0, 4, 8], [("time", times[::4])])
        assert_identical(expected, actual)

        # verify that labels don't use the first value
        actual = array.resample(time="24H").first()
        expected = DataArray(array.to_series().resample("24H").first())
        assert_identical(expected, actual)

        # missing values
        array = array.astype(float)
        array[:2] = np.nan
        actual = array.resample(time="1D").first()
        expected = DataArray([2, 4, 8], [("time", times[::4])])
        assert_identical(expected, actual)

        actual = array.resample(time="1D").first(skipna=False)
        expected = DataArray([np.nan, 4, 8], [("time", times[::4])])
        assert_identical(expected, actual)

        # regression test for http://stackoverflow.com/questions/33158558/
        array = Dataset({"time": times})["time"]
        actual = array.resample(time="1D").last()
        expected_times = pd.to_datetime(
            ["2000-01-01T18", "2000-01-02T18", "2000-01-03T06"]
        )
        expected = DataArray(expected_times, [("time", times[::4])], name="time")
        assert_identical(expected, actual)

    def test_resample_bad_resample_dim(self):
        times = pd.date_range("2000-01-01", freq="6H", periods=10)
        array = DataArray(np.arange(10), [("__resample_dim__", times)])
        with pytest.raises(ValueError, match=r"Proxy resampling dimension"):
            array.resample(**{"__resample_dim__": "1D"}).first()

    @requires_scipy
    def test_resample_drop_nondim_coords(self):
        xs = np.arange(6)
        ys = np.arange(3)
        times = pd.date_range("2000-01-01", freq="6H", periods=5)
        data = np.tile(np.arange(5), (6, 3, 1))
        xx, yy = np.meshgrid(xs * 5, ys * 2.5)
        tt = np.arange(len(times), dtype=int)
        array = DataArray(data, {"time": times, "x": xs, "y": ys}, ("x", "y", "time"))
        xcoord = DataArray(xx.T, {"x": xs, "y": ys}, ("x", "y"))
        ycoord = DataArray(yy.T, {"x": xs, "y": ys}, ("x", "y"))
        tcoord = DataArray(tt, {"time": times}, ("time",))
        ds = Dataset({"data": array, "xc": xcoord, "yc": ycoord, "tc": tcoord})
        ds = ds.set_coords(["xc", "yc", "tc"])

        # Select the data now, with the auxiliary coordinates in place
        array = ds["data"]

        # Re-sample
        actual = array.resample(time="12H", restore_coord_dims=True).mean("time")
        assert "tc" not in actual.coords

        # Up-sample - filling
        actual = array.resample(time="1H", restore_coord_dims=True).ffill()
        assert "tc" not in actual.coords

        # Up-sample - interpolation
        actual = array.resample(time="1H", restore_coord_dims=True).interpolate(
            "linear"
        )
        assert "tc" not in actual.coords

    def test_resample_keep_attrs(self):
        times = pd.date_range("2000-01-01", freq="6H", periods=10)
        array = DataArray(np.ones(10), [("time", times)])
        array.attrs["meta"] = "data"

        result = array.resample(time="1D").mean(keep_attrs=True)
        expected = DataArray([1, 1, 1], [("time", times[::4])], attrs=array.attrs)
        assert_identical(result, expected)

        with pytest.warns(
            UserWarning, match="Passing ``keep_attrs`` to ``resample`` has no effect."
        ):
            array.resample(time="1D", keep_attrs=True)

    def test_resample_skipna(self):
        times = pd.date_range("2000-01-01", freq="6H", periods=10)
        array = DataArray(np.ones(10), [("time", times)])
        array[1] = np.nan

        result = array.resample(time="1D").mean(skipna=False)
        expected = DataArray([np.nan, 1, 1], [("time", times[::4])])
        assert_identical(result, expected)

    def test_upsample(self):
        times = pd.date_range("2000-01-01", freq="6H", periods=5)
        array = DataArray(np.arange(5), [("time", times)])

        # Forward-fill
        actual = array.resample(time="3H").ffill()
        expected = DataArray(array.to_series().resample("3H").ffill())
        assert_identical(expected, actual)

        # Backward-fill
        actual = array.resample(time="3H").bfill()
        expected = DataArray(array.to_series().resample("3H").bfill())
        assert_identical(expected, actual)

        # As frequency
        actual = array.resample(time="3H").asfreq()
        expected = DataArray(array.to_series().resample("3H").asfreq())
        assert_identical(expected, actual)

        # Pad
        actual = array.resample(time="3H").pad()
        expected = DataArray(array.to_series().resample("3H").pad())
        assert_identical(expected, actual)

        # Nearest
        rs = array.resample(time="3H")
        actual = rs.nearest()
        new_times = rs._full_index
        expected = DataArray(array.reindex(time=new_times, method="nearest"))
        assert_identical(expected, actual)

    def test_upsample_nd(self):
        # Same as before, but now we try on multi-dimensional DataArrays.
        xs = np.arange(6)
        ys = np.arange(3)
        times = pd.date_range("2000-01-01", freq="6H", periods=5)
        data = np.tile(np.arange(5), (6, 3, 1))
        array = DataArray(data, {"time": times, "x": xs, "y": ys}, ("x", "y", "time"))

        # Forward-fill
        actual = array.resample(time="3H").ffill()
        expected_data = np.repeat(data, 2, axis=-1)
        expected_times = times.to_series().resample("3H").asfreq().index
        expected_data = expected_data[..., : len(expected_times)]
        expected = DataArray(
            expected_data,
            {"time": expected_times, "x": xs, "y": ys},
            ("x", "y", "time"),
        )
        assert_identical(expected, actual)

        # Backward-fill
        actual = array.resample(time="3H").ffill()
        expected_data = np.repeat(np.flipud(data.T).T, 2, axis=-1)
        expected_data = np.flipud(expected_data.T).T
        expected_times = times.to_series().resample("3H").asfreq().index
        expected_data = expected_data[..., : len(expected_times)]
        expected = DataArray(
            expected_data,
            {"time": expected_times, "x": xs, "y": ys},
            ("x", "y", "time"),
        )
        assert_identical(expected, actual)

        # As frequency
        actual = array.resample(time="3H").asfreq()
        expected_data = np.repeat(data, 2, axis=-1).astype(float)[..., :-1]
        expected_data[..., 1::2] = np.nan
        expected_times = times.to_series().resample("3H").asfreq().index
        expected = DataArray(
            expected_data,
            {"time": expected_times, "x": xs, "y": ys},
            ("x", "y", "time"),
        )
        assert_identical(expected, actual)

        # Pad
        actual = array.resample(time="3H").pad()
        expected_data = np.repeat(data, 2, axis=-1)
        expected_data[..., 1::2] = expected_data[..., ::2]
        expected_data = expected_data[..., :-1]
        expected_times = times.to_series().resample("3H").asfreq().index
        expected = DataArray(
            expected_data,
            {"time": expected_times, "x": xs, "y": ys},
            ("x", "y", "time"),
        )
        assert_identical(expected, actual)

    def test_upsample_tolerance(self):
        # Test tolerance keyword for upsample methods bfill, pad, nearest
        times = pd.date_range("2000-01-01", freq="1D", periods=2)
        times_upsampled = pd.date_range("2000-01-01", freq="6H", periods=5)
        array = DataArray(np.arange(2), [("time", times)])

        # Forward fill
        actual = array.resample(time="6H").ffill(tolerance="12H")
        expected = DataArray([0.0, 0.0, 0.0, np.nan, 1.0], [("time", times_upsampled)])
        assert_identical(expected, actual)

        # Backward fill
        actual = array.resample(time="6H").bfill(tolerance="12H")
        expected = DataArray([0.0, np.nan, 1.0, 1.0, 1.0], [("time", times_upsampled)])
        assert_identical(expected, actual)

        # Nearest
        actual = array.resample(time="6H").nearest(tolerance="6H")
        expected = DataArray([0, 0, np.nan, 1, 1], [("time", times_upsampled)])
        assert_identical(expected, actual)

    @requires_scipy
    def test_upsample_interpolate(self):
        from scipy.interpolate import interp1d

        xs = np.arange(6)
        ys = np.arange(3)
        times = pd.date_range("2000-01-01", freq="6H", periods=5)

        z = np.arange(5) ** 2
        data = np.tile(z, (6, 3, 1))
        array = DataArray(data, {"time": times, "x": xs, "y": ys}, ("x", "y", "time"))

        expected_times = times.to_series().resample("1H").asfreq().index
        # Split the times into equal sub-intervals to simulate the 6 hour
        # to 1 hour up-sampling
        new_times_idx = np.linspace(0, len(times) - 1, len(times) * 5)
        for kind in ["linear", "nearest", "zero", "slinear", "quadratic", "cubic"]:
            actual = array.resample(time="1H").interpolate(kind)
            f = interp1d(
                np.arange(len(times)),
                data,
                kind=kind,
                axis=-1,
                bounds_error=True,
                assume_sorted=True,
            )
            expected_data = f(new_times_idx)
            expected = DataArray(
                expected_data,
                {"time": expected_times, "x": xs, "y": ys},
                ("x", "y", "time"),
            )
            # Use AllClose because there are some small differences in how
            # we upsample timeseries versus the integer indexing as I've
            # done here due to floating point arithmetic
            assert_allclose(expected, actual, rtol=1e-16)

    @requires_scipy
    def test_upsample_interpolate_bug_2197(self):
        dates = pd.date_range("2007-02-01", "2007-03-01", freq="D")
        da = xr.DataArray(np.arange(len(dates)), [("time", dates)])
        result = da.resample(time="M").interpolate("linear")
        expected_times = np.array(
            [np.datetime64("2007-02-28"), np.datetime64("2007-03-31")]
        )
        expected = xr.DataArray([27.0, np.nan], [("time", expected_times)])
        assert_equal(result, expected)

    @requires_scipy
    def test_upsample_interpolate_regression_1605(self):
        dates = pd.date_range("2016-01-01", "2016-03-31", freq="1D")
        expected = xr.DataArray(
            np.random.random((len(dates), 2, 3)),
            dims=("time", "x", "y"),
            coords={"time": dates},
        )
        actual = expected.resample(time="1D").interpolate("linear")
        assert_allclose(actual, expected, rtol=1e-16)

    @requires_dask
    @requires_scipy
    @pytest.mark.parametrize("chunked_time", [True, False])
    def test_upsample_interpolate_dask(self, chunked_time):
        from scipy.interpolate import interp1d

        xs = np.arange(6)
        ys = np.arange(3)
        times = pd.date_range("2000-01-01", freq="6H", periods=5)

        z = np.arange(5) ** 2
        data = np.tile(z, (6, 3, 1))
        array = DataArray(data, {"time": times, "x": xs, "y": ys}, ("x", "y", "time"))
        chunks = {"x": 2, "y": 1}
        if chunked_time:
            chunks["time"] = 3

        expected_times = times.to_series().resample("1H").asfreq().index
        # Split the times into equal sub-intervals to simulate the 6 hour
        # to 1 hour up-sampling
        new_times_idx = np.linspace(0, len(times) - 1, len(times) * 5)
        for kind in ["linear", "nearest", "zero", "slinear", "quadratic", "cubic"]:
            actual = array.chunk(chunks).resample(time="1H").interpolate(kind)
            actual = actual.compute()
            f = interp1d(
                np.arange(len(times)),
                data,
                kind=kind,
                axis=-1,
                bounds_error=True,
                assume_sorted=True,
            )
            expected_data = f(new_times_idx)
            expected = DataArray(
                expected_data,
                {"time": expected_times, "x": xs, "y": ys},
                ("x", "y", "time"),
            )
            # Use AllClose because there are some small differences in how
            # we upsample timeseries versus the integer indexing as I've
            # done here due to floating point arithmetic
            assert_allclose(expected, actual, rtol=1e-16)


class TestDatasetResample:
    def test_resample_and_first(self):
        times = pd.date_range("2000-01-01", freq="6H", periods=10)
        ds = Dataset(
            {
                "foo": (["time", "x", "y"], np.random.randn(10, 5, 3)),
                "bar": ("time", np.random.randn(10), {"meta": "data"}),
                "time": times,
            }
        )

        actual = ds.resample(time="1D").first(keep_attrs=True)
        expected = ds.isel(time=[0, 4, 8])
        assert_identical(expected, actual)

        # upsampling
        expected_time = pd.date_range("2000-01-01", freq="3H", periods=19)
        expected = ds.reindex(time=expected_time)
        actual = ds.resample(time="3H")
        for how in ["mean", "sum", "first", "last"]:
            method = getattr(actual, how)
            result = method()
            assert_equal(expected, result)
        for method in [np.mean]:
            result = actual.reduce(method)
            assert_equal(expected, result)

    def test_resample_min_count(self):
        times = pd.date_range("2000-01-01", freq="6H", periods=10)
        ds = Dataset(
            {
                "foo": (["time", "x", "y"], np.random.randn(10, 5, 3)),
                "bar": ("time", np.random.randn(10), {"meta": "data"}),
                "time": times,
            }
        )
        # inject nan
        ds["foo"] = xr.where(ds["foo"] > 2.0, np.nan, ds["foo"])

        actual = ds.resample(time="1D").sum(min_count=1)
        expected = xr.concat(
            [
                ds.isel(time=slice(i * 4, (i + 1) * 4)).sum("time", min_count=1)
                for i in range(3)
            ],
            dim=actual["time"],
        )
        assert_equal(expected, actual)

    def test_resample_by_mean_with_keep_attrs(self):
        times = pd.date_range("2000-01-01", freq="6H", periods=10)
        ds = Dataset(
            {
                "foo": (["time", "x", "y"], np.random.randn(10, 5, 3)),
                "bar": ("time", np.random.randn(10), {"meta": "data"}),
                "time": times,
            }
        )
        ds.attrs["dsmeta"] = "dsdata"

        resampled_ds = ds.resample(time="1D").mean(keep_attrs=True)
        actual = resampled_ds["bar"].attrs
        expected = ds["bar"].attrs
        assert expected == actual

        actual = resampled_ds.attrs
        expected = ds.attrs
        assert expected == actual

        with pytest.warns(
            UserWarning, match="Passing ``keep_attrs`` to ``resample`` has no effect."
        ):
            ds.resample(time="1D", keep_attrs=True)

    def test_resample_loffset(self):
        times = pd.date_range("2000-01-01", freq="6H", periods=10)
        ds = Dataset(
            {
                "foo": (["time", "x", "y"], np.random.randn(10, 5, 3)),
                "bar": ("time", np.random.randn(10), {"meta": "data"}),
                "time": times,
            }
        )
        ds.attrs["dsmeta"] = "dsdata"

        # Our use of `loffset` may change if we align our API with pandas' changes.
        # ref https://github.com/pydata/xarray/pull/4537
        actual = ds.resample(time="24H", loffset="-12H").mean().bar
        expected_ = ds.bar.to_series().resample("24H").mean()
        expected_.index += to_offset("-12H")
        expected = DataArray.from_series(expected_)
        assert_allclose(actual, expected)

    def test_resample_by_mean_discarding_attrs(self):
        times = pd.date_range("2000-01-01", freq="6H", periods=10)
        ds = Dataset(
            {
                "foo": (["time", "x", "y"], np.random.randn(10, 5, 3)),
                "bar": ("time", np.random.randn(10), {"meta": "data"}),
                "time": times,
            }
        )
        ds.attrs["dsmeta"] = "dsdata"

        resampled_ds = ds.resample(time="1D").mean(keep_attrs=False)

        assert resampled_ds["bar"].attrs == {}
        assert resampled_ds.attrs == {}

    def test_resample_by_last_discarding_attrs(self):
        times = pd.date_range("2000-01-01", freq="6H", periods=10)
        ds = Dataset(
            {
                "foo": (["time", "x", "y"], np.random.randn(10, 5, 3)),
                "bar": ("time", np.random.randn(10), {"meta": "data"}),
                "time": times,
            }
        )
        ds.attrs["dsmeta"] = "dsdata"

        resampled_ds = ds.resample(time="1D").last(keep_attrs=False)

        assert resampled_ds["bar"].attrs == {}
        assert resampled_ds.attrs == {}

    @requires_scipy
    def test_resample_drop_nondim_coords(self):
        xs = np.arange(6)
        ys = np.arange(3)
        times = pd.date_range("2000-01-01", freq="6H", periods=5)
        data = np.tile(np.arange(5), (6, 3, 1))
        xx, yy = np.meshgrid(xs * 5, ys * 2.5)
        tt = np.arange(len(times), dtype=int)
        array = DataArray(data, {"time": times, "x": xs, "y": ys}, ("x", "y", "time"))
        xcoord = DataArray(xx.T, {"x": xs, "y": ys}, ("x", "y"))
        ycoord = DataArray(yy.T, {"x": xs, "y": ys}, ("x", "y"))
        tcoord = DataArray(tt, {"time": times}, ("time",))
        ds = Dataset({"data": array, "xc": xcoord, "yc": ycoord, "tc": tcoord})
        ds = ds.set_coords(["xc", "yc", "tc"])

        # Re-sample
        actual = ds.resample(time="12H").mean("time")
        assert "tc" not in actual.coords

        # Up-sample - filling
        actual = ds.resample(time="1H").ffill()
        assert "tc" not in actual.coords

        # Up-sample - interpolation
        actual = ds.resample(time="1H").interpolate("linear")
        assert "tc" not in actual.coords

    def test_resample_old_api(self):

        times = pd.date_range("2000-01-01", freq="6H", periods=10)
        ds = Dataset(
            {
                "foo": (["time", "x", "y"], np.random.randn(10, 5, 3)),
                "bar": ("time", np.random.randn(10), {"meta": "data"}),
                "time": times,
            }
        )

        with pytest.raises(TypeError, match=r"resample\(\) no longer supports"):
            ds.resample("1D", "time")

        with pytest.raises(TypeError, match=r"resample\(\) no longer supports"):
            ds.resample("1D", dim="time", how="mean")

        with pytest.raises(TypeError, match=r"resample\(\) no longer supports"):
            ds.resample("1D", dim="time")

    def test_resample_ds_da_are_the_same(self):
        time = pd.date_range("2000-01-01", freq="6H", periods=365 * 4)
        ds = xr.Dataset(
            {
                "foo": (("time", "x"), np.random.randn(365 * 4, 5)),
                "time": time,
                "x": np.arange(5),
            }
        )
        assert_identical(
            ds.resample(time="M").mean()["foo"], ds.foo.resample(time="M").mean()
        )

    def test_ds_resample_apply_func_args(self):
        def func(arg1, arg2, arg3=0.0):
            return arg1.mean("time") + arg2 + arg3

        times = pd.date_range("2000", freq="D", periods=3)
        ds = xr.Dataset({"foo": ("time", [1.0, 1.0, 1.0]), "time": times})
        expected = xr.Dataset({"foo": ("time", [3.0, 3.0, 3.0]), "time": times})
        actual = ds.resample(time="D").map(func, args=(1.0,), arg3=1.0)
        assert_identical(expected, actual)


# TODO: move other groupby tests from test_dataset and test_dataarray over here
