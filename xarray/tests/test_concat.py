from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import random

from xarray import DataArray, Dataset, Variable, concat
from xarray.core import dtypes, merge

from . import (
    InaccessibleArray,
    assert_array_equal,
    assert_equal,
    assert_identical,
    raises_regex,
    requires_dask,
)
from .test_dataset import create_test_data


# helper method to create multiple tests datasets to concat
def create_concat_datasets(num_datasets=2, seed=None, include_day=True):
    random.seed(seed)
    result = []
    lat = np.random.randn(1, 4)
    lon = np.random.randn(1, 4)
    for i in range(num_datasets):
        if include_day:
            result.append(
                Dataset(
                    data_vars={
                        "temperature": (["x", "y", "day"], np.random.randn(1, 4, 2)),
                        "pressure": (["x", "y", "day"], np.random.randn(1, 4, 2)),
                        "humidity": (["x", "y", "day"], np.random.randn(1, 4, 2)),
                        "precipitation": (["x", "y", "day"], np.random.randn(1, 4, 2)),
                        "cloud cover": (["x", "y", "day"], np.random.randn(1, 4, 2)),
                    },
                    coords={
                        "lat": (["x", "y"], lat),
                        "lon": (["x", "y"], lon),
                        "day": ["day" + str(i * 2 + 1), "day" + str(i * 2 + 2)],
                    },
                )
            )
        else:
            result.append(
                Dataset(
                    data_vars={
                        "temperature": (["x", "y"], np.random.randn(1, 4)),
                        "pressure": (["x", "y"], np.random.randn(1, 4)),
                        "humidity": (["x", "y"], np.random.randn(1, 4)),
                        "precipitation": (["x", "y"], np.random.randn(1, 4)),
                        "cloud cover": (["x", "y"], np.random.randn(1, 4)),
                    },
                    coords={"lat": (["x", "y"], lat), "lon": (["x", "y"], lon)},
                )
            )

    return result


# helper method to create multiple tests datasets to concat with specific types
def create_typed_datasets(num_datasets=2, seed=None):
    random.seed(seed)
    var_strings = ["a", "b", "c", "d", "e", "f", "g", "h"]
    result = []
    lat = np.random.randn(1, 4)
    lon = np.random.randn(1, 4)
    for i in range(num_datasets):
        result.append(
            Dataset(
                data_vars={
                    "float": (["x", "y", "day"], np.random.randn(1, 4, 2)),
                    "float2": (["x", "y", "day"], np.random.randn(1, 4, 2)),
                    "string": (
                        ["x", "y", "day"],
                        np.random.choice(var_strings, (1, 4, 2)),
                    ),
                    "int": (["x", "y", "day"], np.random.randint(0, 10, (1, 4, 2))),
                    "datetime64": (
                        ["x", "y", "day"],
                        np.arange(
                            np.datetime64("2017-01-01"), np.datetime64("2017-01-09")
                        ).reshape(1, 4, 2),
                    ),
                    "timedelta64": (
                        ["x", "y", "day"],
                        np.reshape([pd.Timedelta(days=i) for i in range(8)], [1, 4, 2]),
                    ),
                },
                coords={
                    "lat": (["x", "y"], lat),
                    "lon": (["x", "y"], lon),
                    "day": ["day" + str(i * 2 + 1), "day" + str(i * 2 + 2)],
                },
            )
        )
    return result


def test_concat_compat():
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
        assert "y" not in result[var]

    with raises_regex(ValueError, "coordinates in some datasets but not others"):
        concat([ds1, ds2], dim="q")

    with raises_regex(ValueError, "coordinates in some datasets but not others"):
        concat([ds2, ds1], dim="q")


def test_concat_missing_var():
    datasets = create_concat_datasets(2, 123)
    vars_to_drop = ["humidity", "precipitation", "cloud cover"]
    datasets[0] = datasets[0].drop_vars(vars_to_drop)
    datasets[1] = datasets[1].drop_vars(vars_to_drop + ["pressure"])

    temperature_result = np.concatenate(
        (datasets[0].temperature.values, datasets[1].temperature.values), axis=2
    )
    pressure_result = np.concatenate(
        (datasets[0].pressure.values, np.full([1, 4, 2], np.nan)), axis=2
    )
    ds_result = Dataset(
        data_vars={
            "temperature": (["x", "y", "day"], temperature_result),
            "pressure": (["x", "y", "day"], pressure_result),
        },
        coords={
            "lat": (["x", "y"], datasets[0].lat.values),
            "lon": (["x", "y"], datasets[0].lon.values),
            "day": ["day1", "day2", "day3", "day4"],
        },
    )
    result = concat(datasets, dim="day")

    r1 = list(result.data_vars.keys())
    r2 = list(ds_result.data_vars.keys())
    assert r1 == r2  # check the variables orders are the same

    assert_equal(result, ds_result)


def test_concat_missing_muliple_consecutive_var():
    datasets = create_concat_datasets(3, 123)
    vars_to_drop = ["pressure", "humidity"]
    datasets[0] = datasets[0].drop_vars(vars_to_drop)
    datasets[1] = datasets[1].drop_vars(vars_to_drop)

    temperature_result = np.concatenate(
        (
            datasets[0].temperature.values,
            datasets[1].temperature.values,
            datasets[2].temperature.values,
        ),
        axis=2,
    )
    pressure_result = np.concatenate(
        (
            np.full([1, 4, 2], np.nan),
            np.full([1, 4, 2], np.nan),
            datasets[2].pressure.values,
        ),
        axis=2,
    )
    humidity_result = np.concatenate(
        (
            np.full([1, 4, 2], np.nan),
            np.full([1, 4, 2], np.nan),
            datasets[2].humidity.values,
        ),
        axis=2,
    )
    precipitation_result = np.concatenate(
        (
            datasets[0].precipitation.values,
            datasets[1].precipitation.values,
            datasets[2].precipitation.values,
        ),
        axis=2,
    )
    cloudcover_result = np.concatenate(
        (
            datasets[0]["cloud cover"].values,
            datasets[1]["cloud cover"].values,
            datasets[2]["cloud cover"].values,
        ),
        axis=2,
    )

    ds_result = Dataset(
        data_vars={
            "temperature": (["x", "y", "day"], temperature_result),
            "precipitation": (["x", "y", "day"], precipitation_result),
            "cloud cover": (["x", "y", "day"], cloudcover_result),
            "pressure": (["x", "y", "day"], pressure_result),
            "humidity": (["x", "y", "day"], humidity_result),
        },
        coords={
            "lat": (["x", "y"], datasets[0].lat.values),
            "lon": (["x", "y"], datasets[0].lon.values),
            "day": ["day1", "day2", "day3", "day4", "day5", "day6"],
        },
    )
    result = concat(datasets, dim="day")
    r1 = list(result.data_vars.keys())
    r2 = list(ds_result.data_vars.keys())
    assert r1 == r2  # check the variables orders are the same
    assert_equal(result, ds_result)


def test_concat_all_empty():
    ds1 = Dataset()
    ds2 = Dataset()
    result = concat([ds1, ds2], dim="new_dim")

    assert_equal(result, Dataset())


def test_concat_second_empty():
    ds1 = Dataset(data_vars={"a": ("y", [0.1])}, coords={"x": 0.1})
    ds2 = Dataset(coords={"x": 0.1})

    ds_result = Dataset(data_vars={"a": ("y", [0.1, np.nan])}, coords={"x": 0.1})
    result = concat([ds1, ds2], dim="y")

    assert_equal(result, ds_result)


def test_multiple_missing_variables():
    datasets = create_concat_datasets(2, 123)
    vars_to_drop = ["pressure", "cloud cover"]
    datasets[1] = datasets[1].drop_vars(vars_to_drop)

    temperature_result = np.concatenate(
        (datasets[0].temperature.values, datasets[1].temperature.values), axis=2
    )
    pressure_result = np.concatenate(
        (datasets[0].pressure.values, np.full([1, 4, 2], np.nan)), axis=2
    )
    humidity_result = np.concatenate(
        (datasets[0].humidity.values, datasets[1].humidity.values), axis=2
    )
    precipitation_result = np.concatenate(
        (datasets[0].precipitation.values, datasets[1].precipitation.values), axis=2
    )
    cloudcover_result = np.concatenate(
        (datasets[0]["cloud cover"].values, np.full([1, 4, 2], np.nan)), axis=2
    )
    ds_result = Dataset(
        data_vars={
            "temperature": (["x", "y", "day"], temperature_result),
            "pressure": (["x", "y", "day"], pressure_result),
            "humidity": (["x", "y", "day"], humidity_result),
            "precipitation": (["x", "y", "day"], precipitation_result),
            "cloud cover": (["x", "y", "day"], cloudcover_result),
        },
        coords={
            "lat": (["x", "y"], datasets[0].lat.values),
            "lon": (["x", "y"], datasets[0].lon.values),
            "day": ["day1", "day2", "day3", "day4"],
        },
    )
    result = concat(datasets, dim="day")

    r1 = list(result.data_vars.keys())
    r2 = list(ds_result.data_vars.keys())
    assert r1 == r2  # check the variables orders are the same

    assert_equal(result, ds_result)


@pytest.mark.xfail(strict=True)
def test_concat_multiple_datasets_missing_vars_and_new_dim():
    vars_to_drop = [
        "temperature",
        "pressure",
        "humidity",
        "precipitation",
        "cloud cover",
    ]
    datasets = create_concat_datasets(len(vars_to_drop), 123, include_day=False)
    # set up the test data
    datasets = [datasets[i].drop_vars(vars_to_drop[i]) for i in range(len(datasets))]

    # set up the validation data
    # the below code just drops one var per dataset depending on the location of the
    # dataset in the list and allows us to quickly catch any boundaries cases across
    # the three equivalence classes of beginning, middle and end of the concat list
    result_vars = dict.fromkeys(vars_to_drop)
    for i in range(len(vars_to_drop)):
        for d in range(len(datasets)):
            if d != i:
                if result_vars[vars_to_drop[i]] is None:
                    result_vars[vars_to_drop[i]] = datasets[d][vars_to_drop[i]].values
                else:
                    result_vars[vars_to_drop[i]] = np.concatenate(
                        (
                            result_vars[vars_to_drop[i]],
                            datasets[d][vars_to_drop[i]].values,
                        ),
                        axis=1,
                    )
            else:
                if result_vars[vars_to_drop[i]] is None:
                    result_vars[vars_to_drop[i]] = np.full([1, 4], np.nan)
                else:
                    result_vars[vars_to_drop[i]] = np.concatenate(
                        (result_vars[vars_to_drop[i]], np.full([1, 4], np.nan)), axis=1,
                    )
    # TODO: this test still has two unexpected errors:

    # 1: concat throws a mergeerror expecting the temperature values to be the same, this doesn't seem to be correct in this case
    #   as we are concating on new dims
    # 2: if the values are the same for a variable (working around #1) then it will likely not correct add the new dim to the first variable
    #   the resulting set

    # ds_result = Dataset(
    #     data_vars={
    #         # pressure will be first in this since the first dataset is missing this var
    #         # and there isn't a good way to determine that this should be first
    #         #this also means temperature will be last as the first data vars will
    #         #determine the order for all that exist in that dataset
    #         "pressure": (["x", "y", "day"], result_vars["pressure"]),
    #         "humidity": (["x", "y", "day"], result_vars["humidity"]),
    #         "precipitation": (["x", "y", "day"], result_vars["precipitation"]),
    #         "cloud cover": (["x", "y", "day"], result_vars["cloud cover"]),
    #         "temperature": (["x", "y", "day"], result_vars["temperature"]),
    #     },
    #     coords={
    #         "lat": (["x", "y"], datasets[0].lat.values),
    #         "lon": (["x", "y"], datasets[0].lon.values),
    #       #  "day": ["day" + str(d + 1) for d in range(2 * len(vars_to_drop))],
    #     },
    # )

    # result = concat(datasets, dim="day")
    # r1 = list(result.data_vars.keys())
    # r2 = list(ds_result.data_vars.keys())
    # assert r1 == r2  # check the variables orders are the same

    # assert_equal(result, ds_result)


def test_multiple_datasets_with_missing_variables():
    vars_to_drop = [
        "temperature",
        "pressure",
        "humidity",
        "precipitation",
        "cloud cover",
    ]
    datasets = create_concat_datasets(len(vars_to_drop), 123)
    # set up the test data
    datasets = [datasets[i].drop_vars(vars_to_drop[i]) for i in range(len(datasets))]

    # set up the validation data
    # the below code just drops one var per dataset depending on the location of the
    # dataset in the list and allows us to quickly catch any boundaries cases across
    # the three equivalence classes of beginning, middle and end of the concat list
    result_vars = dict.fromkeys(vars_to_drop)
    for i in range(len(vars_to_drop)):
        for d in range(len(datasets)):
            if d != i:
                if result_vars[vars_to_drop[i]] is None:
                    result_vars[vars_to_drop[i]] = datasets[d][vars_to_drop[i]].values
                else:
                    result_vars[vars_to_drop[i]] = np.concatenate(
                        (
                            result_vars[vars_to_drop[i]],
                            datasets[d][vars_to_drop[i]].values,
                        ),
                        axis=2,
                    )
            else:
                if result_vars[vars_to_drop[i]] is None:
                    result_vars[vars_to_drop[i]] = np.full([1, 4, 2], np.nan)
                else:
                    result_vars[vars_to_drop[i]] = np.concatenate(
                        (result_vars[vars_to_drop[i]], np.full([1, 4, 2], np.nan)),
                        axis=2,
                    )

    ds_result = Dataset(
        data_vars={
            # pressure will be first in this since the first dataset is missing this var
            # and there isn't a good way to determine that this should be first
            # this also means temperature will be last as the first data vars will
            # determine the order for all that exist in that dataset
            "pressure": (["x", "y", "day"], result_vars["pressure"]),
            "humidity": (["x", "y", "day"], result_vars["humidity"]),
            "precipitation": (["x", "y", "day"], result_vars["precipitation"]),
            "cloud cover": (["x", "y", "day"], result_vars["cloud cover"]),
            "temperature": (["x", "y", "day"], result_vars["temperature"]),
        },
        coords={
            "lat": (["x", "y"], datasets[0].lat.values),
            "lon": (["x", "y"], datasets[0].lon.values),
            "day": ["day" + str(d + 1) for d in range(2 * len(vars_to_drop))],
        },
    )
    result = concat(datasets, dim="day")

    r1 = list(result.data_vars.keys())
    r2 = list(ds_result.data_vars.keys())
    assert r1 == r2  # check the variables orders are the same

    assert_equal(result, ds_result)


def test_multiple_datasets_with_multiple_missing_variables():
    vars_to_drop_in_first = ["temperature", "pressure"]
    vars_to_drop_in_second = ["humidity", "precipitation", "cloud cover"]
    datasets = create_concat_datasets(2, 123)
    # set up the test data
    datasets[0] = datasets[0].drop_vars(vars_to_drop_in_first)
    datasets[1] = datasets[1].drop_vars(vars_to_drop_in_second)

    temperature_result = np.concatenate(
        (np.full([1, 4, 2], np.nan), datasets[1].temperature.values), axis=2
    )
    pressure_result = np.concatenate(
        (np.full([1, 4, 2], np.nan), datasets[1].pressure.values), axis=2
    )
    humidity_result = np.concatenate(
        (datasets[0].humidity.values, np.full([1, 4, 2], np.nan)), axis=2
    )
    precipitation_result = np.concatenate(
        (datasets[0].precipitation.values, np.full([1, 4, 2], np.nan)), axis=2
    )
    cloudcover_result = np.concatenate(
        (datasets[0]["cloud cover"].values, np.full([1, 4, 2], np.nan)), axis=2
    )
    ds_result = Dataset(
        data_vars={
            "humidity": (["x", "y", "day"], humidity_result),
            "precipitation": (["x", "y", "day"], precipitation_result),
            "cloud cover": (["x", "y", "day"], cloudcover_result),
            # these two are at the end of the expected as they are missing from the first
            # dataset in the concat list
            "temperature": (["x", "y", "day"], temperature_result),
            "pressure": (["x", "y", "day"], pressure_result),
        },
        coords={
            "lat": (["x", "y"], datasets[0].lat.values),
            "lon": (["x", "y"], datasets[0].lon.values),
            "day": ["day1", "day2", "day3", "day4"],
        },
    )
    result = concat(datasets, dim="day")

    r1 = list(result.data_vars.keys())
    r2 = list(ds_result.data_vars.keys())
    assert r1 == r2  # check the variables orders are the same

    assert_equal(result, ds_result)


def test_type_of_missing_fill():
    datasets = create_typed_datasets(2, 123)

    vars = ["float", "float2", "string", "int", "datetime64", "timedelta64"]

    # set up the test data
    datasets[1] = datasets[1].drop_vars(vars[1:])

    float_result = np.concatenate(
        (datasets[0].float.values, datasets[1].float.values), axis=2
    )
    float2_result = np.concatenate(
        (datasets[0].float2.values, np.full([1, 4, 2], np.nan)), axis=2
    )
    # to correctly create the expected dataset we need to ensure we promote the string array to
    # object type before filling as it will be promoted to that in the concat case.
    # this matches the behavior of pandas
    string_values = datasets[0].string.values
    string_values = string_values.astype(object)
    string_result = np.concatenate((string_values, np.full([1, 4, 2], np.nan)), axis=2)
    datetime_result = np.concatenate(
        (datasets[0].datetime64.values, np.full([1, 4, 2], np.datetime64("NaT"))),
        axis=2,
    )
    timedelta_result = np.concatenate(
        (datasets[0].timedelta64.values, np.full([1, 4, 2], np.timedelta64("NaT"))),
        axis=2,
    )
    # string_result = string_result.astype(object)
    int_result = np.concatenate(
        (datasets[0].int.values, np.full([1, 4, 2], np.nan)), axis=2
    )
    ds_result = Dataset(
        data_vars={
            "float": (["x", "y", "day"], float_result),
            "float2": (["x", "y", "day"], float2_result),
            "string": (["x", "y", "day"], string_result),
            "int": (["x", "y", "day"], int_result),
            "datetime64": (["x", "y", "day"], datetime_result),
            "timedelta64": (["x", "y", "day"], timedelta_result),
        },
        coords={
            "lat": (["x", "y"], datasets[0].lat.values),
            "lon": (["x", "y"], datasets[0].lon.values),
            "day": ["day1", "day2", "day3", "day4"],
        },
    )
    result = concat(datasets, dim="day", fill_value=dtypes.NA)

    assert_equal(result, ds_result)

    # test in the reverse order
    float_result_rev = np.concatenate(
        (datasets[1].float.values, datasets[0].float.values), axis=2
    )
    float2_result_rev = np.concatenate(
        (np.full([1, 4, 2], np.nan), datasets[0].float2.values), axis=2
    )
    string_result_rev = np.concatenate(
        (np.full([1, 4, 2], np.nan), string_values), axis=2
    )
    datetime_result_rev = np.concatenate(
        (np.full([1, 4, 2], np.datetime64("NaT")), datasets[0].datetime64.values),
        axis=2,
    )
    timedelta_result_rev = np.concatenate(
        (np.full([1, 4, 2], np.timedelta64("NaT")), datasets[0].timedelta64.values),
        axis=2,
    )
    int_result_rev = np.concatenate(
        (np.full([1, 4, 2], np.nan), datasets[0].int.values), axis=2
    )
    ds_result_rev = Dataset(
        data_vars={
            "float": (["x", "y", "day"], float_result_rev),
            "float2": (["x", "y", "day"], float2_result_rev),
            "string": (["x", "y", "day"], string_result_rev),
            "int": (["x", "y", "day"], int_result_rev),
            "datetime64": (["x", "y", "day"], datetime_result_rev),
            "timedelta64": (["x", "y", "day"], timedelta_result_rev),
        },
        coords={
            "lat": (["x", "y"], datasets[0].lat.values),
            "lon": (["x", "y"], datasets[0].lon.values),
            "day": ["day3", "day4", "day1", "day2"],
        },
    )
    result_rev = concat(datasets[::-1], dim="day", fill_value=dtypes.NA)

    assert_equal(result_rev, ds_result_rev)


def test_order_when_filling_missing():
    vars_to_drop_in_first = []
    # drop middle
    vars_to_drop_in_second = ["humidity"]
    datasets = create_concat_datasets(2, 123)
    # set up the test data
    datasets[0] = datasets[0].drop_vars(vars_to_drop_in_first)
    datasets[1] = datasets[1].drop_vars(vars_to_drop_in_second)

    temperature_result = np.concatenate(
        (datasets[0].temperature.values, datasets[1].temperature.values), axis=2
    )
    pressure_result = np.concatenate(
        (datasets[0].pressure.values, datasets[1].pressure.values), axis=2
    )
    humidity_result = np.concatenate(
        (datasets[0].humidity.values, np.full([1, 4, 2], np.nan)), axis=2
    )
    precipitation_result = np.concatenate(
        (datasets[0].precipitation.values, datasets[1].precipitation.values), axis=2
    )
    cloudcover_result = np.concatenate(
        (datasets[0]["cloud cover"].values, datasets[1]["cloud cover"].values), axis=2
    )
    ds_result = Dataset(
        data_vars={
            "temperature": (["x", "y", "day"], temperature_result),
            "pressure": (["x", "y", "day"], pressure_result),
            "precipitation": (["x", "y", "day"], precipitation_result),
            "cloud cover": (["x", "y", "day"], cloudcover_result),
            "humidity": (["x", "y", "day"], humidity_result),
        },
        coords={
            "lat": (["x", "y"], datasets[0].lat.values),
            "lon": (["x", "y"], datasets[0].lon.values),
            "day": ["day1", "day2", "day3", "day4"],
        },
    )
    result = concat(datasets, dim="day")

    assert_equal(result, ds_result)

    result_keys = [
        "temperature",
        "pressure",
        "humidity",
        "precipitation",
        "cloud cover",
    ]
    result_index = 0
    for k in result.data_vars.keys():
        assert k == result_keys[result_index]
        result_index += 1

    result_keys_rev = [
        "temperature",
        "pressure",
        "precipitation",
        "cloud cover",
        "humidity",
    ]
    # test order when concat in reversed order
    rev_result = concat(datasets[::-1], dim="day")
    result_index = 0
    for k in rev_result.data_vars.keys():
        assert k == result_keys_rev[result_index]
        result_index += 1


class TestConcatDataset:
    @pytest.fixture
    def data(self):
        return create_test_data().drop_dims("dim3")

    def rectify_dim_order(self, data, dataset):
        # return a new dataset with all variable dimensions transposed into
        # the order in which they are found in `data`
        return Dataset(
            {k: v.transpose(*data[k].dims) for k, v in dataset.data_vars.items()},
            dataset.coords,
            attrs=dataset.attrs,
        )

    @pytest.mark.parametrize("coords", ["different", "minimal"])
    @pytest.mark.parametrize("dim", ["dim1", "dim2"])
    def test_concat_simple(self, data, dim, coords):
        datasets = [g for _, g in data.groupby(dim, squeeze=False)]
        assert_identical(data, concat(datasets, dim, coords=coords))

    def test_concat_merge_variables_present_in_some_datasets(self, data):
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

    def test_concat_2(self, data):
        dim = "dim2"
        datasets = [g for _, g in data.groupby(dim, squeeze=True)]
        concat_over = [k for k, v in data.coords.items() if dim in v.dims and k != dim]
        actual = concat(datasets, data[dim], coords=concat_over)
        assert_identical(data, self.rectify_dim_order(data, actual))

    @pytest.mark.parametrize("coords", ["different", "minimal", "all"])
    @pytest.mark.parametrize("dim", ["dim1", "dim2"])
    def test_concat_coords_kwarg(self, data, dim, coords):
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

    def test_concat(self, data):
        split_data = [
            data.isel(dim1=slice(3)),
            data.isel(dim1=3),
            data.isel(dim1=slice(4, None)),
        ]
        assert_identical(data, concat(split_data, "dim1"))

    def test_concat_dim_precedence(self, data):
        # verify that the dim argument takes precedence over
        # concatenating dataset variables of the same name
        dim = (2 * data["dim1"]).rename("dim1")
        datasets = [g for _, g in data.groupby("dim1", squeeze=False)]
        expected = data.copy()
        expected["dim1"] = dim
        assert_identical(expected, concat(datasets, dim))

    def test_concat_data_vars(self):
        data = Dataset({"foo": ("x", np.random.randn(10))})
        objs = [data.isel(x=slice(5)), data.isel(x=slice(5, None))]
        for data_vars in ["minimal", "different", "all", [], ["foo"]]:
            actual = concat(objs, dim="x", data_vars=data_vars)
            assert_identical(data, actual)

    def test_concat_coords(self):
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
            with raises_regex(merge.MergeError, "conflicting values"):
                concat(objs, dim="x", coords=coords)

    def test_concat_constant_index(self):
        # GH425
        ds1 = Dataset({"foo": 1.5}, {"y": 1})
        ds2 = Dataset({"foo": 2.5}, {"y": 1})
        expected = Dataset({"foo": ("y", [1.5, 2.5]), "y": [1, 1]})
        for mode in ["different", "all", ["foo"]]:
            actual = concat([ds1, ds2], "y", data_vars=mode)
            assert_identical(expected, actual)
        with raises_regex(merge.MergeError, "conflicting values"):
            # previously dim="y", and raised error which makes no sense.
            # "foo" has dimension "y" so minimal should concatenate it?
            concat([ds1, ds2], "new_dim", data_vars="minimal")

    def test_concat_size0(self):
        data = create_test_data()
        split_data = [data.isel(dim1=slice(0, 0)), data]
        actual = concat(split_data, "dim1")
        assert_identical(data, actual)

        actual = concat(split_data[::-1], "dim1")
        assert_identical(data, actual)

    def test_concat_autoalign(self):
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
        data = create_test_data()
        split_data = [data.isel(dim1=slice(3)), data.isel(dim1=slice(3, None))]

        with raises_regex(ValueError, "must supply at least one"):
            concat([], "dim1")

        with raises_regex(ValueError, "Cannot specify both .*='different'"):
            concat(
                [data, data], dim="concat_dim", data_vars="different", compat="override"
            )

        with raises_regex(ValueError, "must supply at least one"):
            concat([], "dim1")

        with raises_regex(ValueError, "are not coordinates"):
            concat([data, data], "new_dim", coords=["not_found"])

        with raises_regex(ValueError, "global attributes not"):
            data0, data1 = deepcopy(split_data)
            data1.attrs["foo"] = "bar"
            concat([data0, data1], "dim1", compat="identical")
        assert_identical(data, concat([data0, data1], "dim1", compat="equals"))

        with raises_regex(ValueError, "compat.* invalid"):
            concat(split_data, "dim1", compat="foobar")

        with raises_regex(ValueError, "unexpected value for"):
            concat([data, data], "new_dim", coords="foobar")

        with raises_regex(ValueError, "coordinate in some datasets but not others"):
            concat([Dataset({"x": 0}), Dataset({"x": [1]})], dim="z")

        with raises_regex(ValueError, "coordinate in some datasets but not others"):
            concat([Dataset({"x": 0}), Dataset({}, {"x": 1})], dim="z")

    def test_concat_join_kwarg(self):
        ds1 = Dataset({"a": (("x", "y"), [[0]])}, coords={"x": [0], "y": [0]})
        ds2 = Dataset({"a": (("x", "y"), [[0]])}, coords={"x": [1], "y": [0.0001]})

        expected = {}
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

        with raises_regex(ValueError, "indexes along dimension 'y'"):
            actual = concat([ds1, ds2], join="exact", dim="x")

        for join in expected:
            actual = concat([ds1, ds2], join=join, dim="x")
            assert_equal(actual, expected[join])

    def test_concat_promote_shape(self):
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

    def test_concat_do_not_promote(self):
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

    def test_concat_dim_is_variable(self):
        objs = [Dataset({"x": 0}), Dataset({"x": 1})]
        coord = Variable("y", [3, 4])
        expected = Dataset({"x": ("y", [0, 1]), "y": [3, 4]})
        actual = concat(objs, coord)
        assert_identical(actual, expected)

    def test_concat_multiindex(self):
        x = pd.MultiIndex.from_product([[1, 2, 3], ["a", "b"]])
        expected = Dataset({"x": x})
        actual = concat(
            [expected.isel(x=slice(2)), expected.isel(x=slice(2, None))], "x"
        )
        assert expected.equals(actual)
        assert isinstance(actual.x.to_index(), pd.MultiIndex)

    # TODO add parameter for missing var
    @pytest.mark.parametrize("fill_value", [dtypes.NA, 2, 2.0])
    def test_concat_fill_value(self, fill_value):
        datasets = [
            Dataset({"a": ("x", [2, 3]), "x": [1, 2]}),
            Dataset({"a": ("x", [1, 2]), "x": [0, 1]}),
        ]

        if fill_value == dtypes.NA:
            # if we supply the default, we expect the missing value for a
            # float array
            fill_value_expected = np.nan
        else:
            fill_value_expected = fill_value

        expected = Dataset(
            {
                "a": (
                    ("t", "x"),
                    [[fill_value_expected, 2, 3], [1, 2, fill_value_expected]],
                )
            },
            {"x": [0, 1, 2]},
        )
        actual = concat(datasets, dim="t", fill_value=fill_value)
        assert_identical(actual, expected)

        # check that the dtype is as expected
        assert expected.a.dtype == type(fill_value_expected)


class TestConcatDataArray:
    def test_concat(self):
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
        stacked = concat(grouped, ds.indexes["x"])
        assert_identical(foo, stacked)

        actual = concat([foo[0], foo[1]], pd.Index([0, 1])).reset_coords(drop=True)
        expected = foo[:2].rename({"x": "concat_dim"})
        assert_identical(expected, actual)

        # TODO: is it really correct to expect the new dim to be concat_dim in this case
        # I propose its likely better to throw an exception
        actual = concat([foo[0], foo[1]], [0, 1]).reset_coords(drop=True)
        expected = foo[:2].rename({"x": "concat_dim"})
        assert_identical(expected, actual)

        with raises_regex(ValueError, "not identical"):
            concat([foo, bar], dim="w", compat="identical")

        with raises_regex(ValueError, "not a valid argument"):
            concat([foo, bar], dim="w", data_vars="minimal")

    def test_concat_encoding(self):
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
    def test_concat_lazy(self):
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
    def test_concat_fill_value(self, fill_value):
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

    def test_concat_join_kwarg(self):
        ds1 = Dataset(
            {"a": (("x", "y"), [[0]])}, coords={"x": [0], "y": [0]}
        ).to_array()
        ds2 = Dataset(
            {"a": (("x", "y"), [[0]])}, coords={"x": [1], "y": [0.0001]}
        ).to_array()

        expected = {}
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

        with raises_regex(ValueError, "indexes along dimension 'y'"):
            actual = concat([ds1, ds2], join="exact", dim="x")

        for join in expected:
            actual = concat([ds1, ds2], join=join, dim="x")
            assert_equal(actual, expected[join].to_array())
