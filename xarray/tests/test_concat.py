from __future__ import annotations

import random
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd
import pytest

from xarray import DataArray, Dataset, Variable, concat
from xarray.core import dtypes, merge
from xarray.core.indexes import PandasIndex
from xarray.tests import (
    InaccessibleArray,
    assert_array_equal,
    assert_equal,
    assert_identical,
    requires_dask,
)
from xarray.tests.test_dataset import create_test_data

if TYPE_CHECKING:
    from xarray.core.types import CombineAttrsOptions, JoinOptions


# helper method to create multiple tests datasets to concat
def create_concat_datasets(
    num_datasets: int = 2, seed: int | None = None, include_day: bool = True
) -> list[Dataset]:
    rng = default_rng(seed)
    lat = rng.standard_normal(size=(1, 4))
    lon = rng.standard_normal(size=(1, 4))
    result = []
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
def create_typed_datasets(
    num_datasets: int = 2, seed: int | None = None
) -> list[Dataset]:
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


def test_concat_missing_var() -> None:
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


def test_concat_missing_multiple_consecutive_var() -> None:
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
            "humidity": (["x", "y", "day"], humidity_result),
            "pressure": (["x", "y", "day"], pressure_result),
        },
        coords={
            "lat": (["x", "y"], datasets[0].lat.values),
            "lon": (["x", "y"], datasets[0].lon.values),
            "day": ["day1", "day2", "day3", "day4", "day5", "day6"],
        },
    )
    result = concat(datasets, dim="day")
    r1 = [var for var in result.data_vars]
    r2 = [var for var in ds_result.data_vars]
    # check the variables orders are the same for the first three variables
    # TODO: Can all variables become deterministic?
    assert r1[:3] == r2[:3]
    assert set(r1[3:]) == set(r2[3:])  # just check availability for the remaining vars
    assert_equal(result, ds_result)


def test_concat_all_empty() -> None:
    ds1 = Dataset()
    ds2 = Dataset()
    result = concat([ds1, ds2], dim="new_dim")

    assert_equal(result, Dataset())


def test_concat_second_empty() -> None:
    ds1 = Dataset(data_vars={"a": ("y", [0.1])}, coords={"x": 0.1})
    ds2 = Dataset(coords={"x": 0.1})

    ds_result = Dataset(data_vars={"a": ("y", [0.1, np.nan])}, coords={"x": 0.1})
    result = concat([ds1, ds2], dim="y")

    assert_equal(result, ds_result)


def test_multiple_missing_variables() -> None:
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


@pytest.mark.parametrize("include_day", [True, False])
def test_concat_multiple_datasets_missing_vars_and_new_dim(include_day: bool) -> None:
    vars_to_drop = [
        "temperature",
        "pressure",
        "humidity",
        "precipitation",
        "cloud cover",
    ]

    datasets = create_concat_datasets(len(vars_to_drop), 123, include_day=include_day)
    # set up the test data
    datasets = [datasets[i].drop_vars(vars_to_drop[i]) for i in range(len(datasets))]

    dim_size = 2 if include_day else 1

    # set up the validation data
    # the below code just drops one var per dataset depending on the location of the
    # dataset in the list and allows us to quickly catch any boundaries cases across
    # the three equivalence classes of beginning, middle and end of the concat list
    result_vars = dict.fromkeys(vars_to_drop, np.array([]))
    for i in range(len(vars_to_drop)):
        for d in range(len(datasets)):
            if d != i:
                if include_day:
                    ds_vals = datasets[d][vars_to_drop[i]].values
                else:
                    ds_vals = datasets[d][vars_to_drop[i]].values[..., None]
                if not result_vars[vars_to_drop[i]].size:
                    result_vars[vars_to_drop[i]] = ds_vals
                else:
                    result_vars[vars_to_drop[i]] = np.concatenate(
                        (
                            result_vars[vars_to_drop[i]],
                            ds_vals,
                        ),
                        axis=-1,
                    )
            else:
                if not result_vars[vars_to_drop[i]].size:
                    result_vars[vars_to_drop[i]] = np.full([1, 4, dim_size], np.nan)
                else:
                    result_vars[vars_to_drop[i]] = np.concatenate(
                        (
                            result_vars[vars_to_drop[i]],
                            np.full([1, 4, dim_size], np.nan),
                        ),
                        axis=-1,
                    )

    ds_result = Dataset(
        data_vars={
            # pressure will be first here since it is first in first dataset and
            # there isn't a good way to determine that temperature should be first
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
        },
    )
    if include_day:
        ds_result = ds_result.assign_coords(
            {"day": ["day" + str(d + 1) for d in range(2 * len(vars_to_drop))]}
        )
    else:
        ds_result = ds_result.transpose("day", "x", "y")

    result = concat(datasets, dim="day")

    r1 = list(result.data_vars.keys())
    r2 = list(ds_result.data_vars.keys())
    assert r1 == r2  # check the variables orders are the same

    assert_equal(result, ds_result)


def test_multiple_datasets_with_missing_variables() -> None:
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
    result_vars = dict.fromkeys(vars_to_drop, np.array([]))
    for i in range(len(vars_to_drop)):
        for d in range(len(datasets)):
            if d != i:
                if not result_vars[vars_to_drop[i]].size:
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
                if not result_vars[vars_to_drop[i]].size:
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


def test_multiple_datasets_with_multiple_missing_variables() -> None:
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
    # check the variables orders are the same for the first three variables
    assert r1[:3] == r2[:3]
    assert set(r1[3:]) == set(r2[3:])  # just check availability for the remaining vars
    assert_equal(result, ds_result)


def test_type_of_missing_fill() -> None:
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


def test_order_when_filling_missing() -> None:
    vars_to_drop_in_first: list[str] = []
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


@pytest.fixture
def concat_var_names() -> Callable:
    # create var names list with one missing value
    def get_varnames(var_cnt: int = 10, list_cnt: int = 10) -> list[list[str]]:
        orig = [f"d{i:02d}" for i in range(var_cnt)]
        var_names = []
        for i in range(0, list_cnt):
            l1 = orig.copy()
            var_names.append(l1)
        return var_names

    return get_varnames


@pytest.fixture
def create_concat_ds() -> Callable:
    def create_ds(
        var_names: list[list[str]],
        dim: bool = False,
        coord: bool = False,
        drop_idx: list[int] | None = None,
    ) -> list[Dataset]:
        out_ds = []
        ds = Dataset()
        ds = ds.assign_coords({"x": np.arange(2)})
        ds = ds.assign_coords({"y": np.arange(3)})
        ds = ds.assign_coords({"z": np.arange(4)})
        for i, dsl in enumerate(var_names):
            vlist = dsl.copy()
            if drop_idx is not None:
                vlist.pop(drop_idx[i])
            foo_data = np.arange(48, dtype=float).reshape(2, 2, 3, 4)
            dsi = ds.copy()
            if coord:
                dsi = ds.assign({"time": (["time"], [i * 2, i * 2 + 1])})
            for k in vlist:
                dsi = dsi.assign({k: (["time", "x", "y", "z"], foo_data.copy())})
            if not dim:
                dsi = dsi.isel(time=0)
            out_ds.append(dsi)
        return out_ds

    return create_ds


@pytest.mark.parametrize("dim", [True, False])
@pytest.mark.parametrize("coord", [True, False])
def test_concat_fill_missing_variables(
    concat_var_names, create_concat_ds, dim: bool, coord: bool
) -> None:
    var_names = concat_var_names()

    random.seed(42)
    drop_idx = [random.randrange(len(vlist)) for vlist in var_names]
    expected = concat(
        create_concat_ds(var_names, dim=dim, coord=coord), dim="time", data_vars="all"
    )
    for i, idx in enumerate(drop_idx):
        if dim:
            expected[var_names[0][idx]][i * 2 : i * 2 + 2] = np.nan
        else:
            expected[var_names[0][idx]][i] = np.nan

    concat_ds = create_concat_ds(var_names, dim=dim, coord=coord, drop_idx=drop_idx)
    actual = concat(concat_ds, dim="time", data_vars="all")

    for name in var_names[0]:
        assert_equal(expected[name], actual[name])
    assert_equal(expected, actual)


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

    with pytest.raises(ValueError, match=r"'y' not present in all datasets."):
        concat([da1, da2], dim="x", coords="all")

    da1 = DataArray([1, 2, 3], dims="x", coords={"x": [1, 2, 3], "y": 1})
    da2 = DataArray([4, 5, 6], dims="x", coords={"x": [4, 5, 6]})
    da3 = DataArray([7, 8, 9], dims="x", coords={"x": [7, 8, 9], "y": 1})
    for coords in ["different", "all"]:
        with pytest.raises(ValueError, match=r"'y' not present in all datasets"):
            concat([da1, da2, da3], dim="x", coords=coords)


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
