from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import xarray as xr
from xarray.core import indexing

cp = pytest.importorskip("cupy")


@pytest.fixture
def toy_weather_data():
    """Construct the example DataSet from the Toy weather data example.

    https://docs.xarray.dev/en/stable/examples/weather-data.html

    Here we construct the DataSet exactly as shown in the example and then
    convert the numpy arrays to cupy.

    """
    np.random.seed(123)
    times = pd.date_range("2000-01-01", "2001-12-31", name="time")
    annual_cycle = np.sin(2 * np.pi * (times.day_of_year.values / 365.25 - 0.28))

    base = 10 + 15 * annual_cycle.reshape(-1, 1)
    tmin_values = base + 3 * np.random.randn(annual_cycle.size, 3)
    tmax_values = base + 10 + 3 * np.random.randn(annual_cycle.size, 3)

    ds = xr.Dataset(
        {
            "tmin": (("time", "location"), tmin_values),
            "tmax": (("time", "location"), tmax_values),
        },
        {"time": times, "location": ["IA", "IN", "IL"]},
    )

    ds.tmax.data = cp.asarray(ds.tmax.data)
    ds.tmin.data = cp.asarray(ds.tmin.data)

    return ds


def test_cupy_import() -> None:
    """Check the import worked."""
    assert cp


def test_check_data_stays_on_gpu(toy_weather_data) -> None:
    """Perform some operations and check the data stays on the GPU."""
    freeze = (toy_weather_data["tmin"] <= 0).groupby("time.month").mean("time")
    assert isinstance(freeze.data, cp.ndarray)


def test_where() -> None:
    from xarray.core.duck_array_ops import where

    data = cp.zeros(10)

    output = where(data < 1, 1, data).all()
    assert output
    assert isinstance(output, cp.ndarray)


@pytest.mark.parametrize(
    "indexers, expected_dims, expected",
    [
        ({"x": xr.DataArray(cp.array(2))}, ("y",), [12, 13, 14, 15, 16, 17]),
        ({"x": cp.array(2)}, ("y",), [12, 13, 14, 15, 16, 17]),
        (
            {"x": xr.DataArray(cp.array(2)), "y": xr.DataArray(cp.array([0, 2]))},
            ("dim_0",),
            [12, 14],
        ),
        (
            {"x": cp.array(2), "y": xr.DataArray(cp.array([0, 2]), dims="y")},
            ("y",),
            [12, 14],
        ),
    ],
)
def test_isel_0d_cupy_indexer(
    indexers: dict[str, Any], expected_dims: tuple[str, ...], expected: list[int]
) -> None:
    # GH8820
    da = xr.DataArray(cp.arange(24).reshape(4, 6), dims=["x", "y"])
    actual = da.isel(indexers)
    assert actual.dims == expected_dims
    assert isinstance(actual.data, cp.ndarray)
    np.testing.assert_array_equal(actual.data.get(), expected)


def test_isel_cupy_vector_indexer_with_pandas_index() -> None:
    da = xr.DataArray(
        cp.arange(24).reshape(4, 6),
        dims=("x", "y"),
        coords={"x": [10, 20, 30, 40]},
    )

    actual = da.isel(x=cp.array([1, 2]))
    expected_data = cp.arange(24).reshape(4, 6)[cp.array([1, 2])]

    assert actual.get_index("x").equals(pd.Index([20, 30]))
    np.testing.assert_array_equal(actual.data.get(), expected_data.get())


def test_isel_cupy_multidimensional_vector_indexer_with_pandas_index() -> None:
    da = xr.DataArray(
        cp.arange(4),
        dims="x",
        coords={"x": [10, 20, 30, 40]},
    )
    indexer = xr.DataArray(cp.array([[0, 2], [3, 1]]), dims=("row", "col"))

    actual = da.isel(x=indexer)

    assert actual.dims == ("row", "col")
    np.testing.assert_array_equal(actual.data.get(), [[0, 2], [3, 1]])
    np.testing.assert_array_equal(actual.coords["x"].data, [[10, 30], [40, 20]])


def test_isel_cupy_argmax_indexer_with_pandas_index() -> None:
    da = xr.DataArray(
        cp.arange(24).reshape(4, 6),
        dims=("x", "y"),
        coords={"x": [10, 20, 30, 40]},
    )
    indexer = da.argmax("x")

    actual = da.isel(x=indexer)
    expected_indexer = xr.DataArray(indexer.data.get(), dims=indexer.dims)
    expected = da.isel(x=expected_indexer)

    assert actual.dims == expected.dims
    np.testing.assert_array_equal(actual.data.get(), expected.data.get())
    np.testing.assert_array_equal(actual.coords["x"].data, expected.coords["x"].data)


@pytest.mark.parametrize("indexer", [2, slice(1, 3)])
def test_cupy_basic_indexer_assignment(indexer: int | slice) -> None:
    data = cp.arange(12).reshape(4, 3)
    da = xr.DataArray(data.copy(), dims=("x", "y"))
    expected = data.copy()
    expected[indexer] = -1

    da[{"x": indexer}] = -1

    np.testing.assert_array_equal(da.data.get(), expected.get())


def test_cupy_loc_assignment() -> None:
    data = cp.arange(12).reshape(4, 3)
    da = xr.DataArray(data.copy(), dims=("x", "y"), coords={"x": [0, 1, 2, 3]})
    expected = data.copy()
    expected[2] = -1

    da.loc[{"x": 2}] = -1

    np.testing.assert_array_equal(da.data.get(), expected.get())


@pytest.mark.parametrize("value", [True, False])
def test_bool_zero_dim_cupy_dataarray(value: bool) -> None:
    assert bool(xr.DataArray(cp.array(value))) is value


@pytest.mark.parametrize(("method", "expected"), [("idxmax", 1), ("idxmin", 0)])
def test_cupy_idxminmax_skipna(method: str, expected: int) -> None:
    da = xr.DataArray(cp.array([1, 3, 2]), dims="x", coords={"x": [0, 1, 2]})

    result = getattr(da, method)(skipna=True)

    assert result.item() == expected


def test_cupy_assignment_preserves_shape_error() -> None:
    adapter = indexing.as_indexable(cp.zeros(2))
    with pytest.raises(ValueError, match="operands could not be broadcast"):
        adapter[indexing.BasicIndexer((slice(None),))] = cp.arange(3)
