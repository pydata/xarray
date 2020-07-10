import numpy as np
import pandas as pd
import pytest

import xarray as xr

cp = pytest.importorskip("cupy")


@pytest.fixture
def toy_weather_data():
    np.random.seed(123)
    times = pd.date_range("2000-01-01", "2001-12-31", name="time")
    annual_cycle = np.sin(2 * np.pi * (times.dayofyear.values / 365.25 - 0.28))

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


def test_cupy_import():
    assert cp


def test_check_mean(toy_weather_data):
    freeze = (toy_weather_data["tmin"] <= 0).groupby("time.month").mean("time")
    assert isinstance(freeze.data, cp.core.core.ndarray)
