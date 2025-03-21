import numpy as np

import xarray as xr
from xarray.indexes import RangeIndex
from xarray.tests import assert_allclose, assert_equal, assert_identical


def create_dataset_arange(start: float, stop: float, step: float, dim: str = "x"):
    index = RangeIndex.arange(dim, dim, start, stop, step)
    return xr.Dataset(coords=xr.Coordinates.from_xindex(index))


def test_range_index_arange() -> None:
    index = RangeIndex.arange("x", "x", 0.0, 1.0, 0.1)
    actual = xr.Coordinates.from_xindex(index)
    expected = xr.Coordinates({"x": np.arange(0.0, 1.0, 0.1)})
    assert_equal(actual, expected, check_default_indexes=False)


def test_range_index_linspace() -> None:
    index = RangeIndex.linspace("x", "x", 0.0, 1.0, num=10, endpoint=False)
    actual = xr.Coordinates.from_xindex(index)
    expected = xr.Coordinates({"x": np.linspace(0.0, 1.0, num=10, endpoint=False)})
    assert_equal(actual, expected)

    index = RangeIndex.linspace("x", "x", 0.0, 1.0, num=11, endpoint=True)
    actual = xr.Coordinates.from_xindex(index)
    expected = xr.Coordinates({"x": np.linspace(0.0, 1.0, num=11, endpoint=True)})
    assert_allclose(actual, expected, check_default_indexes=False)


def test_range_index_isel() -> None:
    ds = create_dataset_arange(0.0, 1.0, 0.1)

    actual = ds.isel(x=slice(1, 3))
    expected = create_dataset_arange(0.1, 0.3, 0.1)
    assert_identical(actual, expected, check_default_indexes=False)
