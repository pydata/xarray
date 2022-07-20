from typing import Tuple

import pytest

import xarray as xr
from xarray.testing import assert_equal

np = pytest.importorskip("numpy", minversion="1.22")

import numpy.array_api as xp  # isort:skip
from numpy.array_api._array_object import Array  # isort:skip


@pytest.fixture
def arrays() -> Tuple[xr.DataArray, xr.DataArray]:
    np_arr = xr.DataArray(np.ones((2, 3)), dims=("x", "y"), coords={"x": [10, 20]})
    xp_arr = xr.DataArray(xp.ones((2, 3)), dims=("x", "y"), coords={"x": [10, 20]})
    assert isinstance(xp_arr.data, Array)
    return np_arr, xp_arr


def test_arithmetic(arrays) -> None:
    np_arr, xp_arr = arrays
    expected = np_arr + 7
    actual = xp_arr + 7
    assert isinstance(actual.data, Array)
    assert_equal(actual, expected)


def test_aggregation(arrays) -> None:
    np_arr, xp_arr = arrays
    expected = np_arr.sum(skipna=False)
    actual = xp_arr.sum(skipna=False)
    assert isinstance(actual.data, Array)
    assert_equal(actual, expected)


def test_indexing(arrays) -> None:
    np_arr, xp_arr = arrays
    expected = np_arr[:, 0]
    actual = xp_arr[:, 0]
    assert isinstance(actual.data, Array)
    assert_equal(actual, expected)


def test_reorganizing_operation(arrays) -> None:
    np_arr, xp_arr = arrays
    expected = np_arr.transpose()
    actual = xp_arr.transpose()
    assert isinstance(actual.data, Array)
    assert_equal(actual, expected)
