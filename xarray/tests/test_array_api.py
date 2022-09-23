from __future__ import annotations

import warnings

import pytest

import xarray as xr
from xarray.testing import assert_equal

np = pytest.importorskip("numpy", minversion="1.22")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import numpy.array_api as xp  # isort:skip
    from numpy.array_api._array_object import Array  # isort:skip


@pytest.fixture
def arrays() -> tuple[xr.DataArray, xr.DataArray]:
    np_arr = xr.DataArray(np.ones((2, 3)), dims=("x", "y"), coords={"x": [10, 20]})
    xp_arr = xr.DataArray(xp.ones((2, 3)), dims=("x", "y"), coords={"x": [10, 20]})
    assert isinstance(xp_arr.data, Array)
    return np_arr, xp_arr


def test_arithmetic(arrays: tuple[xr.DataArray, xr.DataArray]) -> None:
    np_arr, xp_arr = arrays
    expected = np_arr + 7
    actual = xp_arr + 7
    assert isinstance(actual.data, Array)
    assert_equal(actual, expected)


def test_aggregation(arrays: tuple[xr.DataArray, xr.DataArray]) -> None:
    np_arr, xp_arr = arrays
    expected = np_arr.sum(skipna=False)
    actual = xp_arr.sum(skipna=False)
    assert isinstance(actual.data, Array)
    assert_equal(actual, expected)


def test_indexing(arrays: tuple[xr.DataArray, xr.DataArray]) -> None:
    np_arr, xp_arr = arrays
    expected = np_arr[:, 0]
    actual = xp_arr[:, 0]
    assert isinstance(actual.data, Array)
    assert_equal(actual, expected)


def test_properties(arrays: tuple[xr.DataArray, xr.DataArray]) -> None:
    np_arr, xp_arr = arrays
    assert np_arr.nbytes == np_arr.data.nbytes
    assert xp_arr.nbytes == np_arr.data.nbytes


def test_reorganizing_operation(arrays: tuple[xr.DataArray, xr.DataArray]) -> None:
    np_arr, xp_arr = arrays
    expected = np_arr.transpose()
    actual = xp_arr.transpose()
    assert isinstance(actual.data, Array)
    assert_equal(actual, expected)
