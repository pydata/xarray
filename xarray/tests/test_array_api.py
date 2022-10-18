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
    np_arr = xr.DataArray(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, np.nan]]),
        dims=("x", "y"),
        coords={"x": [10, 20]},
    )
    xp_arr = xr.DataArray(
        xp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, np.nan]]),
        dims=("x", "y"),
        coords={"x": [10, 20]},
    )
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
    expected = np_arr.sum()
    actual = xp_arr.sum()
    assert isinstance(actual.data, Array)
    assert_equal(actual, expected)


def test_aggregation_skipna(arrays) -> None:
    np_arr, xp_arr = arrays
    expected = np_arr.sum(skipna=False)
    actual = xp_arr.sum(skipna=False)
    assert isinstance(actual.data, Array)
    assert_equal(actual, expected)


def test_astype(arrays) -> None:
    np_arr, xp_arr = arrays
    expected = np_arr.astype(np.int64)
    actual = xp_arr.astype(np.int64)
    assert actual.dtype == np.int64
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


def test_stack(arrays: tuple[xr.DataArray, xr.DataArray]) -> None:
    np_arr, xp_arr = arrays
    expected = np_arr.stack(z=("x", "y"))
    actual = xp_arr.stack(z=("x", "y"))
    assert isinstance(actual.data, Array)
    assert_equal(actual, expected)


def test_where() -> None:
    np_arr = xr.DataArray(np.array([1, 0]), dims="x")
    xp_arr = xr.DataArray(xp.asarray([1, 0]), dims="x")
    expected = xr.where(np_arr, 1, 0)
    actual = xr.where(xp_arr, 1, 0)
    assert isinstance(actual.data, Array)
    assert_equal(actual, expected)
