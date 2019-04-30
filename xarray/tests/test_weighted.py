import pytest

import numpy as np

import xarray as xr
from xarray import (
    DataArray,)

from xarray.tests import assert_equal, raises_regex


def test_weigted_non_DataArray_weights():

    da = DataArray([1, 2])
    with raises_regex(AssertionError, "'weights' must be a DataArray"):
        da.weighted([1, 2])


@pytest.mark.parametrize('weights', ([1, 2], [np.nan, 2], [np.nan, np.nan]))
def test_weighted_weights_nan_replaced(weights):
    # make sure nans are removed from weights

    da = DataArray([1, 2])

    expected = DataArray(weights).fillna(0.)
    result = da.weighted(DataArray(weights)).weights

    assert_equal(expected, result)


@pytest.mark.parametrize(('weights', 'expected'), (([1, 2], 3),
                                                   ([0, 2], 2),
                                                   ([0, 0], np.nan),
                                                   ([-1, 1], np.nan)))
def test_weigted_sum_of_weights_no_nan(weights, expected):

    da = DataArray([1, 2])
    weights = DataArray(weights)
    result = da.weighted(weights).sum_of_weights()

    expected = DataArray(expected)

    assert_equal(expected, result)


@pytest.mark.parametrize(('weights', 'expected'), (([1, 2], 2),
                                                   ([0, 2], 2),
                                                   ([0, 0], np.nan),
                                                   ([-1, 1], 1)))
def test_weigted_sum_of_weights_nan(weights, expected):

    da = DataArray([np.nan, 2])
    weights = DataArray(weights)
    result = da.weighted(weights).sum_of_weights()

    expected = DataArray(expected)

    assert_equal(expected, result)


@pytest.mark.parametrize('da', ([1, 2], [1, np.nan], [np.nan, np.nan]))
@pytest.mark.parametrize('factor', [0, 1, 2, 3.14])
@pytest.mark.parametrize('skipna', (True, False))
def test_weighted_sum_equal_weights(da, factor, skipna):
    # if all weights are 'f'; weighted sum is f times the ordinary sum

    da = DataArray(da)
    weights = xr.zeros_like(da) + factor

    expected = da.sum(skipna=skipna) * factor
    result = da.weighted(weights).sum(skipna=skipna)

    assert_equal(expected, result)


@pytest.mark.parametrize(('weights', 'expected'), (([1, 2], 5),
                                                   ([0, 2], 4),
                                                   ([0, 0], 0)))
def test_weighted_sum_no_nan(weights, expected):
    da = DataArray([1, 2])

    weights = DataArray(weights)
    result = da.weighted(weights).sum()
    expected = DataArray(expected)

    assert_equal(expected, result)


@pytest.mark.parametrize(('weights', 'expected'), (([1, 2], 4),
                                                   ([0, 2], 4),
                                                   ([1, 0], 0),
                                                   ([0, 0], 0)))
@pytest.mark.parametrize('skipna', (True, False))
def test_weighted_sum_nan(weights, expected, skipna):
    da = DataArray([np.nan, 2])

    weights = DataArray(weights)
    result = da.weighted(weights).sum(skipna=skipna)

    if skipna:
        expected = DataArray(expected)
    else:
        expected = DataArray(np.nan)

    assert_equal(expected, result)


@pytest.mark.filterwarnings("ignore:Mean of empty slice")
@pytest.mark.parametrize('da', ([1, 2], [1, np.nan], [np.nan, np.nan]))
@pytest.mark.parametrize('skipna', (True, False))
def test_weigted_mean_equal_weights(da, skipna):
    # if all weights are equal, should yield the same result as mean

    da = DataArray(da)

    # all weights as 1.
    weights = xr.zeros_like(da) + 1

    expected = da.mean(skipna=skipna)
    result = da.weighted(weights).mean(skipna=skipna)

    assert_equal(expected, result)


@pytest.mark.parametrize(('weights', 'expected'), (([4, 6], 1.6),
                                                   ([0, 1], 2.0),
                                                   ([0, 2], 2.0),
                                                   ([0, 0], np.nan)))
def test_weigted_mean_no_nan(weights, expected):

    da = DataArray([1, 2])
    weights = DataArray(weights)
    expected = DataArray(expected)

    result = da.weighted(weights).mean()

    assert_equal(expected, result)


@pytest.mark.parametrize(('weights', 'expected'), (([4, 6], 2.0),
                                                   ([0, 1], 2.0),
                                                   ([0, 2], 2.0),
                                                   ([0, 0], np.nan)))
@pytest.mark.parametrize('skipna', (True, False))
def test_weigted_mean_nan(weights, expected, skipna):

    da = DataArray([np.nan, 2])
    weights = DataArray(weights)

    if skipna:
        expected = DataArray(expected)
    else:
        expected = DataArray(np.nan)

    result = da.weighted(weights).mean(skipna=skipna)

    assert_equal(expected, result)
