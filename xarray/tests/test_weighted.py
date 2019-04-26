import pytest

import numpy as np

import xarray as xr
from xarray import (
    DataArray,)

from xarray.tests import assert_equal


def test_weigted_non_DataArray_weights():

    da = DataArray([1, 2])
    with pytest.raises(AssertionError):
        da.weighted([1, 2])


@pytest.mark.parametrize('da', ([1, 2], [1, np.nan]))
@pytest.mark.parametrize('weights', ([1, 2], [np.nan, 2]))
def test_weigted_sum_of_weights(da, weights):

    da = DataArray(da)
    weights = DataArray(weights)

    expected = weights.where(~ da.isnull()).sum()
    result = da.weighted(weights).sum_of_weights()

    assert_equal(expected, result)


@pytest.mark.parametrize('da', ([1, 2], [1, np.nan]))
@pytest.mark.parametrize('skipna', (True, False))
def test_weigted_mean_equal_weights(da, skipna):
    # if all weights are equal, should yield the same result as mean

    da = DataArray(da)

    weights = xr.zeros_like(da) + 1

    expected = da.mean(skipna=skipna)
    result = da.weighted(weights).mean(skipna=skipna)

    assert_equal(expected, result)

def expected_weighted(da, weights, skipna):
    np.warnings.filterwarnings('ignore')

    # all NaN's in weights are replaced
    weights = np.nan_to_num(weights)

    if np.all(np.isnan(da)):
        expected = np.nan
    elif skipna:
        da = np.ma.masked_invalid(da)
        expected = np.ma.average(da, weights=weights)
    else:
        expected = np.ma.average(da, weights=weights)

    expected = np.asarray(expected)

    expected[np.isinf(expected)] = np.nan

    return DataArray(expected)


@pytest.mark.parametrize('da', ([1, 2], [1, np.nan], [np.nan, np.nan]))
@pytest.mark.parametrize('weights', ([4, 6], [-1, 1], [1, 0], [0, 1],
                                     [1, np.nan], [np.nan, np.nan]))
@pytest.mark.parametrize('skipna', (True, False))
def test_weigted_mean(da, weights, skipna):

    expected = expected_weighted(da, weights, skipna)

    da = DataArray(da)
    weights = DataArray(weights)

    result = da.weighted(weights).mean(skipna=skipna)

    assert_equal(expected, result)
