from __future__ import absolute_import

import itertools
import numpy as np
import pandas as pd
import pytest

import xarray as xr
from xarray.tests import assert_array_equal, assert_identical


@pytest.mark.parametrize(('start', 'stop'), [(1, 2), (1, 3)])
def test_nprange(start, stop):
    assert np.all(np.arange(start, stop) == np.arange(start, stop))


@pytest.fixture()
def nprange():
    return np.arange(1, 10)


def test_nprange2(nprange):
    # print(nprange)
    assert np.all(nprange == nprange)


@pytest.fixture()
def downsamp_xr_index():
    # return xr.cftime_range('2000', periods=30, freq='MS')
    return xr.cftime_range('2000', periods=9, freq='T')


@pytest.fixture()
def da(downsamp_xr_index):
    return xr.DataArray(np.arange(100., 100.+downsamp_xr_index.size),
                        coords=[downsamp_xr_index], dims=['time'])


# @pytest.fixture(params=list(itertools.product(['left', 'right'],
#                                               ['left', 'right'],
#                                               ['MS', '3MS', '7MS'])))
@pytest.fixture(params=list(itertools.product(['left', 'right'],
                                              ['left', 'right'],
                                              ['T', '3T', '7T'])))
def da_resampler(request, da):
    # print(request.param)
    # return request.param
    closed = request.param[0]
    label = request.param[1]
    time = request.param[2]
    return da.resample(time=time, closed=closed, label=label).mean().values


def test_resampler(da_resampler):
    # print(da_resampler)
    # assert da_resampler == da_resampler
    # assert np.all(da_resampler.mean().values == da_resampler.mean().values)
    print(da_resampler)
    assert np.all(da_resampler == da_resampler)
