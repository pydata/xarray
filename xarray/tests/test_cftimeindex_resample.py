from __future__ import absolute_import

import pytest

import numpy as np
import pandas as pd
import xarray as xr


@pytest.fixture(
    params=[
        dict(start='2004-01-01T12:07:01', periods=37, freq='A'),
        dict(start='2004-01-01T12:07:01', periods=31, freq='3MS'),
        dict(start='2004-01-01T12:07:01', periods=31, freq='MS'),
        dict(start='2004-01-01T12:07:01', periods=31, freq='3D'),
        dict(start='2004-01-01T12:07:04', periods=901, freq='D'),
        dict(start='2004-01-01T12:07:01', periods=31, freq='D'),
        dict(start='1892-01-01T12:00:00', periods=15, freq='5256113T'),
        dict(start='1892', periods=10, freq='6AS-JUN')
    ],
    ids=['37_A', '31_3MS', '31_MS', '31_3D', '901D', '31D', 'XT', '6AS_JUN']
)
def time_range_kwargs(request):
    return request.param


@pytest.fixture()
def datetime_index(time_range_kwargs):
    return pd.date_range(**time_range_kwargs)


@pytest.fixture()
def cftime_index(time_range_kwargs):
    return xr.cftime_range(**time_range_kwargs)


def da(index):
    return xr.DataArray(np.arange(100., 100. + index.size),
                        coords=[index], dims=['time'])


@pytest.mark.parametrize('freq', [
    '600003T',
    '2H', '5H', '7H', '12H', '8001H',
    'D', '2D', '3D', '4D', '5D', '7D', '8D',
    'MS', 'M', '2MS', '2M', '3MS', '3M', '4MS', '4M',
    '5MS', '5M', '7MS', '7M', '8MS', '8M', '11M', '11MS',
    'AS', 'A', '2AS', '2A', '3AS', '3A', '4AS', '4A'])
@pytest.mark.parametrize('closed', ['left', 'right'])
@pytest.mark.parametrize('label', ['left', 'right'])
@pytest.mark.parametrize('base', [1, 5, 12, 17, 24])
@pytest.mark.xfail(raises=ValueError)
def test_resampler(closed, label, base, freq,
                   datetime_index, cftime_index):
    da_cftime = da(cftime_index).resample(
        time=freq, closed=closed, label=label, base=base).mean()
    da_cftime['time'] = da_cftime.indexes['time'].to_datetimeindex()
    da_datetime = da(datetime_index).resample(
        time=freq, closed=closed, label=label, base=base).mean()
    np.testing.assert_equal(da_cftime.values, da_datetime.values)
    np.testing.assert_allclose(da_cftime['time'].values.astype(np.float64),
                               da_datetime['time'].values.astype(np.float64))
