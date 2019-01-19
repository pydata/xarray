from __future__ import absolute_import

import pytest

import numpy as np
import pandas as pd
import xarray as xr
import pandas.core.resample as pdr
import xarray.core.resample_cftime as xrr
from pandas.tseries.frequencies import to_offset as pd_to_offset
from xarray.coding.cftime_offsets import (to_cftime_datetime,
                                          to_offset as xr_to_offset)


@pytest.fixture(
    params=[
        dict(start='2004-01-01T12:07:01', periods=91, freq='3D'),
        dict(start='1892-01-03T12:07:01', periods=15, freq='41987T'),
        dict(start='2004-01-01T12:07:01', periods=31, freq='2MS'),
        dict(start='1892-01-03T12:07:01', periods=10, freq='3AS-JUN')
    ],
    ids=['2MS', '3D', '41987T', '3AS_JUN']
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
    '700T', '8001T',
    '12H', '8001H',
    '3D', '8D', '8001D',
    '2MS', '2M', '3MS', '3M', '4MS', '4M',
    '3AS', '3A', '4AS', '4A'])
@pytest.mark.parametrize('closed', ['left', 'right'])
@pytest.mark.parametrize('label', ['left', 'right'])
@pytest.mark.parametrize('base', [17, 24])
@pytest.mark.xfail(raises=ValueError)
def test_resampler(freq, closed, label, base,
                   datetime_index, cftime_index):
    da_cftime = da(cftime_index).resample(
        time=freq, closed=closed, label=label, base=base).mean()
    da_cftime['time'] = da_cftime.indexes['time'].to_datetimeindex()
    da_datetime = da(datetime_index).resample(
        time=freq, closed=closed, label=label, base=base).mean()
    np.testing.assert_equal(da_cftime.values, da_datetime.values)
    np.testing.assert_equal(da_cftime['time'].values.astype(np.float64),
                            da_datetime['time'].values.astype(np.float64))


@pytest.mark.parametrize('freq', [
    '700T', '8001T',
    '12H', '8001H',
    '3D', '8D', '8001D',
    '2MS', '2M', '3MS', '3M', '4MS', '4M',
    '3AS', '3A', '4AS', '4A'])
@pytest.mark.parametrize('closed', ['left', 'right'])
@pytest.mark.parametrize('label', ['left', 'right'])
@pytest.mark.parametrize('base', [17, 24])
@pytest.mark.parametrize('first', ['1892-01-03T12:07:01'])
@pytest.mark.parametrize('last', ['1893-01-03T12:07:01'])
def test_get_range_edges(freq, closed, label, base,
                         first, last):
    """
    Test must cover the following cases:
    freq/offset is instance of Tick/CFTIME_TICKS (Day, Hour, Minute, Second)
    so that _adjust_dates_anchored is triggered.
    """
    first_cftime = to_cftime_datetime(first, 'standard')
    first_datetime = pd.Timestamp(first)
    last_cftime = to_cftime_datetime(last, 'standard')
    last_datetime = pd.Timestamp(last)
    first_pd, last_pd = pdr._get_timestamp_range_edges(
        first_datetime, last_datetime, pd_to_offset(freq), closed, base)
    first_xr, last_xr = xrr._get_range_edges(
        first_cftime, last_cftime, xr_to_offset(freq), closed, base)
    np.testing.assert_equal([first_pd, last_pd], [first_xr, last_xr])
    # pdr._adjust_dates_anchored()
