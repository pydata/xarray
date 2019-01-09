from __future__ import absolute_import

import pytest

import numpy as np
import pandas as pd
import xarray as xr
from xarray.tests import assert_equal


@pytest.fixture()
def pd_index():
    return pd.date_range('2004-01-01', periods=31, freq='MS')
    # specifying tz='UTC' results in TypeError: Only valid with DatetimeIndex,
    # TimedeltaIndex or PeriodIndex, but got an instance of 'Index'


@pytest.fixture()
def xr_index():
    return xr.cftime_range('2004-01-01', periods=31, freq='MS')


@pytest.fixture()
def pd_3ms_index():
    return pd.date_range('2004-01-01', periods=31, freq='3MS')
    # specifying tz='UTC' results in TypeError: Only valid with DatetimeIndex,
    # TimedeltaIndex or PeriodIndex, but got an instance of 'Index'


@pytest.fixture()
def xr_3ms_index():
    return xr.cftime_range('2004-01-01', periods=31, freq='3MS')


@pytest.fixture()
def pd_3d_index():
    return pd.date_range('2004-01-01', periods=31, freq='3D')
    # specifying tz='UTC' results in TypeError: Only valid with DatetimeIndex,
    # TimedeltaIndex or PeriodIndex, but got an instance of 'Index'


@pytest.fixture()
def xr_3d_index():
    return xr.cftime_range('2004-01-01', periods=31, freq='3D')


@pytest.fixture()
def daily_pd_index():
    return pd.date_range('2004-01-01', periods=901, freq='D')
    # specifying tz='UTC' results in TypeError: Only valid with DatetimeIndex,
    # TimedeltaIndex or PeriodIndex, but got an instance of 'Index'


@pytest.fixture()
def daily_xr_index():
    return xr.cftime_range('2004-01-01', periods=901, freq='D')


@pytest.fixture()
def base_pd_index():
    return pd.date_range('2004-01-01', periods=31, freq='D')
    # specifying tz='UTC' results in TypeError: Only valid with DatetimeIndex,
    # TimedeltaIndex or PeriodIndex, but got an instance of 'Index'


@pytest.fixture()
def base_xr_index():
    return xr.cftime_range('2004-01-01', periods=31, freq='D')


def da(index):
    return xr.DataArray(np.arange(100., 100. + index.size),
                        coords=[index], dims=['time'])


def series(index):
    return pd.Series(np.arange(100., 100. + index.size), index=index)


@pytest.mark.parametrize('closed', ['left', 'right'])
@pytest.mark.parametrize('label', ['left', 'right'])
@pytest.mark.parametrize('freq', ['2MS', '2M', '3MS', '3M',
                                  '4MS', '4M', '5MS', '5M',
                                  '7MS', '7M', '8MS', '8M'])
def test_downsampler(closed, label, freq, pd_index, xr_index):
    downsamp_pdtime = da(pd_index).resample(
        time=freq, closed=closed, label=label).mean()
    downsamp_cftime = da(xr_index).resample(
        time=freq, closed=closed, label=label).mean()
    downsamp_cftime['time'] = downsamp_cftime.indexes['time'].to_datetimeindex()
    assert_equal(downsamp_pdtime, downsamp_cftime)


@pytest.mark.parametrize('closed', ['left', 'right'])
@pytest.mark.parametrize('label', ['left', 'right'])
@pytest.mark.parametrize('freq', ['2MS', '2M', '3MS', '3M',
                                  '4MS', '4M', '5MS', '5M',
                                  '7MS', '7M', '8MS', '8M',
                                  'AS', 'A', '2AS', '2A',
                                  '3AS', '3A', '4AS', '4A'])
def test_downsampler_daily(closed, label, freq, daily_pd_index, daily_xr_index):
    downsamp_pdtime = da(daily_pd_index).resample(
        time=freq, closed=closed, label=label).mean()
    downsamp_cftime = da(daily_xr_index).resample(
        time=freq, closed=closed, label=label).mean()
    downsamp_cftime['time'] = downsamp_cftime.indexes['time'].to_datetimeindex()
    assert_equal(downsamp_pdtime, downsamp_cftime)


@pytest.mark.parametrize('closed', ['left', 'right'])
@pytest.mark.parametrize('label', ['left', 'right'])
@pytest.mark.parametrize('freq', ['MS', 'M', '7D', '5D', '8D', '4D', 'D'])
def test_upsampler(closed, label, freq, pd_index, xr_index):
    # The testing here covers cases of equal sampling as well.
    # For pandas, --not xarray--, .ffill() and .bfill() gives
    # error (mismatched length).
    upsamp_pdtime = da(pd_index).resample(
        time=freq, closed=closed, label=label).mean()
    upsamp_cftime = da(xr_index).resample(
        time=freq, closed=closed, label=label).mean()
    upsamp_cftime['time'] = upsamp_cftime.indexes['time'].to_datetimeindex()
    assert_equal(upsamp_pdtime, upsamp_cftime)


@pytest.mark.parametrize('closed', ['left', 'right'])
@pytest.mark.parametrize('label', ['left', 'right'])
@pytest.mark.parametrize('freq', ['3MS', '3M', '2MS', '2M', 'MS', 'M',
                                  '7D', '5D', '8D', '4D', 'D'])
def test_upsampler_3ms(closed, label, freq, pd_3ms_index, xr_3ms_index):
    # The testing here covers cases of equal sampling as well.
    upsamp_pdtime = da(pd_3ms_index).resample(
        time=freq, closed=closed, label=label).mean()
    upsamp_cftime = da(xr_3ms_index).resample(
        time=freq, closed=closed, label=label).mean()
    upsamp_cftime['time'] = upsamp_cftime.indexes['time'].to_datetimeindex()
    assert_equal(upsamp_pdtime, upsamp_cftime)


@pytest.mark.parametrize('closed', ['left', 'right'])
@pytest.mark.parametrize('label', ['left', 'right'])
@pytest.mark.parametrize('freq', ['3D', '2D', 'D',
                                  '7H', '5H', '2H'])
def test_upsampler_3d(closed, label, freq, pd_3d_index, xr_3d_index):
    # The testing here covers cases of equal sampling as well.
    upsamp_pdtime = da(pd_3d_index).resample(
        time=freq, closed=closed, label=label).mean()
    upsamp_cftime = da(xr_3d_index).resample(
        time=freq, closed=closed, label=label).mean()
    upsamp_cftime['time'] = upsamp_cftime.indexes['time'].to_datetimeindex()
    assert_equal(upsamp_pdtime, upsamp_cftime)


@pytest.mark.parametrize('closed', ['left', 'right'])
@pytest.mark.parametrize('label', ['left', 'right'])
@pytest.mark.parametrize('base', [1, 5, 12, 17, 24])
@pytest.mark.parametrize('freq', ['12H', '7H', '5H', '2H'])
def test_upsampler_base(closed, label, base, freq,
                        base_pd_index, base_xr_index):
    upsamp_pdtime = da(base_pd_index).resample(
        time=freq, closed=closed, label=label).mean()
    upsamp_cftime = da(base_xr_index).resample(
        time=freq, closed=closed, label=label).mean()
    upsamp_cftime['time'] = upsamp_cftime.indexes['time'].to_datetimeindex()
    assert_equal(upsamp_pdtime, upsamp_cftime)
