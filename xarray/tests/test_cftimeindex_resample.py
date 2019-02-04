import pytest

import datetime
import numpy as np
import pandas as pd
import xarray as xr

pytest.importorskip('cftime')
pytest.importorskip('pandas', minversion='0.24')


@pytest.fixture(
    params=[
        dict(start='2004-01-01T12:07:01', periods=91, freq='3D'),
        dict(start='1892-01-03T12:07:01', periods=15, freq='41987T'),
        dict(start='2004-01-01T12:07:01', periods=31, freq='2MS'),
        dict(start='1892-01-03T12:07:01', periods=10, freq='3AS-JUN')
    ],
    ids=['3D', '41987T', '2MS', '3AS_JUN']
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
@pytest.mark.parametrize('closed', [None, 'left', 'right'])
@pytest.mark.parametrize('label', [None, 'left', 'right'])
@pytest.mark.parametrize('base', [17, 24])
def test_resampler(freq, closed, label, base,
                   datetime_index, cftime_index):
    # Fairly extensive testing for standard/proleptic Gregorian calendar
    loffset = '12H'
    try:
        da_datetime = da(datetime_index).resample(
            time=freq, closed=closed, label=label, base=base,
            loffset=loffset).mean()
    except ValueError:
        with pytest.raises(ValueError):
            da(cftime_index).resample(
                time=freq, closed=closed, label=label, base=base,
                loffset=loffset).mean()
    else:
        da_cftime = da(cftime_index).resample(time=freq, closed=closed,
                                              label=label, base=base,
                                              loffset=loffset).mean()
        da_cftime['time'] = da_cftime.indexes['time'].to_datetimeindex()
        xr.testing.assert_identical(da_cftime, da_datetime)


@pytest.mark.parametrize('calendar', ['gregorian', 'noleap', 'all_leap',
                                      '360_day', 'julian'])
def test_calendars(calendar):
    # Limited testing for non-standard calendars
    freq, closed, label, base = '81T', None, None, 17
    loffset = datetime.timedelta(hours=12)
    xr_index = xr.cftime_range(start='2004-01-01T12:07:01', periods=7,
                               freq='3D', calendar=calendar)
    pd_index = pd.date_range(start='2004-01-01T12:07:01', periods=7,
                             freq='3D')
    da_cftime = da(xr_index).resample(
        time=freq, closed=closed, label=label, base=base, loffset=loffset
    ).mean()
    da_datetime = da(pd_index).resample(
        time=freq, closed=closed, label=label, base=base, loffset=loffset
    ).mean()
    da_cftime['time'] = da_cftime.indexes['time'].to_datetimeindex()
    xr.testing.assert_identical(da_cftime, da_datetime)
