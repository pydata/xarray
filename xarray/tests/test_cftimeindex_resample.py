import pytest

import datetime
import numpy as np
import pandas as pd
import xarray as xr
from xarray.core.resample_cftime import CFTimeGrouper

pytest.importorskip('cftime')
pytest.importorskip('pandas', minversion='0.24')


@pytest.fixture(
    params=[
        dict(start='2004-01-01T12:07:01', periods=91, freq='3D'),
        dict(start='1892-01-03T12:07:01', periods=15, freq='41987T'),
        dict(start='2004-01-01T12:07:01', periods=7, freq='3Q-AUG'),
        dict(start='1892-01-03T12:07:01', periods=10, freq='3AS-JUN')
    ],
    ids=['3D', '41987T', '3Q_AUG', '3AS_JUN']
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
    '8D', '8001D',
    '2MS', '3MS',
    '2QS-AUG', '3QS-SEP',
    '3AS-MAR', '4A-MAY'])
@pytest.mark.parametrize('closed', [None, 'right'])
@pytest.mark.parametrize('label', [None, 'right'])
@pytest.mark.parametrize('base', [24, 31])
def test_resampler(freq, closed, label, base,
                   datetime_index, cftime_index):
    # Fairly extensive testing for standard/proleptic Gregorian calendar
    # For any frequencies which are not greater-than-day and anchored
    # at the end, the default values for closed and label are 'left'.
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


@pytest.mark.parametrize('freq', [
    '2M', '3M',
    '2Q-JUN', '3Q-JUL',
    '3A-FEB', '4A-APR'])
@pytest.mark.parametrize('closed', ['left', None])
@pytest.mark.parametrize('label', ['left', None])
@pytest.mark.parametrize('base', [17, 24])
def test_resampler_end_super_day(freq, closed, label, base,
                                 datetime_index, cftime_index):
    # Fairly extensive testing for standard/proleptic Gregorian calendar.
    # For greater-than-day frequencies anchored at the end, the default values
    # for closed and label are 'right'.
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


@pytest.mark.parametrize(
    ('freq', 'expected'),
    [('S', 'left'), ('T', 'left'), ('H', 'left'), ('D', 'left'),
     ('M', 'right'), ('MS', 'left'), ('Q', 'right'), ('QS', 'left'),
     ('A', 'right'), ('AS', 'left')])
def test_closed_label_defaults(freq, expected):
    assert CFTimeGrouper(freq=freq).closed == expected
    assert CFTimeGrouper(freq=freq).label == expected


@pytest.mark.parametrize('calendar', ['gregorian', 'noleap', 'all_leap',
                                      '360_day', 'julian'])
def test_calendars(calendar):
    # Limited testing for non-standard calendars
    freq, closed, label, base = '8001T', None, None, 17
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
