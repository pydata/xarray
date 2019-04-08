import re

import pytest

import datetime
import numpy as np
import pandas as pd
import xarray as xr
from xarray.core.resample_cftime import CFTimeGrouper
from xarray.coding.cftime_offsets import _PATTERN

pytest.importorskip('cftime')
pytest.importorskip('pandas', minversion='0.24')


def da(index):
    return xr.DataArray(np.arange(100., 100. + index.size),
                        coords=[index], dims=['time'])


def _scale_freq(freq, mode):
    """Scale input to a longer or shorter frequency string"""
    freq_data = re.match(_PATTERN, freq).groupdict()
    if mode == 'longer':
        new_multiple = int(freq_data['multiple']) * 2
    elif mode == 'shorter':
        new_multiple = int(freq_data['multiple']) // 2
    else:
        raise ValueError
    return '{}{}'.format(new_multiple, freq_data['freq'])


@pytest.mark.parametrize('freq', [
    '8001T', '12H',
    '8D', '2MS',
    '7M', '41QS-AUG',
    '11Q-JUN', '3AS-MAR',
    '4A-MAY'])
@pytest.mark.parametrize('closed', [None, 'left', 'right'])
@pytest.mark.parametrize('label', [None, 'left', 'right'])
@pytest.mark.parametrize('base', [24, 31])
@pytest.mark.parametrize(
    'da_freq_length',
    ['longer', 'shorter'],
    ids=['longer_da_freq', 'shorter_da_freq']
)
def test_resample(freq, closed, label, base, da_freq_length):
    da_freq = _scale_freq(freq, da_freq_length)
    index_kwargs = dict(start='2000-01-01T12:07:01', periods=5, freq=da_freq)
    datetime_index = pd.date_range(**index_kwargs)
    cftime_index = xr.cftime_range(**index_kwargs)

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
