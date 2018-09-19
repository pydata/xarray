from __future__ import absolute_import

import pytest

import numpy as np
import pandas as pd
import xarray as xr

from datetime import timedelta
from xarray.coding.cftimeindex import (
    parse_iso8601, CFTimeIndex, assert_all_valid_date_type,
    _parsed_string_to_bounds, _parse_iso8601_with_reso)
from xarray.tests import assert_array_equal, assert_identical

from . import has_cftime, has_cftime_or_netCDF4
from .test_coding_times import _all_cftime_date_types


def date_dict(year=None, month=None, day=None,
              hour=None, minute=None, second=None):
    return dict(year=year, month=month, day=day, hour=hour,
                minute=minute, second=second)


ISO8601_STRING_TESTS = {
    'year': ('1999', date_dict(year='1999')),
    'month': ('199901', date_dict(year='1999', month='01')),
    'month-dash': ('1999-01', date_dict(year='1999', month='01')),
    'day': ('19990101', date_dict(year='1999', month='01', day='01')),
    'day-dash': ('1999-01-01', date_dict(year='1999', month='01', day='01')),
    'hour': ('19990101T12', date_dict(
        year='1999', month='01', day='01', hour='12')),
    'hour-dash': ('1999-01-01T12', date_dict(
        year='1999', month='01', day='01', hour='12')),
    'minute': ('19990101T1234', date_dict(
        year='1999', month='01', day='01', hour='12', minute='34')),
    'minute-dash': ('1999-01-01T12:34', date_dict(
        year='1999', month='01', day='01', hour='12', minute='34')),
    'second': ('19990101T123456', date_dict(
        year='1999', month='01', day='01', hour='12', minute='34',
        second='56')),
    'second-dash': ('1999-01-01T12:34:56', date_dict(
        year='1999', month='01', day='01', hour='12', minute='34',
        second='56'))
}


@pytest.mark.parametrize(('string', 'expected'),
                         list(ISO8601_STRING_TESTS.values()),
                         ids=list(ISO8601_STRING_TESTS.keys()))
def test_parse_iso8601(string, expected):
    result = parse_iso8601(string)
    assert result == expected

    with pytest.raises(ValueError):
        parse_iso8601(string + '3')
        parse_iso8601(string + '.3')


_CFTIME_CALENDARS = ['365_day', '360_day', 'julian', 'all_leap',
                     '366_day', 'gregorian', 'proleptic_gregorian']


@pytest.fixture(params=_CFTIME_CALENDARS)
def date_type(request):
    return _all_cftime_date_types()[request.param]


@pytest.fixture
def index(date_type):
    dates = [date_type(1, 1, 1), date_type(1, 2, 1),
             date_type(2, 1, 1), date_type(2, 2, 1)]
    return CFTimeIndex(dates)


@pytest.fixture
def monotonic_decreasing_index(date_type):
    dates = [date_type(2, 2, 1), date_type(2, 1, 1),
             date_type(1, 2, 1), date_type(1, 1, 1)]
    return CFTimeIndex(dates)


@pytest.fixture
def length_one_index(date_type):
    dates = [date_type(1, 1, 1)]
    return CFTimeIndex(dates)


@pytest.fixture
def da(index):
    return xr.DataArray([1, 2, 3, 4], coords=[index],
                        dims=['time'])


@pytest.fixture
def series(index):
    return pd.Series([1, 2, 3, 4], index=index)


@pytest.fixture
def df(index):
    return pd.DataFrame([1, 2, 3, 4], index=index)


@pytest.fixture
def feb_days(date_type):
    import cftime
    if date_type is cftime.DatetimeAllLeap:
        return 29
    elif date_type is cftime.Datetime360Day:
        return 30
    else:
        return 28


@pytest.fixture
def dec_days(date_type):
    import cftime
    if date_type is cftime.Datetime360Day:
        return 30
    else:
        return 31


@pytest.fixture
def index_with_name(date_type):
    dates = [date_type(1, 1, 1), date_type(1, 2, 1),
             date_type(2, 1, 1), date_type(2, 2, 1)]
    return CFTimeIndex(dates, name='foo')


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize(
    ('name', 'expected_name'),
    [('bar', 'bar'),
     (None, 'foo')])
def test_constructor_with_name(index_with_name, name, expected_name):
    result = CFTimeIndex(index_with_name, name=name).name
    assert result == expected_name


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
def test_assert_all_valid_date_type(date_type, index):
    import cftime
    if date_type is cftime.DatetimeNoLeap:
        mixed_date_types = np.array(
            [date_type(1, 1, 1),
             cftime.DatetimeAllLeap(1, 2, 1)])
    else:
        mixed_date_types = np.array(
            [date_type(1, 1, 1),
             cftime.DatetimeNoLeap(1, 2, 1)])
    with pytest.raises(TypeError):
        assert_all_valid_date_type(mixed_date_types)

    with pytest.raises(TypeError):
        assert_all_valid_date_type(np.array([1, date_type(1, 1, 1)]))

    assert_all_valid_date_type(
        np.array([date_type(1, 1, 1), date_type(1, 2, 1)]))


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize(('field', 'expected'), [
    ('year', [1, 1, 2, 2]),
    ('month', [1, 2, 1, 2]),
    ('day', [1, 1, 1, 1]),
    ('hour', [0, 0, 0, 0]),
    ('minute', [0, 0, 0, 0]),
    ('second', [0, 0, 0, 0]),
    ('microsecond', [0, 0, 0, 0])])
def test_cftimeindex_field_accessors(index, field, expected):
    result = getattr(index, field)
    assert_array_equal(result, expected)


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize(('string', 'date_args', 'reso'), [
    ('1999', (1999, 1, 1), 'year'),
    ('199902', (1999, 2, 1), 'month'),
    ('19990202', (1999, 2, 2), 'day'),
    ('19990202T01', (1999, 2, 2, 1), 'hour'),
    ('19990202T0101', (1999, 2, 2, 1, 1), 'minute'),
    ('19990202T010156', (1999, 2, 2, 1, 1, 56), 'second')])
def test_parse_iso8601_with_reso(date_type, string, date_args, reso):
    expected_date = date_type(*date_args)
    expected_reso = reso
    result_date, result_reso = _parse_iso8601_with_reso(date_type, string)
    assert result_date == expected_date
    assert result_reso == expected_reso


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
def test_parse_string_to_bounds_year(date_type, dec_days):
    parsed = date_type(2, 2, 10, 6, 2, 8, 1)
    expected_start = date_type(2, 1, 1)
    expected_end = date_type(2, 12, dec_days, 23, 59, 59, 999999)
    result_start, result_end = _parsed_string_to_bounds(
        date_type, 'year', parsed)
    assert result_start == expected_start
    assert result_end == expected_end


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
def test_parse_string_to_bounds_month_feb(date_type, feb_days):
    parsed = date_type(2, 2, 10, 6, 2, 8, 1)
    expected_start = date_type(2, 2, 1)
    expected_end = date_type(2, 2, feb_days, 23, 59, 59, 999999)
    result_start, result_end = _parsed_string_to_bounds(
        date_type, 'month', parsed)
    assert result_start == expected_start
    assert result_end == expected_end


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
def test_parse_string_to_bounds_month_dec(date_type, dec_days):
    parsed = date_type(2, 12, 1)
    expected_start = date_type(2, 12, 1)
    expected_end = date_type(2, 12, dec_days, 23, 59, 59, 999999)
    result_start, result_end = _parsed_string_to_bounds(
        date_type, 'month', parsed)
    assert result_start == expected_start
    assert result_end == expected_end


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize(('reso', 'ex_start_args', 'ex_end_args'), [
    ('day', (2, 2, 10), (2, 2, 10, 23, 59, 59, 999999)),
    ('hour', (2, 2, 10, 6), (2, 2, 10, 6, 59, 59, 999999)),
    ('minute', (2, 2, 10, 6, 2), (2, 2, 10, 6, 2, 59, 999999)),
    ('second', (2, 2, 10, 6, 2, 8), (2, 2, 10, 6, 2, 8, 999999))])
def test_parsed_string_to_bounds_sub_monthly(date_type, reso,
                                             ex_start_args, ex_end_args):
    parsed = date_type(2, 2, 10, 6, 2, 8, 123456)
    expected_start = date_type(*ex_start_args)
    expected_end = date_type(*ex_end_args)

    result_start, result_end = _parsed_string_to_bounds(
        date_type, reso, parsed)
    assert result_start == expected_start
    assert result_end == expected_end


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
def test_parsed_string_to_bounds_raises(date_type):
    with pytest.raises(KeyError):
        _parsed_string_to_bounds(date_type, 'a', date_type(1, 1, 1))


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
def test_get_loc(date_type, index):
    result = index.get_loc('0001')
    expected = [0, 1]
    assert_array_equal(result, expected)

    result = index.get_loc(date_type(1, 2, 1))
    expected = 1
    assert result == expected

    result = index.get_loc('0001-02-01')
    expected = 1
    assert result == expected


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize('kind', ['loc', 'getitem'])
def test_get_slice_bound(date_type, index, kind):
    result = index.get_slice_bound('0001', 'left', kind)
    expected = 0
    assert result == expected

    result = index.get_slice_bound('0001', 'right', kind)
    expected = 2
    assert result == expected

    result = index.get_slice_bound(
        date_type(1, 3, 1), 'left', kind)
    expected = 2
    assert result == expected

    result = index.get_slice_bound(
        date_type(1, 3, 1), 'right', kind)
    expected = 2
    assert result == expected


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize('kind', ['loc', 'getitem'])
def test_get_slice_bound_decreasing_index(
        date_type, monotonic_decreasing_index, kind):
    result = monotonic_decreasing_index.get_slice_bound('0001', 'left', kind)
    expected = 2
    assert result == expected

    result = monotonic_decreasing_index.get_slice_bound('0001', 'right', kind)
    expected = 4
    assert result == expected

    result = monotonic_decreasing_index.get_slice_bound(
        date_type(1, 3, 1), 'left', kind)
    expected = 2
    assert result == expected

    result = monotonic_decreasing_index.get_slice_bound(
        date_type(1, 3, 1), 'right', kind)
    expected = 2
    assert result == expected


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize('kind', ['loc', 'getitem'])
def test_get_slice_bound_length_one_index(
        date_type, length_one_index, kind):
    result = length_one_index.get_slice_bound('0001', 'left', kind)
    expected = 0
    assert result == expected

    result = length_one_index.get_slice_bound('0001', 'right', kind)
    expected = 1
    assert result == expected

    result = length_one_index.get_slice_bound(
        date_type(1, 3, 1), 'left', kind)
    expected = 1
    assert result == expected

    result = length_one_index.get_slice_bound(
        date_type(1, 3, 1), 'right', kind)
    expected = 1
    assert result == expected


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
def test_string_slice_length_one_index(length_one_index):
    da = xr.DataArray([1], coords=[length_one_index], dims=['time'])
    result = da.sel(time=slice('0001', '0001'))
    assert_identical(result, da)


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
def test_date_type_property(date_type, index):
    assert index.date_type is date_type


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
def test_contains(date_type, index):
    assert '0001-01-01' in index
    assert '0001' in index
    assert '0003' not in index
    assert date_type(1, 1, 1) in index
    assert date_type(3, 1, 1) not in index


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
def test_groupby(da):
    result = da.groupby('time.month').sum('time')
    expected = xr.DataArray([4, 6], coords=[[1, 2]], dims=['month'])
    assert_identical(result, expected)


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
def test_resample_error(da):
    with pytest.raises(TypeError):
        da.resample(time='Y')


SEL_STRING_OR_LIST_TESTS = {
    'string': '0001',
    'string-slice': slice('0001-01-01', '0001-12-30'),
    'bool-list': [True, True, False, False]
}


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize('sel_arg', list(SEL_STRING_OR_LIST_TESTS.values()),
                         ids=list(SEL_STRING_OR_LIST_TESTS.keys()))
def test_sel_string_or_list(da, index, sel_arg):
    expected = xr.DataArray([1, 2], coords=[index[:2]], dims=['time'])
    result = da.sel(time=sel_arg)
    assert_identical(result, expected)


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
def test_sel_date_slice_or_list(da, index, date_type):
    expected = xr.DataArray([1, 2], coords=[index[:2]], dims=['time'])
    result = da.sel(time=slice(date_type(1, 1, 1), date_type(1, 12, 30)))
    assert_identical(result, expected)

    result = da.sel(time=[date_type(1, 1, 1), date_type(1, 2, 1)])
    assert_identical(result, expected)


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
def test_sel_date_scalar(da, date_type, index):
    expected = xr.DataArray(1).assign_coords(time=index[0])
    result = da.sel(time=date_type(1, 1, 1))
    assert_identical(result, expected)


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize('sel_kwargs', [
    {'method': 'nearest'},
    {'method': 'nearest', 'tolerance': timedelta(days=70)}
])
def test_sel_date_scalar_nearest(da, date_type, index, sel_kwargs):
    expected = xr.DataArray(2).assign_coords(time=index[1])
    result = da.sel(time=date_type(1, 4, 1), **sel_kwargs)
    assert_identical(result, expected)

    expected = xr.DataArray(3).assign_coords(time=index[2])
    result = da.sel(time=date_type(1, 11, 1), **sel_kwargs)
    assert_identical(result, expected)


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize('sel_kwargs', [
    {'method': 'pad'},
    {'method': 'pad', 'tolerance': timedelta(days=365)}
])
def test_sel_date_scalar_pad(da, date_type, index, sel_kwargs):
    expected = xr.DataArray(2).assign_coords(time=index[1])
    result = da.sel(time=date_type(1, 4, 1), **sel_kwargs)
    assert_identical(result, expected)

    expected = xr.DataArray(2).assign_coords(time=index[1])
    result = da.sel(time=date_type(1, 11, 1), **sel_kwargs)
    assert_identical(result, expected)


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize('sel_kwargs', [
    {'method': 'backfill'},
    {'method': 'backfill', 'tolerance': timedelta(days=365)}
])
def test_sel_date_scalar_backfill(da, date_type, index, sel_kwargs):
    expected = xr.DataArray(3).assign_coords(time=index[2])
    result = da.sel(time=date_type(1, 4, 1), **sel_kwargs)
    assert_identical(result, expected)

    expected = xr.DataArray(3).assign_coords(time=index[2])
    result = da.sel(time=date_type(1, 11, 1), **sel_kwargs)
    assert_identical(result, expected)


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize('sel_kwargs', [
    {'method': 'pad', 'tolerance': timedelta(days=20)},
    {'method': 'backfill', 'tolerance': timedelta(days=20)},
    {'method': 'nearest', 'tolerance': timedelta(days=20)},
])
def test_sel_date_scalar_tolerance_raises(da, date_type, sel_kwargs):
    with pytest.raises(KeyError):
        da.sel(time=date_type(1, 5, 1), **sel_kwargs)


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize('sel_kwargs', [
    {'method': 'nearest'},
    {'method': 'nearest', 'tolerance': timedelta(days=70)}
])
def test_sel_date_list_nearest(da, date_type, index, sel_kwargs):
    expected = xr.DataArray(
        [2, 2], coords=[[index[1], index[1]]], dims=['time'])
    result = da.sel(
        time=[date_type(1, 3, 1), date_type(1, 4, 1)], **sel_kwargs)
    assert_identical(result, expected)

    expected = xr.DataArray(
        [2, 3], coords=[[index[1], index[2]]], dims=['time'])
    result = da.sel(
        time=[date_type(1, 3, 1), date_type(1, 12, 1)], **sel_kwargs)
    assert_identical(result, expected)

    expected = xr.DataArray(
        [3, 3], coords=[[index[2], index[2]]], dims=['time'])
    result = da.sel(
        time=[date_type(1, 11, 1), date_type(1, 12, 1)], **sel_kwargs)
    assert_identical(result, expected)


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize('sel_kwargs', [
    {'method': 'pad'},
    {'method': 'pad', 'tolerance': timedelta(days=365)}
])
def test_sel_date_list_pad(da, date_type, index, sel_kwargs):
    expected = xr.DataArray(
        [2, 2], coords=[[index[1], index[1]]], dims=['time'])
    result = da.sel(
        time=[date_type(1, 3, 1), date_type(1, 4, 1)], **sel_kwargs)
    assert_identical(result, expected)


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize('sel_kwargs', [
    {'method': 'backfill'},
    {'method': 'backfill', 'tolerance': timedelta(days=365)}
])
def test_sel_date_list_backfill(da, date_type, index, sel_kwargs):
    expected = xr.DataArray(
        [3, 3], coords=[[index[2], index[2]]], dims=['time'])
    result = da.sel(
        time=[date_type(1, 3, 1), date_type(1, 4, 1)], **sel_kwargs)
    assert_identical(result, expected)


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize('sel_kwargs', [
    {'method': 'pad', 'tolerance': timedelta(days=20)},
    {'method': 'backfill', 'tolerance': timedelta(days=20)},
    {'method': 'nearest', 'tolerance': timedelta(days=20)},
])
def test_sel_date_list_tolerance_raises(da, date_type, sel_kwargs):
    with pytest.raises(KeyError):
        da.sel(time=[date_type(1, 2, 1), date_type(1, 5, 1)], **sel_kwargs)


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
def test_isel(da, index):
    expected = xr.DataArray(1).assign_coords(time=index[0])
    result = da.isel(time=0)
    assert_identical(result, expected)

    expected = xr.DataArray([1, 2], coords=[index[:2]], dims=['time'])
    result = da.isel(time=[0, 1])
    assert_identical(result, expected)


@pytest.fixture
def scalar_args(date_type):
    return [date_type(1, 1, 1)]


@pytest.fixture
def range_args(date_type):
    return ['0001', slice('0001-01-01', '0001-12-30'),
            slice(None, '0001-12-30'),
            slice(date_type(1, 1, 1), date_type(1, 12, 30)),
            slice(None, date_type(1, 12, 30))]


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
def test_indexing_in_series_getitem(series, index, scalar_args, range_args):
    for arg in scalar_args:
        assert series[arg] == 1

    expected = pd.Series([1, 2], index=index[:2])
    for arg in range_args:
        assert series[arg].equals(expected)


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
def test_indexing_in_series_loc(series, index, scalar_args, range_args):
    for arg in scalar_args:
        assert series.loc[arg] == 1

    expected = pd.Series([1, 2], index=index[:2])
    for arg in range_args:
        assert series.loc[arg].equals(expected)


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
def test_indexing_in_series_iloc(series, index):
    expected = 1
    assert series.iloc[0] == expected

    expected = pd.Series([1, 2], index=index[:2])
    assert series.iloc[:2].equals(expected)


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
def test_indexing_in_dataframe_loc(df, index, scalar_args, range_args):
    expected = pd.Series([1], name=index[0])
    for arg in scalar_args:
        result = df.loc[arg]
        assert result.equals(expected)

    expected = pd.DataFrame([1, 2], index=index[:2])
    for arg in range_args:
        result = df.loc[arg]
        assert result.equals(expected)


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
def test_indexing_in_dataframe_iloc(df, index):
    expected = pd.Series([1], name=index[0])
    result = df.iloc[0]
    assert result.equals(expected)
    assert result.equals(expected)

    expected = pd.DataFrame([1, 2], index=index[:2])
    result = df.iloc[:2]
    assert result.equals(expected)


@pytest.mark.skipif(not has_cftime_or_netCDF4, reason='cftime not installed')
@pytest.mark.parametrize('enable_cftimeindex', [False, True])
def test_concat_cftimeindex(date_type, enable_cftimeindex):
    with xr.set_options(enable_cftimeindex=enable_cftimeindex):
        da1 = xr.DataArray(
            [1., 2.], coords=[[date_type(1, 1, 1), date_type(1, 2, 1)]],
            dims=['time'])
        da2 = xr.DataArray(
            [3., 4.], coords=[[date_type(1, 3, 1), date_type(1, 4, 1)]],
            dims=['time'])
        da = xr.concat([da1, da2], dim='time')

    if enable_cftimeindex and has_cftime:
        assert isinstance(da.indexes['time'], CFTimeIndex)
    else:
        assert isinstance(da.indexes['time'], pd.Index)
        assert not isinstance(da.indexes['time'], CFTimeIndex)


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
def test_empty_cftimeindex():
    index = CFTimeIndex([])
    assert index.date_type is None
