import pytest

from itertools import product

import numpy as np
import pandas as pd

from xarray.coding.cftime_offsets import (
    BaseCFTimeOffset, YearBegin, YearEnd, MonthBegin, MonthEnd,
    Day, Hour, Minute, Second, _days_in_month,
    to_offset, get_date_type, _MONTH_ABBREVIATIONS, _cftime_range,
    to_cftime_datetime, cftime_range)
from xarray import CFTimeIndex
from . import has_cftime


_CFTIME_CALENDARS = ['365_day', '360_day', 'julian', 'all_leap',
                     '366_day', 'gregorian', 'proleptic_gregorian', 'standard']


def _id_func(param):
    """Called on each parameter passed to pytest.mark.parametrize"""
    return str(param)


@pytest.fixture(params=_CFTIME_CALENDARS)
def calendar(request):
    return request.param


@pytest.mark.parametrize(
    ('offset', 'expected'),
    [(BaseCFTimeOffset(), None),
     (MonthBegin(), 'MS'),
     (YearBegin(), 'AS-JAN')],
    ids=_id_func
)
def test_rule_code(offset, expected):
    assert offset.rule_code() == expected


@pytest.mark.parametrize(
    ('offset', 'expected'),
    [(BaseCFTimeOffset(), '<BaseCFTimeOffset: n=1>'),
     (YearBegin(), '<YearBegin: n=1, month=1>')],
    ids=_id_func
)
def test_str_and_repr(offset, expected):
    assert str(offset) == expected
    assert repr(offset) == expected


@pytest.mark.parametrize(
    'offset',
    [BaseCFTimeOffset(), MonthBegin(), YearBegin()],
    ids=_id_func
)
def test_to_offset_offset_input(offset):
    assert to_offset(offset) == offset


@pytest.mark.parametrize(
    ('freq', 'expected'),
    [('M', MonthEnd()),
     ('2M', MonthEnd(n=2)),
     ('MS', MonthBegin()),
     ('2MS', MonthBegin(n=2)),
     ('D', Day()),
     ('2D', Day(n=2)),
     ('H', Hour()),
     ('2H', Hour(n=2)),
     ('T', Minute()),
     ('2T', Minute(n=2)),
     ('min', Minute()),
     ('2min', Minute(n=2)),
     ('S', Second()),
     ('2S', Second(n=2))],
    ids=_id_func
)
def test_to_offset_sub_annual(freq, expected):
    assert to_offset(freq) == expected


_ANNUAL_OFFSET_TYPES = {
    'A': YearEnd,
    'AS': YearBegin
}


@pytest.mark.parametrize(('month_int', 'month_label'),
                         list(_MONTH_ABBREVIATIONS.items()) + [('', '')])
@pytest.mark.parametrize('multiple', [None, 2])
@pytest.mark.parametrize('offset_str', ['AS', 'A'])
def test_to_offset_annual(month_label, month_int, multiple, offset_str):
    freq = offset_str
    offset_type = _ANNUAL_OFFSET_TYPES[offset_str]
    if month_label:
        freq = '-'.join([freq, month_label])
    if multiple:
        freq = '{}'.format(multiple) + freq
    result = to_offset(freq)

    if multiple and month_int:
        expected = offset_type(n=multiple, month=month_int)
    elif multiple:
        expected = offset_type(n=multiple)
    elif month_int:
        expected = offset_type(month=month_int)
    else:
        expected = offset_type()
    assert result == expected


@pytest.mark.parametrize('freq', ['Z', '7min2', 'AM', 'M-', 'AS-'])
def test_invalid_to_offset_str(freq):
    with pytest.raises(ValueError):
        to_offset(freq)


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize(
    ('argument', 'expected_date_args'),
    [('2000-01-01', (2000, 1, 1)),
     ((2000, 1, 1), (2000, 1, 1))],
    ids=_id_func
)
def test_to_cftime_datetime(calendar, argument, expected_date_args):
    date_type = get_date_type(calendar)
    expected = date_type(*expected_date_args)
    if isinstance(argument, tuple):
        argument = date_type(*argument)
    result = to_cftime_datetime(argument, calendar=calendar)
    assert result == expected


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize('argument', ['2000', 1])
def test_to_cftime_datetime_error(argument):
    with pytest.raises(ValueError):
        to_cftime_datetime(argument)


_EQ_TESTS_A = [
    BaseCFTimeOffset(), YearBegin(), YearEnd(), YearBegin(month=2),
    YearEnd(month=2), MonthBegin(), MonthEnd(), Day(), Hour(), Minute(),
    Second()
]
_EQ_TESTS_B = [
    BaseCFTimeOffset(n=2), YearBegin(n=2), YearEnd(n=2),
    YearBegin(n=2, month=2), YearEnd(n=2, month=2), MonthBegin(n=2),
    MonthEnd(n=2), Day(n=2), Hour(n=2), Minute(n=2), Second(n=2)
]


@pytest.mark.parametrize(
    ('a', 'b'), product(_EQ_TESTS_A, _EQ_TESTS_B), ids=_id_func
)
def test_neq(a, b):
    assert a != b


_EQ_TESTS_B_COPY = [
    BaseCFTimeOffset(n=2), YearBegin(n=2), YearEnd(n=2),
    YearBegin(n=2, month=2), YearEnd(n=2, month=2), MonthBegin(n=2),
    MonthEnd(n=2), Day(n=2), Hour(n=2), Minute(n=2), Second(n=2)
]


@pytest.mark.parametrize(
    ('a', 'b'), zip(_EQ_TESTS_B, _EQ_TESTS_B_COPY), ids=_id_func
)
def test_eq(a, b):
    assert a == b


_MUL_TESTS = [
    (BaseCFTimeOffset(), BaseCFTimeOffset(n=3)),
    (YearEnd(), YearEnd(n=3)),
    (YearBegin(), YearBegin(n=3)),
    (MonthEnd(), MonthEnd(n=3)),
    (MonthBegin(), MonthBegin(n=3)),
    (Day(), Day(n=3)),
    (Hour(), Hour(n=3)),
    (Minute(), Minute(n=3)),
    (Second(), Second(n=3))
]


@pytest.mark.parametrize(('offset', 'expected'), _MUL_TESTS, ids=_id_func)
def test_mul(offset, expected):
    assert offset * 3 == expected


@pytest.mark.parametrize(('offset', 'expected'), _MUL_TESTS, ids=_id_func)
def test_rmul(offset, expected):
    assert 3 * offset == expected


@pytest.mark.parametrize(
    ('offset', 'expected'),
    [(BaseCFTimeOffset(), BaseCFTimeOffset(n=-1)),
     (YearEnd(), YearEnd(n=-1)),
     (YearBegin(), YearBegin(n=-1)),
     (MonthEnd(), MonthEnd(n=-1)),
     (MonthBegin(), MonthBegin(n=-1)),
     (Day(), Day(n=-1)),
     (Hour(), Hour(n=-1)),
     (Minute(), Minute(n=-1)),
     (Second(), Second(n=-1))],
    ids=_id_func)
def test_neg(offset, expected):
    assert -offset == expected


_ADD_TESTS = [
    (Day(n=2), (1, 1, 3)),
    (Hour(n=2), (1, 1, 1, 2)),
    (Minute(n=2), (1, 1, 1, 0, 2)),
    (Second(n=2), (1, 1, 1, 0, 0, 2))
]


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize(
    ('offset', 'expected_date_args'),
    _ADD_TESTS,
    ids=_id_func
)
def test_add_sub_monthly(offset, expected_date_args, calendar):
    date_type = get_date_type(calendar)
    initial = date_type(1, 1, 1)
    expected = date_type(*expected_date_args)
    result = offset + initial
    assert result == expected


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize(
    ('offset', 'expected_date_args'),
    _ADD_TESTS,
    ids=_id_func
)
def test_radd_sub_monthly(offset, expected_date_args, calendar):
    date_type = get_date_type(calendar)
    initial = date_type(1, 1, 1)
    expected = date_type(*expected_date_args)
    result = initial + offset
    assert result == expected


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize(
    ('offset', 'expected_date_args'),
    [(Day(n=2), (1, 1, 1)),
     (Hour(n=2), (1, 1, 2, 22)),
     (Minute(n=2), (1, 1, 2, 23, 58)),
     (Second(n=2), (1, 1, 2, 23, 59, 58))],
    ids=_id_func
)
def test_rsub_sub_monthly(offset, expected_date_args, calendar):
    date_type = get_date_type(calendar)
    initial = date_type(1, 1, 3)
    expected = date_type(*expected_date_args)
    result = initial - offset
    assert result == expected


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize('offset', _EQ_TESTS_A, ids=_id_func)
def test_sub_error(offset, calendar):
    date_type = get_date_type(calendar)
    initial = date_type(1, 1, 1)
    with pytest.raises(TypeError):
        offset - initial


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize(
    ('a', 'b'),
    zip(_EQ_TESTS_A, _EQ_TESTS_B),
    ids=_id_func
)
def test_minus_offset(a, b):
    result = b - a
    expected = a
    assert result == expected


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize(
    ('a', 'b'),
    list(zip(np.roll(_EQ_TESTS_A, 1), _EQ_TESTS_B)) +
    [(YearEnd(month=1), YearEnd(month=2))],
    ids=_id_func
)
def test_minus_offset_error(a, b):
    with pytest.raises(NotImplementedError):
        b - a


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
def test_days_in_month_non_december(calendar):
    date_type = get_date_type(calendar)
    reference = date_type(1, 4, 1)
    assert _days_in_month(reference) == 30


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
def test_days_in_month_december(calendar):
    if calendar == '360_day':
        expected = 30
    else:
        expected = 31
    date_type = get_date_type(calendar)
    reference = date_type(1, 12, 5)
    assert _days_in_month(reference) == expected


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize(
    ('initial_date_args', 'offset', 'expected_date_args'),
    [((1, 1, 1), MonthBegin(), (1, 2, 1)),
     ((1, 1, 1), MonthBegin(n=2), (1, 3, 1)),
     ((1, 1, 7), MonthBegin(), (1, 2, 1)),
     ((1, 1, 7), MonthBegin(n=2), (1, 3, 1)),
     ((1, 3, 1), MonthBegin(n=-1), (1, 2, 1)),
     ((1, 3, 1), MonthBegin(n=-2), (1, 1, 1)),
     ((1, 3, 3), MonthBegin(n=-1), (1, 3, 1)),
     ((1, 3, 3), MonthBegin(n=-2), (1, 2, 1)),
     ((1, 2, 1), MonthBegin(n=14), (2, 4, 1)),
     ((2, 4, 1), MonthBegin(n=-14), (1, 2, 1)),
     ((1, 1, 1, 5, 5, 5, 5), MonthBegin(), (1, 2, 1, 5, 5, 5, 5)),
     ((1, 1, 3, 5, 5, 5, 5), MonthBegin(), (1, 2, 1, 5, 5, 5, 5)),
     ((1, 1, 3, 5, 5, 5, 5), MonthBegin(n=-1), (1, 1, 1, 5, 5, 5, 5))],
    ids=_id_func
)
def test_add_month_begin(
        calendar, initial_date_args, offset, expected_date_args):
    date_type = get_date_type(calendar)
    initial = date_type(*initial_date_args)
    result = initial + offset
    expected = date_type(*expected_date_args)
    assert result == expected


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize(
    ('initial_date_args', 'offset', 'expected_year_month',
     'expected_sub_day'),
    [((1, 1, 1), MonthEnd(), (1, 1), ()),
     ((1, 1, 1), MonthEnd(n=2), (1, 2), ()),
     ((1, 3, 1), MonthEnd(n=-1), (1, 2), ()),
     ((1, 3, 1), MonthEnd(n=-2), (1, 1), ()),
     ((1, 2, 1), MonthEnd(n=14), (2, 3), ()),
     ((2, 4, 1), MonthEnd(n=-14), (1, 2), ()),
     ((1, 1, 1, 5, 5, 5, 5), MonthEnd(), (1, 1), (5, 5, 5, 5)),
     ((1, 2, 1, 5, 5, 5, 5), MonthEnd(n=-1), (1, 1), (5, 5, 5, 5))],
    ids=_id_func
)
def test_add_month_end(
    calendar, initial_date_args, offset, expected_year_month,
    expected_sub_day
):
    date_type = get_date_type(calendar)
    initial = date_type(*initial_date_args)
    result = initial + offset
    reference_args = expected_year_month + (1,)
    reference = date_type(*reference_args)

    # Here the days at the end of each month varies based on the calendar used
    expected_date_args = (expected_year_month +
                          (_days_in_month(reference),) + expected_sub_day)
    expected = date_type(*expected_date_args)
    assert result == expected


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize(
    ('initial_year_month', 'initial_sub_day', 'offset', 'expected_year_month',
     'expected_sub_day'),
    [((1, 1), (), MonthEnd(), (1, 2), ()),
     ((1, 1), (), MonthEnd(n=2), (1, 3), ()),
     ((1, 3), (), MonthEnd(n=-1), (1, 2), ()),
     ((1, 3), (), MonthEnd(n=-2), (1, 1), ()),
     ((1, 2), (), MonthEnd(n=14), (2, 4), ()),
     ((2, 4), (), MonthEnd(n=-14), (1, 2), ()),
     ((1, 1), (5, 5, 5, 5), MonthEnd(), (1, 2), (5, 5, 5, 5)),
     ((1, 2), (5, 5, 5, 5), MonthEnd(n=-1), (1, 1), (5, 5, 5, 5))],
    ids=_id_func
)
def test_add_month_end_on_offset(
    calendar, initial_year_month, initial_sub_day, offset, expected_year_month,
    expected_sub_day
):
    date_type = get_date_type(calendar)
    reference_args = initial_year_month + (1,)
    reference = date_type(*reference_args)
    initial_date_args = (initial_year_month + (_days_in_month(reference),) +
                         initial_sub_day)
    initial = date_type(*initial_date_args)
    result = initial + offset
    reference_args = expected_year_month + (1,)
    reference = date_type(*reference_args)

    # Here the days at the end of each month varies based on the calendar used
    expected_date_args = (expected_year_month +
                          (_days_in_month(reference),) + expected_sub_day)
    expected = date_type(*expected_date_args)
    assert result == expected


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize(
    ('initial_date_args', 'offset', 'expected_date_args'),
    [((1, 1, 1), YearBegin(), (2, 1, 1)),
     ((1, 1, 1), YearBegin(n=2), (3, 1, 1)),
     ((1, 1, 1), YearBegin(month=2), (1, 2, 1)),
     ((1, 1, 7), YearBegin(n=2), (3, 1, 1)),
     ((2, 2, 1), YearBegin(n=-1), (2, 1, 1)),
     ((1, 1, 2), YearBegin(n=-1), (1, 1, 1)),
     ((1, 1, 1, 5, 5, 5, 5), YearBegin(), (2, 1, 1, 5, 5, 5, 5)),
     ((2, 1, 1, 5, 5, 5, 5), YearBegin(n=-1), (1, 1, 1, 5, 5, 5, 5))],
    ids=_id_func
)
def test_add_year_begin(calendar, initial_date_args, offset,
                        expected_date_args):
    date_type = get_date_type(calendar)
    initial = date_type(*initial_date_args)
    result = initial + offset
    expected = date_type(*expected_date_args)
    assert result == expected


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize(
    ('initial_date_args', 'offset', 'expected_year_month',
     'expected_sub_day'),
    [((1, 1, 1), YearEnd(), (1, 12), ()),
     ((1, 1, 1), YearEnd(n=2), (2, 12), ()),
     ((1, 1, 1), YearEnd(month=1), (1, 1), ()),
     ((2, 3, 1), YearEnd(n=-1), (1, 12), ()),
     ((1, 3, 1), YearEnd(n=-1, month=2), (1, 2), ()),
     ((1, 1, 1, 5, 5, 5, 5), YearEnd(), (1, 12), (5, 5, 5, 5)),
     ((1, 1, 1, 5, 5, 5, 5), YearEnd(n=2), (2, 12), (5, 5, 5, 5))],
    ids=_id_func
)
def test_add_year_end(
    calendar, initial_date_args, offset, expected_year_month,
    expected_sub_day
):
    date_type = get_date_type(calendar)
    initial = date_type(*initial_date_args)
    result = initial + offset
    reference_args = expected_year_month + (1,)
    reference = date_type(*reference_args)

    # Here the days at the end of each month varies based on the calendar used
    expected_date_args = (expected_year_month +
                          (_days_in_month(reference),) + expected_sub_day)
    expected = date_type(*expected_date_args)
    assert result == expected


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize(
    ('initial_year_month', 'initial_sub_day', 'offset', 'expected_year_month',
     'expected_sub_day'),
    [((1, 12), (), YearEnd(), (2, 12), ()),
     ((1, 12), (), YearEnd(n=2), (3, 12), ()),
     ((2, 12), (), YearEnd(n=-1), (1, 12), ()),
     ((3, 12), (), YearEnd(n=-2), (1, 12), ()),
     ((1, 1), (), YearEnd(month=2), (1, 2), ()),
     ((1, 12), (5, 5, 5, 5), YearEnd(), (2, 12), (5, 5, 5, 5)),
     ((2, 12), (5, 5, 5, 5), YearEnd(n=-1), (1, 12), (5, 5, 5, 5))],
    ids=_id_func
)
def test_add_year_end_on_offset(
    calendar, initial_year_month, initial_sub_day, offset, expected_year_month,
    expected_sub_day
):
    date_type = get_date_type(calendar)
    reference_args = initial_year_month + (1,)
    reference = date_type(*reference_args)
    initial_date_args = (initial_year_month + (_days_in_month(reference),) +
                         initial_sub_day)
    initial = date_type(*initial_date_args)
    result = initial + offset
    reference_args = expected_year_month + (1,)
    reference = date_type(*reference_args)

    # Here the days at the end of each month varies based on the calendar used
    expected_date_args = (expected_year_month +
                          (_days_in_month(reference),) + expected_sub_day)
    expected = date_type(*expected_date_args)
    assert result == expected


# Note for all sub-monthly offsets, pandas always returns True for on_offset
@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize(
    ('date_args', 'offset', 'expected'),
    [((1, 1, 1), MonthBegin(), True),
     ((1, 1, 1, 1), MonthBegin(), True),
     ((1, 1, 5), MonthBegin(), False),
     ((1, 1, 5), MonthEnd(), False),
     ((1, 1, 1), YearBegin(), True),
     ((1, 1, 1, 1), YearBegin(), True),
     ((1, 1, 5), YearBegin(), False),
     ((1, 12, 1), YearEnd(), False),
     ((1, 1, 1), Day(), True),
     ((1, 1, 1, 1), Day(), True),
     ((1, 1, 1), Hour(), True),
     ((1, 1, 1), Minute(), True),
     ((1, 1, 1), Second(), True)],
    ids=_id_func
)
def test_on_offset(calendar, date_args, offset, expected):
    date_type = get_date_type(calendar)
    date = date_type(*date_args)
    result = offset.on_offset(date)
    assert result == expected


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize(
    ('year_month_args', 'sub_day_args', 'offset'),
    [((1, 1), (), MonthEnd()),
     ((1, 1), (1,), MonthEnd()),
     ((1, 12), (), YearEnd()),
     ((1, 1), (), YearEnd(month=1))],
    ids=_id_func
)
def test_on_offset_month_or_year_end(
        calendar, year_month_args, sub_day_args, offset):
    date_type = get_date_type(calendar)
    reference_args = year_month_args + (1,)
    reference = date_type(*reference_args)
    date_args = year_month_args + (_days_in_month(reference),) + sub_day_args
    date = date_type(*date_args)
    result = offset.on_offset(date)
    assert result


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize(
    ('offset', 'initial_date_args', 'expected_month_year'),
    [(YearBegin(), (1, 3, 1), (2, 1)),
     (YearBegin(n=2), (1, 3, 1), (2, 1)),
     (YearBegin(n=2, month=2), (1, 3, 1), (2, 2)),
     (YearEnd(), (1, 3, 1), (1, 12)),
     (YearEnd(n=2), (1, 3, 1), (1, 12)),
     (YearEnd(n=2, month=2), (1, 3, 1), (2, 2)),
     (MonthBegin(), (1, 3, 2), (1, 4)),
     (MonthBegin(n=2), (1, 3, 2), (1, 4)),
     (MonthEnd(), (1, 3, 2), (1, 3)),
     (MonthEnd(n=2), (1, 3, 2), (1, 3))],
    ids=_id_func
)
def test_roll_forward(calendar, offset, initial_date_args,
                      expected_month_year):
    date_type = get_date_type(calendar)
    initial = date_type(*initial_date_args)
    if isinstance(offset, (MonthBegin, YearBegin)):
        expected_date_args = expected_month_year + (1,)
    else:
        reference_args = expected_month_year + (1,)
        reference = date_type(*reference_args)
        expected_date_args = expected_month_year + (_days_in_month(reference),)
    expected = date_type(*expected_date_args)
    result = offset.roll_forward(initial)
    assert result == expected


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize(
    ('offset', 'initial_date_args', 'expected_month_year'),
    [(YearBegin(), (1, 3, 1), (1, 1)),
     (YearBegin(n=2), (1, 3, 1), (1, 1)),
     (YearBegin(n=2, month=2), (1, 3, 1), (1, 2)),
     (YearEnd(), (2, 3, 1), (1, 12)),
     (YearEnd(n=2), (2, 3, 1), (1, 12)),
     (YearEnd(n=2, month=2), (2, 3, 1), (2, 2)),
     (MonthBegin(), (1, 3, 2), (1, 3)),
     (MonthBegin(n=2), (1, 3, 2), (1, 3)),
     (MonthEnd(), (1, 3, 2), (1, 2)),
     (MonthEnd(n=2), (1, 3, 2), (1, 2))],
    ids=_id_func
)
def test_roll_backward(calendar, offset, initial_date_args,
                       expected_month_year):
    date_type = get_date_type(calendar)
    initial = date_type(*initial_date_args)
    if isinstance(offset, (MonthBegin, YearBegin)):
        expected_date_args = expected_month_year + (1,)
    else:
        reference_args = expected_month_year + (1,)
        reference = date_type(*reference_args)
        expected_date_args = expected_month_year + (_days_in_month(reference),)
    expected = date_type(*expected_date_args)
    result = offset.roll_backward(initial)
    assert result == expected


_CFTIME_RANGE_TESTS = [
    ('0001-01-01', '0001-01-04', None, 'D', None, False,
     [(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 1, 4)]),
    ('0001-01-01', '0001-01-04', None, 'D', 'left', False,
     [(1, 1, 1), (1, 1, 2), (1, 1, 3)]),
    ('0001-01-01', '0001-01-04', None, 'D', 'right', False,
     [(1, 1, 2), (1, 1, 3), (1, 1, 4)]),
    ('0001-01-01T01:00:00', '0001-01-04', None, 'D', None, False,
     [(1, 1, 1, 1), (1, 1, 2, 1), (1, 1, 3, 1)]),
    ('0001-01-01T01:00:00', '0001-01-04', None, 'D', None, True,
     [(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 1, 4)]),
    ('0001-01-01', None, 4, 'D', None, False,
     [(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 1, 4)]),
    (None, '0001-01-04', 4, 'D', None, False,
     [(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 1, 4)]),
    ((1, 1, 1), '0001-01-04', None, 'D', None, False,
     [(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 1, 4)]),
    ((1, 1, 1), (1, 1, 4), None, 'D', None, False,
     [(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 1, 4)]),
    ('0001-01-30', '0011-02-01', None, '3AS-JUN', None, False,
     [(1, 6, 1), (4, 6, 1), (7, 6, 1), (10, 6, 1)]),
    ('0001-01-04', '0001-01-01', None, 'D', None, False,
     []),
    ('0010', None, 4, YearBegin(n=-2), None, False,
     [(10, 1, 1), (8, 1, 1), (6, 1, 1), (4, 1, 1)]),
    ('0001-01-01', '0001-01-04', 4, None, None, False,
     [(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 1, 4)])
]


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize(
    ('start', 'end', 'periods', 'freq', 'closed', 'normalize',
     'expected_date_args'),
    _CFTIME_RANGE_TESTS, ids=_id_func
)
def test_private_cftime_range(
        start, end, periods, freq, closed, normalize, calendar,
        expected_date_args):
    date_type = get_date_type(calendar)
    if isinstance(start, tuple):
        start = date_type(*start)
    if isinstance(end, tuple):
        end = date_type(*end)
    result = _cftime_range(
        start, end, periods, freq, closed, normalize, calendar)
    expected = [date_type(*args) for args in expected_date_args]
    if freq is not None:
        np.testing.assert_equal(result, expected)
    else:
        # If we create a linear range of dates using cftime.num2date
        # we will not get exact round number dates.  This is because
        # datetime arithmetic in cftime is accurate approximately to
        # 1 millisecond (see https://unidata.github.io/cftime/api.html).
        deltas = result - expected
        deltas = np.array([delta.total_seconds() for delta in deltas])
        assert np.max(np.abs(deltas)) < 0.001


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize(
    ('start', 'end', 'periods', 'freq', 'closed'),
    [(None, None, 5, 'A', None),
     ('2000', None, None, 'A', None),
     (None, '2000', None, 'A', None),
     ('2000', '2001', None, None, None),
     (None, None, None, None, None),
     ('2000', '2001', None, 'A', 'up'),
     ('2000', '2001', 5, 'A', None)]
)
def test_invalid_cftime_range_inputs(start, end, periods, freq, closed):
    with pytest.raises(ValueError):
        _cftime_range(start, end, periods, freq, closed=closed)


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
@pytest.mark.parametrize(
    ('start', 'end', 'periods', 'freq', 'name'),
    [('0001', None, 5, 'A', 'foo'),
     ('2000', None, 5, 'A', 'foo'),
     ('2000', '1999', None, 'A', 'foo')]
)
def test_cftime_range(start, end, periods, freq, name, calendar):
    result = cftime_range(start, end, periods,
                          freq, name=name, calendar=calendar)
    if start == '2000' and calendar == 'standard':
        assert isinstance(result, pd.DatetimeIndex)
    else:
        assert isinstance(result, CFTimeIndex)
    assert result.name == name


@pytest.mark.skipif(not has_cftime, reason='cftime not installed')
def test_cftime_range_invalid_tz_input(calendar):
    with pytest.raises(ValueError):
        cftime_range('0001', '0002', None, 'M', tz='Asia/Hong_Kong',
                     calendar=calendar)
