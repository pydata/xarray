from itertools import product

import numpy as np
import pandas as pd
import pytest

from xarray import CFTimeIndex
from xarray.coding.cftime_offsets import (
    _MONTH_ABBREVIATIONS,
    BaseCFTimeOffset,
    Day,
    Hour,
    Minute,
    MonthBegin,
    MonthEnd,
    QuarterBegin,
    QuarterEnd,
    Second,
    YearBegin,
    YearEnd,
    _days_in_month,
    cftime_range,
    get_date_type,
    to_cftime_datetime,
    to_offset,
)

cftime = pytest.importorskip("cftime")


_CFTIME_CALENDARS = [
    "365_day",
    "360_day",
    "julian",
    "all_leap",
    "366_day",
    "gregorian",
    "proleptic_gregorian",
    "standard",
]


def _id_func(param):
    """Called on each parameter passed to pytest.mark.parametrize"""
    return str(param)


@pytest.fixture(params=_CFTIME_CALENDARS)
def calendar(request):
    return request.param


@pytest.mark.parametrize(
    ("offset", "expected_n"),
    [
        (BaseCFTimeOffset(), 1),
        (YearBegin(), 1),
        (YearEnd(), 1),
        (QuarterBegin(), 1),
        (QuarterEnd(), 1),
        (BaseCFTimeOffset(n=2), 2),
        (YearBegin(n=2), 2),
        (YearEnd(n=2), 2),
        (QuarterBegin(n=2), 2),
        (QuarterEnd(n=2), 2),
    ],
    ids=_id_func,
)
def test_cftime_offset_constructor_valid_n(offset, expected_n):
    assert offset.n == expected_n


@pytest.mark.parametrize(
    ("offset", "invalid_n"),
    [
        (BaseCFTimeOffset, 1.5),
        (YearBegin, 1.5),
        (YearEnd, 1.5),
        (QuarterBegin, 1.5),
        (QuarterEnd, 1.5),
    ],
    ids=_id_func,
)
def test_cftime_offset_constructor_invalid_n(offset, invalid_n):
    with pytest.raises(TypeError):
        offset(n=invalid_n)


@pytest.mark.parametrize(
    ("offset", "expected_month"),
    [
        (YearBegin(), 1),
        (YearEnd(), 12),
        (YearBegin(month=5), 5),
        (YearEnd(month=5), 5),
        (QuarterBegin(), 3),
        (QuarterEnd(), 3),
        (QuarterBegin(month=5), 5),
        (QuarterEnd(month=5), 5),
    ],
    ids=_id_func,
)
def test_year_offset_constructor_valid_month(offset, expected_month):
    assert offset.month == expected_month


@pytest.mark.parametrize(
    ("offset", "invalid_month", "exception"),
    [
        (YearBegin, 0, ValueError),
        (YearEnd, 0, ValueError),
        (YearBegin, 13, ValueError),
        (YearEnd, 13, ValueError),
        (YearBegin, 1.5, TypeError),
        (YearEnd, 1.5, TypeError),
        (QuarterBegin, 0, ValueError),
        (QuarterEnd, 0, ValueError),
        (QuarterBegin, 1.5, TypeError),
        (QuarterEnd, 1.5, TypeError),
        (QuarterBegin, 13, ValueError),
        (QuarterEnd, 13, ValueError),
    ],
    ids=_id_func,
)
def test_year_offset_constructor_invalid_month(offset, invalid_month, exception):
    with pytest.raises(exception):
        offset(month=invalid_month)


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        (BaseCFTimeOffset(), None),
        (MonthBegin(), "MS"),
        (YearBegin(), "AS-JAN"),
        (QuarterBegin(), "QS-MAR"),
    ],
    ids=_id_func,
)
def test_rule_code(offset, expected):
    assert offset.rule_code() == expected


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        (BaseCFTimeOffset(), "<BaseCFTimeOffset: n=1>"),
        (YearBegin(), "<YearBegin: n=1, month=1>"),
        (QuarterBegin(), "<QuarterBegin: n=1, month=3>"),
    ],
    ids=_id_func,
)
def test_str_and_repr(offset, expected):
    assert str(offset) == expected
    assert repr(offset) == expected


@pytest.mark.parametrize(
    "offset",
    [BaseCFTimeOffset(), MonthBegin(), QuarterBegin(), YearBegin()],
    ids=_id_func,
)
def test_to_offset_offset_input(offset):
    assert to_offset(offset) == offset


@pytest.mark.parametrize(
    ("freq", "expected"),
    [
        ("M", MonthEnd()),
        ("2M", MonthEnd(n=2)),
        ("MS", MonthBegin()),
        ("2MS", MonthBegin(n=2)),
        ("D", Day()),
        ("2D", Day(n=2)),
        ("H", Hour()),
        ("2H", Hour(n=2)),
        ("T", Minute()),
        ("2T", Minute(n=2)),
        ("min", Minute()),
        ("2min", Minute(n=2)),
        ("S", Second()),
        ("2S", Second(n=2)),
    ],
    ids=_id_func,
)
def test_to_offset_sub_annual(freq, expected):
    assert to_offset(freq) == expected


_ANNUAL_OFFSET_TYPES = {"A": YearEnd, "AS": YearBegin}


@pytest.mark.parametrize(
    ("month_int", "month_label"), list(_MONTH_ABBREVIATIONS.items()) + [(0, "")]
)
@pytest.mark.parametrize("multiple", [None, 2])
@pytest.mark.parametrize("offset_str", ["AS", "A"])
def test_to_offset_annual(month_label, month_int, multiple, offset_str):
    freq = offset_str
    offset_type = _ANNUAL_OFFSET_TYPES[offset_str]
    if month_label:
        freq = "-".join([freq, month_label])
    if multiple:
        freq = f"{multiple}{freq}"
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


_QUARTER_OFFSET_TYPES = {"Q": QuarterEnd, "QS": QuarterBegin}


@pytest.mark.parametrize(
    ("month_int", "month_label"), list(_MONTH_ABBREVIATIONS.items()) + [(0, "")]
)
@pytest.mark.parametrize("multiple", [None, 2])
@pytest.mark.parametrize("offset_str", ["QS", "Q"])
def test_to_offset_quarter(month_label, month_int, multiple, offset_str):
    freq = offset_str
    offset_type = _QUARTER_OFFSET_TYPES[offset_str]
    if month_label:
        freq = "-".join([freq, month_label])
    if multiple:
        freq = f"{multiple}{freq}"
    result = to_offset(freq)

    if multiple and month_int:
        expected = offset_type(n=multiple, month=month_int)
    elif multiple:
        if month_int:
            expected = offset_type(n=multiple)
        else:
            if offset_type == QuarterBegin:
                expected = offset_type(n=multiple, month=1)
            elif offset_type == QuarterEnd:
                expected = offset_type(n=multiple, month=12)
    elif month_int:
        expected = offset_type(month=month_int)
    else:
        if offset_type == QuarterBegin:
            expected = offset_type(month=1)
        elif offset_type == QuarterEnd:
            expected = offset_type(month=12)
    assert result == expected


@pytest.mark.parametrize("freq", ["Z", "7min2", "AM", "M-", "AS-", "QS-", "1H1min"])
def test_invalid_to_offset_str(freq):
    with pytest.raises(ValueError):
        to_offset(freq)


@pytest.mark.parametrize(
    ("argument", "expected_date_args"),
    [("2000-01-01", (2000, 1, 1)), ((2000, 1, 1), (2000, 1, 1))],
    ids=_id_func,
)
def test_to_cftime_datetime(calendar, argument, expected_date_args):
    date_type = get_date_type(calendar)
    expected = date_type(*expected_date_args)
    if isinstance(argument, tuple):
        argument = date_type(*argument)
    result = to_cftime_datetime(argument, calendar=calendar)
    assert result == expected


def test_to_cftime_datetime_error_no_calendar():
    with pytest.raises(ValueError):
        to_cftime_datetime("2000")


def test_to_cftime_datetime_error_type_error():
    with pytest.raises(TypeError):
        to_cftime_datetime(1)


_EQ_TESTS_A = [
    BaseCFTimeOffset(),
    YearBegin(),
    YearEnd(),
    YearBegin(month=2),
    YearEnd(month=2),
    QuarterBegin(),
    QuarterEnd(),
    QuarterBegin(month=2),
    QuarterEnd(month=2),
    MonthBegin(),
    MonthEnd(),
    Day(),
    Hour(),
    Minute(),
    Second(),
]
_EQ_TESTS_B = [
    BaseCFTimeOffset(n=2),
    YearBegin(n=2),
    YearEnd(n=2),
    YearBegin(n=2, month=2),
    YearEnd(n=2, month=2),
    QuarterBegin(n=2),
    QuarterEnd(n=2),
    QuarterBegin(n=2, month=2),
    QuarterEnd(n=2, month=2),
    MonthBegin(n=2),
    MonthEnd(n=2),
    Day(n=2),
    Hour(n=2),
    Minute(n=2),
    Second(n=2),
]


@pytest.mark.parametrize(("a", "b"), product(_EQ_TESTS_A, _EQ_TESTS_B), ids=_id_func)
def test_neq(a, b):
    assert a != b


_EQ_TESTS_B_COPY = [
    BaseCFTimeOffset(n=2),
    YearBegin(n=2),
    YearEnd(n=2),
    YearBegin(n=2, month=2),
    YearEnd(n=2, month=2),
    QuarterBegin(n=2),
    QuarterEnd(n=2),
    QuarterBegin(n=2, month=2),
    QuarterEnd(n=2, month=2),
    MonthBegin(n=2),
    MonthEnd(n=2),
    Day(n=2),
    Hour(n=2),
    Minute(n=2),
    Second(n=2),
]


@pytest.mark.parametrize(("a", "b"), zip(_EQ_TESTS_B, _EQ_TESTS_B_COPY), ids=_id_func)
def test_eq(a, b):
    assert a == b


_MUL_TESTS = [
    (BaseCFTimeOffset(), BaseCFTimeOffset(n=3)),
    (YearEnd(), YearEnd(n=3)),
    (YearBegin(), YearBegin(n=3)),
    (QuarterEnd(), QuarterEnd(n=3)),
    (QuarterBegin(), QuarterBegin(n=3)),
    (MonthEnd(), MonthEnd(n=3)),
    (MonthBegin(), MonthBegin(n=3)),
    (Day(), Day(n=3)),
    (Hour(), Hour(n=3)),
    (Minute(), Minute(n=3)),
    (Second(), Second(n=3)),
]


@pytest.mark.parametrize(("offset", "expected"), _MUL_TESTS, ids=_id_func)
def test_mul(offset, expected):
    assert offset * 3 == expected


@pytest.mark.parametrize(("offset", "expected"), _MUL_TESTS, ids=_id_func)
def test_rmul(offset, expected):
    assert 3 * offset == expected


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        (BaseCFTimeOffset(), BaseCFTimeOffset(n=-1)),
        (YearEnd(), YearEnd(n=-1)),
        (YearBegin(), YearBegin(n=-1)),
        (QuarterEnd(), QuarterEnd(n=-1)),
        (QuarterBegin(), QuarterBegin(n=-1)),
        (MonthEnd(), MonthEnd(n=-1)),
        (MonthBegin(), MonthBegin(n=-1)),
        (Day(), Day(n=-1)),
        (Hour(), Hour(n=-1)),
        (Minute(), Minute(n=-1)),
        (Second(), Second(n=-1)),
    ],
    ids=_id_func,
)
def test_neg(offset, expected):
    assert -offset == expected


_ADD_TESTS = [
    (Day(n=2), (1, 1, 3)),
    (Hour(n=2), (1, 1, 1, 2)),
    (Minute(n=2), (1, 1, 1, 0, 2)),
    (Second(n=2), (1, 1, 1, 0, 0, 2)),
]


@pytest.mark.parametrize(("offset", "expected_date_args"), _ADD_TESTS, ids=_id_func)
def test_add_sub_monthly(offset, expected_date_args, calendar):
    date_type = get_date_type(calendar)
    initial = date_type(1, 1, 1)
    expected = date_type(*expected_date_args)
    result = offset + initial
    assert result == expected


@pytest.mark.parametrize(("offset", "expected_date_args"), _ADD_TESTS, ids=_id_func)
def test_radd_sub_monthly(offset, expected_date_args, calendar):
    date_type = get_date_type(calendar)
    initial = date_type(1, 1, 1)
    expected = date_type(*expected_date_args)
    result = initial + offset
    assert result == expected


@pytest.mark.parametrize(
    ("offset", "expected_date_args"),
    [
        (Day(n=2), (1, 1, 1)),
        (Hour(n=2), (1, 1, 2, 22)),
        (Minute(n=2), (1, 1, 2, 23, 58)),
        (Second(n=2), (1, 1, 2, 23, 59, 58)),
    ],
    ids=_id_func,
)
def test_rsub_sub_monthly(offset, expected_date_args, calendar):
    date_type = get_date_type(calendar)
    initial = date_type(1, 1, 3)
    expected = date_type(*expected_date_args)
    result = initial - offset
    assert result == expected


@pytest.mark.parametrize("offset", _EQ_TESTS_A, ids=_id_func)
def test_sub_error(offset, calendar):
    date_type = get_date_type(calendar)
    initial = date_type(1, 1, 1)
    with pytest.raises(TypeError):
        offset - initial


@pytest.mark.parametrize(("a", "b"), zip(_EQ_TESTS_A, _EQ_TESTS_B), ids=_id_func)
def test_minus_offset(a, b):
    result = b - a
    expected = a
    assert result == expected


@pytest.mark.parametrize(
    ("a", "b"),
    list(zip(np.roll(_EQ_TESTS_A, 1), _EQ_TESTS_B))
    + [(YearEnd(month=1), YearEnd(month=2))],
    ids=_id_func,
)
def test_minus_offset_error(a, b):
    with pytest.raises(TypeError):
        b - a


def test_days_in_month_non_december(calendar):
    date_type = get_date_type(calendar)
    reference = date_type(1, 4, 1)
    assert _days_in_month(reference) == 30


def test_days_in_month_december(calendar):
    if calendar == "360_day":
        expected = 30
    else:
        expected = 31
    date_type = get_date_type(calendar)
    reference = date_type(1, 12, 5)
    assert _days_in_month(reference) == expected


@pytest.mark.parametrize(
    ("initial_date_args", "offset", "expected_date_args"),
    [
        ((1, 1, 1), MonthBegin(), (1, 2, 1)),
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
        ((1, 1, 3, 5, 5, 5, 5), MonthBegin(n=-1), (1, 1, 1, 5, 5, 5, 5)),
    ],
    ids=_id_func,
)
def test_add_month_begin(calendar, initial_date_args, offset, expected_date_args):
    date_type = get_date_type(calendar)
    initial = date_type(*initial_date_args)
    result = initial + offset
    expected = date_type(*expected_date_args)
    assert result == expected


@pytest.mark.parametrize(
    ("initial_date_args", "offset", "expected_year_month", "expected_sub_day"),
    [
        ((1, 1, 1), MonthEnd(), (1, 1), ()),
        ((1, 1, 1), MonthEnd(n=2), (1, 2), ()),
        ((1, 3, 1), MonthEnd(n=-1), (1, 2), ()),
        ((1, 3, 1), MonthEnd(n=-2), (1, 1), ()),
        ((1, 2, 1), MonthEnd(n=14), (2, 3), ()),
        ((2, 4, 1), MonthEnd(n=-14), (1, 2), ()),
        ((1, 1, 1, 5, 5, 5, 5), MonthEnd(), (1, 1), (5, 5, 5, 5)),
        ((1, 2, 1, 5, 5, 5, 5), MonthEnd(n=-1), (1, 1), (5, 5, 5, 5)),
    ],
    ids=_id_func,
)
def test_add_month_end(
    calendar, initial_date_args, offset, expected_year_month, expected_sub_day
):
    date_type = get_date_type(calendar)
    initial = date_type(*initial_date_args)
    result = initial + offset
    reference_args = expected_year_month + (1,)
    reference = date_type(*reference_args)

    # Here the days at the end of each month varies based on the calendar used
    expected_date_args = (
        expected_year_month + (_days_in_month(reference),) + expected_sub_day
    )
    expected = date_type(*expected_date_args)
    assert result == expected


@pytest.mark.parametrize(
    (
        "initial_year_month",
        "initial_sub_day",
        "offset",
        "expected_year_month",
        "expected_sub_day",
    ),
    [
        ((1, 1), (), MonthEnd(), (1, 2), ()),
        ((1, 1), (), MonthEnd(n=2), (1, 3), ()),
        ((1, 3), (), MonthEnd(n=-1), (1, 2), ()),
        ((1, 3), (), MonthEnd(n=-2), (1, 1), ()),
        ((1, 2), (), MonthEnd(n=14), (2, 4), ()),
        ((2, 4), (), MonthEnd(n=-14), (1, 2), ()),
        ((1, 1), (5, 5, 5, 5), MonthEnd(), (1, 2), (5, 5, 5, 5)),
        ((1, 2), (5, 5, 5, 5), MonthEnd(n=-1), (1, 1), (5, 5, 5, 5)),
    ],
    ids=_id_func,
)
def test_add_month_end_onOffset(
    calendar,
    initial_year_month,
    initial_sub_day,
    offset,
    expected_year_month,
    expected_sub_day,
):
    date_type = get_date_type(calendar)
    reference_args = initial_year_month + (1,)
    reference = date_type(*reference_args)
    initial_date_args = (
        initial_year_month + (_days_in_month(reference),) + initial_sub_day
    )
    initial = date_type(*initial_date_args)
    result = initial + offset
    reference_args = expected_year_month + (1,)
    reference = date_type(*reference_args)

    # Here the days at the end of each month varies based on the calendar used
    expected_date_args = (
        expected_year_month + (_days_in_month(reference),) + expected_sub_day
    )
    expected = date_type(*expected_date_args)
    assert result == expected


@pytest.mark.parametrize(
    ("initial_date_args", "offset", "expected_date_args"),
    [
        ((1, 1, 1), YearBegin(), (2, 1, 1)),
        ((1, 1, 1), YearBegin(n=2), (3, 1, 1)),
        ((1, 1, 1), YearBegin(month=2), (1, 2, 1)),
        ((1, 1, 7), YearBegin(n=2), (3, 1, 1)),
        ((2, 2, 1), YearBegin(n=-1), (2, 1, 1)),
        ((1, 1, 2), YearBegin(n=-1), (1, 1, 1)),
        ((1, 1, 1, 5, 5, 5, 5), YearBegin(), (2, 1, 1, 5, 5, 5, 5)),
        ((2, 1, 1, 5, 5, 5, 5), YearBegin(n=-1), (1, 1, 1, 5, 5, 5, 5)),
    ],
    ids=_id_func,
)
def test_add_year_begin(calendar, initial_date_args, offset, expected_date_args):
    date_type = get_date_type(calendar)
    initial = date_type(*initial_date_args)
    result = initial + offset
    expected = date_type(*expected_date_args)
    assert result == expected


@pytest.mark.parametrize(
    ("initial_date_args", "offset", "expected_year_month", "expected_sub_day"),
    [
        ((1, 1, 1), YearEnd(), (1, 12), ()),
        ((1, 1, 1), YearEnd(n=2), (2, 12), ()),
        ((1, 1, 1), YearEnd(month=1), (1, 1), ()),
        ((2, 3, 1), YearEnd(n=-1), (1, 12), ()),
        ((1, 3, 1), YearEnd(n=-1, month=2), (1, 2), ()),
        ((1, 1, 1, 5, 5, 5, 5), YearEnd(), (1, 12), (5, 5, 5, 5)),
        ((1, 1, 1, 5, 5, 5, 5), YearEnd(n=2), (2, 12), (5, 5, 5, 5)),
    ],
    ids=_id_func,
)
def test_add_year_end(
    calendar, initial_date_args, offset, expected_year_month, expected_sub_day
):
    date_type = get_date_type(calendar)
    initial = date_type(*initial_date_args)
    result = initial + offset
    reference_args = expected_year_month + (1,)
    reference = date_type(*reference_args)

    # Here the days at the end of each month varies based on the calendar used
    expected_date_args = (
        expected_year_month + (_days_in_month(reference),) + expected_sub_day
    )
    expected = date_type(*expected_date_args)
    assert result == expected


@pytest.mark.parametrize(
    (
        "initial_year_month",
        "initial_sub_day",
        "offset",
        "expected_year_month",
        "expected_sub_day",
    ),
    [
        ((1, 12), (), YearEnd(), (2, 12), ()),
        ((1, 12), (), YearEnd(n=2), (3, 12), ()),
        ((2, 12), (), YearEnd(n=-1), (1, 12), ()),
        ((3, 12), (), YearEnd(n=-2), (1, 12), ()),
        ((1, 1), (), YearEnd(month=2), (1, 2), ()),
        ((1, 12), (5, 5, 5, 5), YearEnd(), (2, 12), (5, 5, 5, 5)),
        ((2, 12), (5, 5, 5, 5), YearEnd(n=-1), (1, 12), (5, 5, 5, 5)),
    ],
    ids=_id_func,
)
def test_add_year_end_onOffset(
    calendar,
    initial_year_month,
    initial_sub_day,
    offset,
    expected_year_month,
    expected_sub_day,
):
    date_type = get_date_type(calendar)
    reference_args = initial_year_month + (1,)
    reference = date_type(*reference_args)
    initial_date_args = (
        initial_year_month + (_days_in_month(reference),) + initial_sub_day
    )
    initial = date_type(*initial_date_args)
    result = initial + offset
    reference_args = expected_year_month + (1,)
    reference = date_type(*reference_args)

    # Here the days at the end of each month varies based on the calendar used
    expected_date_args = (
        expected_year_month + (_days_in_month(reference),) + expected_sub_day
    )
    expected = date_type(*expected_date_args)
    assert result == expected


@pytest.mark.parametrize(
    ("initial_date_args", "offset", "expected_date_args"),
    [
        ((1, 1, 1), QuarterBegin(), (1, 3, 1)),
        ((1, 1, 1), QuarterBegin(n=2), (1, 6, 1)),
        ((1, 1, 1), QuarterBegin(month=2), (1, 2, 1)),
        ((1, 1, 7), QuarterBegin(n=2), (1, 6, 1)),
        ((2, 2, 1), QuarterBegin(n=-1), (1, 12, 1)),
        ((1, 3, 2), QuarterBegin(n=-1), (1, 3, 1)),
        ((1, 1, 1, 5, 5, 5, 5), QuarterBegin(), (1, 3, 1, 5, 5, 5, 5)),
        ((2, 1, 1, 5, 5, 5, 5), QuarterBegin(n=-1), (1, 12, 1, 5, 5, 5, 5)),
    ],
    ids=_id_func,
)
def test_add_quarter_begin(calendar, initial_date_args, offset, expected_date_args):
    date_type = get_date_type(calendar)
    initial = date_type(*initial_date_args)
    result = initial + offset
    expected = date_type(*expected_date_args)
    assert result == expected


@pytest.mark.parametrize(
    ("initial_date_args", "offset", "expected_year_month", "expected_sub_day"),
    [
        ((1, 1, 1), QuarterEnd(), (1, 3), ()),
        ((1, 1, 1), QuarterEnd(n=2), (1, 6), ()),
        ((1, 1, 1), QuarterEnd(month=1), (1, 1), ()),
        ((2, 3, 1), QuarterEnd(n=-1), (1, 12), ()),
        ((1, 3, 1), QuarterEnd(n=-1, month=2), (1, 2), ()),
        ((1, 1, 1, 5, 5, 5, 5), QuarterEnd(), (1, 3), (5, 5, 5, 5)),
        ((1, 1, 1, 5, 5, 5, 5), QuarterEnd(n=2), (1, 6), (5, 5, 5, 5)),
    ],
    ids=_id_func,
)
def test_add_quarter_end(
    calendar, initial_date_args, offset, expected_year_month, expected_sub_day
):
    date_type = get_date_type(calendar)
    initial = date_type(*initial_date_args)
    result = initial + offset
    reference_args = expected_year_month + (1,)
    reference = date_type(*reference_args)

    # Here the days at the end of each month varies based on the calendar used
    expected_date_args = (
        expected_year_month + (_days_in_month(reference),) + expected_sub_day
    )
    expected = date_type(*expected_date_args)
    assert result == expected


@pytest.mark.parametrize(
    (
        "initial_year_month",
        "initial_sub_day",
        "offset",
        "expected_year_month",
        "expected_sub_day",
    ),
    [
        ((1, 12), (), QuarterEnd(), (2, 3), ()),
        ((1, 12), (), QuarterEnd(n=2), (2, 6), ()),
        ((1, 12), (), QuarterEnd(n=-1), (1, 9), ()),
        ((1, 12), (), QuarterEnd(n=-2), (1, 6), ()),
        ((1, 1), (), QuarterEnd(month=2), (1, 2), ()),
        ((1, 12), (5, 5, 5, 5), QuarterEnd(), (2, 3), (5, 5, 5, 5)),
        ((1, 12), (5, 5, 5, 5), QuarterEnd(n=-1), (1, 9), (5, 5, 5, 5)),
    ],
    ids=_id_func,
)
def test_add_quarter_end_onOffset(
    calendar,
    initial_year_month,
    initial_sub_day,
    offset,
    expected_year_month,
    expected_sub_day,
):
    date_type = get_date_type(calendar)
    reference_args = initial_year_month + (1,)
    reference = date_type(*reference_args)
    initial_date_args = (
        initial_year_month + (_days_in_month(reference),) + initial_sub_day
    )
    initial = date_type(*initial_date_args)
    result = initial + offset
    reference_args = expected_year_month + (1,)
    reference = date_type(*reference_args)

    # Here the days at the end of each month varies based on the calendar used
    expected_date_args = (
        expected_year_month + (_days_in_month(reference),) + expected_sub_day
    )
    expected = date_type(*expected_date_args)
    assert result == expected


# Note for all sub-monthly offsets, pandas always returns True for onOffset
@pytest.mark.parametrize(
    ("date_args", "offset", "expected"),
    [
        ((1, 1, 1), MonthBegin(), True),
        ((1, 1, 1, 1), MonthBegin(), True),
        ((1, 1, 5), MonthBegin(), False),
        ((1, 1, 5), MonthEnd(), False),
        ((1, 3, 1), QuarterBegin(), True),
        ((1, 3, 1, 1), QuarterBegin(), True),
        ((1, 3, 5), QuarterBegin(), False),
        ((1, 12, 1), QuarterEnd(), False),
        ((1, 1, 1), YearBegin(), True),
        ((1, 1, 1, 1), YearBegin(), True),
        ((1, 1, 5), YearBegin(), False),
        ((1, 12, 1), YearEnd(), False),
        ((1, 1, 1), Day(), True),
        ((1, 1, 1, 1), Day(), True),
        ((1, 1, 1), Hour(), True),
        ((1, 1, 1), Minute(), True),
        ((1, 1, 1), Second(), True),
    ],
    ids=_id_func,
)
def test_onOffset(calendar, date_args, offset, expected):
    date_type = get_date_type(calendar)
    date = date_type(*date_args)
    result = offset.onOffset(date)
    assert result == expected


@pytest.mark.parametrize(
    ("year_month_args", "sub_day_args", "offset"),
    [
        ((1, 1), (), MonthEnd()),
        ((1, 1), (1,), MonthEnd()),
        ((1, 12), (), QuarterEnd()),
        ((1, 1), (), QuarterEnd(month=1)),
        ((1, 12), (), YearEnd()),
        ((1, 1), (), YearEnd(month=1)),
    ],
    ids=_id_func,
)
def test_onOffset_month_or_quarter_or_year_end(
    calendar, year_month_args, sub_day_args, offset
):
    date_type = get_date_type(calendar)
    reference_args = year_month_args + (1,)
    reference = date_type(*reference_args)
    date_args = year_month_args + (_days_in_month(reference),) + sub_day_args
    date = date_type(*date_args)
    result = offset.onOffset(date)
    assert result


@pytest.mark.parametrize(
    ("offset", "initial_date_args", "partial_expected_date_args"),
    [
        (YearBegin(), (1, 3, 1), (2, 1)),
        (YearBegin(), (1, 1, 1), (1, 1)),
        (YearBegin(n=2), (1, 3, 1), (2, 1)),
        (YearBegin(n=2, month=2), (1, 3, 1), (2, 2)),
        (YearEnd(), (1, 3, 1), (1, 12)),
        (YearEnd(n=2), (1, 3, 1), (1, 12)),
        (YearEnd(n=2, month=2), (1, 3, 1), (2, 2)),
        (YearEnd(n=2, month=4), (1, 4, 30), (1, 4)),
        (QuarterBegin(), (1, 3, 2), (1, 6)),
        (QuarterBegin(), (1, 4, 1), (1, 6)),
        (QuarterBegin(n=2), (1, 4, 1), (1, 6)),
        (QuarterBegin(n=2, month=2), (1, 4, 1), (1, 5)),
        (QuarterEnd(), (1, 3, 1), (1, 3)),
        (QuarterEnd(n=2), (1, 3, 1), (1, 3)),
        (QuarterEnd(n=2, month=2), (1, 3, 1), (1, 5)),
        (QuarterEnd(n=2, month=4), (1, 4, 30), (1, 4)),
        (MonthBegin(), (1, 3, 2), (1, 4)),
        (MonthBegin(), (1, 3, 1), (1, 3)),
        (MonthBegin(n=2), (1, 3, 2), (1, 4)),
        (MonthEnd(), (1, 3, 2), (1, 3)),
        (MonthEnd(), (1, 4, 30), (1, 4)),
        (MonthEnd(n=2), (1, 3, 2), (1, 3)),
        (Day(), (1, 3, 2, 1), (1, 3, 2, 1)),
        (Hour(), (1, 3, 2, 1, 1), (1, 3, 2, 1, 1)),
        (Minute(), (1, 3, 2, 1, 1, 1), (1, 3, 2, 1, 1, 1)),
        (Second(), (1, 3, 2, 1, 1, 1, 1), (1, 3, 2, 1, 1, 1, 1)),
    ],
    ids=_id_func,
)
def test_rollforward(calendar, offset, initial_date_args, partial_expected_date_args):
    date_type = get_date_type(calendar)
    initial = date_type(*initial_date_args)
    if isinstance(offset, (MonthBegin, QuarterBegin, YearBegin)):
        expected_date_args = partial_expected_date_args + (1,)
    elif isinstance(offset, (MonthEnd, QuarterEnd, YearEnd)):
        reference_args = partial_expected_date_args + (1,)
        reference = date_type(*reference_args)
        expected_date_args = partial_expected_date_args + (_days_in_month(reference),)
    else:
        expected_date_args = partial_expected_date_args
    expected = date_type(*expected_date_args)
    result = offset.rollforward(initial)
    assert result == expected


@pytest.mark.parametrize(
    ("offset", "initial_date_args", "partial_expected_date_args"),
    [
        (YearBegin(), (1, 3, 1), (1, 1)),
        (YearBegin(n=2), (1, 3, 1), (1, 1)),
        (YearBegin(n=2, month=2), (1, 3, 1), (1, 2)),
        (YearBegin(), (1, 1, 1), (1, 1)),
        (YearBegin(n=2, month=2), (1, 2, 1), (1, 2)),
        (YearEnd(), (2, 3, 1), (1, 12)),
        (YearEnd(n=2), (2, 3, 1), (1, 12)),
        (YearEnd(n=2, month=2), (2, 3, 1), (2, 2)),
        (YearEnd(month=4), (1, 4, 30), (1, 4)),
        (QuarterBegin(), (1, 3, 2), (1, 3)),
        (QuarterBegin(), (1, 4, 1), (1, 3)),
        (QuarterBegin(n=2), (1, 4, 1), (1, 3)),
        (QuarterBegin(n=2, month=2), (1, 4, 1), (1, 2)),
        (QuarterEnd(), (2, 3, 1), (1, 12)),
        (QuarterEnd(n=2), (2, 3, 1), (1, 12)),
        (QuarterEnd(n=2, month=2), (2, 3, 1), (2, 2)),
        (QuarterEnd(n=2, month=4), (1, 4, 30), (1, 4)),
        (MonthBegin(), (1, 3, 2), (1, 3)),
        (MonthBegin(n=2), (1, 3, 2), (1, 3)),
        (MonthBegin(), (1, 3, 1), (1, 3)),
        (MonthEnd(), (1, 3, 2), (1, 2)),
        (MonthEnd(n=2), (1, 3, 2), (1, 2)),
        (MonthEnd(), (1, 4, 30), (1, 4)),
        (Day(), (1, 3, 2, 1), (1, 3, 2, 1)),
        (Hour(), (1, 3, 2, 1, 1), (1, 3, 2, 1, 1)),
        (Minute(), (1, 3, 2, 1, 1, 1), (1, 3, 2, 1, 1, 1)),
        (Second(), (1, 3, 2, 1, 1, 1, 1), (1, 3, 2, 1, 1, 1, 1)),
    ],
    ids=_id_func,
)
def test_rollback(calendar, offset, initial_date_args, partial_expected_date_args):
    date_type = get_date_type(calendar)
    initial = date_type(*initial_date_args)
    if isinstance(offset, (MonthBegin, QuarterBegin, YearBegin)):
        expected_date_args = partial_expected_date_args + (1,)
    elif isinstance(offset, (MonthEnd, QuarterEnd, YearEnd)):
        reference_args = partial_expected_date_args + (1,)
        reference = date_type(*reference_args)
        expected_date_args = partial_expected_date_args + (_days_in_month(reference),)
    else:
        expected_date_args = partial_expected_date_args
    expected = date_type(*expected_date_args)
    result = offset.rollback(initial)
    assert result == expected


_CFTIME_RANGE_TESTS = [
    (
        "0001-01-01",
        "0001-01-04",
        None,
        "D",
        None,
        False,
        [(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 1, 4)],
    ),
    (
        "0001-01-01",
        "0001-01-04",
        None,
        "D",
        "left",
        False,
        [(1, 1, 1), (1, 1, 2), (1, 1, 3)],
    ),
    (
        "0001-01-01",
        "0001-01-04",
        None,
        "D",
        "right",
        False,
        [(1, 1, 2), (1, 1, 3), (1, 1, 4)],
    ),
    (
        "0001-01-01T01:00:00",
        "0001-01-04",
        None,
        "D",
        None,
        False,
        [(1, 1, 1, 1), (1, 1, 2, 1), (1, 1, 3, 1)],
    ),
    (
        "0001-01-01 01:00:00",
        "0001-01-04",
        None,
        "D",
        None,
        False,
        [(1, 1, 1, 1), (1, 1, 2, 1), (1, 1, 3, 1)],
    ),
    (
        "0001-01-01T01:00:00",
        "0001-01-04",
        None,
        "D",
        None,
        True,
        [(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 1, 4)],
    ),
    (
        "0001-01-01",
        None,
        4,
        "D",
        None,
        False,
        [(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 1, 4)],
    ),
    (
        None,
        "0001-01-04",
        4,
        "D",
        None,
        False,
        [(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 1, 4)],
    ),
    (
        (1, 1, 1),
        "0001-01-04",
        None,
        "D",
        None,
        False,
        [(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 1, 4)],
    ),
    (
        (1, 1, 1),
        (1, 1, 4),
        None,
        "D",
        None,
        False,
        [(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 1, 4)],
    ),
    (
        "0001-01-30",
        "0011-02-01",
        None,
        "3AS-JUN",
        None,
        False,
        [(1, 6, 1), (4, 6, 1), (7, 6, 1), (10, 6, 1)],
    ),
    ("0001-01-04", "0001-01-01", None, "D", None, False, []),
    (
        "0010",
        None,
        4,
        YearBegin(n=-2),
        None,
        False,
        [(10, 1, 1), (8, 1, 1), (6, 1, 1), (4, 1, 1)],
    ),
    (
        "0001-01-01",
        "0001-01-04",
        4,
        None,
        None,
        False,
        [(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 1, 4)],
    ),
    (
        "0001-06-01",
        None,
        4,
        "3QS-JUN",
        None,
        False,
        [(1, 6, 1), (2, 3, 1), (2, 12, 1), (3, 9, 1)],
    ),
]


@pytest.mark.parametrize(
    ("start", "end", "periods", "freq", "closed", "normalize", "expected_date_args"),
    _CFTIME_RANGE_TESTS,
    ids=_id_func,
)
def test_cftime_range(
    start, end, periods, freq, closed, normalize, calendar, expected_date_args
):
    date_type = get_date_type(calendar)
    expected_dates = [date_type(*args) for args in expected_date_args]

    if isinstance(start, tuple):
        start = date_type(*start)
    if isinstance(end, tuple):
        end = date_type(*end)

    result = cftime_range(
        start=start,
        end=end,
        periods=periods,
        freq=freq,
        closed=closed,
        normalize=normalize,
        calendar=calendar,
    )
    resulting_dates = result.values

    assert isinstance(result, CFTimeIndex)

    if freq is not None:
        np.testing.assert_equal(resulting_dates, expected_dates)
    else:
        # If we create a linear range of dates using cftime.num2date
        # we will not get exact round number dates.  This is because
        # datetime arithmetic in cftime is accurate approximately to
        # 1 millisecond (see https://unidata.github.io/cftime/api.html).
        deltas = resulting_dates - expected_dates
        deltas = np.array([delta.total_seconds() for delta in deltas])
        assert np.max(np.abs(deltas)) < 0.001


def test_cftime_range_name():
    result = cftime_range(start="2000", periods=4, name="foo")
    assert result.name == "foo"

    result = cftime_range(start="2000", periods=4)
    assert result.name is None


@pytest.mark.parametrize(
    ("start", "end", "periods", "freq", "closed"),
    [
        (None, None, 5, "A", None),
        ("2000", None, None, "A", None),
        (None, "2000", None, "A", None),
        ("2000", "2001", None, None, None),
        (None, None, None, None, None),
        ("2000", "2001", None, "A", "up"),
        ("2000", "2001", 5, "A", None),
    ],
)
def test_invalid_cftime_range_inputs(start, end, periods, freq, closed):
    with pytest.raises(ValueError):
        cftime_range(start, end, periods, freq, closed=closed)


_CALENDAR_SPECIFIC_MONTH_END_TESTS = [
    ("2M", "noleap", [(2, 28), (4, 30), (6, 30), (8, 31), (10, 31), (12, 31)]),
    ("2M", "all_leap", [(2, 29), (4, 30), (6, 30), (8, 31), (10, 31), (12, 31)]),
    ("2M", "360_day", [(2, 30), (4, 30), (6, 30), (8, 30), (10, 30), (12, 30)]),
    ("2M", "standard", [(2, 29), (4, 30), (6, 30), (8, 31), (10, 31), (12, 31)]),
    ("2M", "gregorian", [(2, 29), (4, 30), (6, 30), (8, 31), (10, 31), (12, 31)]),
    ("2M", "julian", [(2, 29), (4, 30), (6, 30), (8, 31), (10, 31), (12, 31)]),
]


@pytest.mark.parametrize(
    ("freq", "calendar", "expected_month_day"),
    _CALENDAR_SPECIFIC_MONTH_END_TESTS,
    ids=_id_func,
)
def test_calendar_specific_month_end(freq, calendar, expected_month_day):
    year = 2000  # Use a leap-year to highlight calendar differences
    result = cftime_range(
        start="2000-02", end="2001", freq=freq, calendar=calendar
    ).values
    date_type = get_date_type(calendar)
    expected = [date_type(year, *args) for args in expected_month_day]
    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    ("calendar", "start", "end", "expected_number_of_days"),
    [
        ("noleap", "2000", "2001", 365),
        ("all_leap", "2000", "2001", 366),
        ("360_day", "2000", "2001", 360),
        ("standard", "2000", "2001", 366),
        ("gregorian", "2000", "2001", 366),
        ("julian", "2000", "2001", 366),
        ("noleap", "2001", "2002", 365),
        ("all_leap", "2001", "2002", 366),
        ("360_day", "2001", "2002", 360),
        ("standard", "2001", "2002", 365),
        ("gregorian", "2001", "2002", 365),
        ("julian", "2001", "2002", 365),
    ],
)
def test_calendar_year_length(calendar, start, end, expected_number_of_days):
    result = cftime_range(start, end, freq="D", closed="left", calendar=calendar)
    assert len(result) == expected_number_of_days


@pytest.mark.parametrize("freq", ["A", "M", "D"])
def test_dayofweek_after_cftime_range(freq):
    pytest.importorskip("cftime", minversion="1.0.2.1")
    result = cftime_range("2000-02-01", periods=3, freq=freq).dayofweek
    expected = pd.date_range("2000-02-01", periods=3, freq=freq).dayofweek
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("freq", ["A", "M", "D"])
def test_dayofyear_after_cftime_range(freq):
    pytest.importorskip("cftime", minversion="1.0.2.1")
    result = cftime_range("2000-02-01", periods=3, freq=freq).dayofyear
    expected = pd.date_range("2000-02-01", periods=3, freq=freq).dayofyear
    np.testing.assert_array_equal(result, expected)


def test_cftime_range_standard_calendar_refers_to_gregorian():
    from cftime import DatetimeGregorian

    (result,) = cftime_range("2000", periods=1)
    assert isinstance(result, DatetimeGregorian)
