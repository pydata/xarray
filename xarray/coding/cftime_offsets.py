"""Time offset classes for use with cftime.datetime objects"""
import re

from datetime import timedelta
from functools import partial

import numpy as np
import pandas as pd

from .cftimeindex import _parse_iso8601_with_reso, CFTimeIndex
from .times import format_cftime_datetime
from ..core.pycompat import basestring

try:
    from pandas.errors import OutOfBoundsDatetime
except ImportError:
    # pandas < 0.20
    from pandas.tslib import OutOfBoundsDatetime


def get_date_type(calendar):
    """Return the cftime date type for a given calendar name."""
    try:
        import cftime
    except ImportError:
        raise ImportError(
            'cftime is required for dates with non-standard calendars')
    else:
        calendars = {
            'noleap': cftime.DatetimeNoLeap,
            '360_day': cftime.Datetime360Day,
            '365_day': cftime.DatetimeNoLeap,
            '366_day': cftime.DatetimeAllLeap,
            'gregorian': cftime.DatetimeGregorian,
            'proleptic_gregorian': cftime.DatetimeProlepticGregorian,
            'julian': cftime.DatetimeJulian,
            'all_leap': cftime.DatetimeAllLeap,
            'standard': cftime.DatetimeProlepticGregorian
        }
        return calendars[calendar]


class BaseCFTimeOffset(object):
    _freq = None

    def __init__(self, n=1):
        self.n = n

    def rule_code(self):
        return self._freq

    def __eq__(self, other):
        return self.n == other.n and self.rule_code() == other.rule_code()

    def __ne__(self, other):
        return not self == other

    def __add__(self, other):
        return self.__apply__(other)

    def __sub__(self, other):
        import cftime

        if isinstance(other, cftime.datetime):
            raise TypeError('Cannot subtract a cftime.datetime '
                            'from a time offset.')
        elif type(other) == type(self):
            return type(self)(self.n - other.n)
        else:
            raise NotImplementedError

    def __mul__(self, other):
        return type(self)(n=other * self.n)

    def __neg__(self):
        return self * -1

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return -self + other

    def __apply__(self):
        raise NotImplementedError

    def on_offset(self, date):
        """Check if the given date is in the set of possible dates created
        using a length-one version of this offset class."""
        test_date = (self + date) - self
        return date == test_date

    def __str__(self):
        return '<{}: n={}>'.format(type(self).__name__, self.n)

    def __repr__(self):
        return str(self)


def _days_in_month(date):
    """The number of days in the month of the given date"""
    if date.month == 12:
        reference = type(date)(date.year + 1, 1, 1)
    else:
        reference = type(date)(date.year, date.month + 1, 1)
    return (reference - timedelta(days=1)).day


def _adjust_n_months(other_day, n, reference_day):
    """Adjust the number of times a monthly offset is applied based
    on the day of a given date, and the reference day provided.
    """
    if n > 0 and other_day < reference_day:
        n = n - 1
    elif n <= 0 and other_day > reference_day:
        n = n + 1
    return n


def _adjust_n_years(other, n, month, reference_day):
    """Adjust the number of times an annual offset is applied based on
    another date, and the reference day provided"""
    if n > 0:
        if other.month < month or (other.month == month and
                                   other.day < reference_day):
            n -= 1
    else:
        if other.month > month or (other.month == month and
                                   other.day > reference_day):
            n += 1
    return n


def _shift_months(date, months, day_option='start'):
    """Shift the date to a month start or end a given number of months away.
    """
    delta_year = (date.month + months) // 12
    month = (date.month + months) % 12

    if month == 0:
        month = 12
        delta_year = delta_year - 1
    year = date.year + delta_year

    if day_option == 'start':
        day = 1
    elif day_option == 'end':
        reference = type(date)(year, month, 1)
        day = _days_in_month(reference)
    else:
        raise ValueError(day_option)
    return date.replace(year=year, month=month, day=day)


class MonthBegin(BaseCFTimeOffset):
    _freq = 'MS'

    def __apply__(self, other):
        n = _adjust_n_months(other.day, self.n, 1)
        return _shift_months(other, n, 'start')

    def on_offset(self, date):
        """Check if the given date is in the set of possible dates created
        using a length-one version of this offset class."""
        return date.day == 1

    def roll_forward(self, date):
        """Roll date forward to nearest start of month"""
        return date + MonthBegin()

    def roll_backward(self, date):
        """Roll date backward to nearest start of month"""
        return date - MonthBegin()


class MonthEnd(BaseCFTimeOffset):
    _freq = 'M'

    def __apply__(self, other):
        n = _adjust_n_months(other.day, self.n, _days_in_month(other))
        return _shift_months(other, n, 'end')

    def on_offset(self, date):
        """Check if the given date is in the set of possible dates created
        using a length-one version of this offset class."""
        return date.day == _days_in_month(date)

    def roll_forward(self, date):
        """Roll date forward to nearest end of month"""
        return date + MonthEnd()

    def roll_backward(self, date):
        """Roll date backward to nearest end of month"""
        return date - MonthEnd()


_MONTH_ABBREVIATIONS = {
    1: 'JAN',
    2: 'FEB',
    3: 'MAR',
    4: 'APR',
    5: 'MAY',
    6: 'JUN',
    7: 'JUL',
    8: 'AUG',
    9: 'SEP',
    10: 'OCT',
    11: 'NOV',
    12: 'DEC'
}


class YearOffset(BaseCFTimeOffset):
    _freq = None
    _day_option = None
    _default_month = None

    def __init__(self, n=1, month=None):
        BaseCFTimeOffset.__init__(self, n)
        if month is None:
            self.month = self._default_month
        else:
            self.month = month

    def __apply__(self, other):
        if self._day_option == 'start':
            reference_day = 1
        elif self._day_option == 'end':
            reference_day = _days_in_month(other)
        else:
            raise ValueError(self._day_option)
        years = _adjust_n_years(other, self.n, self.month, reference_day)
        months = years * 12 + (self.month - other.month)
        return _shift_months(other, months, self._day_option)

    def __sub__(self, other):
        import cftime

        if isinstance(other, cftime.datetime):
            raise TypeError('Cannot subtract cftime.datetime from offset.')
        elif type(other) == type(self) and other.month == self.month:
            return type(self)(self.n - other.n, month=self.month)
        else:
            raise NotImplementedError

    def __mul__(self, other):
        return type(self)(n=other * self.n, month=self.month)

    def rule_code(self):
        return '{}-{}'.format(self._freq, _MONTH_ABBREVIATIONS[self.month])

    def __str__(self):
        return '<{}: n={}, month={}>'.format(
            type(self).__name__, self.n, self.month)


class YearBegin(YearOffset):
    _freq = 'AS'
    _day_option = 'start'
    _default_month = 1

    def on_offset(self, date):
        """Check if the given date is in the set of possible dates created
        using a length-one version of this offset class."""
        return date.day == 1 and date.month == self.month

    def roll_forward(self, date):
        """Roll date forward to nearest start of year"""
        return date + YearBegin(month=self.month)

    def roll_backward(self, date):
        """Roll date backward to nearest start of year"""
        return date - YearBegin(month=self.month)


class YearEnd(YearOffset):
    _freq = 'A'
    _day_option = 'end'
    _default_month = 12

    def on_offset(self, date):
        """Check if the given date is in the set of possible dates created
        using a length-one version of this offset class."""
        return date.day == _days_in_month(date) and date.month == self.month

    def roll_forward(self, date):
        """Roll date forward to nearest end of year"""
        return date + YearEnd(month=self.month)

    def roll_backward(self, date):
        """Roll date backward to nearest end of year"""
        return date - YearEnd(month=self.month)


class Day(BaseCFTimeOffset):
    _freq = 'D'

    def __apply__(self, other):
        return other + timedelta(days=self.n)


class Hour(BaseCFTimeOffset):
    _freq = 'H'

    def __apply__(self, other):
        return other + timedelta(hours=self.n)


class Minute(BaseCFTimeOffset):
    _freq = 'T'

    def __apply__(self, other):
        return other + timedelta(minutes=self.n)


class Second(BaseCFTimeOffset):
    _freq = 'S'

    def __apply__(self, other):
        return other + timedelta(seconds=self.n)


_FREQUENCIES = {
    'A': YearEnd,
    'AS': YearBegin,
    'Y': YearEnd,
    'YS': YearBegin,
    'M': MonthEnd,
    'MS': MonthBegin,
    'D': Day,
    'H': Hour,
    'T': Minute,
    'min': Minute,
    'S': Second,
    'AS-JAN': partial(YearBegin, month=1),
    'AS-FEB': partial(YearBegin, month=2),
    'AS-MAR': partial(YearBegin, month=3),
    'AS-APR': partial(YearBegin, month=4),
    'AS-MAY': partial(YearBegin, month=5),
    'AS-JUN': partial(YearBegin, month=6),
    'AS-JUL': partial(YearBegin, month=7),
    'AS-AUG': partial(YearBegin, month=8),
    'AS-SEP': partial(YearBegin, month=9),
    'AS-OCT': partial(YearBegin, month=10),
    'AS-NOV': partial(YearBegin, month=11),
    'AS-DEC': partial(YearBegin, month=12),
    'A-JAN': partial(YearEnd, month=1),
    'A-FEB': partial(YearEnd, month=2),
    'A-MAR': partial(YearEnd, month=3),
    'A-APR': partial(YearEnd, month=4),
    'A-MAY': partial(YearEnd, month=5),
    'A-JUN': partial(YearEnd, month=6),
    'A-JUL': partial(YearEnd, month=7),
    'A-AUG': partial(YearEnd, month=8),
    'A-SEP': partial(YearEnd, month=9),
    'A-OCT': partial(YearEnd, month=10),
    'A-NOV': partial(YearEnd, month=11),
    'A-DEC': partial(YearEnd, month=12)
}


_FREQUENCY_CONDITION = '|'.join(_FREQUENCIES.keys())
_PATTERN = '^((?P<multiple>\d+)|())(?P<freq>({0}))$'.format(
    _FREQUENCY_CONDITION)


def to_offset(freq):
    """Convert a frequency string to the appropriate subclass of
    BaseCFTimeOffset."""
    if isinstance(freq, BaseCFTimeOffset):
        return freq
    else:
        try:
            freq_data = re.match(_PATTERN, freq).groupdict()
        except AttributeError:
            raise ValueError('Invalid frequency string provided')

    freq = freq_data['freq']
    multiples = freq_data['multiple']
    if multiples is None:
        multiples = 1
    else:
        multiples = int(multiples)

    return _FREQUENCIES[freq](n=multiples)


def to_cftime_datetime(date_str_or_date, calendar=None):
    import cftime

    if isinstance(date_str_or_date, basestring):
        if calendar is None:
            raise ValueError(
                'If converting a string to a cftime.datetime object, '
                'a calendar type must be provided')
        date, _ = _parse_iso8601_with_reso(get_date_type(calendar),
                                           date_str_or_date)
        return date
    elif isinstance(date_str_or_date, cftime.datetime):
        return date_str_or_date
    else:
        raise ValueError('date_str_or_date must be a string or a '
                         'subclass of cftime.datetime; got {}.'.format(
                             date_str_or_date))


def normalize_date(date):
    """Round datetime down to midnight."""
    return date.replace(hour=0, minute=0, second=0, microsecond=0)


def _maybe_normalize_date(date, normalize):
    """Round datetime down to midnight if normalize is True."""
    if normalize:
        return normalize_date(date)
    else:
        return date


def _generate_linear_range(start, end, periods):
    """Generate an equally-spaced sequence of cftime.datetime objects between
    and including two dates (whose length equals the number of periods)."""
    import cftime

    total_seconds = (end - start).total_seconds()
    values = np.linspace(0., total_seconds, periods, endpoint=True)
    units = 'seconds since {}'.format(format_cftime_datetime(start))
    calendar = start.calendar
    return cftime.num2date(values, units=units, calendar=calendar,
                           only_use_cftime_datetimes=True)


def _generate_range(start, end, periods, offset):
    """Generate a regular range of cftime.datetime objects with a
    given time offset.

    Adapted from pandas.tseries.offsets.generate_range.

    Parameters
    ----------
    start : cftime.datetime, or None
        Start of range
    end : cftime.datetime, or None
        End of range
    periods : int, or None
        Number of elements in the sequence
    offset : BaseCFTimeOffset
        An offset class designed for working with cftime.datetime objects

    Returns
    -------
    A generator object
    """
    if start and not offset.on_offset(start):
        start = offset.roll_forward(start)

    if end and not offset.on_offset(end):
        end = offset.roll_backward(end)

    if periods is None and end < start:
        end = None
        periods = 0

    if end is None:
        end = start + (periods - 1) * offset

    if start is None:
        start = end - (periods - 1) * offset

    current = start
    if offset.n >= 0:
        while current <= end:
            yield current

            next_date = current + offset
            if next_date <= current:
                raise ValueError('Offset {offset} did not increment date'
                                 .format(offset=offset))
            current = next_date
    else:
        while current >= end:
            yield current

            next_date = current + offset
            if next_date >= current:
                raise ValueError('Offset {offset} did not decrement date'
                                 .format(offset=offset))
            current = next_date


def _count_not_none(*args):
    """Compute the number of non-None arguments."""
    return sum([arg is not None for arg in args])


def _cftime_range(start=None, end=None, periods=None, freq=None,
                  closed=None, normalize=False, calendar='standard'):
    """Generate a range of cftime.datetime objects from pandas.date_range-style
    input arguments.

    Adapted from pandas.core.indexes.datetimes._generate_range.

    Parameters
    ----------
    start : str or cftime.datetime (optional)
        Starting date for the range.
    end : str or cftime.datetime (optional)
        Ending date for the range.
    periods : int (optional)
        Number of dates in the range.
    freq : str, BaseCFTimeOffset, or None
        Frequency of dates in date range.
    closed : {None, 'left', 'right'}
        Make the interval closed with respect to the given frequency to
        the 'left', 'right', or both sides (None).
    normalize : bool (default False)
        Normalize start/end dates to midnight before generating date range.
    calendar : str (default 'standard')
        Calendar type of the datetimes (default 'standard').

    Returns
    -------
    list of cftime.datetime objects
    """
    if _count_not_none(start, end, periods, freq) != 3:
        raise ValueError(
            "Of the arguments 'start', 'end', 'periods', and 'freq', three "
            "must be specified at a time.")

    if start is not None:
        start = to_cftime_datetime(start, calendar)
        start = _maybe_normalize_date(start, normalize)
    if end is not None:
        end = to_cftime_datetime(end, calendar)
        end = _maybe_normalize_date(end, normalize)

    if freq is None:
        dates = _generate_linear_range(start, end, periods)
    else:
        offset = to_offset(freq)
        dates = np.array(list(_generate_range(start, end, periods, offset)))

    left_closed = False
    right_closed = False

    if closed is None:
        left_closed = True
        right_closed = True
    elif closed == 'left':
        left_closed = True
    elif closed == 'right':
        right_closed = True
    else:
        raise ValueError("Closed must be either 'left', 'right' or None")

    if (not left_closed and len(dates) and
       start is not None and dates[0] == start):
        dates = dates[1:]
    if (not right_closed and len(dates) and
       end is not None and dates[-1] == end):
        dates = dates[:-1]
    return dates


def cftime_range(start=None, end=None, periods=None, freq='D',
                 tz=None, normalize=False, name=None, closed=None,
                 calendar='standard'):
    """Return a fixed frequency DatetimeIndex or CFTimeIndex.

    Parameters
    ----------
    start : str or datetime-like, optional
        Left bound for generating dates.
    end : str or datetime-like, optional
        Right bound for generating dates.
    periods : integer, optional
        Number of periods to generate.
    freq : str, default 'D', DateOffset, BaseCFTimeOffset, or None
       Frequency strings can have multiples, e.g. '5H'.
    tz : str or tzinfo, optional
        Time zone name for returning localized DatetimeIndex, for example
        'Asia/Hong_Kong'.  By default, the resulting DatetimeIndex is
        timezone-naive.  This option is only supported for 'standard'
        calendar dates that result in a DatetimeIndex.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    name : str, default None
        Name of the resulting index
    closed : {None, 'left', 'right'}, optional
        Make the interval closed with respect to the given frequency to the
        'left', 'right', or both sides (None, the default).
    calendar : str
        Calendar type for the datetimes (default 'standard').

    Returns
    -------
    DatetimeIndex or CFTimeIndex

    Notes
    -----

    Valid simple frequency strings for use with ``cftime``-calendars include
    any multiples of the following.

    +--------+-----------------------+
    | Alias  | Description           |
    +========+=======================+
    | A, Y   | Year-end frequency    |
    +--------+-----------------------+
    | AS, YS | Year-start frequency  |
    +--------+-----------------------+
    | M      | Month-end frequency   |
    +--------+-----------------------+
    | MS     | Month-start frequency |
    +--------+-----------------------+
    | D      | Day frequency         |
    +--------+-----------------------+
    | H      | Hour frequency        |
    +--------+-----------------------+
    | T, min | Minute frequency      |
    +--------+-----------------------+
    | S      | Second frequency      |
    +--------+-----------------------+

    Any multiples of the following anchored offsets are also supported.

    +----------+-------------------------------------------------------------------+
    | Alias    | Description                                                       |
    +==========+===================================================================+
    | A(S)-JAN | Annual frequency, anchored at the end (or beginning) of January   |
    +----------+-------------------------------------------------------------------+
    | A(S)-FEB | Annual frequency, anchored at the end (or beginning) of February  |
    +----------+-------------------------------------------------------------------+
    | A(S)-MAR | Annual frequency, anchored at the end (or beginning) of March     |
    +----------+-------------------------------------------------------------------+
    | A(S)-APR | Annual frequency, anchored at the end (or beginning) of April     |
    +----------+-------------------------------------------------------------------+
    | A(S)-MAY | Annual frequency, anchored at the end (or beginning) of May       |
    +----------+-------------------------------------------------------------------+
    | A(S)-JUN | Annual frequency, anchored at the end (or beginning) of June      |
    +----------+-------------------------------------------------------------------+
    | A(S)-JUL | Annual frequency, anchored at the end (or beginning) of July      |
    +----------+-------------------------------------------------------------------+
    | A(S)-AUG | Annual frequency, anchored at the end (or beginning) of August    |
    +----------+-------------------------------------------------------------------+
    | A(S)-SEP | Annual frequency, anchored at the end (or beginning) of September |
    +----------+-------------------------------------------------------------------+
    | A(S)-OCT | Annual frequency, anchored at the end (or beginning) of October   |
    +----------+-------------------------------------------------------------------+
    | A(S)-NOV | Annual frequency, anchored at the end (or beginning) of November  |
    +----------+-------------------------------------------------------------------+
    | A(S)-DEC | Annual frequency, anchored at the end (or beginning) of December  |
    +----------+-------------------------------------------------------------------+

    Finally, the following calendar aliases are supported.

    +----------------------+-------------------------------------------------------------------------+
    | Alias                | Date type                                                               |
    +======================+=========================================================================+
    | standard             | ``np.datetime64`` (falls back to ``cftime.DatetimeProlepticGregorian``) |
    +----------------------+-------------------------------------------------------------------------+
    | gregorian            | ``cftime.DatetimeGregorian``                                            |
    +----------------------+-------------------------------------------------------------------------+
    | proleptic_gregorian  | ``cftime.DatetimeProlepticGregorian``                                   |
    +----------------------+-------------------------------------------------------------------------+
    | noleap, 365_day      | ``cftime.DatetimeNoLeap``                                               |
    +----------------------+-------------------------------------------------------------------------+
    | all_leap, 366_day    | ``cftime.DatetimeAllLeap``                                              |
    +----------------------+-------------------------------------------------------------------------+
    | 360_day              | ``cftime.Datetime360Day``                                               |
    +----------------------+-------------------------------------------------------------------------+
    | julian               | ``cftime.DatetimeJulian``                                               |
    +----------------------+-------------------------------------------------------------------------+

    The ``CFTimeIndex``-enabled portion of this function supports most of the
    features of ``pandas.date_range`` (e.g. specifying how the index is
    ``closed`` on either side, or whether or not to ``normalize`` the start and
    end bounds).  Notable exceptions, however, are:

    - You cannot specify a ``tz`` (time zone) argument for a date range that
      results in a ``CFTimeIndex``.
    - Start or end dates specified as partial-datetime strings must use the
      `ISO8601 format <https://en.wikipedia.org/wiki/ISO_8601>`_.
    - The ``CFTimeIndex``-enabled version of the function supports many, but
      not all frequencies supported by the ``DatetimeIndex`` version.  For
      example it does not currently support any of the business-related,
      semi-monthly, or sub-second frequencies.
    - Compound sub-monthly frequencies are not supported, e.g. '1H1min', as
      these can easily be written in terms of the finest common resolution,
      e.g. '61min'.

    Examples
    --------

    For dates from standard calendars within the ``pandas.Timestamp``-valid
    range, this function operates as a thin wrapper around
    ``pandas.date_range``.

    >>> xr.cftime_range(start='2000', periods=6, freq='2MS')
    DatetimeIndex(['2000-01-01', '2000-03-01', '2000-05-01', '2000-07-01',
                   '2000-09-01', '2000-11-01'],
                  dtype='datetime64[ns]', freq='2MS')

    For dates from non-standard calendars, this function returns a
    ``CFTimeIndex``, populated with ``cftime.datetime`` objects associated with
    the specified calendar type, e.g.

    >>> xr.cftime_range(start='2000', periods=6, freq='2MS', calendar='noleap')
    CFTimeIndex([2000-01-01 00:00:00, 2000-03-01 00:00:00, 2000-05-01 00:00:00,
                 2000-07-01 00:00:00, 2000-09-01 00:00:00, 2000-11-01 00:00:00],
                dtype='object', calendar='noleap')

    If a 'standard' calendar is specified, but the dates are outside the
    ``pandas.Timestamp``-valid range, a ``CFTimeIndex`` composed of
    ``cftime.DatetimeGregorian`` objects will be returned.

    >>> xr.cftime_range(start='0001', periods=6, freq='2MS', calendar='standard')
    CFTimeIndex([0001-01-01 00:00:00, 0001-03-01 00:00:00, 0001-05-01 00:00:00,
                 0001-07-01 00:00:00, 0001-09-01 00:00:00, 0001-11-01 00:00:00],
                dtype='object', calendar='proleptic_gregorian')

    As in the standard pandas function, three of the ``start``, ``end``,
    ``periods``, or ``freq`` arguments must be specified at a given time, with
    the other set to ``None``.  See the `pandas documentation
    <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.date_range.html#pandas.date_range>`_
    for more examples of the behavior of ``date_range`` with each of the
    parameters.

    See Also
    --------
    pandas.date_range
    """  # noqa: E501
    if calendar == 'standard':
        try:
            return pd.date_range(
                start=start, end=end, periods=periods,
                freq=freq, tz=tz, normalize=normalize, name=name,
                closed=closed)
        except OutOfBoundsDatetime:
            if tz is not None:
                raise ValueError(
                    "'tz' can only be specified if the resulting"
                    "index is a DatetimeIndex.")
            return CFTimeIndex(start=start, end=end, periods=periods,
                               freq=freq, closed=closed, normalize=normalize,
                               calendar=calendar, name=name)
    else:
        if tz is not None:
            raise ValueError(
                "'tz' cannot be specified for non-standard calendars."
            )
        return CFTimeIndex(start=start, end=end, periods=periods,
                           freq=freq, closed=closed, normalize=normalize,
                           calendar=calendar, name=name)
