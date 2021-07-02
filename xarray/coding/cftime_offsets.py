"""Time offset classes for use with cftime.datetime objects"""
# The offset classes and mechanisms for generating time ranges defined in
# this module were copied/adapted from those defined in pandas.  See in
# particular the objects and methods defined in pandas.tseries.offsets
# and pandas.core.indexes.datetimes.

# For reference, here is a copy of the pandas copyright notice:

# (c) 2011-2012, Lambda Foundry, Inc. and PyData Development Team
# All rights reserved.

# Copyright (c) 2008-2011 AQR Capital Management, LLC
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

#     * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.

#     * Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided
#        with the distribution.

#     * Neither the name of the copyright holder nor the names of any
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import re
from datetime import timedelta
from distutils.version import LooseVersion
from functools import partial
from typing import ClassVar, Optional

import numpy as np

from ..core.pdcompat import count_not_none
from .cftimeindex import CFTimeIndex, _parse_iso8601_with_reso
from .times import format_cftime_datetime


def get_date_type(calendar):
    """Return the cftime date type for a given calendar name."""
    try:
        import cftime
    except ImportError:
        raise ImportError("cftime is required for dates with non-standard calendars")
    else:
        calendars = {
            "noleap": cftime.DatetimeNoLeap,
            "360_day": cftime.Datetime360Day,
            "365_day": cftime.DatetimeNoLeap,
            "366_day": cftime.DatetimeAllLeap,
            "gregorian": cftime.DatetimeGregorian,
            "proleptic_gregorian": cftime.DatetimeProlepticGregorian,
            "julian": cftime.DatetimeJulian,
            "all_leap": cftime.DatetimeAllLeap,
            "standard": cftime.DatetimeGregorian,
        }
        return calendars[calendar]


class BaseCFTimeOffset:
    _freq: ClassVar[Optional[str]] = None
    _day_option: ClassVar[Optional[str]] = None

    def __init__(self, n=1):
        if not isinstance(n, int):
            raise TypeError(
                "The provided multiple 'n' must be an integer. "
                "Instead a value of type {!r} was provided.".format(type(n))
            )
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
            raise TypeError("Cannot subtract a cftime.datetime from a time offset.")
        elif type(other) == type(self):
            return type(self)(self.n - other.n)
        else:
            return NotImplemented

    def __mul__(self, other):
        return type(self)(n=other * self.n)

    def __neg__(self):
        return self * -1

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        if isinstance(other, BaseCFTimeOffset) and type(self) != type(other):
            raise TypeError("Cannot subtract cftime offsets of differing types")
        return -self + other

    def __apply__(self):
        return NotImplemented

    def onOffset(self, date):
        """Check if the given date is in the set of possible dates created
        using a length-one version of this offset class."""
        test_date = (self + date) - self
        return date == test_date

    def rollforward(self, date):
        if self.onOffset(date):
            return date
        else:
            return date + type(self)()

    def rollback(self, date):
        if self.onOffset(date):
            return date
        else:
            return date - type(self)()

    def __str__(self):
        return "<{}: n={}>".format(type(self).__name__, self.n)

    def __repr__(self):
        return str(self)

    def _get_offset_day(self, other):
        # subclass must implement `_day_option`; calling from the base class
        # will raise NotImplementedError.
        return _get_day_of_month(other, self._day_option)


def _get_day_of_month(other, day_option):
    """Find the day in `other`'s month that satisfies a BaseCFTimeOffset's
    onOffset policy, as described by the `day_option` argument.

    Parameters
    ----------
    other : cftime.datetime
    day_option : 'start', 'end'
        'start': returns 1
        'end': returns last day of the month

    Returns
    -------
    day_of_month : int

    """

    if day_option == "start":
        return 1
    elif day_option == "end":
        return _days_in_month(other)
    elif day_option is None:
        # Note: unlike `_shift_month`, _get_day_of_month does not
        # allow day_option = None
        raise NotImplementedError()
    else:
        raise ValueError(day_option)


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
        if other.month < month or (other.month == month and other.day < reference_day):
            n -= 1
    else:
        if other.month > month or (other.month == month and other.day > reference_day):
            n += 1
    return n


def _shift_month(date, months, day_option="start"):
    """Shift the date to a month start or end a given number of months away."""
    import cftime

    delta_year = (date.month + months) // 12
    month = (date.month + months) % 12

    if month == 0:
        month = 12
        delta_year = delta_year - 1
    year = date.year + delta_year

    if day_option == "start":
        day = 1
    elif day_option == "end":
        reference = type(date)(year, month, 1)
        day = _days_in_month(reference)
    else:
        raise ValueError(day_option)
    if LooseVersion(cftime.__version__) < LooseVersion("1.0.4"):
        # dayofwk=-1 is required to update the dayofwk and dayofyr attributes of
        # the returned date object in versions of cftime between 1.0.2 and
        # 1.0.3.4.  It can be removed for versions of cftime greater than
        # 1.0.3.4.
        return date.replace(year=year, month=month, day=day, dayofwk=-1)
    else:
        return date.replace(year=year, month=month, day=day)


def roll_qtrday(other, n, month, day_option, modby=3):
    """Possibly increment or decrement the number of periods to shift
    based on rollforward/rollbackward conventions.

    Parameters
    ----------
    other : cftime.datetime
    n : number of periods to increment, before adjusting for rolling
    month : int reference month giving the first month of the year
    day_option : 'start', 'end'
        The convention to use in finding the day in a given month against
        which to compare for rollforward/rollbackward decisions.
    modby : int 3 for quarters, 12 for years

    Returns
    -------
    n : int number of periods to increment

    See Also
    --------
    _get_day_of_month : Find the day in a month provided an offset.
    """

    months_since = other.month % modby - month % modby

    if n > 0:
        if months_since < 0 or (
            months_since == 0 and other.day < _get_day_of_month(other, day_option)
        ):
            # pretend to roll back if on same month but
            # before compare_day
            n -= 1
    else:
        if months_since > 0 or (
            months_since == 0 and other.day > _get_day_of_month(other, day_option)
        ):
            # make sure to roll forward, so negate
            n += 1
    return n


def _validate_month(month, default_month):
    result_month = default_month if month is None else month
    if not isinstance(result_month, int):
        raise TypeError(
            "'self.month' must be an integer value between 1 "
            "and 12.  Instead, it was set to a value of "
            "{!r}".format(result_month)
        )
    elif not (1 <= result_month <= 12):
        raise ValueError(
            "'self.month' must be an integer value between 1 "
            "and 12.  Instead, it was set to a value of "
            "{!r}".format(result_month)
        )
    return result_month


class MonthBegin(BaseCFTimeOffset):
    _freq = "MS"

    def __apply__(self, other):
        n = _adjust_n_months(other.day, self.n, 1)
        return _shift_month(other, n, "start")

    def onOffset(self, date):
        """Check if the given date is in the set of possible dates created
        using a length-one version of this offset class."""
        return date.day == 1


class MonthEnd(BaseCFTimeOffset):
    _freq = "M"

    def __apply__(self, other):
        n = _adjust_n_months(other.day, self.n, _days_in_month(other))
        return _shift_month(other, n, "end")

    def onOffset(self, date):
        """Check if the given date is in the set of possible dates created
        using a length-one version of this offset class."""
        return date.day == _days_in_month(date)


_MONTH_ABBREVIATIONS = {
    1: "JAN",
    2: "FEB",
    3: "MAR",
    4: "APR",
    5: "MAY",
    6: "JUN",
    7: "JUL",
    8: "AUG",
    9: "SEP",
    10: "OCT",
    11: "NOV",
    12: "DEC",
}


class QuarterOffset(BaseCFTimeOffset):
    """Quarter representation copied off of pandas/tseries/offsets.py"""

    _freq: ClassVar[str]
    _default_month: ClassVar[int]

    def __init__(self, n=1, month=None):
        BaseCFTimeOffset.__init__(self, n)
        self.month = _validate_month(month, self._default_month)

    def __apply__(self, other):
        # months_since: find the calendar quarter containing other.month,
        # e.g. if other.month == 8, the calendar quarter is [Jul, Aug, Sep].
        # Then find the month in that quarter containing an onOffset date for
        # self.  `months_since` is the number of months to shift other.month
        # to get to this on-offset month.
        months_since = other.month % 3 - self.month % 3
        qtrs = roll_qtrday(
            other, self.n, self.month, day_option=self._day_option, modby=3
        )
        months = qtrs * 3 - months_since
        return _shift_month(other, months, self._day_option)

    def onOffset(self, date):
        """Check if the given date is in the set of possible dates created
        using a length-one version of this offset class."""
        mod_month = (date.month - self.month) % 3
        return mod_month == 0 and date.day == self._get_offset_day(date)

    def __sub__(self, other):
        import cftime

        if isinstance(other, cftime.datetime):
            raise TypeError("Cannot subtract cftime.datetime from offset.")
        elif type(other) == type(self) and other.month == self.month:
            return type(self)(self.n - other.n, month=self.month)
        else:
            return NotImplemented

    def __mul__(self, other):
        return type(self)(n=other * self.n, month=self.month)

    def rule_code(self):
        return "{}-{}".format(self._freq, _MONTH_ABBREVIATIONS[self.month])

    def __str__(self):
        return "<{}: n={}, month={}>".format(type(self).__name__, self.n, self.month)


class QuarterBegin(QuarterOffset):
    # When converting a string to an offset, pandas converts
    # 'QS' to a QuarterBegin offset starting in the month of
    # January.  When creating a QuarterBegin offset directly
    # from the constructor, however, the default month is March.
    # We follow that behavior here.
    _default_month = 3
    _freq = "QS"
    _day_option = "start"

    def rollforward(self, date):
        """Roll date forward to nearest start of quarter"""
        if self.onOffset(date):
            return date
        else:
            return date + QuarterBegin(month=self.month)

    def rollback(self, date):
        """Roll date backward to nearest start of quarter"""
        if self.onOffset(date):
            return date
        else:
            return date - QuarterBegin(month=self.month)


class QuarterEnd(QuarterOffset):
    # When converting a string to an offset, pandas converts
    # 'Q' to a QuarterEnd offset starting in the month of
    # December.  When creating a QuarterEnd offset directly
    # from the constructor, however, the default month is March.
    # We follow that behavior here.
    _default_month = 3
    _freq = "Q"
    _day_option = "end"

    def rollforward(self, date):
        """Roll date forward to nearest end of quarter"""
        if self.onOffset(date):
            return date
        else:
            return date + QuarterEnd(month=self.month)

    def rollback(self, date):
        """Roll date backward to nearest end of quarter"""
        if self.onOffset(date):
            return date
        else:
            return date - QuarterEnd(month=self.month)


class YearOffset(BaseCFTimeOffset):
    _freq: ClassVar[str]
    _day_option: ClassVar[str]
    _default_month: ClassVar[int]

    def __init__(self, n=1, month=None):
        BaseCFTimeOffset.__init__(self, n)
        self.month = _validate_month(month, self._default_month)

    def __apply__(self, other):
        reference_day = _get_day_of_month(other, self._day_option)
        years = _adjust_n_years(other, self.n, self.month, reference_day)
        months = years * 12 + (self.month - other.month)
        return _shift_month(other, months, self._day_option)

    def __sub__(self, other):
        import cftime

        if isinstance(other, cftime.datetime):
            raise TypeError("Cannot subtract cftime.datetime from offset.")
        elif type(other) == type(self) and other.month == self.month:
            return type(self)(self.n - other.n, month=self.month)
        else:
            return NotImplemented

    def __mul__(self, other):
        return type(self)(n=other * self.n, month=self.month)

    def rule_code(self):
        return "{}-{}".format(self._freq, _MONTH_ABBREVIATIONS[self.month])

    def __str__(self):
        return "<{}: n={}, month={}>".format(type(self).__name__, self.n, self.month)


class YearBegin(YearOffset):
    _freq = "AS"
    _day_option = "start"
    _default_month = 1

    def onOffset(self, date):
        """Check if the given date is in the set of possible dates created
        using a length-one version of this offset class."""
        return date.day == 1 and date.month == self.month

    def rollforward(self, date):
        """Roll date forward to nearest start of year"""
        if self.onOffset(date):
            return date
        else:
            return date + YearBegin(month=self.month)

    def rollback(self, date):
        """Roll date backward to nearest start of year"""
        if self.onOffset(date):
            return date
        else:
            return date - YearBegin(month=self.month)


class YearEnd(YearOffset):
    _freq = "A"
    _day_option = "end"
    _default_month = 12

    def onOffset(self, date):
        """Check if the given date is in the set of possible dates created
        using a length-one version of this offset class."""
        return date.day == _days_in_month(date) and date.month == self.month

    def rollforward(self, date):
        """Roll date forward to nearest end of year"""
        if self.onOffset(date):
            return date
        else:
            return date + YearEnd(month=self.month)

    def rollback(self, date):
        """Roll date backward to nearest end of year"""
        if self.onOffset(date):
            return date
        else:
            return date - YearEnd(month=self.month)


class Day(BaseCFTimeOffset):
    _freq = "D"

    def as_timedelta(self):
        return timedelta(days=self.n)

    def __apply__(self, other):
        return other + self.as_timedelta()


class Hour(BaseCFTimeOffset):
    _freq = "H"

    def as_timedelta(self):
        return timedelta(hours=self.n)

    def __apply__(self, other):
        return other + self.as_timedelta()


class Minute(BaseCFTimeOffset):
    _freq = "T"

    def as_timedelta(self):
        return timedelta(minutes=self.n)

    def __apply__(self, other):
        return other + self.as_timedelta()


class Second(BaseCFTimeOffset):
    _freq = "S"

    def as_timedelta(self):
        return timedelta(seconds=self.n)

    def __apply__(self, other):
        return other + self.as_timedelta()


class Millisecond(BaseCFTimeOffset):
    _freq = "L"

    def as_timedelta(self):
        return timedelta(milliseconds=self.n)

    def __apply__(self, other):
        return other + self.as_timedelta()


class Microsecond(BaseCFTimeOffset):
    _freq = "U"

    def as_timedelta(self):
        return timedelta(microseconds=self.n)

    def __apply__(self, other):
        return other + self.as_timedelta()


_FREQUENCIES = {
    "A": YearEnd,
    "AS": YearBegin,
    "Y": YearEnd,
    "YS": YearBegin,
    "Q": partial(QuarterEnd, month=12),
    "QS": partial(QuarterBegin, month=1),
    "M": MonthEnd,
    "MS": MonthBegin,
    "D": Day,
    "H": Hour,
    "T": Minute,
    "min": Minute,
    "S": Second,
    "L": Millisecond,
    "ms": Millisecond,
    "U": Microsecond,
    "us": Microsecond,
    "AS-JAN": partial(YearBegin, month=1),
    "AS-FEB": partial(YearBegin, month=2),
    "AS-MAR": partial(YearBegin, month=3),
    "AS-APR": partial(YearBegin, month=4),
    "AS-MAY": partial(YearBegin, month=5),
    "AS-JUN": partial(YearBegin, month=6),
    "AS-JUL": partial(YearBegin, month=7),
    "AS-AUG": partial(YearBegin, month=8),
    "AS-SEP": partial(YearBegin, month=9),
    "AS-OCT": partial(YearBegin, month=10),
    "AS-NOV": partial(YearBegin, month=11),
    "AS-DEC": partial(YearBegin, month=12),
    "A-JAN": partial(YearEnd, month=1),
    "A-FEB": partial(YearEnd, month=2),
    "A-MAR": partial(YearEnd, month=3),
    "A-APR": partial(YearEnd, month=4),
    "A-MAY": partial(YearEnd, month=5),
    "A-JUN": partial(YearEnd, month=6),
    "A-JUL": partial(YearEnd, month=7),
    "A-AUG": partial(YearEnd, month=8),
    "A-SEP": partial(YearEnd, month=9),
    "A-OCT": partial(YearEnd, month=10),
    "A-NOV": partial(YearEnd, month=11),
    "A-DEC": partial(YearEnd, month=12),
    "QS-JAN": partial(QuarterBegin, month=1),
    "QS-FEB": partial(QuarterBegin, month=2),
    "QS-MAR": partial(QuarterBegin, month=3),
    "QS-APR": partial(QuarterBegin, month=4),
    "QS-MAY": partial(QuarterBegin, month=5),
    "QS-JUN": partial(QuarterBegin, month=6),
    "QS-JUL": partial(QuarterBegin, month=7),
    "QS-AUG": partial(QuarterBegin, month=8),
    "QS-SEP": partial(QuarterBegin, month=9),
    "QS-OCT": partial(QuarterBegin, month=10),
    "QS-NOV": partial(QuarterBegin, month=11),
    "QS-DEC": partial(QuarterBegin, month=12),
    "Q-JAN": partial(QuarterEnd, month=1),
    "Q-FEB": partial(QuarterEnd, month=2),
    "Q-MAR": partial(QuarterEnd, month=3),
    "Q-APR": partial(QuarterEnd, month=4),
    "Q-MAY": partial(QuarterEnd, month=5),
    "Q-JUN": partial(QuarterEnd, month=6),
    "Q-JUL": partial(QuarterEnd, month=7),
    "Q-AUG": partial(QuarterEnd, month=8),
    "Q-SEP": partial(QuarterEnd, month=9),
    "Q-OCT": partial(QuarterEnd, month=10),
    "Q-NOV": partial(QuarterEnd, month=11),
    "Q-DEC": partial(QuarterEnd, month=12),
}


_FREQUENCY_CONDITION = "|".join(_FREQUENCIES.keys())
_PATTERN = fr"^((?P<multiple>\d+)|())(?P<freq>({_FREQUENCY_CONDITION}))$"


# pandas defines these offsets as "Tick" objects, which for instance have
# distinct behavior from monthly or longer frequencies in resample.
CFTIME_TICKS = (Day, Hour, Minute, Second)


def to_offset(freq):
    """Convert a frequency string to the appropriate subclass of
    BaseCFTimeOffset."""
    if isinstance(freq, BaseCFTimeOffset):
        return freq
    else:
        try:
            freq_data = re.match(_PATTERN, freq).groupdict()
        except AttributeError:
            raise ValueError("Invalid frequency string provided")

    freq = freq_data["freq"]
    multiples = freq_data["multiple"]
    multiples = 1 if multiples is None else int(multiples)
    return _FREQUENCIES[freq](n=multiples)


def to_cftime_datetime(date_str_or_date, calendar=None):
    import cftime

    if isinstance(date_str_or_date, str):
        if calendar is None:
            raise ValueError(
                "If converting a string to a cftime.datetime object, "
                "a calendar type must be provided"
            )
        date, _ = _parse_iso8601_with_reso(get_date_type(calendar), date_str_or_date)
        return date
    elif isinstance(date_str_or_date, cftime.datetime):
        return date_str_or_date
    else:
        raise TypeError(
            "date_str_or_date must be a string or a "
            "subclass of cftime.datetime. Instead got "
            "{!r}.".format(date_str_or_date)
        )


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
    values = np.linspace(0.0, total_seconds, periods, endpoint=True)
    units = "seconds since {}".format(format_cftime_datetime(start))
    calendar = start.calendar
    return cftime.num2date(
        values, units=units, calendar=calendar, only_use_cftime_datetimes=True
    )


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
    if start:
        start = offset.rollforward(start)

    if end:
        end = offset.rollback(end)

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
                raise ValueError(f"Offset {offset} did not increment date")
            current = next_date
    else:
        while current >= end:
            yield current

            next_date = current + offset
            if next_date >= current:
                raise ValueError(f"Offset {offset} did not decrement date")
            current = next_date


def cftime_range(
    start=None,
    end=None,
    periods=None,
    freq="D",
    normalize=False,
    name=None,
    closed=None,
    calendar="standard",
):
    """Return a fixed frequency CFTimeIndex.

    Parameters
    ----------
    start : str or cftime.datetime, optional
        Left bound for generating dates.
    end : str or cftime.datetime, optional
        Right bound for generating dates.
    periods : int, optional
        Number of periods to generate.
    freq : str or None, default: "D"
        Frequency strings can have multiples, e.g. "5H".
    normalize : bool, default: False
        Normalize start/end dates to midnight before generating date range.
    name : str, default: None
        Name of the resulting index
    closed : {"left", "right"} or None, default: None
        Make the interval closed with respect to the given frequency to the
        "left", "right", or both sides (None).
    calendar : str, default: "standard"
        Calendar type for the datetimes.

    Returns
    -------
    CFTimeIndex

    Notes
    -----
    This function is an analog of ``pandas.date_range`` for use in generating
    sequences of ``cftime.datetime`` objects.  It supports most of the
    features of ``pandas.date_range`` (e.g. specifying how the index is
    ``closed`` on either side, or whether or not to ``normalize`` the start and
    end bounds); however, there are some notable exceptions:

    - You cannot specify a ``tz`` (time zone) argument.
    - Start or end dates specified as partial-datetime strings must use the
      `ISO-8601 format <https://en.wikipedia.org/wiki/ISO_8601>`_.
    - It supports many, but not all, frequencies supported by
      ``pandas.date_range``.  For example it does not currently support any of
      the business-related or semi-monthly frequencies.
    - Compound sub-monthly frequencies are not supported, e.g. '1H1min', as
      these can easily be written in terms of the finest common resolution,
      e.g. '61min'.

    Valid simple frequency strings for use with ``cftime``-calendars include
    any multiples of the following.

    +--------+--------------------------+
    | Alias  | Description              |
    +========+==========================+
    | A, Y   | Year-end frequency       |
    +--------+--------------------------+
    | AS, YS | Year-start frequency     |
    +--------+--------------------------+
    | Q      | Quarter-end frequency    |
    +--------+--------------------------+
    | QS     | Quarter-start frequency  |
    +--------+--------------------------+
    | M      | Month-end frequency      |
    +--------+--------------------------+
    | MS     | Month-start frequency    |
    +--------+--------------------------+
    | D      | Day frequency            |
    +--------+--------------------------+
    | H      | Hour frequency           |
    +--------+--------------------------+
    | T, min | Minute frequency         |
    +--------+--------------------------+
    | S      | Second frequency         |
    +--------+--------------------------+
    | L, ms  | Millisecond frequency    |
    +--------+--------------------------+
    | U, us  | Microsecond frequency    |
    +--------+--------------------------+

    Any multiples of the following anchored offsets are also supported.

    +----------+--------------------------------------------------------------------+
    | Alias    | Description                                                        |
    +==========+====================================================================+
    | A(S)-JAN | Annual frequency, anchored at the end (or beginning) of January    |
    +----------+--------------------------------------------------------------------+
    | A(S)-FEB | Annual frequency, anchored at the end (or beginning) of February   |
    +----------+--------------------------------------------------------------------+
    | A(S)-MAR | Annual frequency, anchored at the end (or beginning) of March      |
    +----------+--------------------------------------------------------------------+
    | A(S)-APR | Annual frequency, anchored at the end (or beginning) of April      |
    +----------+--------------------------------------------------------------------+
    | A(S)-MAY | Annual frequency, anchored at the end (or beginning) of May        |
    +----------+--------------------------------------------------------------------+
    | A(S)-JUN | Annual frequency, anchored at the end (or beginning) of June       |
    +----------+--------------------------------------------------------------------+
    | A(S)-JUL | Annual frequency, anchored at the end (or beginning) of July       |
    +----------+--------------------------------------------------------------------+
    | A(S)-AUG | Annual frequency, anchored at the end (or beginning) of August     |
    +----------+--------------------------------------------------------------------+
    | A(S)-SEP | Annual frequency, anchored at the end (or beginning) of September  |
    +----------+--------------------------------------------------------------------+
    | A(S)-OCT | Annual frequency, anchored at the end (or beginning) of October    |
    +----------+--------------------------------------------------------------------+
    | A(S)-NOV | Annual frequency, anchored at the end (or beginning) of November   |
    +----------+--------------------------------------------------------------------+
    | A(S)-DEC | Annual frequency, anchored at the end (or beginning) of December   |
    +----------+--------------------------------------------------------------------+
    | Q(S)-JAN | Quarter frequency, anchored at the end (or beginning) of January   |
    +----------+--------------------------------------------------------------------+
    | Q(S)-FEB | Quarter frequency, anchored at the end (or beginning) of February  |
    +----------+--------------------------------------------------------------------+
    | Q(S)-MAR | Quarter frequency, anchored at the end (or beginning) of March     |
    +----------+--------------------------------------------------------------------+
    | Q(S)-APR | Quarter frequency, anchored at the end (or beginning) of April     |
    +----------+--------------------------------------------------------------------+
    | Q(S)-MAY | Quarter frequency, anchored at the end (or beginning) of May       |
    +----------+--------------------------------------------------------------------+
    | Q(S)-JUN | Quarter frequency, anchored at the end (or beginning) of June      |
    +----------+--------------------------------------------------------------------+
    | Q(S)-JUL | Quarter frequency, anchored at the end (or beginning) of July      |
    +----------+--------------------------------------------------------------------+
    | Q(S)-AUG | Quarter frequency, anchored at the end (or beginning) of August    |
    +----------+--------------------------------------------------------------------+
    | Q(S)-SEP | Quarter frequency, anchored at the end (or beginning) of September |
    +----------+--------------------------------------------------------------------+
    | Q(S)-OCT | Quarter frequency, anchored at the end (or beginning) of October   |
    +----------+--------------------------------------------------------------------+
    | Q(S)-NOV | Quarter frequency, anchored at the end (or beginning) of November  |
    +----------+--------------------------------------------------------------------+
    | Q(S)-DEC | Quarter frequency, anchored at the end (or beginning) of December  |
    +----------+--------------------------------------------------------------------+

    Finally, the following calendar aliases are supported.

    +--------------------------------+---------------------------------------+
    | Alias                          | Date type                             |
    +================================+=======================================+
    | standard, gregorian            | ``cftime.DatetimeGregorian``          |
    +--------------------------------+---------------------------------------+
    | proleptic_gregorian            | ``cftime.DatetimeProlepticGregorian`` |
    +--------------------------------+---------------------------------------+
    | noleap, 365_day                | ``cftime.DatetimeNoLeap``             |
    +--------------------------------+---------------------------------------+
    | all_leap, 366_day              | ``cftime.DatetimeAllLeap``            |
    +--------------------------------+---------------------------------------+
    | 360_day                        | ``cftime.Datetime360Day``             |
    +--------------------------------+---------------------------------------+
    | julian                         | ``cftime.DatetimeJulian``             |
    +--------------------------------+---------------------------------------+

    Examples
    --------
    This function returns a ``CFTimeIndex``, populated with ``cftime.datetime``
    objects associated with the specified calendar type, e.g.

    >>> xr.cftime_range(start="2000", periods=6, freq="2MS", calendar="noleap")
    CFTimeIndex([2000-01-01 00:00:00, 2000-03-01 00:00:00, 2000-05-01 00:00:00,
                 2000-07-01 00:00:00, 2000-09-01 00:00:00, 2000-11-01 00:00:00],
                dtype='object', length=6, calendar='noleap', freq='2MS')

    As in the standard pandas function, three of the ``start``, ``end``,
    ``periods``, or ``freq`` arguments must be specified at a given time, with
    the other set to ``None``.  See the `pandas documentation
    <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html>`_
    for more examples of the behavior of ``date_range`` with each of the
    parameters.

    See Also
    --------
    pandas.date_range
    """
    # Adapted from pandas.core.indexes.datetimes._generate_range.
    if count_not_none(start, end, periods, freq) != 3:
        raise ValueError(
            "Of the arguments 'start', 'end', 'periods', and 'freq', three "
            "must be specified at a time."
        )

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
    elif closed == "left":
        left_closed = True
    elif closed == "right":
        right_closed = True
    else:
        raise ValueError("Closed must be either 'left', 'right' or None")

    if not left_closed and len(dates) and start is not None and dates[0] == start:
        dates = dates[1:]
    if not right_closed and len(dates) and end is not None and dates[-1] == end:
        dates = dates[:-1]

    return CFTimeIndex(dates, name=name)
