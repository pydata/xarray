import numpy as np
import pandas as pd

from .cftime_offsets import _MONTH_ABBREVIATIONS
from .cftimeindex import CFTimeIndex

_ONE_MICRO = 1
_ONE_MILLI = _ONE_MICRO * 1000
_ONE_SECOND = _ONE_MILLI * 1000
_ONE_MINUTE = 60 * _ONE_SECOND
_ONE_HOUR = 60 * _ONE_MINUTE
_ONE_DAY = 24 * _ONE_HOUR


def infer_freq(index):
    """
    Infer the most likely frequency given the input index.

    Parameters
    ----------
    index : CFTimeIndex, DataArray, pd.DatetimeIndex or pd.TimedeltaIndex
      If not passed a CFTimeIndex, this simply calls `pandas.infer_freq`.
      If passed a Series or a DataArray will use the values of the series (NOT THE INDEX).

    Returns
    -------
    str or None
        None if no discernible frequency.

    Raises
    ------
    TypeError
        If the index is not datetime-like.
    ValueError
        If there are fewer than three values or the index is not 1D.
    """
    from xarray.core.dataarray import DataArray

    if isinstance(index, DataArray):
        if index.ndim > 1:
            raise ValueError("'index' must be 1D")
        if np.asarray(index).dtype == "datetime64[ns]":
            index = pd.DatetimeIndex(index)
        else:
            index = CFTimeIndex(index)

    if isinstance(index, CFTimeIndex):
        inferer = _CFTimeFrequencyInferer(index)
        return inferer.get_freq()

    return pd.infer_freq(index)


class _CFTimeFrequencyInferer:  # (pd.tseries.frequencies._FrequencyInferer):
    def __init__(self, index):
        self.index = index
        self.values = index.asi8

        if len(index) < 3:
            raise ValueError("Need at least 3 dates to infer frequency")

        self.is_monotonic = (
            self.index.is_monotonic_decreasing or self.index.is_monotonic_increasing
        )

        self._deltas = None
        self._year_deltas = None
        self._month_deltas = None

    def get_freq(self):
        """Find the appropriate frequency string to describe the inferred frequency of self.index

        Adapted from `pandas.tsseries.frequencies._FrequencyInferer.get_freq` for CFTimeIndexes.

        Returns
        -------
        str or None
        """
        if not self.is_monotonic or not self.index.is_unique:
            return None

        delta = self.deltas[0]  # Smallest delta
        if _is_multiple(delta, _ONE_DAY):
            return self._infer_daily_rule()
        # There is not other possible intraday frequency
        # Different from pandas: we don't need to manage DST and business offsets in cftime
        elif not len(self.deltas) == 1:
            return None

        if _is_multiple(delta, _ONE_HOUR):
            # Hours
            return _maybe_add_count("H", delta / _ONE_HOUR)
        elif _is_multiple(delta, _ONE_MINUTE):
            # Minutes
            return _maybe_add_count("T", delta / _ONE_MINUTE)
        elif _is_multiple(delta, _ONE_SECOND):
            # Seconds
            return _maybe_add_count("S", delta / _ONE_SECOND)
        elif _is_multiple(delta, _ONE_MILLI):
            # Milliseconds
            return _maybe_add_count("L", delta / _ONE_MILLI)
        else:
            # Microseconds (smallest CFTime division)
            return _maybe_add_count("U", delta / _ONE_MICRO)

    def _infer_daily_rule(self):
        annual_rule = self._get_annual_rule()
        if annual_rule:
            nyears = self.year_deltas[0]
            month = _MONTH_ABBREVIATIONS[self.index[0].month]
            alias = f"{annual_rule}-{month}"
            return _maybe_add_count(alias, nyears)

        quartely_rule = self._get_quartely_rule()
        if quartely_rule:
            nquarters = self.month_deltas[0] / 3
            mod_dict = {0: 12, 2: 11, 1: 10}
            month = _MONTH_ABBREVIATIONS[mod_dict[self.index[0].month % 3]]
            alias = f"{quartely_rule}-{month}"
            return _maybe_add_count(alias, nquarters)

        monthly_rule = self._get_monthly_rule()
        if monthly_rule:
            return _maybe_add_count(monthly_rule, self.month_deltas[0])

        if len(self.deltas) == 1:
            # Daily as there is no "Weekly" offsets with CFTime
            days = self.deltas[0] / _ONE_DAY
            return _maybe_add_count("D", days)

        # CFTime has no business freq and no "week of month" (WOM)
        return None

    def _get_annual_rule(self):
        if len(self.year_deltas) > 1:
            return None

        if len(np.unique(self.index.month)) > 1:
            return None

        return {"cs": "AS", "ce": "A"}.get(month_anchor_check(self.index))

    def _get_quartely_rule(self):
        if len(self.month_deltas) > 1:
            return None

        if not self.month_deltas[0] % 3 == 0:
            return None

        return {"cs": "QS", "ce": "Q"}.get(month_anchor_check(self.index))

    def _get_monthly_rule(self):
        if len(self.month_deltas) > 1:
            return None

        return {"cs": "MS", "ce": "M"}.get(month_anchor_check(self.index))

    @property
    def deltas(self):
        """Sorted unique timedeltas as microseconds."""
        if self._deltas is None:
            self._deltas = _unique_deltas(self.values)
        return self._deltas

    @property
    def year_deltas(self):
        """Sorted unique year deltas."""
        if self._year_deltas is None:
            self._year_deltas = _unique_deltas(self.index.year)
        return self._year_deltas

    @property
    def month_deltas(self):
        """Sorted unique month deltas."""
        if self._month_deltas is None:
            self._month_deltas = _unique_deltas(self.index.year * 12 + self.index.month)
        return self._month_deltas


def _unique_deltas(arr):
    """Sorted unique deltas of numpy array"""
    return np.sort(np.unique(np.diff(arr)))


def _is_multiple(us, mult: int):
    """Whether us is a multiple of mult"""
    return us % mult == 0


def _maybe_add_count(base: str, count: float):
    """If count is greater than 1, add it to the base offset string"""
    if count != 1:
        assert count == int(count)
        count = int(count)
        return f"{count}{base}"
    else:
        return base


def month_anchor_check(dates):
    """Return the monthly offset string.

    Return "cs" if all dates are the first days of the month,
    "ce" if all dates are the last day of the month,
    None otherwise.

    Replicated pandas._libs.tslibs.resolution.month_position_check
    but without business offset handling.
    """
    calendar_end = True
    calendar_start = True

    for date in dates:
        if calendar_start:
            calendar_start &= date.day == 1

        if calendar_end:
            cal = date.day == date.daysinmonth
            if calendar_end:
                calendar_end &= cal
        elif not calendar_start:
            break

    if calendar_end:
        return "ce"
    elif calendar_start:
        return "cs"
    else:
        return None
