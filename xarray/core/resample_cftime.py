"""
CFTimeIndex port of pandas resampling
(pandas/pandas/core/resample.py)
Does not support non-integer freq
"""
from __future__ import absolute_import, division, print_function

from ..coding.cftimeindex import CFTimeIndex
from ..coding.cftime_offsets import (cftime_range, normalize_date,
                                     Day, MonthEnd, YearEnd,
                                     CFTIME_TICKS, to_offset)
import datetime
import numpy as np
import pandas as pd


class CFTimeGrouper(object):
    """This is a simple container for the grouping parameters that implements a
    single method, the only one required for resampling in xarray.  It cannot
    be used in a call to groupby like a pandas.Grouper object can."""

    def __init__(self, freq, closed, label, base):
        self.freq = to_offset(freq)
        self.closed = closed
        self.label = label
        self.base = base

    def first_items(self, index):
        """Meant to reproduce the results of the following

        grouper = pandas.Grouper(...)
        first_items = pd.Series(np.arange(len(index)), index).groupby(grouper).first()

        with index being a CFTimeIndex instead of a DatetimeIndex.
        """
        datetime_bins, labels = _get_time_bins(index, self.freq, self.closed,
                                               self.label, self.base)

        # check binner fits data
        if index[0] < datetime_bins[0]:
            raise ValueError("Value falls before first bin")
        if index[len(index) - 1] > datetime_bins[len(datetime_bins) - 1]:
            raise ValueError("Value falls after last bin")

        integer_bins = np.searchsorted(index, datetime_bins, side=self.closed)[
                       :-1]
        if len(integer_bins) < len(labels):
            labels = labels[:len(integer_bins)]
        first_items = pd.Series(integer_bins, labels)

        # Mask duplicate values with NaNs, preserving the last values
        non_duplicate = ~first_items.duplicated('last')
        return first_items.where(non_duplicate)


def _get_time_bins(index, freq, closed, label, base):
    """This is basically the same with the exception of the call to
    _adjust_bin_edges."""
    # This portion of code comes from TimeGrouper __init__ #
    if closed is None:
        closed = _default_closed_or_label(freq)

    if label is None:
        label = _default_closed_or_label(freq)
    # This portion of code comes from TimeGrouper __init__ #

    if not isinstance(index, CFTimeIndex):
        raise TypeError('index must be a CFTimeIndex, but got '
                        'an instance of %r' % type(index).__name__)
    if len(index) == 0:
        datetime_bins = labels = CFTimeIndex(data=[], name=index.name)
        return datetime_bins, labels

    first, last = _get_range_edges(index.min(), index.max(), freq,
                                   closed=closed,
                                   base=base)
    datetime_bins = labels = cftime_range(freq=freq,
                                          start=first,
                                          end=last,
                                          name=index.name)
    print('XARRAY-START')
    print(index.min(), index.max())
    print(first, last)
    print('initial range\n', datetime_bins)
    print('len binner: ', len(datetime_bins),
          'len labels: ', len(labels))

    datetime_bins = _adjust_bin_edges(datetime_bins, freq, closed, index)

    print('len datetime_bins: ', len(datetime_bins),
          'len labels: ', len(labels))
    print('_adjust_bin_edges\n', datetime_bins)

    if closed == 'right':
        if label == 'right':
            labels = labels[1:]
    elif label == 'right':
        labels = labels[1:]

        print('len datetime_bins: ', len(datetime_bins),
              'len labels: ', len(labels))

    if index.hasnans:  # cannot be true since CFTimeIndex does not allow NaNs
        datetime_bins = datetime_bins.insert(0, pd.NaT)
        labels = labels.insert(0, pd.NaT)

        print('len binner: ', len(datetime_bins),
              'len labels: ', len(labels))
    print('XARRAY-END')

    return datetime_bins, labels


def _adjust_bin_edges(datetime_bins, offset, closed, index):
    """This is required for determining the bin edges resampling with
    daily frequencies greater than one day, month end, and year end
    frequencies.

    Consider the following example.  Let's say you want to downsample the
    time series with the following coordinates to month end frequency:

    CFTimeIndex([2000-01-01 12:00:00, 2000-01-31 12:00:00, 2000-02-01 12:00:00], dtype='object')

    Without this adjustment, _get_time_bins with month-end frequency will
    return the following index for the bin edges (default closed='right' and
    label='right' in this case):

    CFTimeIndex([1999-12-31 00:00:00, 2000-01-31 00:00:00, 2000-02-29 00:00:00], dtype='object')

    If 2000-01-31 is used as a bound for a bin, the value on
    2000-01-31T12:00:00 (at noon on January 31st), will not be included in the
    month of January.  To account for this, pandas adds a day minus one worth
    of microseconds to the bin edges generated by cftime range, so that we do
    bin the value at noon on January 31st in the January bin.  This results in
    an index with bin edges like the following:

    CFTimeIndex([1999-12-31 23:59:59, 2000-01-31 23:59:59, 2000-02-29 23:59:59], dtype='object')

    The labels are still:

    CFTimeIndex([2000-01-31 00:00:00, 2000-02-29 00:00:00], dtype='object')

    This is also required for daily frequencies longer than one day and
    year-end frequencies.
    """
    is_super_daily = (isinstance(offset, (MonthEnd, YearEnd)) or
                      (isinstance(offset, Day) and offset.n > 1))
    if is_super_daily:
        if closed == 'right':
            datetime_bins = datetime_bins + \
                            datetime.timedelta(days=1, microseconds=-1)
        if datetime_bins[-2] > index.max():
            datetime_bins = datetime_bins[:-1]
    return datetime_bins


def _get_range_edges(first, last, offset, closed='left', base=0):
    """ Get the correct starting and ending datetimes for the resampled
    CFTimeIndex range.

    Parameters
    ----------
    first : cftime.datetime
        Uncorrected starting datetime object for resampled CFTimeIndex range.
        Usually the min of the original CFTimeIndex.
    last : cftime.datetime
        Uncorrected ending datetime object for resampled CFTimeIndex range.
        Usually the max of the original CFTimeIndex.
    offset : xarray.coding.cftime_offsets.BaseCFTimeOffset
        The offset object representing target conversion a.k.a. resampling
        frequency. Contains information on offset type (e.g. Day or 'D') and
        offset magnitude (e.g., n = 3).
    closed : 'left' or 'right', optional
        Which side of bin interval is closed. Defaults to 'left'.
    base : int, optional
        For frequencies that evenly subdivide 1 day, the "origin" of the
        aggregated intervals. For example, for '5min' frequency, base could
        range from 0 through 4. Defaults to 0.

    Returns
    -------
    first : cftime.datetime
        Corrected starting datetime object for resampled CFTimeIndex range.
    last : cftime.datetime
        Corrected ending datetime object for resampled CFTimeIndex range.
    """
    if isinstance(offset, CFTIME_TICKS):
        first, last = _adjust_dates_anchored(first, last, offset,
                                             closed=closed, base=base)
        # if isinstance(offset, Day):
        # first = normalize_date(first)
        # last = normalize_date(last)
        return first, last
    else:
        first = normalize_date(first)
        last = normalize_date(last)

    if closed == 'left':
        first = offset.rollback(first)
    else:
        first = first - offset

    last = last + offset
    return first, last


def _adjust_dates_anchored(first, last, offset, closed='right', base=0):
    """ First and last offsets should be calculated from the start day to fix
    an error cause by resampling across multiple days when a one day period is
    not a multiple of the frequency.
    See https://github.com/pandas-dev/pandas/issues/8683

    Parameters
    ----------
    first : cftime.datetime
        A datetime object representing the start of a CFTimeIndex range.
    last : cftime.datetime
        A datetime object representing the end of a CFTimeIndex range.
    offset : xarray.coding.cftime_offsets.BaseCFTimeOffset
        The offset object representing target conversion a.k.a. resampling
        frequency. Contains information on offset type (e.g. Day or 'D') and
        offset magnitude (e.g., n = 3).
    closed : 'left' or 'right', optional
        Which side of bin interval is closed. Defaults to 'right'.
    base : int, optional
        For frequencies that evenly subdivide 1 day, the "origin" of the
        aggregated intervals. For example, for '5min' frequency, base could
        range from 0 through 4. Defaults to 0.

    Returns
    -------
    fresult : cftime.datetime
        A datetime object representing the start of a date range that has been
        adjusted to fix resampling errors.
    lresult : cftime.datetime
        A datetime object representing the end of a date range that has been
        adjusted to fix resampling errors.
    """

    base = base % offset.n
    start_day = normalize_date(first)
    base_td = type(offset)(n=base).as_timedelta()
    start_day += base_td
    foffset = exact_cftime_datetime_difference(start_day, first) % offset.as_timedelta()
    loffset = exact_cftime_datetime_difference(start_day, last) % offset.as_timedelta()
    if closed == 'right':
        if foffset.total_seconds() > 0:
            fresult = first - foffset
        else:
            fresult = first - offset.as_timedelta()

        if loffset.total_seconds() > 0:
            lresult = last + (offset.as_timedelta() - loffset)
        else:
            lresult = last
    else:
        if foffset.total_seconds() > 0:
            fresult = first - foffset
        else:
            fresult = first

        if loffset.total_seconds() > 0:
            lresult = last + (offset.as_timedelta() - loffset)
        else:
            lresult = last + offset.as_timedelta()
    return fresult, lresult


def exact_cftime_datetime_difference(a, b):
    """Exact computation of b - a"""
    seconds = b.replace(microsecond=0) - a.replace(microsecond=0)
    seconds = int(round(seconds.total_seconds()))
    microseconds = b.microsecond - a.microsecond
    return datetime.timedelta(seconds=seconds, microseconds=microseconds)


def _adjust_binner_for_upsample(binner, closed):
    """ Adjust our binner when upsampling.
        The range of a new index should not be outside specified range

    Parameters
    ----------
    binner : CFTimeIndex
        Defines the edge of resampling bins by which original index values will
        be grouped into. Uncorrected version.
    closed : 'left' or 'right'
        Which side of bin interval is closed.

    Returns
    -------
    binner : CFTimeIndex
        Defines the edge of resampling bins by which original index values will
        be grouped into. Corrected version.
    """

    if closed == 'right':
        binner = binner[1:]
    else:
        binner = binner[:-1]
    return binner


def _default_closed_or_label(freq):
    end_types = {'M', 'A'}
    if freq._freq in end_types:
        return 'right'
    else:
        return 'left'
