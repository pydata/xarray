"""Resampling for CFTimeIndex. Does not support non-integer freq."""
# The mechanisms for resampling CFTimeIndex was copied and adapted from
# the source code defined in pandas.core.resample
#
# For reference, here is a copy of the pandas copyright notice:
#
# BSD 3-Clause License
#
# Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc.
# and PyData Development Team
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

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

    def __init__(self, freq, closed, label, base, loffset):
        self.freq = to_offset(freq)
        self.closed = closed
        self.label = label
        self.base = base
        self.loffset = loffset

        if isinstance(self.freq, (MonthEnd, YearEnd)):
            if self.closed is None:
                self.closed = 'right'
            if self.label is None:
                self.label = 'right'
        else:
            if self.closed is None:
                self.closed = 'left'
            if self.label is None:
                self.label = 'left'

    def first_items(self, index):
        """Meant to reproduce the results of the following

        grouper = pandas.Grouper(...)
        first_items = pd.Series(np.arange(len(index)),
                                index).groupby(grouper).first()

        with index being a CFTimeIndex instead of a DatetimeIndex.
        """

        datetime_bins, labels = _get_time_bins(index, self.freq, self.closed,
                                               self.label, self.base)
        if self.loffset is not None:
            if isinstance(self.loffset, datetime.timedelta):
                labels = labels + self.loffset
            else:
                labels = labels + to_offset(self.loffset)

        # check binner fits data
        if index[0] < datetime_bins[0]:
            raise ValueError("Value falls before first bin")
        if index[-1] > datetime_bins[-1]:
            raise ValueError("Value falls after last bin")

        integer_bins = np.searchsorted(
            index, datetime_bins, side=self.closed)[:-1]
        first_items = pd.Series(integer_bins, labels)

        # Mask duplicate values with NaNs, preserving the last values
        non_duplicate = ~first_items.duplicated('last')
        return first_items.where(non_duplicate)


def _get_time_bins(index, freq, closed, label, base):
    """Obtain the bins and their respective labels for resampling operations.

    Parameters
    ----------
    index : CFTimeIndex
        Index object to be resampled (e.g., CFTimeIndex named 'time').
    freq : xarray.coding.cftime_offsets.BaseCFTimeOffset
        The offset object representing target conversion a.k.a. resampling
        frequency (e.g., 'MS', '2D', 'H', or '3T' with
        coding.cftime_offsets.to_offset() applied to it).
    closed : 'left' or 'right', optional
        Which side of bin interval is closed.
        The default is 'left' for all frequency offsets except for 'M' and 'A',
        which have a default of 'right'.
    label : 'left' or 'right', optional
        Which bin edge label to label bucket with.
        The default is 'left' for all frequency offsets except for 'M' and 'A',
        which have a default of 'right'.
    base : int, optional
        For frequencies that evenly subdivide 1 day, the "origin" of the
        aggregated intervals. For example, for '5min' frequency, base could
        range from 0 through 4. Defaults to 0.

    Returns
    -------
    datetime_bins : CFTimeIndex
        Defines the edge of resampling bins by which original index values will
        be grouped into.
    labels : CFTimeIndex
        Define what the user actually sees the bins labeled as.
    """

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

    datetime_bins, labels = _adjust_bin_edges(datetime_bins, freq, closed,
                                              index, labels)

    if label == 'right':
        labels = labels[1:]
    else:
        labels = labels[:-1]

    # TODO: when CFTimeIndex supports missing values, if the reference index
    # contains missing values, insert the appropriate NaN value at the
    # beginning of the datetime_bins and labels indexes.

    return datetime_bins, labels


def _adjust_bin_edges(datetime_bins, offset, closed, index, labels):
    """This is required for determining the bin edges resampling with
    daily frequencies greater than one day, month end, and year end
    frequencies.

    Consider the following example.  Let's say you want to downsample the
    time series with the following coordinates to month end frequency:

    CFTimeIndex([2000-01-01 12:00:00, 2000-01-31 12:00:00,
                 2000-02-01 12:00:00], dtype='object')

    Without this adjustment, _get_time_bins with month-end frequency will
    return the following index for the bin edges (default closed='right' and
    label='right' in this case):

    CFTimeIndex([1999-12-31 00:00:00, 2000-01-31 00:00:00,
                 2000-02-29 00:00:00], dtype='object')

    If 2000-01-31 is used as a bound for a bin, the value on
    2000-01-31T12:00:00 (at noon on January 31st), will not be included in the
    month of January.  To account for this, pandas adds a day minus one worth
    of microseconds to the bin edges generated by cftime range, so that we do
    bin the value at noon on January 31st in the January bin.  This results in
    an index with bin edges like the following:

    CFTimeIndex([1999-12-31 23:59:59, 2000-01-31 23:59:59,
                 2000-02-29 23:59:59], dtype='object')

    The labels are still:

    CFTimeIndex([2000-01-31 00:00:00, 2000-02-29 00:00:00], dtype='object')

    This is also required for daily frequencies longer than one day and
    year-end frequencies.
    """
    is_super_daily = (isinstance(offset, (MonthEnd, YearEnd)) or
                      (isinstance(offset, Day) and offset.n > 1))
    if is_super_daily:
        if closed == 'right':
            datetime_bins = datetime_bins + datetime.timedelta(days=1,
                                                               microseconds=-1)
        if datetime_bins[-2] > index.max():
            datetime_bins = datetime_bins[:-1]
            labels = labels[:-1]

    return datetime_bins, labels


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
    foffset = exact_cftime_datetime_difference(
        start_day, first) % offset.as_timedelta()
    loffset = exact_cftime_datetime_difference(
        start_day, last) % offset.as_timedelta()
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
    """Exact computation of b - a

    Assumes:

        a = a_0 + a_m
        b = b_0 + b_m

    Here a_0, and b_0 represent the input dates rounded
    down to the nearest second, and a_m, and b_m represent
    the remaining microseconds associated with date a and
    date b.

    We can then express the value of b - a as:

        b - a = (b_0 + b_m) - (a_0 + a_m) = b_0 - a_0 + b_m - a_m

    By construction, we know that b_0 - a_0 must be a round number
    of seconds.  Therefore we can take the result of b_0 - a_0 using
    ordinary cftime.datetime arithmetic and round to the nearest
    second.  b_m - a_m is the remainder, in microseconds, and we
    can simply add this to the rounded timedelta.

    Parameters
    ----------
    a : cftime.datetime
        Input datetime
    b : cftime.datetime
        Input datetime

    Returns
    -------
    datetime.timedelta
    """
    seconds = b.replace(microsecond=0) - a.replace(microsecond=0)
    seconds = int(round(seconds.total_seconds()))
    microseconds = b.microsecond - a.microsecond
    return datetime.timedelta(seconds=seconds, microseconds=microseconds)
