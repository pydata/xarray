"""
CFTimeIndex port of pandas resampling
(pandas/pandas/core/resample.py)
Does not support non-integer freq
"""
from __future__ import absolute_import, division, print_function

from ..coding.cftimeindex import CFTimeIndex
from ..coding.cftime_offsets import (cftime_range, normalize_date,
                                     Day, CFTIME_TICKS)


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
    binner : CFTimeIndex
        Defines the edge of resampling bins by which original index values will
        be grouped into.
    labels : CFTimeIndex
        Define what the user actually sees the bins labeled as.
    """

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
        binner = labels = CFTimeIndex(data=[], name=index.name)
        return binner, [], labels

    first, last = _get_range_edges(index.min(), index.max(), freq,
                                   closed=closed,
                                   base=base)
    binner = labels = cftime_range(freq=freq,
                                   start=first,
                                   end=last,
                                   name=index.name)

    if closed == 'right':
        labels = binner
        if label == 'right':
            labels = labels[1:]
        else:
            labels = labels[:-1]
    else:
        if label == 'right':
            labels = labels[1:]
        else:
            labels = labels[:-1]

    return binner, labels


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
        is_day = isinstance(offset, Day)
        if (is_day and offset.n == 1) or not is_day:
            return _adjust_dates_anchored(first, last, offset,
                                          closed=closed, base=base)
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
    foffset = (first - start_day) % offset.as_timedelta()
    loffset = (last - start_day) % offset.as_timedelta()
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
