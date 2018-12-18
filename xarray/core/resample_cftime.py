"""
CFTimeIndex port of pandas resampling
(pandas/pandas/core/resample.py)
Does not support non-integer freq
"""
from __future__ import absolute_import, division, print_function

import datetime
from ..coding.cftimeindex import CFTimeIndex
from ..coding.cftime_offsets import (cftime_range, normalize_date,
                                     Day, Hour, Minute, Second, CFTIME_TICKS)


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
    print('og_first: ', index.min(), 'og_last: ', index.max())
    print('first: ', first, 'last: ', last)
    print(binner[0], binner[-1], binner[-2], labels[0], labels[-1], len(labels), len(binner))

    # if len(binner) > 1 and binner[-1] < last:
    #     extra_date_range = cftime_range(binner[-1], last + freq,
    #                                     freq=freq, name=index.name)
    #     binner = labels = CFTimeIndex(binner.append(extra_date_range[1:]))
    #     print(extra_date_range)
    # Removing code block helps more tests pass, failed: 8, passed: 152
    # Originally was, failed: 40, passed: 120

    trimmed = False
    # if len(binner) > 2 and binner[-2] == last and closed == 'right':
    #     binner = binner[:-1]
    #     trimmed = True
    # Removing code block above has no effect on tests, failed: 8, passed: 152
    print(binner[0], binner[-1], binner[-2], labels[0], labels[-1], len(labels), len(binner))

    if closed == 'right':
        labels = binner
        if label == 'right':
            labels = labels[1:]
        elif not trimmed:
            labels = labels[:-1]
    else:
        if label == 'right':
            labels = labels[1:]
        elif not trimmed:
            labels = labels[:-1]
    print(binner[0], binner[-1], labels[0], labels[-1], len(labels), len(binner))

    # # Non-pandas logic. Removes extra bins at the end/tail of time range.
    # first_diff = labels[0] - index.min()
    # last1_diff = labels[-1] - index.max()
    # last2_diff = labels[-2] - index.max()
    # first_last1_diff = last1_diff - first_diff
    # first_last2_diff = last2_diff - first_diff
    # # print(labels[0], labels[-1], labels[-2])
    # # print('first: ', first, 'last: ', last)
    # # print(first_diff)
    # # print(labels[0] - first)
    # # print('first_diff: ', first_diff,
    # #       'last1_diff: ', last1_diff,
    # #       'last2_diff: ', last2_diff,
    # #       'first_last1_diff: ', first_last1_diff,
    # #       'first_last2_diff: ', first_last2_diff)
    # print(labels)
    # print(binner)
    # if abs(first_last1_diff) >= abs(first_last2_diff):
    #     print('hello!')
    #     labels = labels[:-1]
    # if len(binner) > (len(labels) + 1):
    #     print('world!')
    #     binner = binner[:(len(labels) + 1)]
    # print(labels)
    # print(binner)
    # print(binner[0], binner[-1], labels[0], labels[-1], len(labels), len(binner))
    # # Non-pandas logic. Removes extra bins at the end/tail of time range.

    return binner, labels


# def _adjust_bin_edges(binner, ax_values, freq):
#     """ Some hacks for > daily data, see pandas GitHub #1471, #1458, #1483
#     Currently --unused-- in xarray/cftime resampling operations.
#
#     Parameters
#     ----------
#     binner : CFTimeIndex
#         Defines the edge of resampling bins by which original index values will
#         be grouped into. Uncorrected version.
#     ax_values : CFTimeIndex
#         Values of the original, un-resampled CFTimeIndex range.
#     freq : xarray.coding.cftime_offsets.BaseCFTimeOffset
#         The offset object representing target conversion a.k.a. resampling
#         frequency (e.g., 'MS', '2D', 'H', or '3T' with
#         coding.cftime_offsets.to_offset() applied to it).
#
#     Returns
#     -------
#     binner : CFTimeIndex
#         Defines the edge of resampling bins by which original index values will
#         be grouped into. Corrected version.
#     """
#     if not isinstance(freq, CFTIME_TICKS):
#         # intraday values on last day
#         if binner[-2] > ax_values.max():
#             binner = binner[:-1]
#     return binner


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
    print('offset freq: ', offset._freq)
    if isinstance(offset, CFTIME_TICKS):
        is_day = isinstance(offset, Day)
        print(is_day)
        if (is_day and offset.n == 1) or not is_day:
            print('double trues')
            return _adjust_dates_anchored(first, last, offset,
                                          closed=closed, base=base)
    else:
        first = normalize_date(first)
        last = normalize_date(last)
        print('last_normed: ', last)

    if closed == 'left':
        first = offset.rollback(first)
    else:
        first = first - offset

    last = last + offset
    print('last+offset: ', last, 'offset: ', offset)
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


# def _offset_timedelta(offset):
#     """ Convert cftime_offsets to timedelta so that timedelta operations
#     can be performed.
#
#     Parameters
#     ----------
#     offset : xarray.coding.cftime_offsets.BaseCFTimeOffset
#         The offset object representing target conversion a.k.a. resampling
#         frequency. Contains information on offset type (e.g. Day or 'D') and
#         offset magnitude (e.g., n = 3).
#
#     Returns
#     -------
#     datetime.timedelta : datetime.timedelta
#         A timedelta object representing the value of the offset.
#     """
#
#     if isinstance(offset, Day):
#         return datetime.timedelta(days=offset.n)
#     elif isinstance(offset, Hour):
#         return datetime.timedelta(hours=offset.n)
#     elif isinstance(offset, Minute):
#         return datetime.timedelta(minutes=offset.n)
#     elif isinstance(offset, Second):
#         return datetime.timedelta(seconds=offset.n)


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
