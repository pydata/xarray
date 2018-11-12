"""
CFTimeIndex port of pandas resampling
(pandas/pandas/core/resample.py)
Does not support non-integer freq
"""
from __future__ import absolute_import, division, print_function

import datetime
from ..coding.cftimeindex import CFTimeIndex
from ..coding.cftime_offsets import (cftime_range, normalize_date,
                                     Day, Hour, Minute, Second)


def _get_time_bins(index, freq, closed, label, base):
    # This portion of code comes from TimeGrouper __init__ #
    end_types = {'M', 'A'}
    if freq._freq in end_types:
        if closed is None:
            closed = 'right'
        if label is None:
            label = 'right'
    else:
        if closed is None:
            closed = 'left'
        if label is None:
            label = 'left'
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

    if len(binner) > 1 and binner[-1] < last:
        extra_date_range = cftime_range(binner[-1], last + freq,
                                        freq=freq, name=index.name)
        binner = labels = CFTimeIndex(binner.append(extra_date_range[1:]))

    trimmed = False
    if len(binner) > 2 and binner[-2] == last and closed == 'right':
        binner = binner[:-1]
        trimmed = True

    # binner = _adjust_bin_edges(binner, index.values, freq)

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
    return binner, labels


def _adjust_bin_edges(binner, ax_values, freq):
    # Some hacks for > daily data, see #1471, #1458, #1483
    if freq._freq not in ['D', 'H', 'T', 'min', 'S']:
        # intraday values on last day
        if binner[-2] > ax_values.max():
            binner = binner[:-1]
    return binner


def _get_range_edges(first, last, offset, closed='left', base=0):
    if offset._freq in ['D', 'H', 'T', 'min', 'S']:
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
    base = base % offset.n
    start_day = normalize_date(first)
    base_td = datetime.timedelta(0)
    if offset._freq == 'D':
        base_td = datetime.timedelta(days=base)
    elif offset._freq == 'H':
        base_td = datetime.timedelta(hours=base)
    elif offset._freq in ['T', 'min']:
        base_td = datetime.timedelta(minutes=base)
    elif offset._freq == 'S':
        base_td = datetime.timedelta(seconds=base)
    offset_td = _offset_timedelta(offset)
    start_day += base_td
    foffset = (first - start_day) % offset_td
    loffset = (last - start_day) % offset_td
    if closed == 'right':
        if foffset.total_seconds() > 0:
            fresult = first - foffset
        else:
            fresult = first - offset_td

        if loffset.total_seconds() > 0:
            lresult = last + (offset_td - loffset)
        else:
            lresult = last
    else:
        if foffset.total_seconds() > 0:
            fresult = first - foffset
        else:
            fresult = first

        if loffset.total_seconds() > 0:
            lresult = last + (offset_td - loffset)
        else:
            lresult = last + offset_td
    return fresult, lresult


def _offset_timedelta(offset):
    if isinstance(offset, Day):
        return datetime.timedelta(days=offset.n)
    elif isinstance(offset, Hour):
        return datetime.timedelta(hours=offset.n)
    elif isinstance(offset, Minute):
        return datetime.timedelta(minutes=offset.n)
    elif isinstance(offset, Second):
        return datetime.timedelta(seconds=offset.n)


def _adjust_binner_for_upsample(binner, closed):
    if closed == 'right':
        binner = binner[1:]
    else:
        binner = binner[:-1]
    return binner
