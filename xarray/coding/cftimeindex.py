"""DatetimeIndex analog for cftime.datetime objects"""
# The pandas.Index subclass defined here was copied and adapted for
# use with cftime.datetime objects based on the source code defining
# pandas.DatetimeIndex.

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

from __future__ import absolute_import
import re
from datetime import timedelta

import numpy as np
import pandas as pd

from xarray.core import pycompat
from xarray.core.utils import is_scalar


def named(name, pattern):
    return '(?P<' + name + '>' + pattern + ')'


def optional(x):
    return '(?:' + x + ')?'


def trailing_optional(xs):
    if not xs:
        return ''
    return xs[0] + optional(trailing_optional(xs[1:]))


def build_pattern(date_sep='\-', datetime_sep='T', time_sep='\:'):
    pieces = [(None, 'year', '\d{4}'),
              (date_sep, 'month', '\d{2}'),
              (date_sep, 'day', '\d{2}'),
              (datetime_sep, 'hour', '\d{2}'),
              (time_sep, 'minute', '\d{2}'),
              (time_sep, 'second', '\d{2}')]
    pattern_list = []
    for sep, name, sub_pattern in pieces:
        pattern_list.append((sep if sep else '') + named(name, sub_pattern))
        # TODO: allow timezone offsets?
    return '^' + trailing_optional(pattern_list) + '$'


_BASIC_PATTERN = build_pattern(date_sep='', time_sep='')
_EXTENDED_PATTERN = build_pattern()
_PATTERNS = [_BASIC_PATTERN, _EXTENDED_PATTERN]


def parse_iso8601(datetime_string):
    for pattern in _PATTERNS:
        match = re.match(pattern, datetime_string)
        if match:
            return match.groupdict()
    raise ValueError('no ISO-8601 match for string: %s' % datetime_string)


def _parse_iso8601_with_reso(date_type, timestr):
    default = date_type(1, 1, 1)
    result = parse_iso8601(timestr)
    replace = {}

    for attr in ['year', 'month', 'day', 'hour', 'minute', 'second']:
        value = result.get(attr, None)
        if value is not None:
            # Note ISO8601 conventions allow for fractional seconds.
            # TODO: Consider adding support for sub-second resolution?
            replace[attr] = int(value)
            resolution = attr

    return default.replace(**replace), resolution


def _parsed_string_to_bounds(date_type, resolution, parsed):
    """Generalization of
    pandas.tseries.index.DatetimeIndex._parsed_string_to_bounds
    for use with non-standard calendars and cftime.datetime
    objects.
    """
    if resolution == 'year':
        return (date_type(parsed.year, 1, 1),
                date_type(parsed.year + 1, 1, 1) - timedelta(microseconds=1))
    elif resolution == 'month':
        if parsed.month == 12:
            end = date_type(parsed.year + 1, 1, 1) - timedelta(microseconds=1)
        else:
            end = (date_type(parsed.year, parsed.month + 1, 1) -
                   timedelta(microseconds=1))
        return date_type(parsed.year, parsed.month, 1), end
    elif resolution == 'day':
        start = date_type(parsed.year, parsed.month, parsed.day)
        return start, start + timedelta(days=1, microseconds=-1)
    elif resolution == 'hour':
        start = date_type(parsed.year, parsed.month, parsed.day, parsed.hour)
        return start, start + timedelta(hours=1, microseconds=-1)
    elif resolution == 'minute':
        start = date_type(parsed.year, parsed.month, parsed.day, parsed.hour,
                          parsed.minute)
        return start, start + timedelta(minutes=1, microseconds=-1)
    elif resolution == 'second':
        start = date_type(parsed.year, parsed.month, parsed.day, parsed.hour,
                          parsed.minute, parsed.second)
        return start, start + timedelta(seconds=1, microseconds=-1)
    else:
        raise KeyError


def get_date_field(datetimes, field):
    """Adapted from pandas.tslib.get_date_field"""
    return np.array([getattr(date, field) for date in datetimes])


def _field_accessor(name, docstring=None):
    """Adapted from pandas.tseries.index._field_accessor"""
    def f(self):
        return get_date_field(self._data, name)

    f.__name__ = name
    f.__doc__ = docstring
    return property(f)


def get_date_type(self):
    if self.data:
        return type(self._data[0])
    else:
        return None


def assert_all_valid_date_type(data):
    import cftime

    if data.size:
        sample = data[0]
        date_type = type(sample)
        if not isinstance(sample, cftime.datetime):
            raise TypeError(
                'CFTimeIndex requires cftime.datetime '
                'objects. Got object of {}.'.format(date_type))
        if not all(isinstance(value, date_type) for value in data):
            raise TypeError(
                'CFTimeIndex requires using datetime '
                'objects of all the same type.  Got\n{}.'.format(data))


class CFTimeIndex(pd.Index):
    """Custom Index for working with CF calendars and dates

    All elements of a CFTimeIndex must be cftime.datetime objects.

    Parameters
    ----------
    data : array or CFTimeIndex
        Sequence of cftime.datetime objects to use in index
    name : str, default None
        Name of the resulting index

    See Also
    --------
    cftime_range
    """
    year = _field_accessor('year', 'The year of the datetime')
    month = _field_accessor('month', 'The month of the datetime')
    day = _field_accessor('day', 'The days of the datetime')
    hour = _field_accessor('hour', 'The hours of the datetime')
    minute = _field_accessor('minute', 'The minutes of the datetime')
    second = _field_accessor('second', 'The seconds of the datetime')
    microsecond = _field_accessor('microsecond',
                                  'The microseconds of the datetime')
    date_type = property(get_date_type)

    def __new__(cls, data, name=None):
        if name is None and hasattr(data, 'name'):
            name = data.name

        result = object.__new__(cls)
        result._data = np.array(data, dtype='O')
        assert_all_valid_date_type(result._data)
        result.name = name
        return result

    def _partial_date_slice(self, resolution, parsed):
        """Adapted from
        pandas.tseries.index.DatetimeIndex._partial_date_slice

        Note that when using a CFTimeIndex, if a partial-date selection
        returns a single element, it will never be converted to a scalar
        coordinate; this is in slight contrast to the behavior when using
        a DatetimeIndex, which sometimes will return a DataArray with a scalar
        coordinate depending on the resolution of the datetimes used in
        defining the index.  For example:

        >>> from cftime import DatetimeNoLeap
        >>> import pandas as pd
        >>> import xarray as xr
        >>> da = xr.DataArray([1, 2],
                              coords=[[DatetimeNoLeap(2001, 1, 1),
                                       DatetimeNoLeap(2001, 2, 1)]],
                              dims=['time'])
        >>> da.sel(time='2001-01-01')
        <xarray.DataArray (time: 1)>
        array([1])
        Coordinates:
          * time     (time) object 2001-01-01 00:00:00
        >>> da = xr.DataArray([1, 2],
                              coords=[[pd.Timestamp(2001, 1, 1),
                                       pd.Timestamp(2001, 2, 1)]],
                              dims=['time'])
        >>> da.sel(time='2001-01-01')
        <xarray.DataArray ()>
        array(1)
        Coordinates:
            time     datetime64[ns] 2001-01-01
        >>> da = xr.DataArray([1, 2],
                              coords=[[pd.Timestamp(2001, 1, 1, 1),
                                       pd.Timestamp(2001, 2, 1)]],
                              dims=['time'])
        >>> da.sel(time='2001-01-01')
        <xarray.DataArray (time: 1)>
        array([1])
        Coordinates:
          * time     (time) datetime64[ns] 2001-01-01T01:00:00
        """
        start, end = _parsed_string_to_bounds(self.date_type, resolution,
                                              parsed)
        lhs_mask = (self._data >= start)
        rhs_mask = (self._data <= end)
        return (lhs_mask & rhs_mask).nonzero()[0]

    def _get_string_slice(self, key):
        """Adapted from pandas.tseries.index.DatetimeIndex._get_string_slice"""
        parsed, resolution = _parse_iso8601_with_reso(self.date_type, key)
        loc = self._partial_date_slice(resolution, parsed)
        return loc

    def get_loc(self, key, method=None, tolerance=None):
        """Adapted from pandas.tseries.index.DatetimeIndex.get_loc"""
        if isinstance(key, pycompat.basestring):
            return self._get_string_slice(key)
        else:
            return pd.Index.get_loc(self, key, method=method,
                                    tolerance=tolerance)

    def _maybe_cast_slice_bound(self, label, side, kind):
        """Adapted from
        pandas.tseries.index.DatetimeIndex._maybe_cast_slice_bound"""
        if isinstance(label, pycompat.basestring):
            parsed, resolution = _parse_iso8601_with_reso(self.date_type,
                                                          label)
            start, end = _parsed_string_to_bounds(self.date_type, resolution,
                                                  parsed)
            if self.is_monotonic_decreasing and len(self) > 1:
                return end if side == 'left' else start
            return start if side == 'left' else end
        else:
            return label

    # TODO: Add ability to use integer range outside of iloc?
    # e.g. series[1:5].
    def get_value(self, series, key):
        """Adapted from pandas.tseries.index.DatetimeIndex.get_value"""
        if not isinstance(key, slice):
            return series.iloc[self.get_loc(key)]
        else:
            return series.iloc[self.slice_indexer(
                key.start, key.stop, key.step)]

    def __contains__(self, key):
        """Adapted from
        pandas.tseries.base.DatetimeIndexOpsMixin.__contains__"""
        try:
            result = self.get_loc(key)
            return (is_scalar(result) or type(result) == slice or
                    (isinstance(result, np.ndarray) and result.size))
        except (KeyError, TypeError, ValueError):
            return False

    def contains(self, key):
        """Needed for .loc based partial-string indexing"""
        return self.__contains__(key)
