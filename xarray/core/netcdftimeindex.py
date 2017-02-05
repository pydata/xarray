import re
from datetime import timedelta

import numpy as np
import pandas as pd

from pandas.lib import isscalar


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
              (time_sep, 'second', '\d{2}' + optional('\.\d+'))]
    pattern_list = []
    for sep, name, sub_pattern in pieces:
        pattern_list.append((sep if sep else '') + named(name, sub_pattern))
        # TODO: allow timezone offsets?
    return '^' + trailing_optional(pattern_list) + '$'


def parse_iso8601(datetime_string):
    basic_pattern = build_pattern(date_sep='', time_sep='')
    extended_pattern = build_pattern()
    patterns = [basic_pattern, extended_pattern]
    for pattern in patterns:
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
            replace[attr] = int(value)
            resolution = attr

    return default.replace(**replace), resolution


def _parsed_string_to_bounds(date_type, resolution, parsed):
    if resolution == 'year':
        return (date_type(parsed.year, 1, 1),
                date_type(parsed.year + 1, 1, 1) - timedelta(microseconds=1))
    if resolution == 'month':
        if parsed.month == 12:
            end = date_type(parsed.year + 1, 1, 1) - timedelta(microseconds=1)
        else:
            end = (date_type(parsed.year, parsed.month + 1, 1) -
                   timedelta(microseconds=1))
        return date_type(parsed.year, parsed.month, 1), end
    if resolution == 'day':
        start = date_type(parsed.year, parsed.month, parsed.day)
        return start, start + timedelta(days=1, microseconds=-1)
    if resolution == 'hour':
        start = date_type(parsed.year, parsed.month, parsed.day, parsed.hour)
        return start, start + timedelta(hours=1, microseconds=-1)
    if resolution == 'minute':
        start = date_type(parsed.year, parsed.month, parsed.day, parsed.hour,
                          parsed.minute)
        return start, start + timedelta(minutes=1, microseconds=-1)
    if resolution == 'second':
        start = date_type(parsed.year, parsed.month, parsed.day, parsed.hour,
                          parsed.minute, parsed.second)
        return start, start + timedelta(seconds=1, microseconds=-1)
    else:
        raise KeyError


def get_date_field(datetimes, field):
    return [getattr(date, field) for date in datetimes]


def _field_accessor(name, docstring=None):
    def f(self):
        return get_date_field(self._data, name)

    f.__name__ = name
    f.__doc__ = docstring
    return property(f)


def get_date_type(self):
    return type(self._data[0])


class NetCDFTimeIndex(pd.Index):
    def __new__(cls, data):
        result = object.__new__(cls)
        result._data = np.array(data)
        return result

    year = _field_accessor('year', 'The year of the datetime')
    month = _field_accessor('month', 'The month of the datetime')
    day = _field_accessor('day', 'The days of the datetime')
    hour = _field_accessor('hour', 'The hours of the datetime')
    minute = _field_accessor('minute', 'The minutes of the datetime')
    second = _field_accessor('second', 'The seconds of the datetime')
    microsecond = _field_accessor('microsecond',
                                  'The microseconds of the datetime')
    date_type = property(get_date_type)

    def _partial_date_slice(self, resolution, parsed,
                            use_lhs=True, use_rhs=True):
        start, end = _parsed_string_to_bounds(self.date_type, resolution,
                                              parsed)
        lhs_mask = (self._data >= start) if use_lhs else True
        rhs_mask = (self._data <= end) if use_rhs else True
        return (lhs_mask & rhs_mask).nonzero()[0]

    def _get_string_slice(self, key, use_lhs=True, use_rhs=True):
        parsed, resolution = _parse_iso8601_with_reso(self.date_type, key)
        loc = self._partial_date_slice(resolution, parsed, use_lhs, use_rhs)
        return loc

    def get_loc(self, key, method=None, tolerance=None):
        if isinstance(key, pd.compat.string_types):
            result = self._get_string_slice(key)
            # Prevents problem with __contains__ if key corresponds to only
            # the first element in index (if we leave things as a list,
            # np.any([0]) is False).
            # Also coerces things to scalar coords in xarray if possible,
            # which is consistent with the behavior with a DatetimeIndex.
            if len(result) == 1:
                return result[0]
            else:
                return result
        else:
            return pd.Index.get_loc(self, key, method=method,
                                    tolerance=tolerance)

    def _maybe_cast_slice_bound(self, label, side, kind):
        if isinstance(label, pd.compat.string_types):
            parsed, resolution = _parse_iso8601_with_reso(self.date_type,
                                                          label)
            start, end = _parsed_string_to_bounds(self.date_type, resolution,
                                                  parsed)
            if self.is_monotonic_decreasing and len(self):
                return end if side == 'left' else start
            return start if side == 'left' else end
        else:
            return label

    # TODO: Add ability to use integer range outside of iloc?
    # e.g. series[1:5].
    def get_value(self, series, key):
        if not isinstance(key, slice):
            return series.iloc[self.get_loc(key)]
        else:
            return series.iloc[self.slice_indexer(
                key.start, key.stop, key.step)]

    def __contains__(self, key):
        try:
            result = self.get_loc(key)
            return isscalar(result) or type(result) == slice or np.any(result)
        except (KeyError, TypeError, ValueError):
            return False
