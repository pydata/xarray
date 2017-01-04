"""String formatting routines for __repr__.

For the sake of sanity, we only do internal formatting with unicode, which can
be returned by the __unicode__ special method. We use ReprMixin to provide the
__repr__ method so that things can work on Python 2.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime, timedelta
import functools

import numpy as np
import pandas as pd
from pandas.tslib import OutOfBoundsDatetime

from .options import OPTIONS
from .pycompat import (
    PY2, unicode_type, bytes_type, dask_array_type, OrderedDict, basestring)


def pretty_print(x, numchars):
    """Given an object `x`, call `str(x)` and format the returned string so
    that it is numchars long, padding with trailing spaces or truncating with
    ellipses as necessary
    """
    s = maybe_truncate(x, numchars)
    return s + ' ' * max(numchars - len(s), 0)


def maybe_truncate(obj, maxlen=500):
    s = unicode_type(obj)
    if len(s) > maxlen:
        s = s[:(maxlen - 3)] + u'...'
    return s


def wrap_indent(text, start='', length=None):
    if length is None:
        length = len(start)
    indent = '\n' + ' ' * length
    return start + indent.join(x for x in text.splitlines())


def ensure_valid_repr(string):
    """Ensure that the given value is valid for the result of __repr__.

    On Python 2, this means we need to convert unicode to bytes. We won't need
    this function once we drop Python 2.7 support.
    """
    if PY2 and isinstance(string, unicode):
        string = string.encode('utf-8')
    return string


class ReprMixin(object):
    """Mixin that defines __repr__ for a class that already has __unicode__."""
    def __repr__(self):
        return ensure_valid_repr(self.__unicode__())


def _get_indexer_at_least_n_items(shape, n_desired):
    assert 0 < n_desired <= np.prod(shape)
    cum_items = np.cumprod(shape[::-1])
    n_steps = np.argmax(cum_items >= n_desired)
    stop = int(np.ceil(float(n_desired) / np.r_[1, cum_items][n_steps]))
    indexer = ((0, ) * (len(shape) - 1 - n_steps) + (slice(stop), ) +
               (slice(None), ) * n_steps)
    return indexer


def first_n_items(x, n_desired):
    """Returns the first n_desired items of an array"""
    # Unfortunately, we can't just do x.flat[:n_desired] here because x might
    # not be a numpy.ndarray. Moreover, access to elements of x could be very
    # expensive (e.g. if it's only available over DAP), so go out of our way to
    # get them in a single call to __getitem__ using only slices.
    if n_desired < 1:
        raise ValueError('must request at least one item')

    if x.size == 0:
        # work around for https://github.com/numpy/numpy/issues/5195
        return []

    if n_desired < x.size:
        indexer = _get_indexer_at_least_n_items(x.shape, n_desired)
        x = x[indexer]
    return np.asarray(x).flat[:n_desired]


def last_item(x):
    """Returns the last item of an array"""
    if x.size == 0:
        # work around for https://github.com/numpy/numpy/issues/5195
        return []

    indexer = (slice(-1, None), ) * x.ndim
    return np.array(x[indexer], ndmin=1)


def format_timestamp(t):
    """Cast given object to a Timestamp and return a nicely formatted string"""
    # Timestamp is only valid for 1678 to 2262
    try:
        datetime_str = unicode_type(pd.Timestamp(t))
    except OutOfBoundsDatetime:
        datetime_str = unicode_type(t)

    try:
        date_str, time_str = datetime_str.split()
    except ValueError:
        # catch NaT and others that don't split nicely
        return datetime_str
    else:
        if time_str == '00:00:00':
            return date_str
        else:
            return '%sT%s' % (date_str, time_str)


def format_timedelta(t, timedelta_format=None):
    """Cast given object to a Timestamp and return a nicely formatted string"""
    timedelta_str = unicode_type(pd.Timedelta(t))
    try:
        days_str, time_str = timedelta_str.split(' days ')
    except ValueError:
        # catch NaT and others that don't split nicely
        return timedelta_str
    else:
        if timedelta_format == 'date':
            return days_str + ' days'
        elif timedelta_format == 'time':
            return time_str
        else:
            return timedelta_str


def format_item(x, timedelta_format=None, quote_strings=True):
    """Returns a succinct summary of an object as a string"""
    if isinstance(x, (np.datetime64, datetime)):
        return format_timestamp(x)
    if isinstance(x, (np.timedelta64, timedelta)):
        return format_timedelta(x, timedelta_format=timedelta_format)
    elif isinstance(x, (unicode_type, bytes_type)):
        return repr(x) if quote_strings else x
    elif isinstance(x, (float, np.float)):
        return u'{0:.4}'.format(x)
    else:
        return unicode_type(x)


def format_items(x):
    """Returns a succinct summaries of all items in a sequence as strings"""
    x = np.asarray(x)
    timedelta_format = 'datetime'
    if np.issubdtype(x.dtype, np.timedelta64):
        x = np.asarray(x, dtype='timedelta64[ns]')
        day_part = (x[~pd.isnull(x)]
                    .astype('timedelta64[D]')
                    .astype('timedelta64[ns]'))
        time_needed = x != day_part
        day_needed = day_part != np.timedelta64(0, 'ns')
        if np.logical_not(day_needed).all():
            timedelta_format = 'time'
        elif np.logical_not(time_needed).all():
            timedelta_format = 'date'

    formatted = [format_item(xi, timedelta_format) for xi in x]
    return formatted


def format_array_flat(items_ndarray, max_width):
    """Return a formatted string for as many items in the flattened version of
    items_ndarray that will fit within max_width characters
    """
    # every item will take up at least two characters, but we always want to
    # print at least one item
    max_possibly_relevant = max(int(np.ceil(max_width / 2.0)), 1)
    relevant_items = first_n_items(items_ndarray, max_possibly_relevant)
    pprint_items = format_items(relevant_items)

    cum_len = np.cumsum([len(s) + 1 for s in pprint_items]) - 1
    if (max_possibly_relevant < items_ndarray.size or
            (cum_len > max_width).any()):
        end_padding = u' ...'
        count = max(np.argmax((cum_len + len(end_padding)) > max_width), 1)
        pprint_items = pprint_items[:count]
    else:
        end_padding = u''

    pprint_str = u' '.join(pprint_items) + end_padding
    return pprint_str


def _summarize_var_or_coord(name, var, col_width, show_values=True,
                            marker=' ', max_width=None):
    if max_width is None:
        max_width = OPTIONS['display_width']
    first_col = pretty_print(u'  %s %s ' % (marker, name), col_width)
    if var.dims:
        dims_str = u'(%s) ' % u', '.join(map(unicode_type, var.dims))
    else:
        dims_str = u''
    front_str = u'%s%s%s ' % (first_col, dims_str, var.dtype)
    if show_values:
        values_str = format_array_flat(var, max_width - len(front_str))
    else:
        values_str = u'...'

    return front_str + values_str


def _summarize_dummy_var(name, col_width, marker=u'o', values=u'-'):
    """Used if there is no coordinate for a dimension."""
    first_col = pretty_print(u'  %s %s ' % (marker, name), col_width)
    dims_str = u'(%s) ' % unicode_type(name)
    return u'%s%s%s' % (first_col, dims_str, values)


def _summarize_coord_multiindex(coord, col_width, marker):
    first_col = pretty_print(u'  %s %s ' % (marker, coord.name), col_width)
    return u'%s(%s) MultiIndex' % (first_col, unicode_type(coord.dims[0]))


def _summarize_coord_levels(coord, col_width, marker=u'-'):
    relevant_coord = coord[:30]
    return u'\n'.join(
        [_summarize_var_or_coord(lname,
                                 relevant_coord.get_level_variable(lname),
                                 col_width, marker=marker)
         for lname in coord.level_names])


def _not_remote(var):
    """Helper function to identify if array is positively identifiable as
    coming from a remote source.
    """
    source = var.encoding.get('source')
    if source and source.startswith('http') and not var._in_memory:
        return False
    return True


def summarize_var(name, var, col_width):
    show_values = _not_remote(var)
    return _summarize_var_or_coord(name, var, col_width, show_values)


def summarize_coord(name, var, col_width):
    if var is None:
        return _summarize_dummy_var(name, col_width)
    is_index = name in var.dims
    show_values = is_index or _not_remote(var)
    marker = u'*' if is_index else u' '
    if is_index:
        coord = var.variable.to_index_variable()
        if coord.level_names is not None:
            return u'\n'.join(
                [_summarize_coord_multiindex(coord, col_width, marker),
                 _summarize_coord_levels(coord, col_width)])
    return _summarize_var_or_coord(name, var, col_width, show_values, marker)


def summarize_attr(key, value, col_width=None):
    # ignore col_width for now to more clearly distinguish attributes
    return u'    %s: %s' % (key, maybe_truncate(value))


EMPTY_REPR = u'    *empty*'


def _get_col_items(mapping):
    """Get all column items to format, including both keys of `mapping`
    and MultiIndex levels if any.
    """
    from .variable import IndexVariable

    col_items = []
    for k, v in mapping.items():
        col_items.append(k)
        var = getattr(v, 'variable', v)
        if isinstance(var, IndexVariable):
            level_names = var.to_index_variable().level_names
            if level_names is not None:
                col_items += list(level_names)
    return col_items


def _calculate_col_width(col_items):
    max_name_length = (max(len(unicode_type(s)) for s in col_items)
                       if col_items else 0)
    col_width = max(max_name_length, 7) + 6
    return col_width


def _mapping_repr(mapping, title, summarizer, col_width=None):
    if col_width is None:
        col_width = _calculate_col_width(mapping)
    summary = [u'%s:' % title]
    if mapping:
        summary += [summarizer(k, v, col_width) for k, v in mapping.items()]
    else:
        summary += [EMPTY_REPR]
    return u'\n'.join(summary)


vars_repr = functools.partial(_mapping_repr, title=u'Data variables',
                              summarizer=summarize_var)


attrs_repr = functools.partial(_mapping_repr, title=u'Attributes',
                               summarizer=summarize_attr)


def coords_repr(coords, col_width=None):
    if col_width is None:
        col_width = _calculate_col_width(_get_col_items(coords))
    # augment coordinates to include markers for missing coordinates
    augmented_coords = OrderedDict(coords)
    for dim in coords.dims:
        if dim not in augmented_coords:
            augmented_coords[dim] = None
    return _mapping_repr(augmented_coords, title=u'Coordinates',
                         summarizer=summarize_coord, col_width=col_width)


def indexes_repr(indexes):
    summary = []
    for k, v in indexes.items():
        summary.append(wrap_indent(repr(v), '%s: ' % k))
    return u'\n'.join(summary)


def dim_summary(obj):
    elements = [u'%s: %s' % (k, v) for k, v in obj.sizes.items()]
    return u', '.join(elements)


def array_repr(arr):
    # used for DataArray, Variable and IndexVariable
    if hasattr(arr, 'name') and arr.name is not None:
        name_str = '%r ' % arr.name
    else:
        name_str = u''

    summary = [u'<xarray.%s %s(%s)>'
               % (type(arr).__name__, name_str, dim_summary(arr))]

    if isinstance(getattr(arr, 'variable', arr)._data, dask_array_type):
        summary.append(repr(arr.data))
    elif arr._in_memory or arr.size < 1e5:
        summary.append(repr(arr.values))
    else:
        summary.append(u'[%s values with dtype=%s]' % (arr.size, arr.dtype))

    if hasattr(arr, 'coords'):
        if arr.coords:
            summary.append(repr(arr.coords))

    if arr.attrs:
        summary.append(attrs_repr(arr.attrs))

    return u'\n'.join(summary)


def dataset_repr(ds):
    summary = [u'<xarray.%s>' % type(ds).__name__]

    col_width = _calculate_col_width(_get_col_items(ds))

    dims_start = pretty_print(u'Dimensions:', col_width)
    summary.append(u'%s(%s)' % (dims_start, dim_summary(ds)))

    summary.append(coords_repr(ds.coords, col_width=col_width))
    summary.append(vars_repr(ds.data_vars, col_width=col_width))
    if ds.attrs:
        summary.append(attrs_repr(ds.attrs))

    return u'\n'.join(summary)
