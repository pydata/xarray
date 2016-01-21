from datetime import datetime, timedelta
import functools

import numpy as np
import pandas as pd

from .options import OPTIONS
from .pycompat import iteritems, unicode_type, bytes_type, dask_array_type


def pretty_print(x, numchars):
    """Given an object `x`, call `str(x)` and format the returned string so
    that it is numchars long, padding with trailing spaces or truncating with
    ellipses as necessary
    """
    s = str(x)
    if len(s) > numchars:
        return s[:(numchars - 3)] + '...'
    else:
        return s + ' ' * (numchars - len(s))


def wrap_indent(text, start='', length=None):
    if length is None:
        length = len(start)
    indent = '\n' + ' ' * length
    return start + indent.join(x for x in text.splitlines())


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


def format_timestamp(t):
    """Cast given object to a Timestamp and return a nicely formatted string"""
    datetime_str = str(pd.Timestamp(t))
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
    timedelta_str = str(pd.Timedelta(t))
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
        return '{0:.4}'.format(x)
    else:
        return str(x)


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
        end_padding = ' ...'
        count = max(np.argmax((cum_len + len(end_padding)) > max_width), 1)
        pprint_items = pprint_items[:count]
    else:
        end_padding = ''

    pprint_str = ' '.join(pprint_items) + end_padding
    return pprint_str


def _summarize_var_or_coord(name, var, col_width, show_values=True,
                            marker=' ', max_width=None):
    if max_width is None:
        max_width = OPTIONS['display_width']
    first_col = pretty_print('  %s %s ' % (marker, name), col_width)
    dims_str = '(%s) ' % ', '.join(map(str, var.dims)) if var.dims else ''
    front_str = first_col + dims_str + ('%s ' % var.dtype)
    if show_values:
        values_str = format_array_flat(var, max_width - len(front_str))
    else:
        values_str = '...'
    return front_str + values_str


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
    is_index = name in var.dims
    show_values = is_index or _not_remote(var)
    marker = '*' if is_index else ' '
    return _summarize_var_or_coord(name, var, col_width, show_values, marker)


def _maybe_truncate(obj, maxlen=500):
    s = str(obj)
    if len(s) > maxlen:
        s = s[:(maxlen - 3)] + '...'
    return s


def summarize_attr(key, value, col_width=None):
    # ignore col_width for now to more clearly distinguish attributes
    return '    %s: %s' % (key, _maybe_truncate(value))


EMPTY_REPR = '    *empty*'


def _calculate_col_width(mapping):
    max_name_length = max(len(str(k)) for k in mapping) if mapping else 0
    col_width = max(max_name_length, 7) + 6
    return col_width


def _mapping_repr(mapping, title, summarizer, col_width=None):
    if col_width is None:
        col_width = _calculate_col_width(mapping)
    summary = ['%s:' % title]
    if mapping:
        summary += [summarizer(k, v, col_width) for k, v in mapping.items()]
    else:
        summary += [EMPTY_REPR]
    return '\n'.join(summary)


coords_repr = functools.partial(_mapping_repr, title='Coordinates',
                                summarizer=summarize_coord)


vars_repr = functools.partial(_mapping_repr, title='Data variables',
                              summarizer=summarize_var)


attrs_repr = functools.partial(_mapping_repr, title='Attributes',
                               summarizer=summarize_attr)


def indexes_repr(indexes):
    summary = []
    for k, v in indexes.items():
        summary.append(wrap_indent(repr(v), '%s: ' % k))
    return '\n'.join(summary)


def array_repr(arr):
    # used for DataArray, Variable and Coordinate
    if hasattr(arr, 'name') and arr.name is not None:
        name_str = '%r ' % arr.name
    else:
        name_str = ''
    dim_summary = ', '.join('%s: %s' % (k, v) for k, v
                            in zip(arr.dims, arr.shape))

    summary = ['<xarray.%s %s(%s)>'
               % (type(arr).__name__, name_str, dim_summary)]

    if isinstance(getattr(arr, 'variable', arr)._data, dask_array_type):
        summary.append(repr(arr.data))
    elif arr._in_memory or arr.size < 1e5:
        summary.append(repr(arr.values))
    else:
        summary.append('[%s values with dtype=%s]' % (arr.size, arr.dtype))

    if hasattr(arr, 'coords'):
        if arr.coords:
            summary.append(repr(arr.coords))

    if arr.attrs:
        summary.append(attrs_repr(arr.attrs))

    return '\n'.join(summary)


def dataset_repr(ds):
    summary = ['<xarray.%s>' % type(ds).__name__]

    col_width = _calculate_col_width(ds)

    dims_start = pretty_print('Dimensions:', col_width)
    all_dim_strings = ['%s: %s' % (k, v) for k, v in iteritems(ds.dims)]
    summary.append('%s(%s)' % (dims_start, ', '.join(all_dim_strings)))

    summary.append(coords_repr(ds.coords, col_width=col_width))
    summary.append(vars_repr(ds.data_vars, col_width=col_width))
    if ds.attrs:
        summary.append(attrs_repr(ds.attrs))

    return '\n'.join(summary)
